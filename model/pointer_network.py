import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
from model.ISAB import ISAB
from model.position import PositionalEncoding

# 假设这些模块在指定的路径下
# from utils.grouping import grouping
# from model.ISAB import ISAB
import pdb # 用于调试

class MAB(nn.Module):
    """
    Multihead Attention Block (MAB)
    实现多头注意力机制。
    """
    def __init__(self, d_model=512, n_heads=2, d_k=64, d_v=64):
        """
        初始化 MAB 层。
        :param int d_model: 输入和输出的特征维度。
        :param int n_heads: 注意力头的数量。
        :param int d_k: 每个注意力头中 Key 和 Query 的维度。
        :param int d_v: 每个注意力头中 Value 的维度。
        """
        super(MAB, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.C = 10 # 注意力分数的缩放因子

        # 定义线性变换层
        self.W_q = nn.Linear(d_model, n_heads * d_k)
        self.W_k = nn.Linear(d_model, n_heads * d_k)
        self.W_v = nn.Linear(d_model, n_heads * d_v)
        self.W_o = nn.Linear(n_heads * d_v, d_model) # 输出线性层

        # 缩放因子，用于稳定梯度
        self.scale = math.sqrt(d_k)

    def forward(self, Q, K, V, mask):
        """
        MAB 的前向传播。
        :param Tensor Q: Query 张量，形状 [B, 1, D] 或 [B, L_q, D]
        :param Tensor K: Key 张量，形状 [B, L_k, D]
        :param Tensor V: Value 张量，形状 [B, L_k, D]
        :param ByteTensor mask: 掩码张量，形状 [B, L_k]，标记哪些 Key/Value 需要被忽略。
        :return: 输出张量 [B, 1, D] 或 [B, L_q, D]，注意力权重 [B, h, L_q, L_k]
        """
        batch_size = Q.size(0)
        len_q = Q.size(1) # Query 的序列长度

        # 1. 线性变换并分头
        # Q: [B, L_q, D] -> [B, L_q, h*d_k] -> [B, h, L_q, d_k]
        Q_proj = self.W_q(Q).view(batch_size, len_q, self.n_heads, self.d_k).transpose(1, 2)
        # K: [B, L_k, D] -> [B, L_k, h*d_k] -> [B, h, L_k, d_k]
        K_proj = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V: [B, L_k, D] -> [B, L_k, h*d_v] -> [B, h, L_k, d_v]
        V_proj = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # 2. 计算注意力分数
        # scores: [B, h, L_q, d_k] * [B, h, d_k, L_k] -> [B, h, L_q, L_k]
        scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / self.scale
        # 使用 tanh 激活并乘以常数 C
        scores = torch.tanh(scores) * self.C
        
        
        # 3. 应用掩码
        if mask is not None:
            # mask: [B, L_k] -> [B, 1, 1, L_k] -> [B, h, L_q, L_k] (通过广播)
            mask = mask.unsqueeze(1).unsqueeze(2).expand_as(scores)
            # 将掩码位置的分数设置为一个非常小的值
            scores = scores.masked_fill(mask, -10) # -1e9 是一个很小的值，避免数值不稳定


        # 4. 计算注意力权重 (Softmax)
        # attn: [B, h, L_q, L_k]
        #attn = F.softmax(scores, dim=-1)
        attn = F.log_softmax(scores, dim=-1) 
        #pdb.set_trace()

        # 5. 计算加权和 (输出)
        # output: [B, h, L_q, L_k] * [B, h, L_k, d_v] -> [B, h, L_q, d_v]
        output = torch.matmul(attn, V_proj)

        # 6. 合并多头并进行最终线性变换
        # output: [B, h, L_q, d_v] -> [B, L_q, h, d_v] -> [B, L_q, h*d_v]
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        # output: [B, L_q, h*d_v] -> [B, L_q, D]
        output = self.W_o(output)

        # 如果 L_q == 1，则移除该维度
        if len_q == 1:
            output = output.squeeze(1)
            attn = attn.squeeze(2) # attn: [B, h, L_k]

        return output, attn

class PointerEncoder(nn.Module):
    """
    Pointer Network 的编码器部分。
    使用 LSTM 对输入序列进行编码。
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout, bidir):
        """
        初始化编码器。
        :param int embedding_dim: 输入嵌入的维度。
        :param int hidden_dim: LSTM 隐藏层的维度。
        :param int n_layers: LSTM 的层数。
        :param float dropout: LSTM 层之间的 dropout 概率。
        :param bool bidir: 是否使用双向 LSTM。
        """
        super(PointerEncoder, self).__init__()
        # 如果是双向 LSTM，实际隐藏维度需要减半，层数加倍
        self.hidden_dim = hidden_dim // 2 if bidir else hidden_dim
        self.n_layers = n_layers * 2 if bidir else n_layers
        self.bidir = bidir

        # 定义 LSTM 层
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers, # 注意：这里传入的是原始层数，bidir 会在内部处理
                            dropout=dropout,
                            bidirectional=bidir,
                            batch_first=True) # 输入和输出张量的第一个维度是 batch

        # 初始化隐藏状态和细胞状态的参数（设为不可训练）
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs, hidden):
        """
        编码器的前向传播。
        :param Tensor embedded_inputs: 嵌入后的输入序列，形状 [B, L, D_emb]。
        :param tuple(Tensor, Tensor) hidden: 初始的隐藏状态和细胞状态 (h0, c0)。
        :return: LSTM 的输出序列 [B, L, D_hid * num_directions] 和最后的隐藏状态 (h_n, c_n)。
        """
        # 设置默认数据类型为 float64 (根据原代码保留，但通常 float32 足够)
        # torch.set_default_dtype(torch.float64) # 建议在模型外部或训练脚本中设置

        # LSTM 前向传播
        outputs, hidden = self.lstm(embedded_inputs, hidden)

        return outputs, hidden

    def init_hidden(self, embedded_inputs):
        """
        初始化 LSTM 的隐藏状态和细胞状态。
        :param Tensor embedded_inputs: 嵌入后的输入序列，用于获取 batch_size。
        :return: tuple(Tensor, Tensor) 初始化的隐藏状态 (h0) 和细胞状态 (c0)。
                 形状为 [num_layers * num_directions, B, D_hid]。
        """
        batch_size = embedded_inputs.size(0)

        # 扩展 h0 和 c0 以匹配 LSTM 输入所需的形状
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.c0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0

class PointerAttention(nn.Module):
    """
    Pointer Network 的注意力机制。
    用于计算解码器在每个时间步对编码器输出的注意力权重。
    """
    def __init__(self, input_dim, hidden_dim):
        """
        初始化注意力模块。
        :param int input_dim: 输入特征维度（通常是编码器/解码器的隐藏维度）。
        :param int hidden_dim: 注意力机制内部的隐藏维度。
        """
        super(PointerAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 线性层，用于变换解码器隐藏状态
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        # 1x1 卷积层（等效于线性层），用于变换编码器输出
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, kernel_size=1, stride=1)
        # 注意力向量 V
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        # 用于掩码的负无穷大值
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        # 使用 log_Softmax 
        self.soft = nn.Softmax(dim=1)
        #self.soft = log_softmax(dim=1) # 对序列长度维度进行 Softmax

        # 初始化注意力向量 V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, decoder_hidden, context, mask):
        """
        注意力机制的前向传播。
        :param Tensor decoder_hidden: 当前解码步的隐藏状态，形状 [B, D_hid]。
        :param Tensor context: 编码器的输出序列（作为注意力上下文），形状 [B, L, D_enc]。
        :param ByteTensor mask: 掩码张量，形状 [B, L]，标记哪些上下文位置已被选择或无效。
        :return: tuple: (加权的上下文向量 [B, D_hid], 注意力权重 alpha [B, L])
        """
        # decoder_hidden: (B, D_hid)
        # context: (B, L, D_enc) - 假设 D_enc == D_hid
        batch_size = context.size(0)
        seq_len = context.size(1)

        # 1. 变换解码器隐藏状态
        # inp: (B, D_hid) -> (B, hidden_dim) -> (B, hidden_dim, 1) -> (B, hidden_dim, L)
        inp = self.input_linear(decoder_hidden).unsqueeze(2).expand(-1, -1, seq_len)

        # 2. 变换编码器输出 (上下文)
        # context: (B, L, D_enc) -> (B, D_enc, L)
        context_permuted = context.permute(0, 2, 1)
        # ctx: (B, D_enc, L) -> (B, hidden_dim, L)
        ctx = self.context_linear(context_permuted)

        # 3. 计算注意力能量 (Bahdanau-style attention)
        # V: (hidden_dim) -> (1, hidden_dim) -> (B, 1, hidden_dim)
        V_expanded = self.V.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        # att: (B, 1, hidden_dim) * tanh( (B, hidden_dim, L) + (B, hidden_dim, L) ) -> (B, 1, L) -> (B, L)
        att_energy = torch.bmm(V_expanded, torch.tanh(inp + ctx)).squeeze(1)

        # 4. 应用掩码
        # 注意：原代码中 mask 的判断和赋值方式可能存在问题
        # if len(att_energy[mask]) > 0: # 检查 mask 是否非空
        #     att_energy[mask] = self.inf[mask] # 使用预定义的 inf 填充
        # 更健壮的方式是直接使用 masked_fill
        if mask is not None:
             att_energy = att_energy.masked_fill(mask, -10) # 直接填充负无穷

        # 5. 计算注意力权重 (Softmax)
        # alpha: (B, L)
        alpha = self.soft(att_energy)

        # 6. 计算加权的上下文向量
        # weighted_context: (B, hidden_dim, L) * (B, L, 1) -> (B, hidden_dim, 1) -> (B, hidden_dim)
        # 注意：这里使用变换后的 ctx 而不是原始 context
        weighted_context = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return weighted_context, alpha

    def init_inf(self, mask_size):
        """
        初始化用于掩码的负无穷大张量。
        :param tuple mask_size: 掩码张量的目标尺寸 (B, L)。
        """
        # 确保 self.inf 具有正确的形状和设备
        if not hasattr(self, 'inf') or self.inf.size() != mask_size:
             self.inf = self._inf.expand(*mask_size).to(self.V.device) # 移动到正确的设备


class PointerDecoder(nn.Module):
    """
    Pointer Network 的解码器部分。
    使用 LSTM 和注意力机制在每个时间步选择输入序列中的一个元素。
    """
    def __init__(self, embedding_dim, hidden_dim, masking=True, output_length=None, dropout=0.2):
        """
        初始化解码器。
        :param int embedding_dim: 输入嵌入的维度。
        :param int hidden_dim: 解码器 LSTM 的隐藏维度。
        :param bool masking: 是否在解码过程中屏蔽已选择的元素。
        :param int output_length: 指定解码的固定步数，如果为 None，则解码步数等于输入长度。
        :param float dropout: （原代码未使用 dropout，保留参数以备将来使用）
        """
        super(PointerDecoder, self).__init__()
        self.masking = masking
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_length = output_length # 解码步数

        # LSTM 单元的线性层 (模拟 LSTMCell)
        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim) # 输入门、遗忘门、细胞门、输出门
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim) # 隐藏状态到门的连接

        # 结合注意力和隐藏状态的输出层
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim) # (attention_context + hidden_state) -> new_hidden_state

        # 注意力模块
        self.att = PointerAttention(hidden_dim, hidden_dim)
        # 多头注意力块 (MAB) - 替代原有的 LSTM + Attention 步骤
        self.MAB = MAB(d_model=hidden_dim, n_heads=1, d_k=256, d_v=256) # 注意力头数为 1

        # 不可训练的参数，用于掩码和索引
        self.mask = Parameter(torch.ones(1), requires_grad=False) # 初始掩码值
        self.runner = Parameter(torch.zeros(1), requires_grad=False) # 用于生成索引
        self.C = 10

    def forward(self, embedded_inputs, decoder_input, hidden, context):
        """
        解码器的前向传播。
        :param Tensor embedded_inputs: 编码器输入的嵌入，形状 [B, L, D_emb]。
        :param Tensor decoder_input: 解码器的初始输入（或上一步选择的元素的嵌入），形状 [B, D_emb]。
        :param tuple(Tensor, Tensor) hidden: 解码器的初始隐藏状态 (h0, c0)，形状 [B, D_hid]。
        :param Tensor context: 编码器的输出序列（注意力上下文），形状 [B, L, D_enc]。
        :return: tuple: ((输出概率 [B, T, L], 选择的索引 [B, T]), 最后的隐藏状态 (h_n, c_n))
        """
        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # --- 初始化 ---
        # 动态确定解码步数
        current_output_length = self.output_length if self.output_length is not None else input_length

        # 初始化掩码张量 (全 1，表示所有位置都可选)
        # mask: [B, L]
        current_mask = self.mask.repeat(batch_size, input_length).to(embedded_inputs.device)

        # 初始化注意力模块的负无穷大张量
        # self.att.init_inf(current_mask.size()) # 如果使用 PointerAttention 需要初始化

        # 生成索引序列 [0, 1, ..., L-1] 并扩展到 batch_size
        # runner: [B, L]
        runner_base = torch.arange(input_length, device=embedded_inputs.device, dtype=torch.long)
        runner = runner_base.unsqueeze(0).expand(batch_size, -1)

        outputs = [] # 存储每一步的注意力分布 (概率)
        pointers = [] # 存储每一步选择的索引

        # --- LSTM 解码步骤 (原代码中的一个选项) ---
        def step_lstm(x, current_hidden, current_context, current_step_mask):
            """
            单个 LSTM 解码步。
            :param Tensor x: 当前时间步的输入, [B, D_emb]。
            :param tuple(Tensor, Tensor) current_hidden: 上一时间步的隐藏状态 (h, c), [B, D_hid]。
            :param Tensor current_context: 编码器输出, [B, L, D_enc]。
            :param Tensor current_step_mask: 当前可选位置的掩码, [B, L]。
            :return: tuple: (下一隐藏状态 h_t [B, D_hid], 下一细胞状态 c_t [B, D_hid], 注意力权重 alpha [B, L])
            """
            h_prev, c_prev = current_hidden
            h_prev = h_prev.to(x.dtype)
            c_prev = c_prev.to(x.dtype)
            
            # 计算 LSTM 门控单元
            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h_prev)
            i, f, g, o = gates.chunk(4, 1) # input, forget, cell, output gates

            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g) # Candidate cell state
            o = torch.sigmoid(o)

            # 计算当前细胞状态和隐藏状态
            c_t = (f * c_prev) + (i * g)
            h_t = o * torch.tanh(c_t)

            # 计算注意力
            # 注意：这里的 mask 需要反转，PointerAttention 期望 mask=True 的位置被忽略
            attention_mask = (current_step_mask == 0) # True 表示被屏蔽
            # self.att.init_inf(attention_mask.size()) # 确保 inf 张量尺寸正确
            attention_weighted_context, alpha = self.att(h_t, current_context, attention_mask)
            
            alpha = F.log_softmax(alpha, dim=1) # 对注意力权重进行 log_softmax
            # 结合注意力和隐藏状态得到最终输出隐藏状态
            combined_hidden = torch.cat((attention_weighted_context, h_t), dim=1)
            output_hidden = torch.tanh(self.hidden_out(combined_hidden)) *self.C

            # 返回新的隐藏状态和注意力权重
            # 注意：原代码返回的是 output_hidden, c_t, alpha。但 LSTM 步通常只更新 h 和 c。
            # 如果要用 output_hidden 作为下一步的 h，需要修改。这里遵循标准 LSTM。
            return h_t, c_t, alpha # 返回标准 LSTM 的 h_t, c_t 和计算出的 alpha

        # --- MHA 解码步骤 (原代码中实际使用的选项) ---
        def step_mha(query_input, current_context, current_step_mask):
            """
            单个 Multihead Attention 解码步。
            :param Tensor query_input: 当前时间步的查询输入 (通常是上一步隐藏状态或特定输入), [B, D_hid]。
            :param Tensor current_context: 编码器输出 (作为 Key 和 Value), [B, L, D_enc]。
            :param Tensor current_step_mask: 当前可选位置的掩码, [B, L]。
            :return: Tensor 注意力权重 alpha [B, L]。
            """
            # 将输入 reshape 成 MAB 需要的 Query 格式 [B, 1, D]
            query = query_input.unsqueeze(1)
            # 注意：MAB 的 mask 含义与 PointerAttention 相反，mask=True 表示忽略
            # 因此直接使用 current_step_mask == 0
            mha_mask = (current_step_mask == 0)

            # 使用 MAB 计算输出和注意力权重
            # 输出 output_mha 形状 [B, 1, D_hid], alpha_mha 形状 [B, h, 1, L] (h=1)
            output_mha, alpha_mha = self.MAB(
                Q=query,
                K=current_context,
                V=current_context,
                mask=mha_mask # True 表示要屏蔽的位置
            )

            # 提取注意力权重并调整形状
            # alpha_mha: [B, 1, 1, L] -> [B, L]
            alpha = alpha_mha.squeeze(1).squeeze(1)
            return alpha # 只返回注意力权重

        # --- 解码循环 ---
        current_decoder_input = decoder_input # 初始化解码器输入
        current_hidden = hidden # 初始化隐藏状态 (如果使用 LSTM)

       
        for t in range(current_output_length):
            # --- 选择解码方式 ---
            # 方式一：使用 LSTM + Attention
            h_t, c_t, step_output_probs = step_lstm(current_decoder_input, current_hidden, context, current_mask)
            current_hidden = (h_t, c_t) # 更新隐藏状态

            # 方式二：使用 MHA (Multihead Attention Block) - 原代码实际使用的
            #step_output_probs = step_mha(current_decoder_input, context, current_mask) # MHA 的输入是 current_decoder_input
           

            # --- 处理当前步的输出 ---
            # 将已被屏蔽的位置的概率设置为 0 (或负无穷，取决于后续操作)
            # masked_outs: [B, L]
            masked_outs = step_output_probs.masked_fill(current_mask == 0, -1e9) # 用负无穷填充，配合 max
            #masked_outs = step_output_probs * current_mask # 直接乘以掩码 (0 或 1)

            # 找到概率最大的索引
            # max_probs: [B], indices: [B]
            max_probs, indices = masked_outs.max(dim=1)

            # --- 更新掩码和下一步输入 ---
            # 创建当前选中索引的 one-hot 编码
            # one_hot_pointers: [B, L], 在选中索引位置为 1，其余为 0
            one_hot_pointers = (runner == indices.unsqueeze(1)).float()

            # 更新掩码，1表示未选择,0表示已选择
            if self.masking:
                current_mask = current_mask * (1 - one_hot_pointers)

            # 获取下一步的解码器输入 (即当前选中元素的嵌入)
            # embedding_mask: [B, L, D_emb], 在选中索引位置为 True
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).bool()
            # next_decoder_input: [B, D_emb]
            # 使用 masked_select 比直接索引更安全
            next_decoder_input = embedded_inputs[embedding_mask].view(batch_size, self.embedding_dim) # 可能因 batch 中 mask 情况不同而出错

            # --- 原代码中计算未选中节点嵌入的逻辑 ---
            # 获取未选中位置的掩码 [B,L,D_emb] True表示未选中,False表示选中
            no_select_mask_bool = (current_mask == 1).unsqueeze(2).expand(-1, -1, self.embedding_dim).bool()
            
            
            # 下一个时刻t的嵌入 True表示选中,False表示未选中
            t_embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).bool()
            next_t_embeds = embedded_inputs[t_embedding_mask].view(batch_size, self.embedding_dim)

            # 未选择节点的嵌入 
            no_selected_embeds = embedded_inputs[no_select_mask_bool.data].view(batch_size, -1, self.embedding_dim)
            # 最大池化
            if no_selected_embeds.size(1) > 0:
                not_select_embed = F.max_pool1d(no_selected_embeds.transpose(1,2),kernel_size=no_selected_embeds.size(1)).squeeze(-1)
            else:
                not_select_embed = torch.zeros(batch_size, self.embedding_dim).to(embedded_inputs.device)
           
            # 更新解码器输入为未选中节点的最大池化嵌入
            current_decoder_input =  next_t_embeds


            # 存储当前步的结果
            outputs.append(step_output_probs.unsqueeze(1)) # 存储概率分布 [B, 1, L]
            pointers.append(indices.unsqueeze(1)) # 存储选中的索引 [B, 1]
            

    
        # --- 循环结束 ---
        # 合并所有时间步的结果
        # outputs: [T, B, 1, L] -> [B, T, L]
        outputs = torch.cat(outputs, dim=1)
        # pointers: [T, B, 1] -> [B, T]
        pointers = torch.cat(pointers, dim=1)

        # 返回最终结果和最后的隐藏状态 (如果使用 LSTM)
        # 注意：如果使用 MHA，没有明确的“最后隐藏状态”传递
        final_hidden = current_hidden if 'current_hidden' in locals() else None # 只有 LSTM 模式有 final_hidden

        return (outputs, pointers), final_hidden


class PointerNetwork(nn.Module):
    """
    完整的 Pointer Network 模型。
    包含 Embedding 层、Encoder、Decoder。
    """
    def __init__(self,
                 elem_dims,        # 输入元素本身的维度 (如果不用 embedding dict)
                 embedding_dim,    # 嵌入维度
                 hidden_dim,       # Encoder/Decoder 隐藏层维度
                 lstm_layers,      # Encoder LSTM 层数
                 dropout,          # Dropout 概率
                 bidir=False,      # Encoder 是否双向
                 masking=True,     # Decoder 是否屏蔽已选项
                 output_length=None, # 固定输出长度 (否则等于输入长度)
                 embedding_by_dict=False, # 是否使用 Embedding 字典 (针对离散输入)
                 embedding_by_dict_size=None,# Embedding 字典大小
                 max_seq_len=100
                 
                 ): 
        """
        初始化 Pointer Network 模型。
        参数含义见类注释。
        """
        super(PointerNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.masking = masking
        self.output_length = output_length
        self.embedding_by_dict = embedding_by_dict
        self.embedding_by_dict_size = embedding_by_dict_size
        

        # --- 可选的注意力机制 (在 Encoder 之前或替代 Encoder) ---
        self.MAB = nn.MultiheadAttention(embedding_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.ISAB = ISAB(embedding_dim, embedding_dim, 1, 1) # 假设 ISAB 存在

        # --- Embedding 层 ---
        if embedding_by_dict:
            assert embedding_by_dict_size is not None, "embedding_by_dict_size must be provided if embedding_by_dict is True"
            self.embedding = nn.Embedding(embedding_by_dict_size, embedding_dim)
        else:
            self.embedding = nn.Linear(elem_dims, embedding_dim)


        # --- 位置编码层 ---
        # **** 实例化 PositionalEncoding ****
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len=max_seq_len)

        # --- Encoder ---
        # 注意：原代码在 forward 中直接使用了 ISAB 或 MAB，没有使用 PointerEncoder
        # 如果需要使用 LSTM Encoder，取消下面的注释
        self.encoder = PointerEncoder(embedding_dim,
                                      hidden_dim,
                                      lstm_layers,
                                      dropout,
                                      bidir)

        # --- Decoder ---
        self.decoder = PointerDecoder(embedding_dim, hidden_dim,
                                      masking=self.masking,
                                      output_length=self.output_length)

        # --- Decoder 的初始输入 ---
        # 可训练或不可训练的初始输入向量
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim), requires_grad=False)
        # 初始化 decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        Pointer Network 的前向传播。
        :param Tensor inputs: 输入序列，形状 [B, L, D_elem] 或 [B, L] (如果 embedding_by_dict=True)。
        :return: tuple: (输出概率 [B, T, L], 选择的索引 [B, T])
        """
        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        # --- 1. Embedding ---
        # 根据输入类型调整形状
        if self.embedding_by_dict:
            # 输入是索引 [B, L]
            embedded_inputs = self.embedding(inputs.long()) # [B, L, D_emb]
        else:
            input_flat = inputs.view(batch_size * input_length, -1)
            embedded_inputs = self.embedding(input_flat.float()).view(batch_size, input_length, -1)

        # --- Add Positional Encoding ---
        #embedded_coords: [B, L, embedding_dim] -> embedded_inputs: [B, L, embedding_dim]
        # embedded_inputs = self.pos_encoder(embedded_inputs)



        # --- 2. Encoder (或替代方案) ---
        # 原代码直接使用 ISAB 或 MAB 作为编码器输出
        # 选项 A: 使用 ISAB (假设已定义)
        #encoder_outputs = self.ISAB(embedded_inputs) # [B, L, D_hid]

        # 选项 B: 使用 MAB (MultiheadAttention)
        encoder_outputs, _ = self.MAB(
            query=embedded_inputs,
            key=embedded_inputs,
            value=embedded_inputs,
            need_weights=False,
            attn_mask=None 
        ) # [B, L, D_emb] -> [B, L, D_hid] (如果 D_emb == D_hid)

        # 选项 C: 使用 LSTM Encoder (如果定义了 self.encoder)
        # encoder_hidden_init = self.encoder.init_hidden(embedded_inputs)
        # encoder_outputs, encoder_hidden_final = self.encoder(embedded_inputs, encoder_hidden_init)
        # # encoder_outputs: [B, L, D_hid * num_directions]

        # **** 重要: 根据原代码逻辑，这里假设 encoder_outputs = embedded_inputs ****
        # **** 或者某个注意力模块的输出，其维度需要与 decoder 的 hidden_dim 匹配 ****
        # **** 这里暂时使用 embedded_inputs 作为 context，需要根据实际情况调整 ****
        # **** 如果 embedding_dim != hidden_dim，需要添加一个线性层转换 ****
        if self.embedding_dim != self.decoder.hidden_dim:
             if not hasattr(self, 'enc_out_proj'):
                  self.enc_out_proj = nn.Linear(self.embedding_dim, self.decoder.hidden_dim).to(inputs.device)
             encoder_outputs_for_decoder = self.enc_out_proj(embedded_inputs)
        else:
             encoder_outputs_for_decoder = embedded_inputs # 假设维度匹配

        # --- 3. Decoder 初始化 ---
        # 初始化 Decoder 的第一个输入
        # decoder_input0_batch: [B, D_emb]
        decoder_input0_batch = self.decoder_input0.unsqueeze(0).expand(batch_size, -1).to(inputs.device)

        # 初始化 Decoder 的隐藏状态 (对于 MHA Decoder，这更像是一个初始查询)
        # 原代码使用随机初始化的隐藏状态，这里遵循
        decoder_hidden_dim = self.decoder.hidden_dim
        
        device = inputs.device
        dtype = inputs.dtype
        # 注意：如果 Decoder 使用 LSTM，需要 (h0, c0)
        # 如果 Decoder 仅使用 MHA，则可能只需要一个初始状态向量
        # decoder_hidden0 = (torch.randn(batch_size, decoder_hidden_dim, device=device, dtype=dtype),
        #                    torch.randn(batch_size, decoder_hidden_dim, device=device, dtype=dtype))
        # 根据 Decoder 的 step_mha，它需要一个 [B, D_hid] 的输入，而不是 (h,c)
        # 因此，初始化一个与 decoder_input0_batch 类似的随机状态
        # 或者直接使用 decoder_input0_batch 作为 MHA 的第一个查询输入
        # 这里我们遵循原代码，创建一个随机的初始 "隐藏状态" 作为 MHA 的第一个查询
        initial_mha_query = torch.randn(batch_size, decoder_hidden_dim, device=device, dtype=dtype)
        # 注意：Decoder 的 forward 函数期望 hidden 参数，即使它在 MHA 模式下可能不直接使用 h,c
        # 为了兼容性，我们传递一个虚拟的元组
        decoder_hidden0_dummy = (initial_mha_query, torch.zeros_like(initial_mha_query)) # (h_dummy, c_dummy)

        # --- 4. Decoder 前向传播 ---
        # (outputs, pointers), decoder_hidden_final = self.decoder(
        #     embedded_inputs=embedded_inputs,          # 用于查找下一步输入的嵌入
        #     decoder_input=decoder_input0_batch,       # 解码器第一个时间步的输入
        #     hidden=decoder_hidden0_dummy,             # 解码器初始状态 (或 MHA 的初始查询包装)
        #     context=encoder_outputs_for_decoder       # 编码器输出 (注意力上下文)
        # )
        # **** 根据 Decoder 的 step_mha 实现，它使用 decoder_input 作为查询 ****
        # **** 因此，初始查询应该是 decoder_input0_batch ****
        (outputs, pointers), decoder_hidden_final = self.decoder(
            embedded_inputs=embedded_inputs,
            decoder_input=decoder_input0_batch, # 使用初始输入作为第一个查询
            hidden=decoder_hidden0_dummy,       # 传递虚拟状态
            context=encoder_outputs_for_decoder
        )


        return outputs.permute(0, 2, 1), pointers  #
