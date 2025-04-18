import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

from utils.grouping import grouping
from model.ISAB import ISAB
import pdb


class MAB(nn.Module):
    def __init__(self, d_model=512, n_heads=2, d_k=64, d_v=64):
        super(MAB, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        # 定义线性变换层
        self.W_q = nn.Linear(d_model, n_heads * d_k)
        self.W_k = nn.Linear(d_model, n_heads * d_k)
        self.W_v = nn.Linear(d_model, n_heads * d_v)
        self.W_o = nn.Linear(n_heads * d_v, d_model)
        
        # 缩放因子
        self.scale = math.sqrt(d_k)

    def forward(self, Q, K, V, mask):
        """
        :param Tensor Q: Query [B,1,D]
        :param Tensor K: Key [B,L,D]
        :param Tensor V: Value [B,L,D]
        :param ByteTensor mask: [B,L]
        :return: Output [B,1,D], Attention weights [B,L,L]
        """
        batch_size = Q.size(0)
        
        # 线性变换并分头
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [B,h,1,d_k]
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [B,h,L,d_k]
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # [B,h,L,d_v]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B,h,1,L]
        
        
        # 扩展mask维度以匹配scores
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B,h,1,L]
            scores = scores.masked_fill(mask, -1e9)
        
        
        # 计算注意力权重
        attn = F.softmax(scores, dim=-1)  # [B,h,1,L]
        
        # 计算输出
        output = torch.matmul(attn, V)  # [B,h,1,d_v]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)  # [B,1,h*d_v]
        output = self.W_o(output).squeeze(1)
        
        attn = attn.squeeze()  
       
        return output, attn
        
class PointerEncoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        Initiate Encoder
        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerEncoder, self).__init__()
        self.hidden_dim = hidden_dim // 2 if bidir else hidden_dim
        self.n_layers = n_layers * 2 if bidir else n_layers
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidir,
                            batch_first=True)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs, hidden):
        """
        Encoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """

        # if batch_first = True, not needed:
        # embedded_inputs = embedded_inputs.permute(1, 0, 2)
        torch.set_default_dtype(torch.float64)
        outputs, hidden = self.lstm(embedded_inputs, hidden)

        return outputs, hidden

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units
        :param Tensor embedded_inputs: The embedded input of Pointer-Net
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.c0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0

class PointerAttention(nn.Module):
    """
    Attention model for Pointer-Net.
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention
        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(PointerAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]),
                              requires_grad=False)
        self.soft = torch.nn.Softmax(dim=1)

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass
        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """
        # input: (batch, hidden)
        # context: (batch, seq_len, hidden)

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1,
                                                           context.size(1))

        # context: (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)

        # ctx: (batch, hidden_dim, seq_len)
        ctx = self.context_linear(context)

        # V: (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # att: (batch, seq_len)
        att = torch.bmm(V, torch.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]

        # alpha: (batch, seq_len)
        alpha = self.soft(att)

        # hidden_state: (batch, hidden)
        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)

class PointerDecoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim, masking=True,
                 output_length=None,
                 dropout=0.2):
        """
        Initiate Decoder
        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(PointerDecoder, self).__init__()
        self.masking = masking
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_length = output_length

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = PointerAttention(hidden_dim, hidden_dim)
        self.MAB = MAB(d_model=hidden_dim, n_heads=1, d_k=256, d_v=256)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context):
        """
        Decoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """
        
        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long() #[B,L]:[0,1,2...]
       

        outputs = []
        pointers = []

        def step_lstm(x, hidden):
            """
            Recurrence step function
            :param Tensor x: t 时刻的输入   [batch, embedding]
            :param tuple(Tensor, Tensor) hidden: t-1 时刻的隐藏状态 ([batch, hidden], [batch, hidden])
            :return: t 时刻的隐藏状态 (h, c), Attention probabilities (Alpha)
            """

            # 使用LSTM计算t时刻的隐藏状态
            h, c = hidden  
            # gates: (batch, hidden * 4)  
            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)

            # input, forget, cell, out: (batch, hidden)
            input, forget, cell, out = gates.chunk(4, 1)

            input = torch.sigmoid(input)    #输入门
            forget = torch.sigmoid(forget)  # 遗忘门
            cell = torch.tanh(cell)         #候选细胞状态
            out = torch.sigmoid(out)        #输出门

            c_t = (forget * c) + (input * cell)  #[batch, hidden]
            h_t = out * torch.tanh(c_t)  #[batch, hidden]
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = torch.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))
        
            return hidden_t, c_t, output 
            
        def step_mha(x):
            h_t = x.unsqueeze(1)
            # 使用MHA计算t时刻的隐藏状态 
            h_t, output = self.MAB(
                Q=h_t,
                K=context,
                V=context,
                mask=torch.eq(mask, 0)
            )
            output = output.squeeze(1)
            return  output
           
    

        # Recurrence loop
        output_length = input_length
        if self.output_length:
            output_length = self.output_length
    
        for _ in range(output_length):
            """
            :param Tensor decoder_input: 初始化的decoder 输入
            :param tuple(Tensor, Tensor) hidden: decoder 隐藏状态(encoder最后一个隐藏状态)
            :return: 
            """
            
            # h_t, c_t, outs = step_lstm(decoder_input, hidden)
            # hidden = (h_t, c_t)
            outs = step_mha(decoder_input)
            
           
            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)

            #[B,L]; value = 1 means the pointer is selected at t-th step
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1,outs.size()[1])).float()

            # Update mask to ignore seen indices, if masking is enabled
            if self.masking:
                mask = mask * (1 - one_hot_pointers)

            # Get embedded mask at t-th step; [B,L,D]; value = 1 means the pointer is selected at t-th step
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1,self.embedding_dim).byte()

            embedding_mask = embedding_mask.bool()

            # [B,D] ; T-th step's input 这里的为下一时刻的输入
            next_t_embed = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim)
            
            """ 没有选中节点的embedding"""
            no_select_mask = mask.unsqueeze(2).expand(-1, -1,self.embedding_dim).byte()
            no_select_mask = no_select_mask.bool()
            no_selected_emd = embedded_inputs[no_select_mask.data].view(batch_size, -1, self.embedding_dim)
            # 这里表示没有选择节点的embedding,使用最大池化
            if no_selected_emd.size(1) > 0:
                # 池化
                not_select_embed = F.max_pool1d(no_selected_emd.transpose(1,2),kernel_size=no_selected_emd.size(1)).squeeze(-1)
            else:
                not_select_embed = torch.zeros(batch_size, self.embedding_dim).to(embedded_inputs.device)
            
          
            decoder_input = not_select_embed 

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))  #indices 为当前t时刻的输出，[B,1]
        
       


        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden

class PointerNetwork(nn.Module):
    """
    Pointer-Net, with optional masking to prevent
    pointing to the same element twice (and never pointing to another).
    """

    def __init__(self,
                 elem_dims,
                 embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 bidir=False,
                 masking=True,
                 output_length=None,
                 embedding_by_dict=False,
                 embedding_by_dict_size=None):
        """
        Initiate Pointer-Net
        :param int elem_dims: dimensions of each invdividual set element
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.masking = masking
        self.output_length = output_length
        self.embedding_by_dict = embedding_by_dict
        self.embedding_by_dict_size = embedding_by_dict_size

        self.MAB = nn.MultiheadAttention(embedding_dim,num_heads=4,dropout=dropout,batch_first=True)
        self.ISAB = ISAB(embedding_dim, embedding_dim, 1, 1)
        
        if embedding_by_dict:                   
            self.embedding = nn.Embedding(embedding_by_dict_size, embedding_dim)
        else:
            self.embedding = nn.Linear(elem_dims, embedding_dim)

        self.encoder = PointerEncoder(embedding_dim,
                                      hidden_dim,
                                      lstm_layers,
                                      dropout,
                                      bidir)
        self.decoder = PointerDecoder(embedding_dim, hidden_dim,
                                      masking=self.masking,
                                      output_length=self.output_length)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim),
                                        requires_grad=False)
     

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence (batch x seq_len x elem_dim)
        :return: Pointers probabilities and indices
        """
        # inputs: (batch x seq_len x elem_dim)
        # for TSP in 2D, elem_dim = 2
        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        # decoder_input0: (batch,  embedding)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)
       
        # inputs: (batch * seq_len, elem_dim)
        input = inputs.view(batch_size * input_length, -1)

        # embedded_inputs: (batch, seq_len, embedding)
        if self.embedding_by_dict:
            input = input.long()
        else:
            input = input.float()
        
        # [B*L,D] -> [B,L,hid_dim]
        embedded_inputs = self.embedding(input).view(batch_size, input_length, -1)

        
        """
        K-means聚类
        :param Tensor embedded_inputs: 输入序列
        :param Tensor mask: 掩码

        mask = torch.zeros(batch_size, input_length).bool()
        #embedded_inputs1, mask, count = grouping(embedded_inputs, mask.to(embedded_inputs.device))
        """
        """ 使用ISAB """

        
        encoder_outputs = self.ISAB(embedded_inputs)
       
        # encoder_outputs, _ = self.MAB(
        #     query=embedded_inputs,
        #     key=embedded_inputs,
        #     value=embedded_inputs,
        #     need_weights=False,
        #     attn_mask=None
        # )

  

        """
        # 初始化hidden,并使用LSTM进行编码
        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)
        lstm_encoder_outputs, encoder_hidden = self.encoder(embedded_inputs,
                                                       encoder_hidden0)
        if self.bidir:
            # last layer's h and c only, concatenated
            decoder_hidden0 = (
                torch.cat(
                    (encoder_hidden[0][-2:][0], encoder_hidden[0][-2:][1]),
                    dim=-1),
                torch.cat(
                    (encoder_hidden[1][-2:][0], encoder_hidden[1][-2:][1]),
                    dim=-1))
        else:
            # decoder_hidden0: ((batch, hidden),
            #                   (batch, hidden))
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])
        encoder_outputs = lstm_encoder_outputs
       
        """

        # 随机初始化input0和hidden0,测试对性能的影响
        # 
        hidden_dim = embedded_inputs.size(2)
        device = embedded_inputs.device
        dtype = embedded_inputs.dtype
        #decoder_input0 = torch.randn(batch_size, self.embedding_dim,device=h.device,dtype=h.dtype)

        decoder_hidden0 = (torch.randn(batch_size, hidden_dim,device=device,dtype=dtype), 
                            torch.randn(batch_size, hidden_dim,device=device,dtype=dtype))


        (outputs, pointers), decoder_hidden = self.decoder( embedded_inputs,  
                                                           decoder_input0,
                                                           decoder_hidden0,
                                                            encoder_outputs)

        return outputs, pointers
