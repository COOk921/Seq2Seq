import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
        Copied from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: 模型的嵌入维度 (embedding dimension)。
            dropout: Dropout rate.
            max_len: 支持的最大序列长度。
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # [d_model / 2]
        pe = torch.zeros(max_len, 1, d_model) # Shape: [max_len, 1, d_model]

        # 分别计算偶数和奇数维度的 PE 值
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # 将 pe 注册为 buffer，这样它不会被视为模型参数，但会随模型移动 (e.g., to device)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将位置编码添加到输入张量中。

        Args:
            x: 输入张量，形状通常为 [batch_size, seq_len, embedding_dim]
               (假设 embedding 层或输入处理后是 batch_first=True 格式)

        Returns:
            添加了位置编码的张量，形状与输入 x 相同。
        """
        # x shape: [batch_size, seq_len, embedding_dim]
        # self.pe shape: [max_len, 1, d_model]
        # 需要从 self.pe 中取出与 x 的 seq_len 匹配的部分，并调整形状以进行广播相加

        # self.pe[:x.size(1)] -> shape [seq_len, 1, embedding_dim]
        # .transpose(0, 1) -> shape [1, seq_len, embedding_dim]
        # 这样可以和 x [batch_size, seq_len, embedding_dim] 进行广播相加
        pe_for_x = self.pe[:x.size(1)].transpose(0, 1)
        x = x + pe_for_x
        return self.dropout(x)