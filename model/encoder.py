import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb
from utils.config import Config


config = Config()

def init_wt_normal(wt):
    wt.data.normal_(std=0.02)

def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-0.08, 0.08)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    #seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input) #[B,L,]
        
       
        output, hidden = self.lstm(embedded)        #[B,L,hidden_dim * num_directions]
        pdb.set_trace()
        encoder_outputs = output.contiguous()
        
        encoder_feature = encoder_outputs.view(-1, 2*config.hidden_dim)  # [B*L,2*hidden_dim]
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden


