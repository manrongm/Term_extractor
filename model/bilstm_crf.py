import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=256):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2,
            num_layers=1, bidirectional=True, batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentences, tags=None, mask=None):
        embeds = self.embedding(sentences)              # [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embeds)                 # [batch_size, seq_len, hidden_dim]
        emissions = self.hidden2tag(lstm_out)           # [batch_size, seq_len, tagset_size]

        if tags is not None:
            # Training mode: return the negative log-likelihood loss
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            # Inference mode: return best path (viterbi decoding)
            return self.crf.decode(emissions, mask=mask)
