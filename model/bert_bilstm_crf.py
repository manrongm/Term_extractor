import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
from transformers import DistilBertModel
from transformers import AutoModel
from torchcrf import CRF


class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, bert_model_name, num_tags, lstm_hidden_dim=256, dropout_rate=0.7):
        super(BERT_BiLSTM_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        # self.bert = DistilBertModel.from_pretrained(bert_model_name)
        # self.bert = AutoModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # frozen first 6 BERT layer
        # for name, param in self.bert.named_parameters():
        #     if "encoder.layer." in name:
        #         layer_num = int(name.split(".")[2])
        #         if layer_num < 6:
        #             param.requires_grad = False

        self.bilstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=lstm_hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout_rate)
        # self.norm = nn.LayerNorm(lstm_hidden_dim)  # add LayerNorm
        self.fc = nn.Linear(lstm_hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_dim]

        lstm_out, _ = self.bilstm(sequence_output)
        lstm_out = self.dropout(lstm_out)
        # lstm_out = self.norm(lstm_out)  #  LayerNorm applied after Dropout
        emissions = self.fc(lstm_out)

        if labels is not None:
            # Return negative log-likelihood loss
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            # Inference: return best tag path
            return self.crf.decode(emissions, mask=attention_mask.bool())