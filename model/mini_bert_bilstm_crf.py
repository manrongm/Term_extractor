import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
from torchcrf import CRF
class LightBERT_BiLSTM_CRF(nn.Module):
    def __init__(self, bert_model_name, num_tags, lstm_hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        # 只冻结前 8 层
        for name, param in self.bert.named_parameters():
            if "encoder.layer." in name:
                layer_num = int(name.split(".")[2])
                if layer_num < 8:
                    param.requires_grad = False

        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.layernorm = nn.LayerNorm(lstm_hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        lstm_out, _ = self.bilstm(sequence_output)
        lstm_out = self.layernorm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        emissions = self.fc(lstm_out)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.bool())
