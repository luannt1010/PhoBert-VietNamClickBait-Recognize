import torch.nn as nn
from transformers import AutoModel

class Model(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base-v2")

        self.dropout = nn.Dropout(dropout_rate)

        self.classify1 = nn.Sequential(nn.Linear(self.bert.config.hidden_size, 256, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(),
                                       nn.Dropout(0.1))

        self.classify2 = nn.Sequential(nn.Linear(256, 128, bias=False),
                                       nn.BatchNorm1d(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))

        self.classify3 = nn.Sequential(nn.Linear(128, 64, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(),
                                       nn.Dropout(0.1))
        self.classify4 = nn.Sequential(nn.Linear(64, 1, bias=False),
                                       nn.BatchNorm1d(1))

        self._init_weight()

    def _init_weight(self):
        for module in [self.classify1, self.classify2, self.classify3, self.classify4]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)


    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False)  # Dropout will errors if without this

        output = self.dropout(output)
        output = self.classify1(output)
        output = self.classify2(output)
        output = self.classify3(output)
        out = self.classify4(output)
        return out


