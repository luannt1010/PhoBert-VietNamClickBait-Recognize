import torch
import torch.nn as nn
from transformers import AutoModel

class Model(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False  # Dropout will errors if without this
        )
        out = self.dropout(output)
        out = self.classifier(out)
        return out

# if __name__ == "__main__":
#     model = Model()
#     print(1)
#     from clickbait_dataset import ClickBaitDataset
#     from transformers import AutoTokenizer
#     from torch.utils.data import DataLoader
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
#     d = ClickBaitDataset(r"D:\private\clickbait_detect_proj\data\raw\train_clickbait.csv")
#     d = DataLoader(d, batch_size=16, shuffle=True)
#     for target, input_ids, attention_mask in d:
#         print(input_ids.shape)
#         outputs = torch.sigmoid(model(input_ids, attention_mask))
#         print("Outputs:", outputs.squeeze(1))
#
#         logits = outputs.squeeze(1)
#         preds = (logits >= 0.5).to(torch.int64)
#
#
#         num_correct = (preds==target).sum().item()
#         print(preds)
#         print(target)
#         print(num_correct)
#         break
