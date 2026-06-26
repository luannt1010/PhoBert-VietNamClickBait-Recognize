from clickbait_detector.preprocessing import prepare_data
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ClickBaitDataset(Dataset):
    def __init__(self, df, max_len=50):
        super().__init__()

        self.df = df
        self.labels = self.df["final_label"].tolist()
        self.headlines = self.df["title"].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.input_ids, self.attention_mask = prepare_data(self.headlines, self.tokenizer, max_len)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target = self.labels[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        return target, input_ids, attention_mask

# if __name__ == "__main__":
#     import pandas as pd
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
#     train_df = pd.read_csv(r"D:\private\clickbait_detect_proj\data\raw\train_clickbait.csv")
#     d = ClickBaitDataset(train_df, tokenizer)
#     print("1")
#
#     print(d[1])
