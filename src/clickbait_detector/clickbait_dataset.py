from torch.utils.data import Dataset
from transformers import AutoTokenizer
from clickbait_detector.preprocessing import prepare_data

class ClickBaitDataset(Dataset):
    def __init__(self, df, max_len=50, tokenizer_dir="vinai/phobert-base-v2"):
        super().__init__()

        self.df = df
        self.labels = self.df["label"].tolist()
        self.headlines = self.df["title_combined"].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.input_ids, self.attention_mask = prepare_data(self.headlines, self.tokenizer, max_len)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target = self.labels[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        return target, input_ids, attention_mask

