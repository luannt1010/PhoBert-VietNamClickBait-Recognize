import torch
import argparse
import pandas as pd
from transformers import AutoTokenizer
from clickbait_detector.preprocessing import prepare_data
from clickbait_detector.utils import load_model_from_state_dict
from clickbait_detector.crawl_data import get_article

class ClickBaitPredictor:
    def __init__(self, config_dir, weight_path, max_len=256, threshold=0.5):
        self.max_len = max_len
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model_from_state_dict(weight_path, config_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config_dir)

    @torch.no_grad()
    def predict_one_title(self, title):
        input_ids, attention_mask = prepare_data([title], self.tokenizer, self.max_len)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        self.model.eval()
        outputs = self.model(input_ids, attention_mask)
        score = torch.sigmoid(outputs).squeeze(1).item()
        return {"sentence": title, "label": 'clickbait' if score >= self.threshold else 'non-clickbait', "score": round(score, 4)}

    @torch.no_grad()
    def predict_file(self, df: pd.DataFrame):
        titles = df['title'].tolist()
        results = [self.predict_one_title(title) for title in titles]
        return results

    @torch.no_grad()
    def predict_url(self, url):
        results = get_article(url, crawl_title=True)
        if "title" not in results:
            return None
        else:
            title = results["title"]
            return self.predict_one_title(title)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, default="vinai/phobert-base-v2")
    parser.add_argument("--weight_path", type=str, default=r".\artifacts\models\last.pth")
    parser.add_argument("--input_sentence", type=str)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_len", type=int, default=256)

    return parser.parse_args()

def main():
    args = get_args()

    config_dir = args.config_dir
    weight_path = args.weight_path
    input_sentence = args.input_sentence
    threshold = args.threshold
    max_len = args.max_len

    predictor = ClickBaitPredictor(config_dir, weight_path, max_len, threshold)
    results = predictor.predict_one_title(input_sentence)
    print(results)
    # predictor = ClickBaitPredictor(r"D:\private\clickbait_detect_proj\models\phobert-base-v2", r"D:\private\clickbait_detect_proj\artifacts\phobert-base-v2-v2\models\last.pth")
    # res = predictor.predict_url("https://vietnamnet.vn/lu-tren-song-sau-bao-so-1-co-the-dang-toi-6m-8-tinh-khan-truong-ung-pho-2532699.html")
    # print(res)
if __name__ == "__main__":
    main()

