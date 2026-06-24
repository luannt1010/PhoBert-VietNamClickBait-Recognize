import torch
import argparse
from transformers import AutoTokenizer
from clickbait_detector.preprocessing import prepare_data
from clickbait_detector.utils import load_model

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weight_path", type=str, default=r"./artifacts/models/best.pth")
    parser.add_argument("--input_sentence", type=str)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_len", type=int, default=50)

    return parser.parse_args()

def infer(model, input_sentence, threshold=0.5, max_len=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    input_ids, attention_mask = prepare_data([input_sentence], tokenizer, max_len)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        score = torch.sigmoid(outputs).squeeze(1).item()
    print(f"Sentence: {input_sentence}")
    print(f"Prediction: {"clickbait" if score >= threshold else "non-clickbait"}")
    print(f"Score: {score}")

if __name__ == "__main__":
    args = get_args()

    weight_path = args.weight_path
    input_sentence = args.input_sentence
    threshold = args.threshold
    max_len = args.max_len

    model = load_model(weight_path)
    infer(model, input_sentence, threshold, max_len)

