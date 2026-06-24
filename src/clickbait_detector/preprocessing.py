import re
import string
from pyvi import ViTokenizer

def preprocess(text: str):
    punc = string.punctuation
    text = text.lower().strip()
    text = re.sub(f"[{re.escape(punc)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def word_segmentation(text: str):
    return ViTokenizer.tokenize(text)

def preprocess_headlines(headlines):
    results = []
    for headline in headlines:
        cleaned = preprocess(headline)
        segmented = word_segmentation(cleaned)
        results.append(segmented)
    return results

def prepare_data(headlines, tokenizer, max_len=50):
    if len(headlines) == 0:
        return None, None

    processed_headlines = preprocess_headlines(headlines)
    encoded = tokenizer(processed_headlines, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    return input_ids, attention_mask

