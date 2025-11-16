from textblob import TextBlob
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Optional

# --- Load model and tokenizer once on module import ---
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def predict_sentiment(texts):
    # This function now uses the globally loaded tokenizer and model for efficiency.
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return [sentiment_map[p.item()] for p in torch.argmax(probabilities, dim=-1)]


def sentiment_analysis(filename="data/comments.json", output_filename="data/sentiment_results.json") -> Optional[list[dict]]:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            comments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Nem található megfelelő JSON fájl!")
        return None

    # Process all comments in a single batch for efficiency
    results = predict_sentiment(comments) if comments else []
    
    df = pd.DataFrame(comments, columns=["Comment"])
    df["Polarity"] = 0
    df["Sentiment"] = results

    sentiment_results = df.to_dict(orient="records")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, ensure_ascii=False, indent=4)
    
    print("Sentiment elemzés eredménye mentve:", output_filename)
    return sentiment_results
    
if __name__ == "__main__":
    sentiment_analysis()
