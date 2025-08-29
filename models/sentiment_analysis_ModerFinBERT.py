from textblob import TextBlob
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def predict_sentiment(texts):
    model_name = "tabularisai/multilingual-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = {0: "Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Positive"}
    return [sentiment_map[p] for p in torch.argmax(probabilities, dim=-1).tolist()]


def sentiment_analysis(filename="data/comments.json", output_filename="data/sentiment_results.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            comments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Nem található megfelelő JSON fájl!")
        return

    results =[]
    for comment in comments:
        results.append(predict_sentiment(comment)[0])

    
    df = pd.DataFrame(comments,columns=["Comment"])
    df["Polarity"] = 0
    df["Sentiment"] = results

    sentiment_results = df.to_dict(orient="records")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, ensure_ascii=False, indent=4)
    
    print("Sentiment elemzés eredménye mentve:", output_filename)
    



sentiment_analysis()