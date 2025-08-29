import json
import pandas as pd
from transformers import pipeline


def sentiment_analysis(filename="data/comments.json", output_filename="data/sentiment_results.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            comments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Nem található megfelelő JSON fájl!")
        return
    sentiment_analyzer = pipeline("text-classification", model="Varnikasiva/sentiment-classification-bert-mini")

    result_labels = []
    result_polarity = []

    for comment in comments:
        result = sentiment_analyzer(comment)
        result_labels.append(result[0]["label"])
        result_polarity.append(result[0]["score"])
    
    df = pd.DataFrame(comments,columns=["Comment"])
    df["Polarity"] = result_polarity
    df["Sentiment"] = result_labels

    sentiment_results = df.to_dict(orient="records")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, ensure_ascii=False, indent=4)
    
    print("Sentiment elemzés eredménye mentve:", output_filename)
    

sentiment_analysis()