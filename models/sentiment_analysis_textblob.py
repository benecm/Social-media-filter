from textblob import TextBlob
import json
import pandas as pd
from typing import Optional

def sentiment_analysis(filename="data/comments.json", output_filename="data/sentiment_results.json") -> Optional[list[dict]]:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            comments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Nem található megfelelő JSON fájl!")
        return None
    
    df = pd.DataFrame(comments, columns=["Comment"])
    df["Polarity"] = df["Comment"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["Sentiment"] = df["Polarity"].apply(lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral")
    
    sentiment_results = df.to_dict(orient="records")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, ensure_ascii=False, indent=4)
    
    print("Sentiment elemzés eredménye mentve:", output_filename)
    return sentiment_results
    
if __name__ == "__main__":
    sentiment_analysis()