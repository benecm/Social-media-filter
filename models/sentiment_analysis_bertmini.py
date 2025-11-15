import json
import pandas as pd
from transformers import pipeline
from typing import Optional


# Emotion label to sentiment mapping
EMOTION_TO_SENTIMENT = {
    "sadness": "Negative",
    "anger": "Negative",
    "love": "Positive",
    "surprise": "Positive",
    "fear": "Negative",
    "happiness": "Positive",
    "neutral": "Neutral",
    "disgust": "Negative",
    "shame": "Negative",
    "guilt": "Negative",
    "confusion": "Negative",
    "desire": "Positive",
    "sarcasm": "Negative"
}


def sentiment_analysis(filename="data/comments.json", output_filename="data/sentiment_results.json", use_original_tags=True) -> Optional[list[dict]]:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            comments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Nem található megfelelő JSON fájl!")
        return None
    sentiment_analyzer = pipeline("text-classification", model="Varnikasiva/sentiment-classification-bert-mini")

    result_labels = []
    result_polarity = []

    for comment in comments:
        result = sentiment_analyzer(comment)
        emotion_label = result[0]["label"].lower()
        # Use original emotion tags or map to 3-class sentiment
        if use_original_tags:
            sentiment_label = emotion_label
        else:
            # Map emotion to sentiment (Positive, Negative, Neutral)
            sentiment_label = EMOTION_TO_SENTIMENT.get(emotion_label, "Neutral")
        result_labels.append(sentiment_label)
        result_polarity.append(result[0]["score"])
    
    df = pd.DataFrame(comments,columns=["Comment"])
    df["Polarity"] = result_polarity
    df["Sentiment"] = result_labels

    sentiment_results = df.to_dict(orient="records")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, ensure_ascii=False, indent=4)
    
    print("Sentiment elemzés eredménye mentve:", output_filename)
    return sentiment_results
    
if __name__ == "__main__":
    sentiment_analysis()