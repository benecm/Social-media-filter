import json
from langchain.tools import tool
from textblob import TextBlob

@tool
def analyze_sentiment(input_path: str, output_path: str):
    """Elemzi a kommentek sentimentjét és elmenti az eredményeket."""
    with open(input_path, "r", encoding="utf-8") as f:
        comments = json.load(f)

    results = {}
    for comment in comments:
        sentiment_score = TextBlob(comment).sentiment.polarity
        sentiment_label = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
        results[comment] = sentiment_label

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    return f"Sentiment analízis befejezve. Eredmény mentve: {output_path}"
