import json
import os

# Fájl elérési utak
DATA_DIR = "data"

# JSON betöltése
def load_json(file_path):
    if not os.path.exists(file_path):
        # Return an empty dictionary if the file doesn't exist
        return {}
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Reasoning függvény agent számára
def summarize_comments(sentiment_path, output_path="data/summary.json"):
    sentiment_results = load_json(sentiment_path)
    
    # Simple summary based on sentiment results
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for result in sentiment_results:
        if result['Sentiment'] == 'Positive':
            positive_count += 1
        elif result['Sentiment'] == 'Negative':
            negative_count += 1
        else:
            neutral_count += 1
            
    summary = {
        "total_comments": len(sentiment_results),
        "positive_comments": positive_count,
        "negative_comments": negative_count,
        "neutral_comments": neutral_count
    }

    return summary

#teszthez
if __name__ == "__main__":
    summary = summarize_comments("data/sentiment_results.json")
    print("Reasoning folyamat sikeresen lefutott!")
    print("Summary:", summary)
