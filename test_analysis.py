import os
import json
from main import run_analysis
from research.Functions import get_youtube_comments, save_comments_to_json

def test_pipeline():
    # Create a dummy comments file
    comments = ["This is a great video!", "I learned a lot.", "This is spam."]
    if not os.path.exists("data"):
        os.makedirs("data")
    save_comments_to_json(comments, os.path.join("data", "comments.json"))

    # Run the analysis
    run_analysis()

    # Check the results
    with open(os.path.join("data", "sentiment_results.json"), "r", encoding="utf-8") as f:
        sentiment_results = json.load(f)
        print("Sentiment results:", sentiment_results)

    with open(os.path.join("data", "summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)
        print("Summary:", summary)

if __name__ == "__main__":
    import sys
    with open("test_output.log", "w") as f:
        sys.stdout = f
        sys.stderr = f
        test_pipeline()
