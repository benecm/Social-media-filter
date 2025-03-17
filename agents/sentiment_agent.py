import json
import os
from models.sentiment_analysis import sentiment_analysis

# Data mappa elérési útvonala
DATA_DIR = "data"

def sentiment_tool(input_file):
    input_path = os.path.join(DATA_DIR, input_file)

    with open(input_path, "r", encoding="utf-8") as f:
        comments = json.load(f)
    
    results = sentiment_analysis(comments)

    output_file = "sentiment_results.json"
    output_path = os.path.join(DATA_DIR, output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    return output_path