import json
import os
from models.bot_detection_modell import detect_bots

DATA_DIR = "data"

def bot_detection_tool(input_file):
    input_path = os.path.join(DATA_DIR, input_file)

    with open(input_path, "r", encoding="utf-8") as f:
        comments = json.load(f)
    
    results = detect_bots(comments)

    output_file = "bot_detection_results.json"
    output_path = os.path.join(DATA_DIR, output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    return output_path