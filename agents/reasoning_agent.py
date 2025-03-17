import os
from models.reasoning import summarize_comments

DATA_DIR = "data"

def reasoning_tool(sentiment_file, bot_file):
    sentiment_path = os.path.join(DATA_DIR, sentiment_file)
    bot_path = os.path.join(DATA_DIR, bot_file)

    summary = summarize_comments(sentiment_path, bot_path)

    output_file = "summary.json"
    output_path = os.path.join(DATA_DIR, output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)

    return output_path