import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# NOTE: The junaid1993/distilroberta-bot-detection model is faster and more
# memory-efficient than the tdrenis/finetuned-bot-detector model.

# --- Load model and tokenizer once on module import ---
MODEL_NAME = "junaid1993/distilroberta-bot-detection"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def run_bot_detection(input_path="data/comments.json", output_path="data/bot_detection_results.json"):
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            comments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Hiba: A(z) {input_path} fájl nem található vagy hibás.")
        return []

    # Filter out empty or whitespace-only comments to avoid model errors
    valid_comments = [c for c in comments if c and c.strip()]
    if not valid_comments:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        return []

    inputs = tokenizer(valid_comments, padding=True, truncation=True, return_tensors='pt', max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    labels = ["human", "bot"]
    results = [{"Comment": comment, "Prediction": labels[pred.item()]} for comment, pred in zip(valid_comments, predictions)]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("Bot detection results saved to:", output_path)
    return results

if __name__ == "__main__":
    run_bot_detection()
