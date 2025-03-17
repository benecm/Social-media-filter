import json
from langchain.tools import tool
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "tdrenis/finetuned-bot-detector"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

@tool
def detect_bots(input_path: str, output_path: str):
    """Elemzi, hogy a kommentek botok által íródtak-e."""
    with open(input_path, "r", encoding="utf-8") as f:
        comments = json.load(f)

    results = {}
    for comment in comments:
        inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        bot_score = probs[0][1].item()
        label = "bot" if bot_score > 0.5 else "human"
        results[comment] = label

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    return f"Bot detekció befejezve. Eredmény mentve: {output_path}"
