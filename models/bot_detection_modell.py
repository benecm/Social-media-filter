import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "tdrenis/finetuned-bot-detector"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

with open("comments.json", "r", encoding="utf-8") as f:
    comments = json.load(f)
inputs = tokenizer(comments, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

labels = ["human", "bot"]  # Az osztályok, lehet, hogy módosítani kell a modell alapján
results = [{"comment": comment, "prediction": labels[pred.item()]} for comment, pred in zip(comments, predictions)]

with open("bot_detection_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
