import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from pathlib import Path


def Read_comments():
    DATA_DIR = "data"
    input_path = os.path.join(DATA_DIR, "comments.json")
    with open(input_path, "r", encoding="utf-8") as f:
        comments = json.load(f)
    return comments


def detect_bots(comments):
    MODEL_NAME = "tdrenis/finetuned-bot-detector"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    inputs = tokenizer(comments, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    labels = ["human", "bot"]  # Az osztályok, lehet, hogy módosítani kell a modell alapján
    results = [{"comment": comment, "prediction": labels[pred.item()]} for comment, pred in zip(comments, predictions)]
    return results