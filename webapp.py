from flask import Flask, request, render_template, jsonify
import json
import re
import pandas as pd
from research.Functions import get_youtube_comments, save_comments_to_json
from textblob import TextBlob
from main import run_analysis
import os

#pelda link: https://www.youtube.com/watch?v=89LOsf8pDhY

app = Flask(__name__)
"""
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_url = request.form['video_url']
        api_key = request.form['api_key']
        comments = get_youtube_comments(video_url, api_key)
        save_comments_to_json(comments)
        results = sentiment_analysis()
        return jsonify(results)
    return render_template('index.html')
"""
@app.route("/")
def index():
    return render_template("index.html")  # Ez megnyitja az index.html-t

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        video_url = data.get("video_url")

        if not video_url:
            return jsonify({"error": "Nincs megadva YouTube link!"}), 400

        comments = get_youtube_comments(video_url)
        save_comments_to_json(comments, os.path.join("data", "comments.json"))

        run_analysis()

        # Eredmények beolvasása
        with open(os.path.join("data", "sentiment_results.json"), "r", encoding="utf-8") as f:
            sentiment_results = json.load(f)

        # with open(os.path.join("data", "bot_detection_results.json"), "r", encoding="utf-8") as f:
        #     bot_results = json.load(f)

        with open(os.path.join("data", "summary.json"), "r", encoding="utf-8") as f:
            summary = json.load(f)

        return jsonify({
            "sentiment": sentiment_results,
            "bots": {},
            "summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500





if __name__ == '__main__':
    app.run(debug=True)
