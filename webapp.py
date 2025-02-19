from flask import Flask, request, render_template, jsonify
import json
import re
import pandas as pd
from Functions import get_youtube_comments, sentiment_analysis,save_comments_to_json
from textblob import TextBlob

#pleda link: https://www.youtube.com/watch?v=89LOsf8pDhY

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
        save_comments_to_json(comments)

        sentiment_analysis()
        #results = json.load('sentiment_results.json')
        try:
            with open('sentiment_results.json', "r", encoding="utf-8") as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("Nem található megfelelő JSON fájl!")
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500





if __name__ == '__main__':
    app.run(debug=True)
