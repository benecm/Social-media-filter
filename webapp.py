from flask import Flask, request, render_template, jsonify, send_file
import json
import os
from research.Functions import get_youtube_comments, save_comments_to_json
from models.sentiment_analysis_ModerFinBERT import sentiment_analysis
from models.reasoning import summarize_comments
from diagram import generate_diagram

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        video_url = data.get("video_url")
        comment_count = data.get("comment_count")

        if not video_url:
            return jsonify({"error": "Nincs megadva YouTube link!"}), 400

        if comment_count:
            comments = get_youtube_comments(video_url, max_results=comment_count)
        else:
            comments = get_youtube_comments(video_url)
        
        comments_path = os.path.join("data", "comments.json")
        sentiment_results_path = os.path.join("data", "sentiment_results.json")
        summary_path = os.path.join("data", "summary.json")

        save_comments_to_json(comments, comments_path)
        
        sentiment_analysis(comments_path, sentiment_results_path)
        summary = summarize_comments(sentiment_results_path)

        with open(sentiment_results_path, "r", encoding="utf-8") as f:
            sentiment_results = json.load(f)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)

        return jsonify({
            "sentiment": sentiment_results,
            "summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/diagram")
def diagram():
    try:
        data_path = os.path.join('data', 'sentiment_results.json')
        output_path = os.path.join('static', 'sentiment_diagram.png')
        if not os.path.exists('static'):
            os.makedirs('static')
        generate_diagram(data_path, output_path)
        return send_file(output_path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
