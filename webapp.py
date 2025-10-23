from flask import Flask, request, render_template, jsonify, send_file
import json
import os
from research.Functions import get_youtube_comments, save_comments_to_json
from models.sentiment_analysis_ModerFinBERT import sentiment_analysis
from models.bot_detection_modell import run_bot_detection
from models.reasoning import get_quantitative_summary, summarize_with_rag
from diagram import generate_diagram

app = Flask(__name__)

# Define constants for file paths
DATA_DIR = "data"
COMMENTS_PATH = os.path.join(DATA_DIR, "comments.json")
SENTIMENT_RESULTS_PATH = os.path.join(DATA_DIR, "sentiment_results.json")
BOT_DETECTION_RESULTS_PATH = os.path.join(DATA_DIR, "bot_detection_results.json")
SUMMARY_PATH = os.path.join(DATA_DIR, "summary.json")
STATIC_DIR = "static"
DIAGRAM_OUTPUT_PATH = os.path.join(STATIC_DIR, 'sentiment_diagram.png')
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
        
        save_comments_to_json(comments, COMMENTS_PATH)
        
        # Run analyses
        sentiment_analysis(COMMENTS_PATH, SENTIMENT_RESULTS_PATH)
        run_bot_detection(COMMENTS_PATH, BOT_DETECTION_RESULTS_PATH)

        # Get quantitative summary for charts
        quantitative_summary = get_quantitative_summary(SENTIMENT_RESULTS_PATH, BOT_DETECTION_RESULTS_PATH)


        # Load results for merging
        with open(SENTIMENT_RESULTS_PATH, "r", encoding="utf-8") as f:
            sentiment_results = json.load(f)
        with open(BOT_DETECTION_RESULTS_PATH, "r", encoding="utf-8") as f:
            bot_results = json.load(f)

        # Merge results for frontend display
        bot_prediction_map = {item['Comment']: item['Prediction'] for item in bot_results}
        
        combined_results = []
        for item in sentiment_results:
            comment_text = item['Comment']
            item['Prediction'] = bot_prediction_map.get(comment_text, 'human') # Default to human
            combined_results.append(item)

        # Get qualitative summary from RAG pipeline
        llm_summary = summarize_with_rag(combined_results)

        # Combine summaries
        final_summary = quantitative_summary
        final_summary['llm_summary'] = llm_summary

        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=4)

        return jsonify({
            "results": combined_results,
            "summary": final_summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/diagram")
def diagram():
    try:
        if not os.path.exists(STATIC_DIR):
            os.makedirs(STATIC_DIR)
        generate_diagram(SENTIMENT_RESULTS_PATH, DIAGRAM_OUTPUT_PATH)
        return send_file(DIAGRAM_OUTPUT_PATH, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
