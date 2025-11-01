from flask import Flask, request, render_template, jsonify, send_file
import json
import os
from langchain_agent import AnalysisAgent
from diagram import generate_diagram

app = Flask(__name__)

# Define constants for file paths
DATA_DIR = "data"
SENTIMENT_RESULTS_PATH = os.path.join(DATA_DIR, "sentiment_results.json")
STATIC_DIR = "static"
DIAGRAM_OUTPUT_PATH = os.path.join(STATIC_DIR, 'sentiment_diagram.png')

# Initialize LangChain agent
agent = None

def init_agent():
    global agent
    if agent is None:
        try:
            agent = AnalysisAgent()
        except Exception as e:
            raise ValueError(f"Failed to initialize Ollama LLM. Make sure Ollama server is running at http://127.0.0.1:11500/. Error: {str(e)}")
    return agent

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        video_url = data.get("video_url")
        comment_count = int(data.get("comment_count", 100))  # Ensure it's an integer

        if not video_url:
            return jsonify({"error": "Nincs megadva YouTube link!"}), 400
        
        if comment_count <= 0:
            return jsonify({"error": "A kommentek száma nem lehet 0 vagy negatív!"}), 400

        # Initialize agent and perform analysis
        current_agent = init_agent()
        analysis_result = current_agent.analyze_video(video_url, comment_count)
        
        if "error" in analysis_result:
            return jsonify({"error": analysis_result["error"]}), 500

        return jsonify(analysis_result)

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
