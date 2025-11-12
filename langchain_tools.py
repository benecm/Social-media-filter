from langchain.tools import BaseTool
from research.Functions import get_youtube_comments, save_comments_to_json
from models.sentiment_analysis_ModerFinBERT import sentiment_analysis
from models.bot_detection_modell import run_bot_detection
from models.reasoning import get_quantitative_summary, summarize_with_rag
import os
import json
from typing import Optional, Type
from pydantic import BaseModel, Field

DATA_DIR = "data"
COMMENTS_PATH = os.path.join(DATA_DIR, "comments.json")
SENTIMENT_RESULTS_PATH = os.path.join(DATA_DIR, "sentiment_results.json")
BOT_DETECTION_RESULTS_PATH = os.path.join(DATA_DIR, "bot_detection_results.json")
SUMMARY_PATH = os.path.join(DATA_DIR, "summary.json")

class YouTubeCommentsTool(BaseTool):
    name: str = "youtube_comments_collector"
    description: str = "Collects YouTube comments from a video URL. Input should be in format 'URL|COUNT' where URL is the YouTube video URL and COUNT is the number of comments to collect."

    def _run(self, input_str: str) -> str:
        try:
            parts = input_str.split('|')
            video_url = parts[0].strip()
            max_results = int(parts[1].strip()) if len(parts) > 1 else 100
            
            comments = get_youtube_comments(video_url, max_results=max_results)
            save_comments_to_json(comments, COMMENTS_PATH)
            return f"Successfully collected {len(comments)} comments"
        except Exception as e:
            return f"Error collecting comments: {str(e)}"

class SentimentAnalysisTool(BaseTool):
    name: str = "sentiment_analyzer"
    description: str = "Analyzes sentiment of collected comments. Input should be 'analyze' to start analysis."

    def _run(self, input_str: str) -> str:
        # Accept a few variants (extra whitespace or surrounding text) to be robust when called by the agent
        if not isinstance(input_str, str) or "analyze" not in input_str.strip().lower():
            return "To analyze sentiments, just input 'analyze'"
        try:
            sentiment_analysis(COMMENTS_PATH, SENTIMENT_RESULTS_PATH)
            return "Successfully analyzed sentiment of comments"
        except Exception as e:
            return f"Error analyzing sentiment: {str(e)}"

class BotDetectionTool(BaseTool):
    name: str = "bot_detector"
    description: str = "Detects bot-generated comments. Input should be 'detect' to start detection."

    def _run(self, input_str: str) -> str:
        # Be lenient with the agent's input formatting. If the input clearly isn't a detect command,
        # still attempt detection to avoid missing the save step when the agent passes slightly different text.
        try:
            run_bot_detection(COMMENTS_PATH, BOT_DETECTION_RESULTS_PATH)
            return "Successfully analyzed comments for bot detection"
        except Exception as e:
            return f"Error in bot detection: {str(e)}"

class ResultsSummarizerTool(BaseTool):
    name: str = "results_summarizer"
    description: str = "Summarizes the analysis results. Input should be 'summarize' to generate summary."

    def _run(self, input_str: str) -> str:
        # Allow slight variations from the agent (whitespace, extra tokens)
        if not isinstance(input_str, str) or "summarize" not in input_str.strip().lower():
            return "To generate summary, just input 'summarize'"
        try:
            # Load results
            with open(SENTIMENT_RESULTS_PATH, "r", encoding="utf-8") as f:
                sentiment_results = json.load(f)
            with open(BOT_DETECTION_RESULTS_PATH, "r", encoding="utf-8") as f:
                bot_results = json.load(f)

            # Get quantitative summary
            quantitative_summary = get_quantitative_summary(SENTIMENT_RESULTS_PATH, BOT_DETECTION_RESULTS_PATH)

            # Merge results
            bot_prediction_map = {item['Comment']: item['Prediction'] for item in bot_results}
            combined_results = []
            for item in sentiment_results:
                comment_text = item['Comment']
                item['Prediction'] = bot_prediction_map.get(comment_text, 'human')
                combined_results.append(item)

            # Get qualitative summary
            llm_summary = summarize_with_rag(combined_results)
            
            # Combine summaries
            final_summary = quantitative_summary
            final_summary['llm_summary'] = llm_summary

            # Save final results
            final_results = {
                "results": combined_results,
                "summary": final_summary
            }
            with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
                json.dump(final_results, f, ensure_ascii=False, indent=4)

            return f"Analysis complete. Found {len(combined_results)} comments with {final_summary['positive_comments']} positive, {final_summary['negative_comments']} negative, and {final_summary['neutral_comments']} neutral. {final_summary['bot_comments']} potential bot comments detected."
        except Exception as e:
            return f"Error generating summary: {str(e)}"