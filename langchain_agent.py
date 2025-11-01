from langchain.agents import initialize_agent, AgentType
from langchain_ollama.llms import OllamaLLM
from langchain_tools import (
    YouTubeCommentsTool,
    SentimentAnalysisTool,
    BotDetectionTool,
    ResultsSummarizerTool,
    COMMENTS_PATH,
    SENTIMENT_RESULTS_PATH,
    BOT_DETECTION_RESULTS_PATH,
    SUMMARY_PATH
)
import os
import json

class AnalysisAgent:
    def __init__(self):
        # Initialize tools
        self.tools = [
            YouTubeCommentsTool(),
            SentimentAnalysisTool(),
            BotDetectionTool(),
            ResultsSummarizerTool()
        ]

        # Initialize Ollama LLM
        OLLAMA_BASE_URL = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11500/')
        llm = OllamaLLM(
            model="llama3.2",
            base_url=OLLAMA_BASE_URL,
            temperature=0
        )

        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def analyze_video(self, video_url: str, comment_count: int = 100) -> dict:
        """
        Analyze a YouTube video's comments using the LangChain agent.
        
        Args:
            video_url (str): The URL of the YouTube video
            comment_count (int): Number of comments to analyze
            
        Returns:
            dict: Analysis results and summary
        """
        try:
            # Execute sequential analysis steps
            input_str = f"{video_url}|{comment_count}"  # Combine URL and count
            self.agent.run(input_str)  # Pass URL and comment count
            self.agent.run("analyze")  # Run sentiment analysis
            self.agent.run("detect")   # Run bot detection
            summary_response = self.agent.run("summarize")  # Generate summary
            
            # Load the final results from the saved file
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                results = json.load(f)
            
            # If results is a string (summary text), wrap it in a proper structure
            if isinstance(results, str):
                with open(COMMENTS_PATH, 'r', encoding='utf-8') as f:
                    comments = json.load(f)
                with open(SENTIMENT_RESULTS_PATH, 'r', encoding='utf-8') as f:
                    sentiment_results = json.load(f)
                with open(BOT_DETECTION_RESULTS_PATH, 'r', encoding='utf-8') as f:
                    bot_results = json.load(f)

                bot_prediction_map = {item['Comment']: item['Prediction'] for item in bot_results}
                combined_results = []
                for item in sentiment_results:
                    comment_text = item['Comment']
                    item['Prediction'] = bot_prediction_map.get(comment_text, 'human')
                    combined_results.append(item)

                return {
                    "results": combined_results,
                    "summary": {
                        "llm_summary": results,
                        "total_comments": len(comments)
                    }
                }
            return results
            
        except Exception as e:
            return {"error": str(e)}