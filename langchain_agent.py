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
            input_str = f"{video_url}|{comment_count}"
            
            # 1. Kommentek gyűjtése
            collection_result = self.agent.run(input_str)
            if "Error" in collection_result:
                return {"error": f"Kommentgyűjtés sikertelen: {collection_result}"}

            # 2. Sentiment elemzés futtatása
            sentiment_result = self.agent.run("analyze")
            if "Error" in sentiment_result:
                return {"error": f"Sentiment elemzés sikertelen: {sentiment_result}"}

            # 3. Bot detektálás futtatása
            bot_detection_result = self.agent.run("detect")
            if "Error" in bot_detection_result:
                return {"error": f"Bot detektálás sikertelen: {bot_detection_result}"}

            # 4. Összegzés generálása
            summary_generation_result = self.agent.run("summarize")
            if "Error" in summary_generation_result:
                return {"error": f"Összegzés generálás sikertelen: {summary_generation_result}"}
            
            # Load the final results from the saved file
            if not os.path.exists(SUMMARY_PATH):
                return {"error": f"Az összegző fájl nem található a várt helyen: {SUMMARY_PATH}"}

            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                results = json.load(f)

            return results
            
        except FileNotFoundError as e:
            return {"error": f"Hiányzó fájl a feldolgozás során: {e}"}
        except json.JSONDecodeError as e:
            return {"error": f"Hiba a JSON fájl dekódolásakor: {e}. Lehet, hogy a fájl sérült vagy üres."}
        except Exception as e:
            return {"error": f"Váratlan hiba történt az elemzés során: {str(e)}"}