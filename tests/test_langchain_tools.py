import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
from langchain_tools import (
    YouTubeCommentsTool,
    SentimentAnalysisTool,
    BotDetectionTool,
    ResultsSummarizerTool,
    COMMENTS_PATH,
    SENTIMENT_RESULTS_PATH,
    BOT_DETECTION_RESULTS_PATH,
    SUMMARY_PATH,
    DATA_DIR
)

class TestLangChainTools(unittest.TestCase):

    def setUp(self):
        """Tesztkörnyezet beállítása minden teszt előtt."""
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        # Dummy adatok a tesztekhez
        self.dummy_comments = [{"comment_id": "1", "text": "Ez egy teszt komment."}]
        self.dummy_sentiment = [{"Comment": "Ez egy teszt komment.", "Sentiment": "Neutral", "Polarity": 0.0}]
        self.dummy_bot = [{"Comment": "Ez egy teszt komment.", "Prediction": "human"}]

    def tearDown(self):
        """Tesztkörnyezet lebontása minden teszt után."""
        for path in [COMMENTS_PATH, SENTIMENT_RESULTS_PATH, BOT_DETECTION_RESULTS_PATH, SUMMARY_PATH]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(DATA_DIR):
            if not os.listdir(DATA_DIR):
                os.rmdir(DATA_DIR)

    @patch('langchain_tools.save_comments_to_json')
    @patch('langchain_tools.get_youtube_comments')
    def test_youtube_comments_tool(self, mock_get_comments, mock_save_json):
        """A YouTubeCommentsTool működésének tesztelése."""
        mock_get_comments.return_value = self.dummy_comments
        tool = YouTubeCommentsTool()

        # Tesztelés érvényes bemenettel
        result = tool._run("https://youtube.com/watch?v=123|50")
        mock_get_comments.assert_called_with("https://youtube.com/watch?v=123", max_results=50)
        mock_save_json.assert_called_with(self.dummy_comments, COMMENTS_PATH)
        self.assertEqual(result, f"Successfully collected {len(self.dummy_comments)} comments")

        # Tesztelés kommentek száma nélkül (alapértelmezett 100)
        tool._run("https://youtube.com/watch?v=123")
        mock_get_comments.assert_called_with("https://youtube.com/watch?v=123", max_results=100)

        # Tesztelés hibakezeléssel
        mock_get_comments.side_effect = Exception("API Hiba")
        result = tool._run("https://youtube.com/watch?v=123|50")
        self.assertTrue(result.startswith("Error collecting comments:"))

    @patch('langchain_tools.sentiment_analysis')
    def test_sentiment_analysis_tool(self, mock_sentiment_analysis):
        """A SentimentAnalysisTool működésének tesztelése."""
        # Dummy komment fájl létrehozása
        with open(COMMENTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.dummy_comments, f)

        tool = SentimentAnalysisTool()

        # Tesztelés érvényes "analyze" bemenettel
        result = tool._run("analyze")
        mock_sentiment_analysis.assert_called_with(COMMENTS_PATH, SENTIMENT_RESULTS_PATH)
        self.assertEqual(result, "Successfully analyzed sentiment of comments")

        # Tesztelés érvénytelen bemenettel
        result = tool._run("valami_mas")
        self.assertEqual(result, "To analyze sentiments, just input 'analyze'")

        # Tesztelés hibakezeléssel
        mock_sentiment_analysis.side_effect = Exception("Elemzési hiba")
        result = tool._run("analyze")
        self.assertTrue(result.startswith("Error analyzing sentiment:"))

    @patch('langchain_tools.run_bot_detection')
    def test_bot_detection_tool(self, mock_bot_detection):
        """A BotDetectionTool működésének tesztelése."""
        # Dummy komment fájl létrehozása
        with open(COMMENTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.dummy_comments, f)

        tool = BotDetectionTool()

        # Tesztelés érvényes "detect" bemenettel
        result = tool._run("detect")
        mock_bot_detection.assert_called_with(COMMENTS_PATH, BOT_DETECTION_RESULTS_PATH)
        self.assertEqual(result, "Successfully analyzed comments for bot detection")

        # Tesztelés érvénytelen bemenettel
        result = tool._run("valami_mas")
        self.assertEqual(result, "To detect bots, just input 'detect'")

        # Tesztelés hibakezeléssel
        mock_bot_detection.side_effect = Exception("Detektálási hiba")
        result = tool._run("detect")
        self.assertTrue(result.startswith("Error in bot detection:"))

    @patch('langchain_tools.summarize_with_rag')
    @patch('langchain_tools.get_quantitative_summary')
    def test_results_summarizer_tool(self, mock_get_quantitative, mock_summarize_rag):
        """A ResultsSummarizerTool működésének tesztelése."""
        # Dummy eredményfájlok létrehozása
        with open(SENTIMENT_RESULTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.dummy_sentiment, f)
        with open(BOT_DETECTION_RESULTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.dummy_bot, f)

        # Mock-ok visszatérési értékeinek beállítása
        quantitative_summary = {
            "total_comments": 1,
            "positive_comments": 0,
            "negative_comments": 0,
            "neutral_comments": 1,
            "human_comments": 1,
            "bot_comments": 0
        }
        mock_get_quantitative.return_value = quantitative_summary
        mock_summarize_rag.return_value = "Ez egy LLM által generált összefoglaló."

        tool = ResultsSummarizerTool()

        # Tesztelés érvényes "summarize" bemenettel
        result = tool._run("summarize")

        # Ellenőrzések
        mock_get_quantitative.assert_called_with(SENTIMENT_RESULTS_PATH, BOT_DETECTION_RESULTS_PATH)
        self.assertTrue(mock_summarize_rag.called)
        
        # Ellenőrizzük, hogy a summary.json létrejött és a tartalma helyes
        self.assertTrue(os.path.exists(SUMMARY_PATH))
        with open(SUMMARY_PATH, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        self.assertIn("results", summary_data)
        self.assertIn("summary", summary_data)
        self.assertEqual(summary_data["summary"]["llm_summary"], "Ez egy LLM által generált összefoglaló.")
        self.assertEqual(summary_data["summary"]["total_comments"], 1)
        self.assertEqual(len(summary_data["results"]), 1)
        self.assertEqual(summary_data["results"][0]["Prediction"], "human")

        # Ellenőrizzük a visszatérési stringet
        expected_result_str = (
            "Analysis complete. Found 1 comments with 0 positive, 0 negative, and 1 neutral. "
            "0 potential bot comments detected."
        )
        self.assertEqual(result, expected_result_str)

        # Tesztelés érvénytelen bemenettel
        result = tool._run("valami_mas")
        self.assertEqual(result, "To generate summary, just input 'summarize'")

        # Tesztelés hibakezeléssel
        mock_get_quantitative.side_effect = Exception("Összegzési hiba")
        result = tool._run("summarize")
        self.assertTrue(result.startswith("Error generating summary:"))
