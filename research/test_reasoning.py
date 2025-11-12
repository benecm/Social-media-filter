import unittest
import os
import json
from unittest.mock import patch, MagicMock

from models.reasoning import get_quantitative_summary, summarize_with_rag

class TestReasoning(unittest.TestCase):

    def setUp(self):
        """Tesztkörnyezet beállítása."""
        self.test_dir = 'test_data_reasoning'
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        self.sentiment_path = os.path.join(self.test_dir, 'sentiment.json')
        self.bot_path = os.path.join(self.test_dir, 'bot.json')

    def tearDown(self):
        """Tesztkörnyezet eltakarítása."""
        for path in [self.sentiment_path, self.bot_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    def test_get_quantitative_summary(self):
        """A kvantitatív összegző függvény tesztelése."""
        sentiment_data = [
            {"Sentiment": "Positive"},
            {"Sentiment": "Negative"},
            {"Sentiment": "Positive"},
            {"Sentiment": "Neutral"}
        ]
        bot_data = [
            {"Prediction": "human"},
            {"Prediction": "bot"},
            {"Prediction": "human"},
            {"Prediction": "human"}
        ]
        with open(self.sentiment_path, 'w') as f:
            json.dump(sentiment_data, f)
        with open(self.bot_path, 'w') as f:
            json.dump(bot_data, f)

        summary = get_quantitative_summary(self.sentiment_path, self.bot_path)

        self.assertEqual(summary['total_comments'], 4)
        self.assertEqual(summary['positive_comments'], 2)
        self.assertEqual(summary['negative_comments'], 1)
        self.assertEqual(summary['neutral_comments'], 1)
        self.assertEqual(summary['bot_comments'], 1)
        self.assertEqual(summary['human_comments'], 3)

    @patch('models.reasoning.RetrievalQA')
    @patch('models.reasoning.Chroma')
    def test_summarize_with_rag(self, mock_chroma, mock_retrieval_qa):
        """A RAG-alapú összegző függvény tesztelése."""
        # Mock-ok beállítása
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"result": "This is a mock summary."}
        mock_retrieval_qa.from_chain_type.return_value = mock_chain

        analysis_results = [
            {"Comment": "Great video!", "Sentiment": "Positive", "Polarity": 0.8, "Prediction": "human"},
            {"Comment": "Not good.", "Sentiment": "Negative", "Polarity": -0.5, "Prediction": "human"}
        ]

        summary = summarize_with_rag(analysis_results)

        self.assertEqual(summary, "This is a mock summary.")
        mock_chroma.from_documents.assert_called_once()
        mock_retrieval_qa.from_chain_type.assert_called_once()
        mock_chain.invoke.assert_called_once_with("Provide a summary of the YouTube comments based on the retrieved context.")

    def test_summarize_with_rag_no_input(self):
        """A RAG összegző tesztelése üres bemenettel."""
        summary = summarize_with_rag([])
        self.assertEqual(summary, "No comments were provided to analyze.")
