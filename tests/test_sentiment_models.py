import unittest
import os
import json
from unittest.mock import patch, mock_open

# Import the functions to be tested
from models.sentiment_analysis_ModerFinBERT import sentiment_analysis as sentiment_moderfinbert
from models.sentiment_analysis_bertmini import sentiment_analysis as sentiment_bertmini
from models.sentiment_analysis_textblob import sentiment_analysis as sentiment_textblob

TEST_COMMENTS_PATH = "test_comments.json"
TEST_RESULTS_PATH = "test_results.json"

class TestSentimentModels(unittest.TestCase):

    def setUp(self):
        """Set up a dummy comments file for each test."""
        self.comments = ["This is a great video!", "This is terrible."]
        with open(TEST_COMMENTS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.comments, f)

    def tearDown(self):
        """Clean up created files after each test."""
        for path in [TEST_COMMENTS_PATH, TEST_RESULTS_PATH]:
            if os.path.exists(path):
                os.remove(path)

    @patch('models.sentiment_analysis_ModerFinBERT.predict_sentiment')
    def test_sentiment_moderfinbert_success(self, mock_predict):
        """Test successful run of ModerFinBERT sentiment analysis."""
        mock_predict.return_value = ["Positive", "Negative"]
        
        sentiment_moderfinbert(TEST_COMMENTS_PATH, TEST_RESULTS_PATH)
        
        mock_predict.assert_called_once_with(self.comments)
        self.assertTrue(os.path.exists(TEST_RESULTS_PATH))
        with open(TEST_RESULTS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['Sentiment'], 'Positive')

    @patch('builtins.print')
    def test_sentiment_moderfinbert_file_not_found(self, mock_print):
        """Test FileNotFoundError for ModerFinBERT."""
        os.remove(TEST_COMMENTS_PATH)
        sentiment_moderfinbert(TEST_COMMENTS_PATH, TEST_RESULTS_PATH)
        mock_print.assert_called_with("Nem található megfelelő JSON fájl!")

    @patch('models.sentiment_analysis_bertmini.pipeline')
    def test_sentiment_bertmini_success(self, mock_pipeline):
        """Test successful run of bert-mini sentiment analysis."""
        mock_analyzer = mock_pipeline.return_value
        mock_analyzer.side_effect = [
            [{"label": "Positive", "score": 0.99}],
            [{"label": "Negative", "score": 0.98}]
        ]
        
        sentiment_bertmini(TEST_COMMENTS_PATH, TEST_RESULTS_PATH)
        
        self.assertTrue(os.path.exists(TEST_RESULTS_PATH))
        with open(TEST_RESULTS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['Sentiment'], 'Positive')

    def test_sentiment_textblob_success(self):
        """Test successful run of TextBlob sentiment analysis."""
        sentiment_textblob(TEST_COMMENTS_PATH, TEST_RESULTS_PATH)
        
        self.assertTrue(os.path.exists(TEST_RESULTS_PATH))
        with open(TEST_RESULTS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['Sentiment'], 'Positive')
        self.assertEqual(data[1]['Sentiment'], 'Negative')

if __name__ == '__main__':
    unittest.main()
