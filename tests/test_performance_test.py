import unittest
import os
import json
from unittest.mock import patch, mock_open

from performance_test import load_ground_truth, calculate_metrics, run_performance_test

class TestPerformanceTest(unittest.TestCase):

    def setUp(self):
        """Tesztkörnyezet beállítása."""
        self.test_dir = 'test_data_perf'
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        self.ground_truth_path = os.path.join(self.test_dir, 'ground_truth.json')

    def tearDown(self):
        """Tesztkörnyezet eltakarítása."""
        if os.path.exists(self.ground_truth_path):
            os.remove(self.ground_truth_path)
        # A run_performance_test által létrehozott fájlok törlése
        for fname in ["temp_test_comments.json", "temp_sentiment_results.json", "temp_bot_results.json", "performance_report.json"]:
            fpath = os.path.join('data', fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    def test_load_ground_truth_success(self):
        """Sikeres ground truth betöltés tesztelése."""
        dummy_data = [{"Comment": "test", "Sentiment": "Positive", "Prediction": "human"}]
        with open(self.ground_truth_path, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f)
        data = load_ground_truth(self.ground_truth_path)
        self.assertEqual(data, dummy_data)

    def test_calculate_metrics(self):
        """A metrikaszámító függvény tesztelése."""
        y_true = ["Positive", "Negative", "Positive"]
        y_pred = ["Positive", "Positive", "Positive"]
        labels = ["Positive", "Negative"]
        metrics = calculate_metrics(y_true, y_pred, labels)
        self.assertAlmostEqual(metrics['accuracy'], 2/3)
        self.assertIn('classification_report', metrics)
        self.assertIn('confusion_matrix', metrics)

    @patch('performance_test.run_bot_detection')
    @patch('performance_test.analyze_sentiment')
    @patch('performance_test.load_ground_truth')
    @patch('builtins.print')
    def test_run_performance_test(self, mock_print, mock_load_gt, mock_analyze, mock_run_bot):
        """A teljes performancia teszt folyamat tesztelése."""
        # Mock-ok beállítása
        gt_data = [
            {"Comment": "Great video!", "Sentiment_true": "Positive", "Prediction_true": "human"},
            {"Comment": "This is spam.", "Sentiment_true": "Negative", "Prediction_true": "bot"}
        ]
        mock_load_gt.return_value = gt_data

        # A mockolt analízis függvényeknek létre kell hozniuk a várt kimeneti fájlokat
        def create_sentiment_file(*args):
            with open(args[1], 'w') as f:
                json.dump([{"Comment": "Great video!", "Sentiment_pred": "Positive"}, {"Comment": "This is spam.", "Sentiment_pred": "Negative"}], f)
        def create_bot_file(*args):
            with open(args[1], 'w') as f:
                json.dump([{"Comment": "Great video!", "Prediction_pred": "human"}, {"Comment": "This is spam.", "Prediction_pred": "bot"}], f)

        mock_analyze.side_effect = create_sentiment_file
        mock_run_bot.side_effect = create_bot_file

        run_performance_test()
        self.assertTrue(os.path.exists(os.path.join('data', 'performance_report.json')))