import unittest
import os
import json
from unittest.mock import patch, mock_open

from main import load_json, run_analysis

class TestMain(unittest.TestCase):

    def setUp(self):
        """Tesztkörnyezet beállítása."""
        self.test_dir = 'test_data_main'
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        self.comments_path = os.path.join(self.test_dir, 'comments.json')

    def tearDown(self):
        """Tesztkörnyezet eltakarítása."""
        if os.path.exists(self.comments_path):
            os.remove(self.comments_path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    def test_load_json_success(self):
        """Sikeres JSON betöltés tesztelése."""
        dummy_data = {"key": "value"}
        with open(self.comments_path, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f)
        data = load_json(self.comments_path)
        self.assertEqual(data, dummy_data)

    @patch('builtins.print')
    def test_load_json_file_not_found(self, mock_print):
        """Nem létező fájl betöltésének tesztelése."""
        data = load_json("non_existent_file.json")
        self.assertIsNone(data)
        mock_print.assert_called_with("Hiba a(z) non_existent_file.json beolvasásakor: [Errno 2] No such file or directory: 'non_existent_file.json'")

    @patch('main.summarize_with_rag')
    @patch('main.analyze_sentiment')
    @patch('main.load_json')
    @patch('builtins.open', new_callable=mock_open)
    @patch('builtins.print')
    def test_run_analysis(self, mock_print, mock_file, mock_load_json, mock_analyze, mock_summarize):
        """A run_analysis folyamat tesztelése."""
        # Mock-ok beállítása
        mock_load_json.side_effect = [
            ["comment1", "comment2"],  # Első hívás (comments_data)
            [{"Sentiment": "Positive"}], # Második hívás (sentiment_result)
            [{"Sentiment": "Positive"}]  # Harmadik hívás (sentiment_data)
        ]
        mock_summarize.return_value = "This is a summary." # A summarize_with_rag egy stringet ad vissza

        # A függvény futtatása
        run_analysis()

        # Ellenőrzések
        self.assertEqual(mock_load_json.call_count, 3)
        mock_analyze.assert_called_once()
        mock_summarize.assert_called_once_with([{"Sentiment": "Positive"}])
        
        # Ellenőrizzük, hogy a summary mentésre került-e
        expected_summary_json = json.dumps({"llm_summary": "This is a summary."}, ensure_ascii=False, indent=4)
        handle = mock_file()
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        self.assertEqual(written_data, expected_summary_json)