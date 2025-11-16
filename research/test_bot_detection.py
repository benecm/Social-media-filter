import unittest
import os
import json
from unittest.mock import patch, mock_open

from models.bot_detection_tdrenis import run_bot_detection

class TestBotDetection(unittest.TestCase):

    def setUp(self):
        """Tesztkörnyezet beállítása."""
        self.test_dir = 'test_data_bot'
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        self.input_path = os.path.join(self.test_dir, 'comments.json')
        self.output_path = os.path.join(self.test_dir, 'bot_results.json')

    def tearDown(self):
        """Tesztkörnyezet eltakarítása."""
        for path in [self.input_path, self.output_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    @patch('models.bot_detection_modell.AutoModelForSequenceClassification.from_pretrained')
    @patch('models.bot_detection_modell.AutoTokenizer.from_pretrained')
    def test_run_bot_detection_success(self, mock_tokenizer, mock_model):
        """Sikeres bot detektálás tesztelése."""
        # Mock-ok beállítása
        mock_tokenizer_instance = mock_tokenizer.return_value
        mock_model_instance = mock_model.return_value
        
        # Dummy adatok
        comments = ["This is a normal comment.", "Click here for free stuff!"]
        with open(self.input_path, 'w', encoding='utf-8') as f:
            json.dump(comments, f)

        # A modell kimenetének mockolása
        import torch
        mock_model_instance.return_value.logits = torch.tensor([[0.9, 0.1], [0.2, 0.8]])

        run_bot_detection(self.input_path, self.output_path)

        self.assertTrue(os.path.exists(self.output_path))
        with open(self.output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['Prediction'], 'human')
        self.assertEqual(results[1]['Prediction'], 'bot')

    def test_run_bot_detection_file_not_found(self):
        """Nem létező bemeneti fájl esetének tesztelése."""
        result = run_bot_detection("non_existent.json", self.output_path)
        self.assertEqual(result, [])
