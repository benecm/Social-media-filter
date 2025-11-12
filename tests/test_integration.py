import os
import json
import unittest
from unittest.mock import patch, MagicMock
import shutil

# Mielőtt importáljuk az appot, beállítjuk a környezeti változót a teszteléshez
os.environ['FLASK_ENV'] = 'testing'
from webapp import app

# Teszt konstansok
TEST_DATA_DIR = "test_data"
TEST_STATIC_DIR = "test_static"

# Mock adatok
MOCK_COMMENTS = ["This is a great video!", "I learned a lot.", "Could be better."]
MOCK_SENTIMENT_RESULTS = [
    {"Comment": "This is a great video!", "Polarity": 0.8, "Sentiment": "Positive"},
    {"Comment": "I learned a lot.", "Polarity": 0.5, "Sentiment": "Positive"},
    {"Comment": "Could be better.", "Polarity": 0.2, "Sentiment": "Positive"}, # TextBlob ezt pozitívnak értékeli
]
MOCK_BOT_RESULTS = [
    {"Comment": "This is a great video!", "Prediction": "human"},
    {"Comment": "I learned a lot.", "Prediction": "human"},
    {"Comment": "Could be better.", "Prediction": "human"},
]
MOCK_SUMMARY = "This is a great summary of the comments."

class IntegrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Osztályszintű beállítás a tesztek futása előtt."""
        # Teszt könyvtárak létrehozása
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        os.makedirs(TEST_STATIC_DIR, exist_ok=True)

        # Patch-ek beállítása, hogy a tesztkönyvtárakat használjuk
        cls.patchers = [
            patch('webapp.DATA_DIR', TEST_DATA_DIR),
            patch('webapp.SENTIMENT_RESULTS_PATH', os.path.join(TEST_DATA_DIR, "sentiment_results.json")),
            patch('webapp.STATIC_DIR', TEST_STATIC_DIR),
            patch('webapp.DIAGRAM_OUTPUT_PATH', os.path.join(TEST_STATIC_DIR, 'sentiment_diagram.png')),
            patch('langchain_tools.DATA_DIR', TEST_DATA_DIR),
            patch('langchain_tools.COMMENTS_PATH', os.path.join(TEST_DATA_DIR, "comments.json")),
            patch('langchain_tools.SENTIMENT_RESULTS_PATH', os.path.join(TEST_DATA_DIR, "sentiment_results.json")),
            patch('langchain_tools.BOT_DETECTION_RESULTS_PATH', os.path.join(TEST_DATA_DIR, "bot_detection_results.json")),
            patch('langchain_tools.SUMMARY_PATH', os.path.join(TEST_DATA_DIR, "summary.json")),
            patch('diagram.generate_diagram', MagicMock()), # A diagram generálást mockoljuk
        ]
        for patcher in cls.patchers:
            patcher.start()

    @classmethod
    def tearDownClass(cls):
        """Osztályszintű takarítás a tesztek futása után."""
        for patcher in cls.patchers:
            patcher.stop()
        # Teszt könyvtárak törlése
        shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
        shutil.rmtree(TEST_STATIC_DIR, ignore_errors=True)

    def setUp(self):
        """Minden teszt előtt lefut."""
        app.config['TESTING'] = True
        self.app = app.test_client()
        # Takarítsuk ki a tesztkönyvtárakat minden teszt előtt
        if os.path.exists(TEST_DATA_DIR):
            for f in os.listdir(TEST_DATA_DIR):
                os.remove(os.path.join(TEST_DATA_DIR, f))

    @patch('webapp.AnalysisAgent')
    def test_analyze_endpoint_success(self, MockAnalysisAgent):
        """Teszteli az /analyze végpont sikeres működését."""
        # Mock AnalysisAgent beállítása
        mock_agent_instance = MockAnalysisAgent.return_value
        mock_agent_instance.analyze_video.return_value = {
            "results": MOCK_SENTIMENT_RESULTS,
            "summary": {"llm_summary": MOCK_SUMMARY}
        }

        # POST kérés küldése
        response = self.app.post('/analyze',
                                 data=json.dumps({'video_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'comment_count': 3}),
                                 content_type='application/json')

        # Válasz ellenőrzése
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('results', data)
        self.assertIn('summary', data)
        self.assertEqual(len(data['results']), 3)
        self.assertEqual(data['summary']['llm_summary'], MOCK_SUMMARY)

        # Ellenőrizzük, hogy az agent a megfelelő paraméterekkel lett-e hívva
        mock_agent_instance.analyze_video.assert_called_once_with('https://www.youtube.com/watch?v=dQw4w9WgXcQ', 3)

    def test_analyze_endpoint_invalid_input(self):
        """Teszteli az /analyze végpont hibakezelését érvénytelen bemenettel."""
        # Teszt üres URL-lel
        response = self.app.post('/analyze',
                                 data=json.dumps({'video_url': '', 'comment_count': 10}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Nincs megadva YouTube link!')

        # Teszt negatív komment számmal
        response = self.app.post('/analyze',
                                 data=json.dumps({'video_url': 'some_url', 'comment_count': -5}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'A kommentek száma nem lehet 0 vagy negatív!')

    @patch('webapp.init_agent')
    @patch('webapp.generate_diagram')
    @patch('webapp.send_file')
    def test_diagram_endpoint(self, mock_send_file, mock_generate_diagram, mock_init_agent):
        """Teszteli a /diagram végpont működését."""
        # Mock sentiment results fájl létrehozása
        sentiment_file_path = os.path.join(TEST_DATA_DIR, "sentiment_results.json")
        with open(sentiment_file_path, 'w', encoding='utf-8') as f:
            json.dump(MOCK_SENTIMENT_RESULTS, f)

        # A diagram végpont hívása
        response = self.app.get('/diagram')

        # Ellenőrzések
        mock_generate_diagram.assert_called_once()
        mock_send_file.assert_called_once()
        self.assertEqual(response.status_code, 200)

    def test_diagram_endpoint_no_file(self):
        """Teszteli a /diagram végpontot, ha hiányzik az adatfájl."""
        # Győződjünk meg róla, hogy a fájl nem létezik
        sentiment_file_path = os.path.join(TEST_DATA_DIR, "sentiment_results.json")
        if os.path.exists(sentiment_file_path):
            os.remove(sentiment_file_path)

        response = self.app.get('/diagram')
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()