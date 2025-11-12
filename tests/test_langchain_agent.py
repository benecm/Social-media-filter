import unittest
from unittest.mock import patch, mock_open
import json
from langchain_agent import AnalysisAgent, SUMMARY_PATH

class TestAnalysisAgent(unittest.TestCase):

    @patch('langchain_agent.initialize_agent')
    def setUp(self, mock_initialize_agent):
        """Minden teszt előtt beállítja a tesztkörnyezetet."""
        self.mock_agent = mock_initialize_agent.return_value
        self.analysis_agent = AnalysisAgent()

    def test_analyze_video_success(self):
        """Teszteli az analyze_video metódus sikeres lefutását."""
        video_url = "https://youtube.com/watch?v=123"
        comment_count = 50
        expected_input_str = f"{video_url}|{comment_count}"
        
        # Mock-ok beállítása
        self.mock_agent.run.return_value = "Successfully processed" # Simulate a success message from the tool
        mock_summary_data = {"results": [], "summary": "Test summary"}
        
        # A `run` hívások sorrendjének ellenőrzése és a `summary.json` olvasásának mockolása
        with patch('os.path.exists', return_value=True):  # Mock os.path.exists to ensure file is "found"
            with patch('builtins.open', mock_open(read_data=json.dumps(mock_summary_data))) as mock_file:
                result = self.analysis_agent.analyze_video(video_url, comment_count)

        # Ellenőrzések
        self.assertEqual(self.mock_agent.run.call_count, 4)
        self.mock_agent.run.assert_any_call(expected_input_str)
        self.mock_agent.run.assert_any_call("analyze")
        self.mock_agent.run.assert_any_call("detect")
        self.mock_agent.run.assert_any_call("summarize")
        
        mock_file.assert_called_with(SUMMARY_PATH, "r", encoding="utf-8")
        self.assertEqual(result, mock_summary_data)

    def test_analyze_video_exception(self):
        """Teszteli az analyze_video metódus hibakezelését."""
        video_url = "https://youtube.com/watch?v=123"
        comment_count = 50
        
        # Hiba szimulálása
        self.mock_agent.run.side_effect = Exception("Agent failed")
        
        result = self.analysis_agent.analyze_video(video_url, comment_count)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Váratlan hiba történt az elemzés során: Agent failed")