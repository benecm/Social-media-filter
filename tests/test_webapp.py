import pytest
from unittest.mock import patch, MagicMock
import json
from webapp import app

@pytest.fixture
def client():
    """Pytest fixture a Flask teszt kliens létrehozásához."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@patch('webapp.init_agent') # Patch init_agent directly
def test_analyze_success(mock_init_agent, client):
    """Teszteli az /analyze végpont sikeres működését."""
    # Mock beállítása
    mock_agent_class = MagicMock() # Create a mock for AnalysisAgent if needed for its methods
    mock_agent_instance = mock_agent_class.return_value
    mock_analysis_result = {
        "results": [{"Comment": "Test comment", "Sentiment": "Positive"}],
        "summary": {"llm_summary": "Great video!"}
    }
    mock_agent_instance.analyze_video.return_value = mock_analysis_result

    mock_init_agent.return_value = mock_agent_instance # Ensure init_agent returns our mocked instance
    # Kérés küldése
    response = client.post('/analyze', json={
        'video_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'comment_count': 10
    })

    # Ellenőrzések
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data == mock_analysis_result
    mock_agent_instance.analyze_video.assert_called_once_with('https://www.youtube.com/watch?v=dQw4w9WgXcQ', 10)

def test_analyze_missing_url(client):
    """Teszteli az /analyze végpontot hiányzó URL esetén."""
    response = client.post('/analyze', json={'comment_count': 10})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'Nincs megadva YouTube link!'

def test_analyze_invalid_comment_count(client):
    """Teszteli az /analyze végpontot érvénytelen komment szám esetén."""
    response = client.post('/analyze', json={
        'video_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'comment_count': 0
    })
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'A kommentek száma nem lehet 0 vagy negatív!'

@patch('webapp.init_agent')
def test_analyze_agent_error(mock_init_agent, client):
    """Teszteli az /analyze végpontot, ha az agent hibát ad vissza."""
    # Mock beállítása az init_agent által visszaadott agent-re
    mock_agent_instance = MagicMock()
    mock_agent_instance.analyze_video.return_value = {"error": "Internal agent error"}
    mock_init_agent.return_value = mock_agent_instance

    response = client.post('/analyze', json={
        'video_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'comment_count': 10
    })
    assert response.status_code == 500
    data = json.loads(response.data)
    assert data['error'] == 'Internal agent error'