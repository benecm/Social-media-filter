import unittest
from unittest.mock import patch, mock_open
import json
import os
from research.Functions import extract_video_id, remove_emojis, get_youtube_comments, save_comments_to_json

class TestFunctions(unittest.TestCase):

    def test_extract_video_id(self):
        self.assertEqual(extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertIsNone(extract_video_id("https://www.google.com"))

    def test_remove_emojis(self):
        self.assertEqual(remove_emojis("Ez egy teszt szöveg 👍"), "Ez egy teszt szöveg ")
        self.assertEqual(remove_emojis("😀😃😄😁😆😅😂🤣😊😇"), "")
        self.assertEqual(remove_emojis("Nincs emoji"), "Nincs emoji")

    @patch('research.Functions.build')
    def test_get_youtube_comments(self, mock_build):
        # Mock YouTube API response
        mock_youtube_instance = mock_build.return_value
        mock_comment_threads = mock_youtube_instance.commentThreads.return_value
        mock_list = mock_comment_threads.list
        mock_list.return_value.execute.return_value = {
            "items": [
                {
                    "snippet": {
                        "topLevelComment": {
                            "snippet": {
                                "textDisplay": "Ez az első komment."
                            }
                        }
                    }
                },
                {
                    "snippet": {
                        "topLevelComment": {
                            "snippet": {
                                "textDisplay": "Ez a második komment. 👍"
                            }
                        }
                    }
                }
            ]
        }
        mock_comment_threads.list_next.return_value = None

        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        comments = get_youtube_comments(video_url, api_key="test_key", max_results=2)

        self.assertEqual(len(comments), 2)
        self.assertIn("Ez az első komment.", comments)
        self.assertIn("Ez a második komment. ", comments) # Emoji removed

    def test_save_comments_to_json(self):
        comments = ["komment 1", "komment 2"]
        filename = "test_comments.json"

        # Mock open to avoid actual file I/O
        m = mock_open()
        with patch('builtins.open', m):
            save_comments_to_json(comments, filename)

        # Check if open was called correctly
        m.assert_called_once_with(filename, "w", encoding="utf-8")

        # Check what was written to the file
        handle = m()
        
        # json.dump calls handle.write multiple times
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        
        # The output of json.dump with indent=4 will have specific formatting
        expected_json_string = json.dumps(comments, ensure_ascii=False, indent=4)

        self.assertEqual(written_data, expected_json_string)