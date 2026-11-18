import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


class ChatTests(unittest.TestCase):
    def test_empty_index_returns_an_ingestion_hint(self):
        # Both OpenAI clients are initialized when the app is imported.
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unused-in-this-test"}):
            import main

        with (
            patch.object(main, "embed_text", return_value=[0.0]),
            patch.object(main, "retrieve_chunks", return_value=[]),
            patch.object(main, "client") as chat_client,
            TestClient(main.app) as http,
        ):
            response = http.post("/chat", json={"question": "Where can I start?"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["sources"], [])
        self.assertIn("ingest.py", body["answer"])
        chat_client.chat.completions.create.assert_not_called()
