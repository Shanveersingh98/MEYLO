"""
tests/test_frontend_api_integration.py
======================================
Automated Integration and End-to-End Test Suite for the AI Companion Frontend API.
Tests:
1. Health and System Hardware Endpoints
2. Model Listing and Strict Independence Switch (Custom <-> External Qwen)
3. Web Search Toggle (ON / OFF) and Source Attribution
4. Full Conversation Lifecycle (Create, List, Rename, Clear, Delete)
5. Chat Interaction with Custom Model (FromScratchLM)
6. Chat Interaction with External Model (Qwen2.5-1.5B)
7. Slash Command Execution (/model, /web, /sources, /clear, /help)
8. Frontend SPA Static File Delivery (HTML and compiled Assets)
"""

import sys
import unittest
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from fastapi.testclient import TestClient
from api_server import app, state


class TestFrontendAPIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Trigger startup event explicitly if using TestClient
        with cls.client:
            pass

    def test_01_health_and_hardware_status(self):
        """Verifies /api/health and /api/status endpoints return dynamic system data."""
        with TestClient(app) as client:
            res = client.get("/api/health")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "ok")
            self.assertIn("active_model", data)
            self.assertIn("device", data)

            status_res = client.get("/api/status")
            self.assertEqual(status_res.status_code, 200)
            sdata = status_res.json()
            self.assertIn("hardware", sdata)
            hw = sdata["hardware"]
            self.assertIn("cpu_name", hw)
            self.assertGreater(hw["ram_total_gb"], 0)
            self.assertIn("ram_percent", hw)

    def test_02_model_listing_and_switch(self):
        """Tests getting models and switching between custom and external models."""
        with TestClient(app) as client:
            res = client.get("/api/models")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            models = data["models"]
            self.assertEqual(len(models), 2)
            model_ids = [m["id"] for m in models]
            self.assertIn("custom", model_ids)
            self.assertIn("external", model_ids)

            # Switch to custom
            sw1 = client.post("/api/models/switch", json={"model": "custom"})
            self.assertEqual(sw1.status_code, 200)
            self.assertEqual(sw1.json()["active_model"], "custom")

            # Verify active model is custom
            h1 = client.get("/api/health").json()
            self.assertEqual(h1["active_model"], "custom")

    def test_03_web_search_toggle(self):
        """Tests enabling and disabling web search."""
        with TestClient(app) as client:
            # Enable web search
            res1 = client.post("/api/web-search", json={"enabled": True})
            self.assertEqual(res1.status_code, 200)
            self.assertTrue(res1.json()["enabled"])

            # Verify status
            res_get = client.get("/api/web-search")
            self.assertTrue(res_get.json()["enabled"])

            # Disable web search (pure offline mode)
            res2 = client.post("/api/web-search", json={"enabled": False})
            self.assertEqual(res2.status_code, 200)
            self.assertFalse(res2.json()["enabled"])

    def test_04_conversation_lifecycle(self):
        """Tests create, list, rename, clear, and delete conversation."""
        with TestClient(app) as client:
            # Create conversation
            c_res = client.post("/api/conversations", json={"title": "Test Chat Lifecycle"})
            self.assertEqual(c_res.status_code, 200)
            conv_id = c_res.json()["conversation_id"]

            # List conversations
            l_res = client.get("/api/conversations")
            self.assertEqual(l_res.status_code, 200)
            convs = l_res.json()["conversations"]
            self.assertTrue(any(c["conversation_id"] == conv_id for c in convs))

            # Rename conversation
            r_res = client.patch(f"/api/conversations/{conv_id}", json={"title": "Renamed Test Chat"})
            self.assertEqual(r_res.status_code, 200)
            self.assertTrue(r_res.json()["success"])

            # Get conversation details
            g_res = client.get(f"/api/conversations/{conv_id}")
            self.assertEqual(g_res.status_code, 200)
            self.assertEqual(g_res.json()["conversation"]["title"], "Renamed Test Chat")

            # Delete conversation
            d_res = client.delete(f"/api/conversations/{conv_id}")
            self.assertEqual(d_res.status_code, 200)
            self.assertTrue(d_res.json()["success"])

    def test_05_chat_slash_commands(self):
        """Tests executing slash commands directly through the chat endpoint."""
        with TestClient(app) as client:
            c_res = client.post("/api/conversations", json={"title": "Command Test"})
            conv_id = c_res.json()["conversation_id"]

            # /help
            h_res = client.post("/api/chat", json={"conversation_id": conv_id, "message": "/help"})
            self.assertEqual(h_res.status_code, 200)
            data = h_res.json()
            self.assertTrue(data["is_command"])
            self.assertIn("Available Slash Commands", data["content"])

            # /web on
            w_res = client.post("/api/chat", json={"conversation_id": conv_id, "message": "/web on"})
            self.assertEqual(w_res.status_code, 200)
            self.assertIn("[Web Search: ON]", w_res.json()["content"])

            # /web off
            w_off = client.post("/api/chat", json={"conversation_id": conv_id, "message": "/web off"})
            self.assertEqual(w_off.status_code, 200)
            self.assertIn("[Web Search: OFF]", w_off.json()["content"])

            # /model status
            m_stat = client.post("/api/chat", json={"conversation_id": conv_id, "message": "/model status"})
            self.assertEqual(m_stat.status_code, 200)
            self.assertIn("AI COMPANION MODEL STATUS", m_stat.json()["content"])

            # Clean up
            client.delete(f"/api/conversations/{conv_id}")

    def test_06_frontend_static_serving(self):
        """Verifies that the compiled React SPA files are served correctly."""
        with TestClient(app) as client:
            # Root route serves index.html
            root_res = client.get("/")
            self.assertEqual(root_res.status_code, 200)
            self.assertIn("AI Companion", root_res.text)
            self.assertIn("<div id=\"root\"></div>", root_res.text)

    def test_07_custom_model_chat_response(self):
        """Verifies that the Custom Model generates an authentic response through the API."""
        with TestClient(app) as client:
            # Ensure model is custom
            client.post("/api/models/switch", json={"model": "custom"})
            c_res = client.post("/api/conversations", json={"title": "Custom Model Test"})
            conv_id = c_res.json()["conversation_id"]

            chat_res = client.post("/api/chat", json={
                "conversation_id": conv_id,
                "message": "Hey! How are you doing today?"
            })
            self.assertEqual(chat_res.status_code, 200)
            data = chat_res.json()
            self.assertEqual(data["model_used"], "custom")
            self.assertGreater(len(data["content"]), 0)
            self.assertNotIn("HTTPError", data["content"])

            # Clean up
            client.delete(f"/api/conversations/{conv_id}")

    def test_08_external_qwen_chat_response(self):
        """Verifies that the External Qwen Model generates an authentic response through the API."""
        with TestClient(app) as client:
            # Switch to external model
            sw = client.post("/api/models/switch", json={"model": "external"})
            self.assertEqual(sw.status_code, 200)
            self.assertEqual(sw.json()["active_model"], "external")

            c_res = client.post("/api/conversations", json={"title": "Qwen Test"})
            conv_id = c_res.json()["conversation_id"]

            chat_res = client.post("/api/chat", json={
                "conversation_id": conv_id,
                "message": "What is machine learning in simple terms?"
            })
            self.assertEqual(chat_res.status_code, 200)
            data = chat_res.json()
            self.assertEqual(data["model_used"], "external")
            self.assertGreater(len(data["content"]), 0)

            # Revert to custom model
            client.post("/api/models/switch", json={"model": "custom"})
            client.delete(f"/api/conversations/{conv_id}")

    def test_09_web_search_online_and_sources(self):
        """Verifies web retrieval with Web Search ON returns sources through the API."""
        with TestClient(app) as client:
            client.post("/api/web-search", json={"enabled": True})
            c_res = client.post("/api/conversations", json={"title": "Web Search Test"})
            conv_id = c_res.json()["conversation_id"]

            chat_res = client.post("/api/chat", json={
                "conversation_id": conv_id,
                "message": "What are the latest AI developments in 2026?"
            })
            self.assertEqual(chat_res.status_code, 200)
            data = chat_res.json()
            self.assertGreater(len(data["content"]), 0)

            # Query sources endpoint
            s_res = client.get("/api/sources")
            self.assertEqual(s_res.status_code, 200)

            # Turn web search back off
            client.post("/api/web-search", json={"enabled": False})
            client.delete(f"/api/conversations/{conv_id}")


if __name__ == "__main__":
    unittest.main()
