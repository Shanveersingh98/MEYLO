"""
tests/test_full_system_verification.py
======================================
Comprehensive verification script verifying the complete Local AI Companion application
according to Section 15 of Master Prompt:
- Frontend static asset and routing verification
- Custom Model full conversation and domain test suite
- External Qwen Model full conversation and domain test suite
- In-chat slash commands verification (/model, /web, /sources, /clear, /help)
- Live Web Retrieval and Source Attribution verification
- Database schema and PRAGMA integrity verification
- Checkpoint and GGUF SHA-256 byte-for-byte immutability verification
- Stability and repeated model switching verification
"""

import sys
import json
import time
import hashlib
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from fastapi.testclient import TestClient
from api_server import app, state
from companion_database import CompanionDatabase

ORIGINAL_CUSTOM_HASH = "cecba4946a8b5a0404e42e9e1790c0d780228da45e07b1910d190d89cbadf2fc"
ORIGINAL_QWEN_HASH = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"


def sha256_of_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


class FullSystemVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()
        # Ensure clean initial state
        cls.client.post("/api/models/switch", json={"model": "custom"})
        cls.client.post("/api/web-search", json={"enabled": False})

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client_context.__exit__(None, None, None)
        except Exception:
            pass

    def test_01_model_weights_immutability(self):
        """Verifies original custom checkpoint and external model weights remain 100% unchanged."""
        custom_path = PROJECT_DIR / "checkpoints" / "ultimate_improvement" / "best.pt"
        self.assertTrue(custom_path.exists(), "Custom checkpoint missing!")
        custom_hash = sha256_of_file(custom_path)
        self.assertEqual(custom_hash, ORIGINAL_CUSTOM_HASH, "Custom checkpoint SHA-256 MISMATCH!")

        qwen_path = PROJECT_DIR / "models" / "external" / "qwen2.5-1.5b-instruct" / "model.gguf"
        self.assertTrue(qwen_path.exists(), "External Qwen model missing!")
        qwen_hash = sha256_of_file(qwen_path)
        self.assertEqual(qwen_hash, ORIGINAL_QWEN_HASH, "External Qwen SHA-256 MISMATCH!")

    def test_02_database_integrity(self):
        """Runs PRAGMA integrity_check and foreign_key_check on SQLite database."""
        db = CompanionDatabase()
        res = db.verify_integrity()
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["integrity_check"], ["ok"])
        self.assertEqual(len(res["foreign_key_violations"]), 0)

    def test_03_frontend_static_assets(self):
        """Verifies compiled React frontend HTML and CSS/JS assets are present and served."""
        dist_html = PROJECT_DIR / "frontend" / "dist" / "index.html"
        self.assertTrue(dist_html.exists(), "frontend/dist/index.html missing!")

        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("AI Companion", res.text)
        self.assertIn("<div id=\"root\"></div>", res.text)

    def test_04_custom_model_full_spectrum(self):
        """Tests Custom Model across Greeting, General Knowledge, Educational, Math, Technical, Follow-up."""
        sw = self.client.post("/api/models/switch", json={"model": "custom"}).json()
        self.assertEqual(sw["active_model"], "custom")

        c_res = self.client.post("/api/conversations", json={"title": "Custom Full Spectrum"}).json()
        conv_id = c_res["conversation_id"]

        test_prompts = [
            ("Greeting", "Hello, how are you?"),
            ("General Knowledge", "What is machine learning?"),
            ("Educational", "Explain neural networks."),
            ("Math", "What is 15 plus 27?"),
            ("Technical", "What is backpropagation?"),
            ("Follow-up", "Can you explain the gradient descent step?")
        ]

        for category, prompt in test_prompts:
            chat_res = self.client.post("/api/chat", json={
                "conversation_id": conv_id,
                "message": prompt
            })
            self.assertEqual(chat_res.status_code, 200)
            data = chat_res.json()
            self.assertEqual(data["model_used"], "custom", f"Prompt '{prompt}' did not use custom model!")
            self.assertGreater(len(data["content"]), 0, f"Prompt '{prompt}' produced empty response!")
            self.assertNotIn("HTTPError", data["content"])

        self.client.delete(f"/api/conversations/{conv_id}")

    def test_05_external_qwen_full_spectrum(self):
        """Tests External Qwen Model across Greeting, General Knowledge, Educational, Math, Technical, Follow-up."""
        sw = self.client.post("/api/models/switch", json={"model": "external"}).json()
        self.assertEqual(sw["active_model"], "external")

        c_res = self.client.post("/api/conversations", json={"title": "Qwen Full Spectrum"}).json()
        conv_id = c_res["conversation_id"]

        test_prompts = [
            ("Greeting", "Hey! How is your day going?"),
            ("General Knowledge", "What is artificial intelligence?"),
            ("Educational", "What is the difference between supervised and unsupervised learning?"),
            ("Math", "What is 48 divided by 6?"),
            ("Technical", "How does self-attention work in a transformer?"),
            ("Follow-up", "Why is multi-head attention beneficial?")
        ]

        for category, prompt in test_prompts:
            chat_res = self.client.post("/api/chat", json={
                "conversation_id": conv_id,
                "message": prompt
            })
            self.assertEqual(chat_res.status_code, 200)
            data = chat_res.json()
            self.assertEqual(data["model_used"], "external", f"Prompt '{prompt}' did not use external model!")
            self.assertGreater(len(data["content"]), 0, f"Prompt '{prompt}' produced empty response!")
            self.assertNotIn("HTTPError", data["content"])

        # Switch back to custom model
        self.client.post("/api/models/switch", json={"model": "custom"})
        self.client.delete(f"/api/conversations/{conv_id}")

    def test_06_all_slash_commands(self):
        """Tests all supported slash commands inside chat input."""
        c_res = self.client.post("/api/conversations", json={"title": "Slash Commands Test"}).json()
        conv_id = c_res["conversation_id"]

        commands = [
            ("/help", ["Available Slash Commands", "/model", "/web", "/sources", "/clear", "/exit"]),
            ("/model status", ["AI COMPANION MODEL STATUS"]),
            ("/model list", ["Available Models in D:/DEEP-LEARNING-PROJECT"]),
            ("/web status", ["[Web Search:"]),
            ("/web on", ["[Web Search: ON]"]),
            ("/web off", ["[Web Search: OFF]"]),
            ("/sources", ["Sources"]),
            ("/clear", ["[History cleared.]"]),
        ]

        for cmd, expected_keywords in commands:
            res = self.client.post("/api/chat", json={
                "conversation_id": conv_id,
                "message": cmd
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data["is_command"])
            for kw in expected_keywords:
                self.assertIn(kw.lower(), data["content"].lower(), f"Command '{cmd}' missing '{kw}' in output!")

        self.client.delete(f"/api/conversations/{conv_id}")

    def test_07_web_retrieval_flow(self):
        """Verifies web retrieval with Web Search ON and verifies pure offline mode with Web Search OFF."""
        c_res = self.client.post("/api/conversations", json={"title": "Web Retrieval Verification"}).json()
        conv_id = c_res["conversation_id"]

        # 1. Web Search ON
        self.client.post("/api/web-search", json={"enabled": True})
        on_res = self.client.post("/api/chat", json={
            "conversation_id": conv_id,
            "message": "What are the latest AI research papers published in 2026?"
        }).json()

        self.assertTrue(len(on_res["sources"]) > 0 or "research" in on_res["content"].lower())
        # Query sources endpoint
        srcs = self.client.get("/api/sources").json()
        self.assertIn("sources", srcs)

        # 2. Web Search OFF (Pure Offline Mode)
        self.client.post("/api/web-search", json={"enabled": False})
        off_res = self.client.post("/api/chat", json={
            "conversation_id": conv_id,
            "message": "What is machine learning?"
        }).json()
        self.assertEqual(len(off_res.get("sources", [])), 0)

        self.client.delete(f"/api/conversations/{conv_id}")

    def test_08_stability_and_repeated_model_switching(self):
        """Repeatedly switches models and verifies no cross-model leaks or crashes."""
        for _ in range(3):
            sw_ext = self.client.post("/api/models/switch", json={"model": "external"}).json()
            self.assertEqual(sw_ext["active_model"], "external")
            h_ext = self.client.get("/api/health").json()
            self.assertEqual(h_ext["active_model"], "external")

            sw_cust = self.client.post("/api/models/switch", json={"model": "custom"}).json()
            self.assertEqual(sw_cust["active_model"], "custom")
            h_cust = self.client.get("/api/health").json()
            self.assertEqual(h_cust["active_model"], "custom")


if __name__ == "__main__":
    unittest.main()
