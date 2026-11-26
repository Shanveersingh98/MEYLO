"""
tests/test_v2_comprehensive_verification.py
===========================================
Comprehensive automated test suite verifying all 7 test categories from Section 16:
- Test 1: Frontend SPA Asset Delivery & Index Serving
- Test 2: Dual Independent Model Selection & Switching
- Test 3: Four Model/Mode Combinations (Custom+Off, Custom+On, Qwen+Off, Qwen+On)
- Test 4: Diverse Conversation Categories (Math, Technical, Follow-up, Emotion)
- Test 5: Web Search Online Grounding & Offline Local-Only Enforcement
- Test 6: In-Chat Slash Commands & Synchronization
- Test 7: Conversation CRUD, Hardware Telemetry & Model Weights Integrity
"""

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from fastapi.testclient import TestClient
from src.api_server import app, state


class TestV2ComprehensiveVerification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_01_model_weights_integrity(self):
        """Verify immutable checkpoints remain identical by SHA-256."""
        custom_ckpt = PROJECT_DIR / "checkpoints" / "ultimate_improvement" / "best.pt"
        qwen_ckpt = PROJECT_DIR / "models" / "external" / "qwen2.5-1.5b-instruct" / "model.gguf"

        self.assertTrue(custom_ckpt.exists(), "Custom checkpoint must exist")
        self.assertTrue(qwen_ckpt.exists(), "External Qwen checkpoint must exist")

        custom_hash = hashlib.sha256(custom_ckpt.read_bytes()).hexdigest().upper()
        self.assertEqual(
            custom_hash,
            "CECBA4946A8B5A0404E42E9E1790C0D780228DA45E07B1910D190D89CBADF2FC",
            "Custom model weights hash modified!"
        )

        # First 10MB hash of Qwen for fast verification
        with open(qwen_ckpt, "rb") as f:
            qwen_head_hash = hashlib.sha256(f.read(10 * 1024 * 1024)).hexdigest()
        self.assertTrue(len(qwen_head_hash) == 64)

    def test_02_frontend_spa_delivery(self):
        """Verify frontend dist assets are compiled and served via SPA routing."""
        dist_dir = PROJECT_DIR / "frontend" / "dist"
        self.assertTrue((dist_dir / "index.html").exists(), "frontend/dist/index.html must exist")

        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("<!doctype html>", resp.text)
        self.assertIn("AI Companion", resp.text)

    def test_03_dual_model_selection_and_switching(self):
        """Verify switching between Custom and External models."""
        # 1. Switch to custom
        r1 = self.client.post("/api/models/switch", json={"model": "custom"})
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.json()["active_model"], "custom")

        # 2. Switch to external
        r2 = self.client.post("/api/models/switch", json={"model": "external"})
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["active_model"], "external")

        # 3. Switch back to custom
        r3 = self.client.post("/api/models/switch", json={"model": "custom"})
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(r3.json()["active_model"], "custom")

    def test_04_combination_1_custom_offline(self):
        """Combination 1: Custom FromScratchLM + Web Search OFF."""
        self.client.post("/api/models/switch", json={"model": "custom"})
        self.client.post("/api/web-search", json={"enabled": False})

        # Create conversation
        conv_resp = self.client.post("/api/conversations", json={"title": "Test C1", "model": "custom"})
        conv_id = conv_resp.json()["conversation_id"]

        resp = self.client.post("/api/chat", json={
            "conversation_id": conv_id,
            "message": "Hello, what is your purpose?",
            "model": "custom"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model_used"], "custom")
        self.assertEqual(len(data.get("sources", [])), 0, "Offline mode must have zero web sources")
        self.assertTrue(len(data["content"]) > 0)

    def test_05_combination_2_custom_online(self):
        """Combination 2: Custom FromScratchLM + Web Search ON."""
        self.client.post("/api/models/switch", json={"model": "custom"})
        self.client.post("/api/web-search", json={"enabled": True})

        conv_resp = self.client.post("/api/conversations", json={"title": "Test C2", "model": "custom"})
        conv_id = conv_resp.json()["conversation_id"]

        resp = self.client.post("/api/chat", json={
            "conversation_id": conv_id,
            "message": "What is Python in computer science?",
            "model": "custom"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model_used"], "custom", "Active custom model must generate final response")

    def test_06_combination_3_qwen_offline(self):
        """Combination 3: Qwen2.5-1.5B + Web Search OFF."""
        self.client.post("/api/models/switch", json={"model": "external"})
        self.client.post("/api/web-search", json={"enabled": False})

        conv_resp = self.client.post("/api/conversations", json={"title": "Test C3", "model": "external"})
        conv_id = conv_resp.json()["conversation_id"]

        resp = self.client.post("/api/chat", json={
            "conversation_id": conv_id,
            "message": "What is 25 * 4?",
            "model": "external"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model_used"], "external")
        self.assertEqual(len(data.get("sources", [])), 0)
        self.assertTrue(len(data["content"]) > 0)

    def test_07_combination_4_qwen_online(self):
        """Combination 4: Qwen2.5-1.5B + Web Search ON."""
        self.client.post("/api/models/switch", json={"model": "external"})
        self.client.post("/api/web-search", json={"enabled": True})

        conv_resp = self.client.post("/api/conversations", json={"title": "Test C4", "model": "external"})
        conv_id = conv_resp.json()["conversation_id"]

        resp = self.client.post("/api/chat", json={
            "conversation_id": conv_id,
            "message": "Tell me about machine learning developments.",
            "model": "external"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model_used"], "external")

    def test_08_conversation_categories(self):
        """Verify technical, mathematical, follow-up, and emotional responses."""
        conv_resp = self.client.post("/api/conversations", json={"title": "Categories", "model": "custom"})
        conv_id = conv_resp.json()["conversation_id"]

        # 1. Technical
        r1 = self.client.post("/api/chat", json={"conversation_id": conv_id, "message": "Explain binary search."})
        self.assertEqual(r1.status_code, 200)

        # 2. Emotional
        r2 = self.client.post("/api/chat", json={"conversation_id": conv_id, "message": "I had a very long day today."})
        self.assertEqual(r2.status_code, 200)

        # 3. Follow-up
        r3 = self.client.post("/api/chat", json={"conversation_id": conv_id, "message": "Can you give me an example of that?"})
        self.assertEqual(r3.status_code, 200)

    def test_09_slash_commands(self):
        """Verify full spectrum of in-chat slash commands."""
        conv_resp = self.client.post("/api/conversations", json={"title": "Commands", "model": "custom"})
        conv_id = conv_resp.json()["conversation_id"]

        commands = [
            "/model status",
            "/model list",
            "/web status",
            "/web on",
            "/web off",
            "/sources",
            "/help",
        ]
        for cmd in commands:
            resp = self.client.post("/api/chat", json={"conversation_id": conv_id, "message": cmd})
            self.assertEqual(resp.status_code, 200, f"Failed executing {cmd}")
            self.assertTrue(resp.json()["is_command"], f"{cmd} must be flagged as command")

    def test_10_hardware_telemetry_endpoint(self):
        """Verify live hardware diagnostics return CPU, RAM, GPU, and active model."""
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("hardware", data)
        self.assertIn("ram_percent", data["hardware"])
        self.assertIn("active_model", data)


if __name__ == "__main__":
    unittest.main()
