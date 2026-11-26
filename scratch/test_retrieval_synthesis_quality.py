"""
scratch/test_retrieval_synthesis_quality.py
============================================
Focused verification for Web Retrieval & Answer Synthesis Quality.
Tests the four previously failed cases plus emotional & offline safeguards:
1. "Why does the human brain need sleep?"
2. "Why do flamingos stand on one leg?"
3. "Compare two smartphone processors with actual specifications."
4. "Find recent research papers about AI and explain the findings."
5. Multi-turn emotional continuity safeguard
6. Offline mode educational question
"""

import sys
import os
import unittest
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

import torch
from chat import load_chatbot, generate_chat_response


class TestRetrievalSynthesisQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== Initializing Chatbot for Quality Verification ===")
        cls.device = torch.device("cpu")
        cls.model, cls.tokenizer, cls.retrieval_engine, cls.ckpt = load_chatbot(cls.device)
        print("Chatbot loaded successfully.\n")

    def _respond(self, prompt, history=None, web_search=True):
        if history is None:
            history = []
        history.append({"role": "user", "text": prompt})
        reply, route = generate_chat_response(
            model=self.model,
            tokenizer=self.tokenizer,
            history=list(history),
            device=self.device,
            retrieval_engine=self.retrieval_engine,
            tool_assisted=True,
            web_search_enabled=web_search
        )
        history.append({"role": "assistant", "text": reply})
        return reply, route

    def test_01_why_human_brain_needs_sleep(self):
        print("\n--- Test 1: Why does the human brain need sleep? ---")
        reply, route = self._respond("Why does the human brain need sleep?", web_search=True)
        print(f"Route: {route}")
        print(f"Reply:\n{reply}\n")

        self.assertEqual(route, "web_grounded_evidence")
        reply_low = reply.lower()
        # Verify it did not fall back to emotional filler
        self.assertNotIn("i'm right here with you", reply_low)
        self.assertNotIn("feel free to share whatever is on your mind", reply_low)

        # Verify biological necessity explanation (glymphatic waste, restoration, memory consolidation)
        has_biological_reason = any(
            k in reply_low for k in [
                "restore", "restoration", "waste", "glymphatic", "metabolic",
                "memory consolidation", "anabolic", "cognitive", "repair"
            ]
        )
        self.assertTrue(has_biological_reason, f"Response must explain biological necessity/function of sleep. Got: {reply[:300]}")

    def test_02_why_flamingos_stand_on_one_leg(self):
        print("\n--- Test 2: Why do flamingos stand on one leg? ---")
        reply, route = self._respond("Why do flamingos stand on one leg?", web_search=True)
        print(f"Route: {route}")
        print(f"Reply:\n{reply}\n")

        self.assertEqual(route, "web_grounded_evidence")
        reply_low = reply.lower()
        # Verify it did not fall back to emotional filler
        self.assertNotIn("i'm right here with you", reply_low)
        self.assertNotIn("feel free to share whatever is on your mind", reply_low)

        # Verify biological explanation (thermoregulation/heat conservation or biomechanical balance/muscle energy)
        has_flamingo_reason = any(
            k in reply_low for k in [
                "heat", "conserve", "thermoregulation", "muscle", "energy",
                "fatigue", "balance", "sway", "wading in cold water"
            ]
        )
        self.assertTrue(has_flamingo_reason, f"Response must explain biological reasons why flamingos stand on one leg. Got: {reply[:300]}")

    def test_03_compare_smartphone_processors_with_actual_specs(self):
        print("\n--- Test 3: Compare two smartphone processors with actual specifications. ---")
        reply, route = self._respond("Compare two smartphone processors with actual specifications.", web_search=True)
        print(f"Route: {route}")
        print(f"Reply:\n{reply}\n")

        reply_low = reply.lower()
        # Verify it did not emit placeholder boilerplate
        self.assertNotIn("focuses on apple a-series principles", reply_low)
        self.assertNotIn("focuses on qualcomm snapdragon principles", reply_low)

        # Verify comparison table exists
        self.assertIn("|", reply, "Response must contain a comparison table.")

        # Verify concrete models and real specs
        has_concrete_models = any(k in reply_low for k in ["a18 pro", "a18", "a17", "snapdragon 8 elite", "8 gen 3", "snapdragon 8"])
        self.assertTrue(has_concrete_models, f"Response must resolve concrete processor models. Got: {reply[:300]}")

        has_actual_specs = any(k in reply_low for k in ["3nm", "cpu", "gpu", "npu", "tops", "oryon", "modem"])
        self.assertTrue(has_actual_specs, f"Response must provide actual verified hardware specifications. Got: {reply[:300]}")

    def test_04_find_recent_research_papers_about_ai(self):
        print("\n--- Test 4: Find recent research papers about AI and explain the findings. ---")
        reply, route = self._respond("Find recent research papers about AI and explain the findings.", web_search=True)
        print(f"Route: {route}")
        print(f"Reply:\n{reply}\n")

        reply_low = reply.lower()
        # Verify it does NOT hardcode the old stale DeepSeek-R1 from January 2025
        self.assertNotIn("published in january 2025, deepseek-r1", reply_low)

        # Verify dynamic research evidence
        has_arxiv_or_papers = any(k in reply_low for k in ["arxiv", "2026", "findings", "abstract", "paper", "preprints", "research"])
        self.assertTrue(has_arxiv_or_papers, f"Response must synthesize recent research papers. Got: {reply[:300]}")

    def test_05_emotional_continuity_safeguard(self):
        print("\n--- Test 5: Emotional continuity safeguard ---")
        hist = []
        # Turn 1: Interpersonal emotional distress
        r1, route1 = self._respond("I had a disagreement with my best friend today.", history=hist, web_search=True)
        print(f"Turn 1 route: {route1}, response: {r1[:120]}...")
        self.assertTrue(any(k in r1.lower() for k in ["friend", "close to you", "matters that much", "conflict", "disagreement", "sorry to hear"]))

        # Turn 2: Follow-up asking for interpersonal advice
        r2, route2 = self._respond("What should I do?", history=hist, web_search=True)
        print(f"Turn 2 route: {route2}, response: {r2[:120]}...")
        # Must give friendly companion advice, NOT search or educational definition
        r2_low = r2.lower()
        self.assertNotIn("wikipedia", r2_low)
        self.assertTrue(any(k in r2_low for k in ["talk", "space", "perspective", "listen", "message", "hear them", "conversation"]))

    def test_06_offline_educational_definition(self):
        print("\n--- Test 6: Offline educational definition ---")
        reply, route = self._respond("What is machine learning?", web_search=False)
        print(f"Offline ML route: {route}, response: {reply[:120]}...")
        self.assertEqual(route, "offline_educational_grounded")
        self.assertTrue(any(k in reply.lower() for k in ["machine learning", "algorithms", "data", "learn", "patterns"]))


if __name__ == "__main__":
    unittest.main()
