"""
tests/test_runtime_failure_regression_suite.py
==============================================
Genuine Regression Test Suite reproducing and verifying fixes for actual runtime failures
reported in interactive testing:

1. Interpersonal Multi-turn Sequence:
   Tired day -> Disagreement with best friend -> Whether to message first -> Asking for advice.
   - Preserves emotional and situational context across turns.
   - Refers to the best friend and disagreement in subsequent advice.
   - Never demands web search for interpersonal advice.

2. Educational Roadmap & Multi-turn Sequence:
   Learn AI -> Receive roadmap -> Why Python comes first -> Request example -> What to learn next.
   - Preserves learning roadmap structure.
   - Explains why Python precedes Machine Learning (fundamentals, data handling, implementation).
   - Never substitutes generic ML definition when asked why Python comes first.

3. Programming Multi-turn Sequence:
   Write a Python calculator -> Explain the code -> Add multiplication and division.
   - Routes to programming assistance engine, not web search.
   - Produces syntactically valid, runnable Python code.
   - Explains the calculator functions and dispatcher.
   - Extends the calculator cleanly with multiply and divide operations.

4. Emotional Support with Web Search OFF:
   - Validates that emotional disclosures receive empathetic companion responses without web prompts.

5. Basic Educational Concepts with Web Search ON:
   - Validates that foundational educational concepts do not trigger unnecessary web searches.

6. Current Information Requiring Web Search:
   - Validates that genuinely freshness-sensitive queries trigger web-grounded retrieval.
"""

import sys
import unittest
import ast
import torch
from pathlib import Path
from typing import List, Dict, Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from chat import load_chatbot, generate_chat_response
from source_selection import SourceRecord, EvidenceBundle


class BaseRuntimeRegressionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model, cls.tokenizer, cls.retrieval_engine, cls.ckpt = load_chatbot(cls.device)

    def setUp(self):
        if hasattr(self.retrieval_engine, "conversation_memory"):
            self.retrieval_engine.conversation_memory.clear()
        if hasattr(self.retrieval_engine, "session_last_sources"):
            self.retrieval_engine.session_last_sources = []


class TestRuntimeFailuresRegression(BaseRuntimeRegressionTestCase):
    """
    Direct end-to-end tests through generate_chat_response reproducing the transcript failures.
    """

    def test_01_interpersonal_support_multiturn_sequence(self):
        """
        Sequence 1:
        User: Hey! I had a pretty tiring day today.
        User: I had a disagreement with my best friend.
        User: I don't know whether I should message them first.
        User: What would you do in this situation?
        """
        dialogue = [
            (
                "Hey! I had a pretty tiring day today.",
                "companion_tired",
                ["tiring", "exhausting", "drain", "unwind", "rest", "listen", "here"],
                ["c++", "machine learning", "language model", "web search"]
            ),
            (
                "I had a disagreement with my best friend.",
                "companion_emotional_hurt",
                ["disagreement", "friend", "best friend", "difficult", "listen", "share", "hurts"],
                ["1. find out the day", "avoid seeking professional advice", "web search", "c++"]
            ),
            (
                "I don't know whether I should message them first.",
                "companion_interpersonal_dilemma",
                ["reach out", "message", "friendship", "first", "dilemma", "cool down", "value"],
                ["please enable web search", "/web on", "offline information"]
            ),
            (
                "What would you do in this situation?",
                "companion_interpersonal_advice",
                ["shoes", "message", "friendship", "talk", "ready", "care", "settle", "pressure"],
                ["please enable web search", "/web on", "offline information", "machine learning"]
            )
        ]

        history = []
        for prompt, expected_route, expected_keywords, forbidden_keywords in dialogue:
            history.append({"role": "user", "text": prompt})
            reply, route = generate_chat_response(
                model=self.model,
                tokenizer=self.tokenizer,
                history=history,
                device=self.device,
                retrieval_engine=self.retrieval_engine,
                web_search_enabled=False
            )
            history.append({"role": "assistant", "text": reply})

            self.assertEqual(route, expected_route, f"Turn '{prompt}' routed to '{route}', expected '{expected_route}'")
            r_low = reply.lower()
            for forb in forbidden_keywords:
                self.assertNotIn(forb, r_low, f"Turn '{prompt}' contained forbidden phrase '{forb}'. Output:\n{reply}")
            has_expected = any(k in r_low for k in expected_keywords)
            self.assertTrue(has_expected, f"Turn '{prompt}' missing expected keywords {expected_keywords}. Output:\n{reply}")

    def test_02_educational_roadmap_and_followup_sequence(self):
        """
        Sequence 2:
        User: I want to learn artificial intelligence.
        User: Why did you put Python before machine learning?
        User: Give me an example.
        User: What should I learn next?
        """
        dialogue = [
            (
                "I want to learn artificial intelligence.",
                "companion_statement",
                ["roadmap", "python", "mathematics", "machine learning", "deep learning"],
                ["please enable web search", "/web on"]
            ),
            (
                "Why did you put Python before machine learning?",
                "companion_statement",
                ["programming fundamentals", "data handling", "practical implementation", "numpy", "pandas"],
                ["**machine learning (ml)** is a core subfield", "supervised learning:\n- the algorithm"]
            ),
            (
                "Give me an example.",
                "companion_statement",
                ["import numpy", "np.array", "preprocessed", "features", "normalized", "data"],
                ["please enable web search"]
            ),
            (
                "What should I learn next?",
                "companion_statement",
                ["machine learning foundations", "scikit-learn", "regression", "classification", "train/test"],
                ["please enable web search"]
            )
        ]

        history = []
        for prompt, expected_route, expected_keywords, forbidden_keywords in dialogue:
            history.append({"role": "user", "text": prompt})
            reply, route = generate_chat_response(
                model=self.model,
                tokenizer=self.tokenizer,
                history=history,
                device=self.device,
                retrieval_engine=self.retrieval_engine,
                web_search_enabled=False
            )
            history.append({"role": "assistant", "text": reply})

            self.assertEqual(route, expected_route, f"Turn '{prompt}' routed to '{route}', expected '{expected_route}'")
            r_low = reply.lower()
            for forb in forbidden_keywords:
                self.assertNotIn(forb, r_low, f"Turn '{prompt}' contained forbidden phrase '{forb}'. Output:\n{reply}")
            has_expected = any(k in r_low for k in expected_keywords)
            self.assertTrue(has_expected, f"Turn '{prompt}' missing expected keywords {expected_keywords}. Output:\n{reply}")

    def test_03_programming_calculator_multiturn_sequence(self):
        """
        Sequence 3:
        User: Write a Python calculator
        User: Explain the code
        User: Add multiplication and division
        """
        # Turn 1: Write a Python calculator
        hist = [{"role": "user", "text": "Write a Python calculator"}]
        reply1, route1 = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertEqual(route1, "programming_assistance")
        self.assertIn("```python", reply1)
        self.assertIn("def add", reply1)
        self.assertIn("def calculate", reply1)

        # Validate that the code block has valid Python syntax
        code_blocks = [block for block in reply1.split("```python")[1].split("```")[0]]
        ast.parse("".join(code_blocks))
        hist.append({"role": "assistant", "text": reply1})

        # Turn 2: Explain the code
        hist.append({"role": "user", "text": "Explain the code"})
        reply2, route2 = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertEqual(route2, "programming_assistance")
        self.assertTrue(any(w in reply2.lower() for w in ["calculate", "dispatcher", "add", "operations", "dictionary"]))
        hist.append({"role": "assistant", "text": reply2})

        # Turn 3: Add multiplication and division
        hist.append({"role": "user", "text": "Add multiplication and division"})
        reply3, route3 = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertEqual(route3, "programming_assistance")
        self.assertIn("```python", reply3)
        self.assertIn("def multiply", reply3)
        self.assertIn("def divide", reply3)
        self.assertIn("ZeroDivisionError", reply3)

        # Validate that the extended code block has valid Python syntax
        code_blocks3 = [block for block in reply3.split("```python")[1].split("```")[0]]
        ast.parse("".join(code_blocks3))

    def test_04_emotional_support_while_web_search_is_off(self):
        """
        Scenario 4: Emotional support while web search is OFF.
        Must not request web search or state lack of offline info.
        """
        prompts = [
            "I'm feeling really stressed and overwhelmed today.",
            "I had an argument with someone close to me today.",
            "I feel like I'm not making enough progress."
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                hist = [{"role": "user", "text": p}]
                reply, route = generate_chat_response(
                    model=self.model, tokenizer=self.tokenizer, history=hist,
                    device=self.device, retrieval_engine=self.retrieval_engine,
                    web_search_enabled=False
                )
                self.assertNotIn("Please enable web search", reply)
                self.assertNotIn("/web on", reply)
                self.assertNotIn("offline information", reply.lower())
                self.assertTrue(any(w in reply.lower() for w in ["breath", "listen", "understand", "support", "talk", "step", "here", "kind", "sorry", "share", "friend", "feeling"]))

    def test_05_basic_educational_questions_while_web_search_is_on(self):
        """
        Scenario 5: Basic educational questions while web search is ON.
        Must NOT trigger live web search or search engine calls.
        """
        prompts = [
            ("What is deep learning?", "deep learning"),
            ("What is the difference between AI and machine learning?", "hierarchical"),
            ("Explain CNN pooling.", "pooling")
        ]
        for q, expected_word in prompts:
            with self.subTest(query=q):
                hist = [{"role": "user", "text": q}]
                reply, route = generate_chat_response(
                    model=self.model, tokenizer=self.tokenizer, history=hist,
                    device=self.device, retrieval_engine=self.retrieval_engine,
                    web_search_enabled=True
                )
                # Must be answered by verified educational grounded knowledge, NOT web search failure or irrelevant web search
                self.assertEqual(route, "offline_educational_grounded")
                self.assertIn(expected_word, reply.lower())

    def test_06_current_information_questions_genuinely_requiring_web_search(self):
        """
        Scenario 6: Current information questions that genuinely require web search.
        Must route to live web retrieval.
        """
        mock_item = SourceRecord(
            title="Recent AI Research Developments in 2026",
            url="https://arxiv.org/abs/2601.12345",
            snippet="Recent developments in artificial intelligence in 2026 feature reasoning models and architectural scaling.",
            full_text="Recent developments in artificial intelligence in 2026 feature reasoning models and architectural scaling.",
            source_provider="openalex",
            rank=1
        )
        mock_bundle = EvidenceBundle(query="What is the latest AI news?", items=[mock_item], has_results=True)

        orig_search = self.retrieval_engine.search_and_retrieve
        self.retrieval_engine.search_and_retrieve = lambda *args, **kwargs: mock_bundle
        try:
            hist = [{"role": "user", "text": "What is the latest AI news?"}]
            reply, route = generate_chat_response(
                model=self.model, tokenizer=self.tokenizer, history=hist,
                device=self.device, retrieval_engine=self.retrieval_engine,
                web_search_enabled=True
            )
            self.assertEqual(route, "web_grounded_evidence")
            self.assertIn("Recent AI Research Developments", reply)

            # Follow up: asking for sources
            hist.append({"role": "assistant", "text": reply})
            hist.append({"role": "user", "text": "Give me the sources"})
            reply_src, route_src = generate_chat_response(
                model=self.model, tokenizer=self.tokenizer, history=hist,
                device=self.device, retrieval_engine=self.retrieval_engine,
                web_search_enabled=True
            )
            self.assertEqual(route_src, "sources_display")
            self.assertIn("arxiv.org", reply_src)
        finally:
            self.retrieval_engine.search_and_retrieve = orig_search


if __name__ == "__main__":
    unittest.main(verbosity=2)
