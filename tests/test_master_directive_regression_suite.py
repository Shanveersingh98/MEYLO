"""
tests/test_master_directive_regression_suite.py
===============================================
Comprehensive End-to-End Regression Suite for AI Companion Chatbot
covering all 30 regression test cases defined in Master Engineering Directive:

- Category 1: Conversation (Cases 1-4)
- Category 2: Emotions & Situations (Cases 5-9)
- Category 3: Multi-turn Context & Python Progression (Cases 10-13)
- Category 4: Educational Questions (Cases 14-17)
- Category 5: Programming & Technical Assistance (Cases 18-20)
- Category 6: Typo & Ambiguity Handling (Cases 21-24)
- Category 7: Web Retrieval & Gating (Cases 25-30)
- Both Offline Mode (Web OFF) and Online Mode (Web ON)
"""

import sys
import unittest
import torch
from pathlib import Path
from typing import List, Dict, Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from chat import load_chatbot, generate_chat_response
from companion_rules import normalize_spelling, detect_ambiguous_phrase
from source_selection import SourceRecord, EvidenceBundle


class BaseChatbotTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model, cls.tokenizer, cls.retrieval_engine, cls.ckpt = load_chatbot(cls.device)

    def setUp(self):
        if hasattr(self.retrieval_engine, "conversation_memory"):
            self.retrieval_engine.conversation_memory.clear()
        if hasattr(self.retrieval_engine, "session_last_sources"):
            self.retrieval_engine.session_last_sources = []


class TestConversationCategory(BaseChatbotTestCase):
    """Cases 1-4: Normal Conversation"""

    def test_case_01_hey_bro_how_was_your_day(self):
        hist = [{"role": "user", "text": "Hey bro, how was your day?"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertNotIn("Please enable web search", reply)
        self.assertTrue(any(w in reply.lower() for w in ["going well", "great", "chat", "how was", "day"]))

    def test_case_02_what_are_you_doing(self):
        hist = [{"role": "user", "text": "What are you doing?"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertNotIn("Please enable web search", reply)
        self.assertTrue(any(w in reply.lower() for w in ["here", "chatting", "help", "ready"]))

    def test_case_03_tell_me_something_interesting(self):
        hist = [{"role": "user", "text": "Tell me something interesting."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertNotIn("Please enable web search", reply)
        self.assertTrue(any(w in reply.lower() for w in ["honey", "octopus", "moth", "fact", "did you know"]))

    def test_case_04_i_am_bored(self):
        hist = [{"role": "user", "text": "I am bored."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertNotIn("Please enable web search", reply)
        self.assertTrue(any(w in reply.lower() for w in ["help", "boredom", "explore", "project", "riddle", "fact", "vibe"]))


class TestEmotionsCategory(BaseChatbotTestCase):
    """Cases 5-9: Emotional Understanding & Companionship"""

    def test_case_05_i_am_sad(self):
        hist = [{"role": "user", "text": "I am sad."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("companion_sad", route)
        self.assertTrue(any(w in reply.lower() for w in ["listen", "troubling", "bothering", "rough", "heavy"]))

    def test_case_06_friend_ignored_me_feel_hurt(self):
        hist = [{"role": "user", "text": "My best friend ignored me today and I feel really hurt."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("emotional_hurt", route)
        self.assertNotIn("doctor", reply.lower())
        self.assertNotIn("first aid", reply.lower())
        self.assertTrue(any(w in reply.lower() for w in ["hurt", "cuts deep", "painful", "listen", "vent", "share"]))

    def test_case_07_i_feel_confused(self):
        hist = [{"role": "user", "text": "I feel confused."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("companion_confused", route)
        self.assertTrue(any(w in reply.lower() for w in ["confused", "okay", "understandable", "step by step", "together"]))

    def test_case_08_i_am_happy_today(self):
        hist = [{"role": "user", "text": "I am happy today."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("companion_happy", route)
        self.assertTrue(any(w in reply.lower() for w in ["wonderful", "fantastic", "special", "celebrate"]))

    def test_case_09_got_hurt_by_friend(self):
        hist = [{"role": "user", "text": "I got hurt by my friend."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        # Must be treated as emotional hurt, NOT physical injury!
        self.assertIn("emotional_hurt", route)
        self.assertNotIn("doctor", reply.lower())
        self.assertNotIn("first aid", reply.lower())
        self.assertTrue(any(w in reply.lower() for w in ["hurt", "care", "listen", "vent", "disappointing"]))


class TestMultiTurnContextCategory(BaseChatbotTestCase):
    """Cases 10-13: Context & Multi-turn Python Progression"""

    def test_cases_10_to_13_python_progression(self):
        dialogue = [
            ("I am learning Python.", ["build", "language", "data science"]),
            ("What should I learn next?", ["variables", "control flow", "functions", "data structures", "oop"]),
            ("Why should I learn that?", ["building blocks", "scalable", "master", "debug", "logic"]),
            ("Explain it simply.", ["kitchen", "cooking", "recipe", "jars", "ingredients"]),
            ("Give me an example.", ["def ", "return", "if", "temperature", "check_temperature"])
        ]

        hist = []
        for user_msg, expected_keywords in dialogue:
            hist.append({"role": "user", "text": user_msg})
            reply, route = generate_chat_response(
                model=self.model, tokenizer=self.tokenizer, history=hist,
                device=self.device, retrieval_engine=self.retrieval_engine,
                web_search_enabled=False
            )
            hist.append({"role": "assistant", "text": reply})
            r_low = reply.lower()
            matched = any(k in r_low for k in expected_keywords)
            self.assertTrue(matched, f"Turn '{user_msg}' failed to match keywords {expected_keywords}. Got:\n{reply}")


class TestEducationCategory(BaseChatbotTestCase):
    """Cases 14-17: Educational Explanations (Non-interchangeable & Grounded)"""

    def test_case_14_what_is_deep_learning(self):
        hist = [{"role": "user", "text": "What is deep learning?"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        r_low = reply.lower()
        self.assertIn("deep learning", r_low)
        self.assertTrue("neural network" in r_low or "layers" in r_low or "hierarchical" in r_low)

    def test_case_15_explain_deep_learning_in_detail(self):
        hist = [{"role": "user", "text": "Explain deep learning in detail."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        r_low = reply.lower()
        self.assertIn("deep learning", r_low)
        self.assertTrue("convolutional" in r_low or "transformers" in r_low or "architectures" in r_low)

    def test_case_16_ai_vs_machine_learning_difference(self):
        hist = [{"role": "user", "text": "What is the difference between AI and machine learning?"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        r_low = reply.lower()
        self.assertIn("artificial intelligence", r_low)
        self.assertIn("machine learning", r_low)
        self.assertTrue("subfield" in r_low or "branch" in r_low or "all machine learning is" in r_low)

    def test_case_17_explain_cnn_pooling(self):
        hist = [{"role": "user", "text": "Explain CNN pooling."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        r_low = reply.lower()
        self.assertIn("pooling", r_low)
        self.assertTrue("downsampling" in r_low or "max pooling" in r_low or "dimensions" in r_low)


class TestProgrammingCategory(BaseChatbotTestCase):
    """Cases 18-20: Programming and Technical Assistance"""

    def test_case_18_write_python_program_for_addition(self):
        hist = [{"role": "user", "text": "Write a Python program for addition."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertEqual(route, "programming_assistance")
        self.assertIn("def add_numbers", reply)
        self.assertIn("return num1 + num2", reply)

    def test_case_19_explain_every_line(self):
        # Multi-turn: Ask for code, then ask to explain every line
        hist = [
            {"role": "user", "text": "Write a Python program for addition."},
            {"role": "assistant", "text": "```python\ndef add_numbers(num1: float, num2: float) -> float:\n    return num1 + num2\n```"},
            {"role": "user", "text": "Explain every line."}
        ]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertEqual(route, "programming_assistance")
        self.assertIn("def add_numbers", reply)
        self.assertTrue("float" in reply.lower() or "function" in reply.lower())

    def test_case_20_fix_python_error_typeerror(self):
        hist = [{"role": "user", "text": "Fix this Python error: TypeError: unsupported operand type(s) for +: 'int' and 'str'"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertEqual(route, "programming_assistance")
        r_low = reply.lower()
        self.assertIn("typeerror", r_low)
        self.assertTrue("f-string" in r_low or "str(" in r_low or "int(" in r_low)


class TestTyposCategory(BaseChatbotTestCase):
    """Cases 21-24: Typos and Informal Language"""

    def test_case_21_hey_how_are_yoyu(self):
        hist = [{"role": "user", "text": "hey how are yoyu"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertNotIn("Please enable web search", reply)
        self.assertTrue(any(w in reply.lower() for w in ["well", "chat", "good", "how are you"]))

    def test_case_22_what_is_deep_lerning(self):
        hist = [{"role": "user", "text": "what is deep lerning"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("deep learning", reply.lower())

    def test_case_23_write_a_code_for_adition(self):
        hist = [{"role": "user", "text": "write a code for adition"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertEqual(route, "programming_assistance")
        self.assertIn("def add_numbers", reply)

    def test_case_24_i_and_old_ambiguity(self):
        hist = [{"role": "user", "text": "i and old"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("companion_ambiguity", route)
        self.assertEqual(reply, "Did you mean 'I am old' or something else?")


class TestWebRetrievalCategory(BaseChatbotTestCase):
    """Cases 25-30: Web Retrieval & Mode Gating"""

    def test_case_25_and_26_ai_news_with_sources(self):
        # Mocked web search for latest AI news
        mock_item = SourceRecord(
            title="Recent AI Research Developments in 2026",
            url="https://arxiv.org/abs/2601.12345",
            snippet="Recent developments in artificial intelligence feature reasoning models and architectural scaling.",
            full_text="Recent developments in artificial intelligence in 2026 feature advanced reasoning models and efficient parameter scaling.",
            source_provider="openalex",
            rank=1
        )
        mock_bundle = EvidenceBundle(query="latest AI news", items=[mock_item], has_results=True)

        original_search = self.retrieval_engine.search_and_retrieve
        self.retrieval_engine.search_and_retrieve = lambda *args, **kwargs: mock_bundle
        try:
            hist = [{"role": "user", "text": "What is the latest AI news?"}]
            reply, route = generate_chat_response(
                model=self.model, tokenizer=self.tokenizer, history=hist,
                device=self.device, retrieval_engine=self.retrieval_engine,
                web_search_enabled=True
            )
            self.assertEqual(route, "web_grounded_evidence")
            self.assertIn("AI", reply)
        finally:
            self.retrieval_engine.search_and_retrieve = original_search

    def test_case_27_conversational_question_while_web_on(self):
        # Asking "How was your day?" while web search is ON must NOT search or fail!
        hist = [{"role": "user", "text": "How was your day?"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=True
        )
        self.assertNotEqual(route, "search_failure")
        self.assertNotEqual(route, "search_relevance_rejected")
        self.assertTrue(any(w in reply.lower() for w in ["great", "going well", "day"]))

    def test_case_28_emotional_question_while_web_on(self):
        # Asking an emotional question while web search is ON must NOT trigger web search!
        hist = [{"role": "user", "text": "I am sad today."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=True
        )
        self.assertIn("companion_sad", route)
        self.assertNotEqual(route, "web_grounded_evidence")
        self.assertTrue(any(w in reply.lower() for w in ["listen", "troubling", "bothering", "rough", "heavy", "sorry"]))

    def test_case_29_reject_unrelated_search_result(self):
        # Deliberately mock an unrelated search result (e.g. Argentine economic plan for an unrelated query)
        mock_unrelated = SourceRecord(
            title="Argentine Economy Reform Plan Simple",
            url="https://example.com/argentine-plan",
            snippet="The Argentine economic plan details inflation control policies in South America.",
            full_text="The Argentine economic plan details inflation control policies in South America.",
            source_provider="web",
            rank=1
        )
        mock_bundle = EvidenceBundle(query="quantum computing breakthroughs", items=[mock_unrelated], has_results=True)

        original_search = self.retrieval_engine.search_and_retrieve
        self.retrieval_engine.search_and_retrieve = lambda *args, **kwargs: mock_bundle
        try:
            hist = [{"role": "user", "text": "Search the web for quantum computing breakthroughs"}]
            reply, route = generate_chat_response(
                model=self.model, tokenizer=self.tokenizer, history=hist,
                device=self.device, retrieval_engine=self.retrieval_engine,
                web_search_enabled=True
            )
            # System must REJECT the unrelated search result!
            self.assertEqual(route, "search_relevance_rejected")
            self.assertIn("retrieved sources did not sufficiently match", reply)
        finally:
            self.retrieval_engine.search_and_retrieve = original_search

    def test_case_30_followup_after_web_researched_answer(self):
        # Multi-turn follow-up referring to sources
        mock_item = SourceRecord(
            title="Transformer Architecture in Neural Networks",
            url="https://arxiv.org/abs/1706.03762",
            snippet="Attention Is All You Need introduces the Transformer based entirely on attention mechanisms.",
            full_text="Attention Is All You Need introduces the Transformer model based entirely on self-attention mechanisms.",
            source_provider="openalex",
            rank=1
        )
        self.retrieval_engine.session_last_sources = [mock_item]

        hist = [
            {"role": "user", "text": "What is the Transformer architecture?"},
            {"role": "assistant", "text": "The Transformer uses self-attention mechanisms to process tokens in parallel."},
            {"role": "user", "text": "Give me the sources"}
        ]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=True
        )
        self.assertEqual(route, "sources_display")
        self.assertIn("Transformer Architecture", reply)
        self.assertIn("arxiv.org", reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
