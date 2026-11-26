"""
tests/test_complete_repair_suite.py
===================================
Comprehensive Automated Verification & Regression Test Suite.
Validates all 9 Test Groups required by the AI Companion Chatbot specification:
- Test Group 1: Basic Conversation (Offline)
- Test Group 2: Short & Incomplete Messages
- Test Group 3: Educational Questions (Offline & Topic Specific)
- Test Group 4: Typos & Informal Language
- Test Group 5: Emotional Conversations
- Test Group 6: Conversation History & Context Tracking
- Test Group 7: Mode Switching & Robust Slash Commands
- Test Group 8: Web Retrieval & Synthesis (Mocked & Grounded)
- Test Group 9: Failure Handling & Graceful Recovery
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

import torch
from chat import (
    load_chatbot, generate_chat_response, parse_command,
    validate_response_quality
)
from companion_rules import (
    normalize_spelling, detect_greeting, detect_identity_query,
    detect_short_or_incomplete_message, detect_conversational_statement,
    get_offline_educational_answer, detect_emotion_and_intent,
    resolve_context
)
from web_retrieval import (
    EvidenceBundle, EvidenceItem, WebRetrievalEngine,
    synthesize_grounded_answer
)
from question_understanding import (
    QuestionUnderstandingEngine, QuestionInterpretation,
    INTENT_CONVERSATION, INTENT_EDUCATIONAL, INTENT_COMPARISON
)


class BaseChatbotTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model, cls.tokenizer, cls.retrieval_engine, cls.ckpt = load_chatbot(cls.device)


class TestGroup1_BasicConversation(BaseChatbotTestCase):
    """TEST GROUP 1: BASIC CONVERSATION (OFFLINE)"""

    def test_basic_greetings_and_conversation(self):
        prompts = [
            "hello",
            "hi",
            "hey",
            "good morning",
            "how are you",
            "what are you doing",
            "what's your name",
            "who are you",
            "are you there",
            "can you help me",
            "I need help"
        ]

        refusal_phrase = "Please enable web search"

        for p in prompts:
            with self.subTest(prompt=p):
                if hasattr(self.retrieval_engine, "conversation_memory"):
                    self.retrieval_engine.conversation_memory.clear()
                hist = [{"role": "user", "text": p}]
                reply, route = generate_chat_response(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    history=hist,
                    device=self.device,
                    retrieval_engine=self.retrieval_engine,
                    web_search_enabled=False
                )
                self.assertIsNotNone(reply)
                self.assertGreater(len(reply.strip()), 5)
                # Must not demand web search for basic conversation
                self.assertNotIn(refusal_phrase, reply, f"Prompt '{p}' incorrectly demanded web search!")

    def test_identity_introductions(self):
        name_reply, _ = generate_chat_response(
            model=self.model,
            tokenizer=self.tokenizer,
            history=[{"role": "user", "text": "what's your name"}],
            device=self.device,
            retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("AI Companion", name_reply)

        help_reply, _ = generate_chat_response(
            model=self.model,
            tokenizer=self.tokenizer,
            history=[{"role": "user", "text": "can you help me"}],
            device=self.device,
            retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertTrue(any(w in help_reply.lower() for w in ["help", "best to help", "tell me"]))


class TestGroup2_ShortAndIncompleteMessages(BaseChatbotTestCase):
    """TEST GROUP 2: SHORT AND INCOMPLETE MESSAGES"""

    def test_short_inputs(self):
        cases = {
            "pls": "help you with",
            "plz": "help you with",
            "help": "what you need help with",
            "explain": "what would you like me to explain",
            "tell me": "what would you like me to tell you about",
            "please explain": "explain",
            "can u help": "help",
            "yes": "proceed",
            "no": "understood",
            "okay": "what's next",
            "thanks": "welcome"
        }

        for p, expected_frag in cases.items():
            with self.subTest(prompt=p):
                if hasattr(self.retrieval_engine, "conversation_memory"):
                    self.retrieval_engine.conversation_memory.clear()
                hist = [{"role": "user", "text": p}]
                reply, route = generate_chat_response(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    history=hist,
                    device=self.device,
                    retrieval_engine=self.retrieval_engine,
                    web_search_enabled=False
                )
                self.assertIsNotNone(reply)
                self.assertIn(expected_frag.lower(), reply.lower())
                self.assertNotIn("Please enable web search", reply)


class TestGroup3_EducationalQuestions(BaseChatbotTestCase):
    """TEST GROUP 3: EDUCATIONAL QUESTIONS (OFFLINE & NON-INTERCHANGEABLE)"""

    def test_machine_learning_response(self):
        hist = [{"role": "user", "text": "What is machine learning?"}]
        reply, route = generate_chat_response(
            model=self.model,
            tokenizer=self.tokenizer,
            history=hist,
            device=self.device,
            retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        r_low = reply.lower()
        self.assertIn("machine learning", r_low)
        self.assertTrue(any(k in r_low for k in ["data", "algorithms", "patterns", "statistical"]))
        # Must not confuse ML with Deep Learning architecture exclusively
        self.assertNotIn("Please enable web search", reply)

    def test_deep_learning_response(self):
        hist = [{"role": "user", "text": "What is deep learning?"}]
        reply, route = generate_chat_response(
            model=self.model,
            tokenizer=self.tokenizer,
            history=hist,
            device=self.device,
            retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        r_low = reply.lower()
        self.assertIn("deep learning", r_low)
        self.assertTrue(any(k in r_low for k in ["neural", "layers", "hierarchical", "deep"]))
        self.assertNotIn("Please enable web search", reply)

    def test_non_interchangeability_ml_and_dl(self):
        if hasattr(self.retrieval_engine, "conversation_memory"):
            self.retrieval_engine.conversation_memory.clear()
        hist_ml = [{"role": "user", "text": "Explain machine learning in detail."}]
        reply_ml, _ = generate_chat_response(
            model=self.model,
            tokenizer=self.tokenizer,
            history=hist_ml,
            device=self.device,
            retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )

        if hasattr(self.retrieval_engine, "conversation_memory"):
            self.retrieval_engine.conversation_memory.clear()
        hist_dl = [{"role": "user", "text": "Explain deep learning in detail."}]
        reply_dl, _ = generate_chat_response(
            model=self.model,
            tokenizer=self.tokenizer,
            history=hist_dl,
            device=self.device,
            retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )

        self.assertNotEqual(reply_ml.strip(), reply_dl.strip(), "ML and DL received identical responses!")
        self.assertTrue(reply_ml.strip().lower().startswith("**machine learning") or "machine learning" in reply_ml.lower()[:50])
        self.assertTrue(reply_dl.strip().lower().startswith("**deep learning") or "deep learning" in reply_dl.lower()[:50])

    def test_ai_definition(self):
        hist = [{"role": "user", "text": "What is artificial intelligence?"}]
        reply, _ = generate_chat_response(
            model=self.model,
            tokenizer=self.tokenizer,
            history=hist,
            device=self.device,
            retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("artificial intelligence", reply.lower())

    def test_ai_vs_ml_difference(self):
        hist = [{"role": "user", "text": "What is the difference between AI and ML?"}]
        reply, _ = generate_chat_response(
            model=self.model,
            tokenizer=self.tokenizer,
            history=hist,
            device=self.device,
            retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        r_low = reply.lower()
        self.assertIn("artificial intelligence", r_low)
        self.assertIn("machine learning", r_low)
        self.assertTrue("subfield" in r_low or "branch" in r_low or "relationship" in r_low or "all machine learning is" in r_low)

    def test_cnn_and_pooling(self):
        hist_cnn = [{"role": "user", "text": "What is CNN?"}]
        reply_cnn, _ = generate_chat_response(
            model=self.model,
            tokenizer=self.tokenizer,
            history=hist_cnn,
            device=self.device,
            retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertTrue("convolutional" in reply_cnn.lower() or "cnn" in reply_cnn.lower())

        hist_pool = [{"role": "user", "text": "Explain pooling in CNN."}]
        reply_pool, _ = generate_chat_response(
            model=self.model,
            tokenizer=self.tokenizer,
            history=hist_pool,
            device=self.device,
            retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertTrue("pooling" in reply_pool.lower())
        self.assertTrue("downsampling" in reply_pool.lower() or "dimensions" in reply_pool.lower())


class TestGroup4_TyposAndInformalLanguage(BaseChatbotTestCase):
    """TEST GROUP 4: TYPOS AND INFORMAL LANGUAGE"""

    def test_typo_normalization(self):
        cases = [
            ("what iks machine learning", "machine learning"),
            ("what is machne learning", "machine learning"),
            ("explain deep lernng", "deep learning"),
            ("can u plss explain AI", "AI"),
            ("hii how r u", "how are you"),
            ("i ned help", "need help")
        ]
        for prompt, expected_intent in cases:
            with self.subTest(prompt=prompt):
                norm = normalize_spelling(prompt)
                self.assertNotIn("iks", norm.split())
                self.assertNotIn("lernng", norm)
                self.assertNotIn("ned", norm.split())
                self.assertNotIn("plss", norm)

                hist = [{"role": "user", "text": prompt}]
                reply, route = generate_chat_response(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    history=hist,
                    device=self.device,
                    retrieval_engine=self.retrieval_engine,
                    web_search_enabled=False
                )
                self.assertIsNotNone(reply)
                self.assertNotIn("Please enable web search", reply)


class TestGroup5_EmotionalConversations(BaseChatbotTestCase):
    """TEST GROUP 5: EMOTIONAL CONVERSATIONS"""

    def test_emotions(self):
        cases = [
            ("I am sad today.", ["sorry", "listen", "troubling", "bothering", "rough", "heavy"]),
            ("I am happy today.", ["wonderful", "fantastic", "special", "celebrate", "energy", "happy"]),
            ("I feel lonely.", ["lonely", "right here", "chat", "not alone", "company"]),
            ("I am stressed.", ["breath", "breathe", "stress", "one step", "smaller parts", "piece by piece", "worried", "pressure", "overwhelming"]),
            ("I am frustrated.", ["frustrating", "frustrated", "calmly", "hear you", "vent", "upset"]),
            ("I don't understand anything.", ["explain", "step by step", "plain english", "unclear", "no worries", "break it down"]),
            ("I had a bad day.", ["sorry", "listen", "share", "heavy", "rough", "bothering"]),
            ("I am feeling excited.", ["exciting", "energy", "news", "enthusiasm", "thrilled", "celebrate"])
        ]

        for p, expected_keywords in cases:
            with self.subTest(prompt=p):
                if hasattr(self.retrieval_engine, "conversation_memory"):
                    self.retrieval_engine.conversation_memory.clear()
                hist = [{"role": "user", "text": p}]
                reply, route = generate_chat_response(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    history=hist,
                    device=self.device,
                    retrieval_engine=self.retrieval_engine,
                    web_search_enabled=False
                )
                r_low = reply.lower()
                matched = any(k in r_low for k in expected_keywords)
                self.assertTrue(matched, f"Emotion reply for '{p}' did not contain expected keywords. Reply was: {reply}")


class TestGroup6_ConversationHistory(BaseChatbotTestCase):
    """TEST GROUP 6: CONVERSATION HISTORY & CONTEXT RESOLUTION"""

    def test_multiturn_dialogue_flow(self):
        turns = [
            ("I am learning machine learning.", ["machine learning", "great", "studying"]),
            ("Deep learning.", ["deep learning", "machine learning", "neural networks"]),
            ("Explain it.", ["deep learning", "neural networks", "hierarchical", "layers"])
        ]

        hist = []
        for user_msg, expected_words in turns:
            hist.append({"role": "user", "text": user_msg})
            reply, route = generate_chat_response(
                model=self.model,
                tokenizer=self.tokenizer,
                history=hist,
                device=self.device,
                retrieval_engine=self.retrieval_engine,
                web_search_enabled=False
            )
            hist.append({"role": "assistant", "text": reply})
            r_low = reply.lower()
            self.assertTrue(
                any(w in r_low for w in expected_words),
                f"Turn '{user_msg}' produced: {reply}"
            )


class TestGroup7_ModeSwitching(unittest.TestCase):
    """TEST GROUP 7: MODE SWITCHING & COMMAND PARSING"""

    def test_command_parser_whitespace_and_case(self):
        cases = [
            ("/web on", ("web_on", None)),
            ("/web  on", ("web_on", None)),
            ("/WEB ON", ("web_on", None)),
            ("/web off", ("web_off", None)),
            ("/web   off", ("web_off", None)),
            ("/WEB OFF", ("web_off", None)),
            ("/web status", ("web_status", None)),
            ("/help", ("help", None)),
            ("/exit", ("exit", None)),
            ("/quit", ("exit", None)),
            ("/clear", ("clear", None)),
            ("/sources", ("sources", None)),
            ("/tool on", ("tool_on", None)),
            ("/tool off", ("tool_off", None)),
            ("/debug on", ("debug_on", None)),
            ("hello world", (None, None))
        ]
        for cmd_str, expected in cases:
            with self.subTest(command=cmd_str):
                res = parse_command(cmd_str)
                self.assertEqual(res, expected)

    def test_unknown_commands_not_leaked(self):
        action, msg = parse_command("/unknown_command_test")
        self.assertEqual(action, "unknown_command")
        self.assertIn("Type /help", msg)


class TestGroup8_WebRetrieval(unittest.TestCase):
    """TEST GROUP 8: WEB RETRIEVAL (MOCKED & DETERMINISTIC)"""

    def test_mocked_web_retrieval_non_interchangeability(self):
        engine = WebRetrievalEngine()
        qu_engine = QuestionUnderstandingEngine()

        # Query 1: Machine Learning
        interp_ml = qu_engine.parse("What is machine learning in detail?")
        self.assertEqual(interp_ml.main_subject, "machine learning")
        self.assertIn("Machine learning", interp_ml.search_queries[0])

        bundle_ml = EvidenceBundle(
            query="machine learning",
            items=[
                EvidenceItem(
                    title="Machine learning",
                    url="https://en.wikipedia.org/wiki/Machine_learning",
                    snippet="Machine learning is an umbrella term for solving problems...",
                    full_text="Machine learning is a field of inquiry devoted to understanding data patterns.",
                    rank=1
                )
            ],
            has_results=True
        )
        ans_ml = synthesize_grounded_answer("what is machine learning", bundle_ml)
        self.assertIn("Machine learning", ans_ml)
        self.assertNotIn("Artificial intelligence is the broad field", ans_ml[:30])

        # Query 2: Deep Learning
        interp_dl = qu_engine.parse("What is deep learning in detail?")
        self.assertEqual(interp_dl.main_subject, "deep learning")
        self.assertIn("Deep learning", interp_dl.search_queries[0])

        bundle_dl = EvidenceBundle(
            query="deep learning",
            items=[
                EvidenceItem(
                    title="Deep learning",
                    url="https://en.wikipedia.org/wiki/Deep_learning",
                    snippet="Deep learning uses artificial neural networks with multiple layers...",
                    full_text="Deep learning is a subset of machine learning based on artificial neural networks with representation learning.",
                    rank=1
                )
            ],
            has_results=True
        )
        ans_dl = synthesize_grounded_answer("what is deep learning", bundle_dl)
        self.assertIn("Deep learning", ans_dl)
        self.assertNotEqual(ans_ml, ans_dl)


class TestGroup9_FailureHandling(BaseChatbotTestCase):
    """TEST GROUP 9: FAILURE HANDLING & RECOVERY"""

    def test_empty_response_handling(self):
        is_valid, reason = validate_response_quality("What is AI?", "")
        self.assertFalse(is_valid)
        self.assertEqual(reason, "Response is empty.")

    def test_repetitive_loop_detection(self):
        loop_resp = "Machine learning is data from data and learn from data and learn from data and learn from data."
        is_valid, reason = validate_response_quality("What is machine learning?", loop_resp)
        self.assertFalse(is_valid)

    def test_stutter_detection(self):
        stutter_resp = "No, CNo, CNo, is a CNo-No-No-No-No-No-No which is a network."
        is_valid, reason = validate_response_quality("What is CNN?", stutter_resp)
        self.assertFalse(is_valid)

    def test_self_referential_boilerplate_rejection(self):
        bp_resp = "As a language model, I do not possess personal opinions on this matter."
        is_valid, reason = validate_response_quality("Explain recursion.", bp_resp)
        self.assertFalse(is_valid)

    def test_graceful_recovery_on_model_fallback(self):
        # Even if neural generation is corrupt, offline mode provides high-quality educational fallback
        hist = [{"role": "user", "text": "What is machine learning?"}]
        reply, route = generate_chat_response(
            model=self.model,
            tokenizer=self.tokenizer,
            history=hist,
            device=self.device,
            retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIsNotNone(reply)
        self.assertGreater(len(reply), 30)
        self.assertNotIn("CNo-No", reply)
        self.assertNotIn("Please enable web search", reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
