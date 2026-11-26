"""
tests/test_companion_functionality_suite.py
===========================================
Realistic conversation test suite for the AI Companion.
Tests both offline and online modes:

OFFLINE TESTS (Web Search OFF):
1. Greeting and casual chat
2. Happy user (celebrating good news)
3. Sad user (empathy, listening without unsolicited advice)
4. Angry user (validating frustration, calming presence)
5. Stressed user (soothing, calming, gentle pacing)
6. Lonely user (warm companionship, presence)
7. Fear and nervousness (grounding reassurance)
8. Friendship disagreement (interpersonal conflict understanding)
9. Personal advice (thoughtful interpersonal perspective without robotic lists)
10. Multi-turn emotional conversation with memory continuity

ONLINE TESTS (Web Search ON):
11. Reruns offline companion tests with web search ON (verifies NO search is triggered for companion talk)
12. Basic educational questions (answered naturally without web search)
13. General knowledge (natural explanation without unnecessary search)
14. Current AI developments (requires web retrieval, synthesized in friendly style)
15. Source attribution and synthesis
"""

import unittest
import torch
from pathlib import Path
import sys

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from src.chat import load_chatbot, generate_chat_response

class TestCompanionFunctionalitySuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model, cls.tokenizer, cls.retrieval_engine, cls.ckpt = load_chatbot(cls.device)

    def setUp(self):
        if hasattr(self.retrieval_engine, "conversation_memory"):
            self.retrieval_engine.conversation_memory.clear()
        if hasattr(self.retrieval_engine, "session_last_sources"):
            self.retrieval_engine.session_last_sources = []

    # -------------------------------------------------------------
    # SECTION 1: OFFLINE TESTS (Web Search OFF)
    # -------------------------------------------------------------
    def test_offline_01_greetings_and_casual_chat(self):
        """User: Hey, how are you? / How was your day? -> Friendly conversation, no search."""
        hist = [{"role": "user", "text": "Hey! How are you doing today?"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertNotIn("How can I assist you today", reply)
        self.assertNotIn("/web on", reply)
        self.assertTrue(any(w in reply.lower() for w in ["doing", "well", "great", "chat", "good to see you", "how are you"]))

        # Follow up
        hist.append({"role": "assistant", "text": reply})
        hist.append({"role": "user", "text": "How was your day?"})
        reply2, route2 = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertNotIn("/web on", reply2)
        self.assertTrue(any(w in reply2.lower() for w in ["day", "chat", "chatting", "well", "brightened", "great"]))

    def test_offline_02_happy_user_celebration(self):
        """User: I got a promotion at work! -> Celebrates warmly, positive energy."""
        hist = [{"role": "user", "text": "I got the job offer I was interviewing for today!"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("happy", route.lower())
        self.assertTrue(any(w in reply.lower() for w in ["congratulations", "happy", "celebrat", "fantastic", "wonderful", "achievement"]))
        # No robotic list
        self.assertNotIn("1. ", reply)

    def test_offline_03_sad_user_empathy_without_unsolicited_lectures(self):
        """User: I feel really sad today -> Empathetic listening, no unsolicited lecturing."""
        hist = [{"role": "user", "text": "I'm feeling really sad and down today."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("sad", route.lower())
        self.assertTrue(any(w in reply.lower() for w in ["sorry", "down", "listen", "heavy", "right here", "with you"]))
        # Should not force unsolicited numbered advice
        self.assertNotIn("1. ", reply)
        self.assertNotIn("2. ", reply)

    def test_offline_04_angry_user_calming_space(self):
        """User: Furious about unfair treatment -> Validating frustration, calm presence."""
        hist = [{"role": "user", "text": "I'm so frustrated and angry, my manager blamed me for someone else's mistake!"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("angry", route.lower())
        self.assertTrue(any(w in reply.lower() for w in ["frustrat", "angry", "hear you", "valid", "listen", "vent", "bother"]))

    def test_offline_05_stressed_user_soothing_presence(self):
        """User: Overwhelmed with stress and exams -> Soothing, gentle pacing."""
        hist = [{"role": "user", "text": "I'm so stressed out with too much pressure from my exams."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("stressed", route.lower())
        self.assertTrue(any(w in reply.lower() for w in ["breath", "stress", "overwhelm", "pressure", "step", "weight", "pause", "shoulders", "worry", "with you"]))

    def test_offline_06_lonely_user_warm_companionship(self):
        """User: Feeling lonely -> True companionship, warmth, not generic deflection."""
        hist = [{"role": "user", "text": "I feel really lonely lately, like nobody has time for me."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("lonely", route.lower())
        self.assertTrue(any(w in reply.lower() for w in ["lonely", "alone", "right here", "company", "chat", "with you"]))

    def test_offline_07_fear_and_nervousness(self):
        """User: Nervous about tomorrow's presentation -> Grounding reassurance."""
        hist = [{"role": "user", "text": "I'm so nervous about my presentation tomorrow morning."}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("fear_nervous", route.lower())
        self.assertTrue(any(w in reply.lower() for w in ["nervous", "breath", "capable", "with you", "uneasy", "tension"]))

    def test_offline_08_multi_turn_friendship_disagreement_and_advice(self):
        """
        Multi-turn sequence:
        Tired day -> Disagreement with best friend -> Message first? -> Advice
        """
        # Turn 1: Tiring day
        hist = [{"role": "user", "text": "Hey! I had a pretty tiring day today."}]
        r1, route1 = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("tired", route1.lower())
        self.assertTrue(any(w in r1.lower() for w in ["tiring", "exhausting", "drain", "unwind", "rest"]))

        # Turn 2: Disagreement with best friend
        hist.append({"role": "assistant", "text": r1})
        hist.append({"role": "user", "text": "I had a disagreement with my best friend."})
        r2, route2 = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("emotional_hurt", route2.lower())
        self.assertTrue(any(w in r2.lower() for w in ["sorry", "disagreement", "friend", "conflict", "listen"]))

        # Turn 3: Should I message first?
        hist.append({"role": "assistant", "text": r2})
        hist.append({"role": "user", "text": "I don't know whether I should message them first."})
        r3, route3 = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("dilemma", route3.lower())
        self.assertTrue(any(w in r3.lower() for w in ["reach out", "first", "message", "friendship", "text"]))

        # Turn 4: What would you do in this situation?
        hist.append({"role": "assistant", "text": r3})
        hist.append({"role": "user", "text": "What would you do in this situation?"})
        r4, route4 = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=False
        )
        self.assertIn("interpersonal_advice", route4.lower())
        self.assertNotIn("/web on", r4)
        self.assertTrue(any(w in r4.lower() for w in ["shoes", "message", "friendship", "care", "settle", "approach"]))

    # -------------------------------------------------------------
    # SECTION 2: ONLINE TESTS (Web Search ON)
    # -------------------------------------------------------------
    def test_online_01_all_companion_conversations_with_web_on(self):
        """
        Critical check: Enabling Web Search must NOT turn chatbot into a search engine.
        Casual and emotional messages must receive companion responses without web searching.
        """
        # Casual greeting
        hist = [{"role": "user", "text": "Hey, how are you?"}]
        reply1, route1 = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=True
        )
        self.assertFalse(self.retrieval_engine.session_last_sources)
        self.assertNotIn("Search results", reply1)
        self.assertTrue(any(w in reply1.lower() for w in ["doing", "well", "great", "chat", "how are you"]))

        # Sad disclosure
        hist = [{"role": "user", "text": "I feel sad today."}]
        reply2, route2 = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=True
        )
        self.assertFalse(self.retrieval_engine.session_last_sources)
        self.assertIn("sad", route2.lower())
        self.assertTrue(any(w in reply2.lower() for w in ["sorry", "down", "listen", "heavy", "with you"]))

        # Interpersonal dispute
        hist = [{"role": "user", "text": "I had a fight with my best friend."}]
        reply3, route3 = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=True
        )
        self.assertFalse(self.retrieval_engine.session_last_sources)
        self.assertIn("emotional_hurt", route3.lower())

    def test_online_02_basic_educational_questions(self):
        """User: What is machine learning? -> Natural educational answer, NO web search needed."""
        hist = [{"role": "user", "text": "What is machine learning?"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=True
        )
        self.assertIn("offline_educational_grounded", route)
        self.assertTrue(any(w in reply.lower() for w in ["machine learning", "data", "algorithms", "patterns"]))

    def test_online_03_general_knowledge_concept(self):
        """User: What is photosynthesis? -> Answered naturally without web search."""
        hist = [{"role": "user", "text": "What is photosynthesis?"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=True
        )
        # Should provide clear explanation
        self.assertTrue(any(w in reply.lower() for w in ["photosynthesis", "plants", "sunlight", "energy", "chlorophyll", "chemical", "food", "process"]))

    def test_online_04_current_ai_developments_triggers_search(self):
        """User: What are the latest developments in AI? -> Genuinely triggers search and synthesizes answer."""
        hist = [{"role": "user", "text": "What are the latest developments in AI?"}]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=True
        )
        # Verify it routed to web grounded retrieval or executed search
        self.assertTrue("web" in route.lower() or "grounded" in route.lower() or "search" in route.lower())
        self.assertTrue(len(reply) > 50)

    def test_online_05_sources_request(self):
        """User: Give me the sources -> Displays links cleanly."""
        hist = [
            {"role": "user", "text": "What are the latest developments in AI?"},
            {"role": "assistant", "text": "Here are some recent developments in AI..."},
            {"role": "user", "text": "Give me the sources"}
        ]
        reply, route = generate_chat_response(
            model=self.model, tokenizer=self.tokenizer, history=hist,
            device=self.device, retrieval_engine=self.retrieval_engine,
            web_search_enabled=True
        )
        self.assertIn("sources", route.lower())
        self.assertTrue("sources" in reply.lower() or "http" in reply.lower() or "no external" in reply.lower())

if __name__ == "__main__":
    unittest.main()
