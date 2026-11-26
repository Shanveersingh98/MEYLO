"""
src/intent_router.py
====================
Universal Intent Understanding & Query Routing Subsystem for From-Scratch Chatbot.
Powered by QuestionUnderstandingEngine.

Provides robust classification across 12 distinct intents:
1. general_conversation
2. educational_concept
3. person_profile
4. current_events
5. technical_programming
6. health_safety (urgent_emergency vs general_health_info)
7. product_service
8. deterministic_tool
9. document_summary
10. comparison
11. followup_reference
12. ambiguous_unspecified

For each request, determines:
- Intent, sub-intent, and action
- Whether live web retrieval is necessary
- Which search provider / strategy to prioritize
- Tailored search query generation
- Immediate safety guidance for urgent health symptoms
- Structured QuestionInterpretation representation
"""

import re
import html
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

try:
    from src.question_understanding import (
        QuestionUnderstandingEngine, QuestionInterpretation,
        INTENT_CONVERSATION, INTENT_EDUCATIONAL, INTENT_PERSON_PROFILE,
        INTENT_CURRENT_EVENTS, INTENT_TECHNICAL, INTENT_HEALTH_SAFETY,
        INTENT_PRODUCT_SERVICE, INTENT_TOOL, INTENT_DOCUMENT_SUMMARY,
        INTENT_COMPARISON, INTENT_FOLLOWUP, INTENT_AMBIGUOUS,
        SUB_HEALTH_URGENT, SUB_HEALTH_GENERAL
    )
except ImportError:
    from question_understanding import (
        QuestionUnderstandingEngine, QuestionInterpretation,
        INTENT_CONVERSATION, INTENT_EDUCATIONAL, INTENT_PERSON_PROFILE,
        INTENT_CURRENT_EVENTS, INTENT_TECHNICAL, INTENT_HEALTH_SAFETY,
        INTENT_PRODUCT_SERVICE, INTENT_TOOL, INTENT_DOCUMENT_SUMMARY,
        INTENT_COMPARISON, INTENT_FOLLOWUP, INTENT_AMBIGUOUS,
        SUB_HEALTH_URGENT, SUB_HEALTH_GENERAL
    )


@dataclass
class IntentDecision:
    intent: str
    sub_intent: Optional[str] = None
    needs_web_search: bool = False
    needs_tool: bool = False
    search_queries: List[str] = field(default_factory=list)
    preferred_provider: str = "local_neural"  # 'general_web', 'wikipedia', 'local_neural', 'calculator', etc.
    immediate_safety_response: Optional[str] = None
    confidence: float = 1.0
    reasoning: str = ""
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    interpretation: Optional[QuestionInterpretation] = None


class IntentRouter:
    """
    Modular, deterministic intent router backed by the QuestionUnderstandingEngine.
    Produces internal structured semantic representations and maps to IntentDecision.
    """

    def __init__(self):
        self.engine = QuestionUnderstandingEngine()

    def classify(
        self,
        user_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        has_cached_web: bool = False,
        cached_source_count: int = 0
    ) -> IntentDecision:
        p_raw = user_prompt.strip()
        eff_cached_count = cached_source_count if cached_source_count > 0 else (1 if has_cached_web else 0)

        interpretation = self.engine.parse(
            user_message=p_raw,
            conversation_history=conversation_history,
            cached_source_count=eff_cached_count
        )

        intent = interpretation.intent
        sub_intent = interpretation.sub_intent
        needs_tool = False
        needs_web_search = interpretation.requires_external_research and (intent != INTENT_CONVERSATION)

        if sub_intent == "mathematics":
            preferred_provider = "local_neural"
            needs_web_search = False
        elif intent == INTENT_HEALTH_SAFETY and sub_intent == "emergency_helpline":
            preferred_provider = "emergency_helpline"
            needs_web_search = False
        elif intent == INTENT_CONVERSATION:
            preferred_provider = "local_neural"
            needs_web_search = False
        elif intent == INTENT_FOLLOWUP:
            preferred_provider = "web_cache_followup"
            needs_web_search = False
        elif intent == INTENT_AMBIGUOUS:
            preferred_provider = "local_neural"
            needs_web_search = False
        elif intent == INTENT_TECHNICAL and not interpretation.requires_current_info:
            preferred_provider = "local_neural"
            needs_web_search = False
        elif intent == INTENT_EDUCATIONAL and sub_intent == "learning_roadmap":
            preferred_provider = "local_neural"
            needs_web_search = False
        elif "official_python_releases" in interpretation.preferred_sources:
            preferred_provider = "general_web"
        elif "official_pytorch_docs" in interpretation.preferred_sources:
            preferred_provider = "general_web"
        elif interpretation.preferred_sources:
            preferred_provider = interpretation.preferred_sources[0]
        else:
            preferred_provider = "general_web"

        # Build extracted entities dict with backward-compatible aliases
        extracted_entities = interpretation.to_dict()
        if intent == INTENT_PERSON_PROFILE:
            extracted_entities["person_name"] = interpretation.main_subject
            extracted_entities["platform"] = interpretation.platform or "web"
        elif intent == INTENT_TECHNICAL:
            extracted_entities["tech_subject"] = interpretation.main_subject
            extracted_entities["technical_identifiers"] = interpretation.technical_identifiers
        elif intent == INTENT_COMPARISON and interpretation.comparison_entities:
            extracted_entities["concept_a"] = interpretation.comparison_entities[0]
            extracted_entities["concept_b"] = interpretation.comparison_entities[1] if len(interpretation.comparison_entities) > 1 else ""
        elif intent == INTENT_HEALTH_SAFETY:
            extracted_entities["medical_topic"] = interpretation.main_subject
            extracted_entities["triage_level"] = "EMERGENCY" if sub_intent == SUB_HEALTH_URGENT else "INFORMATIONAL"
        elif intent == INTENT_EDUCATIONAL:
            extracted_entities["concept"] = interpretation.main_subject

        return IntentDecision(
            intent=intent,
            sub_intent=sub_intent,
            needs_web_search=needs_web_search,
            needs_tool=needs_tool,
            search_queries=interpretation.search_queries,
            preferred_provider=preferred_provider,
            immediate_safety_response=interpretation.immediate_safety_response,
            confidence=interpretation.confidence,
            reasoning=interpretation.reasoning,
            extracted_entities=extracted_entities,
            interpretation=interpretation
        )
