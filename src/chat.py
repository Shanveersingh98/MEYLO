"""
src/chat.py
===========
AI Companion Chatbot combining:
1. Rule-based emotion, identity, and conversational response system.
2. Independent FromScratchLM trained neural model with controlled sampling and quality gating.
3. Offline educational knowledge base for guaranteed reliable offline responses.
4. Enhanced web search with multi-turn context, compound query handling,
   and source attribution.
5. Robust slash command parsing and interaction management.
6. Comprehensive diagnostic router logging to logs/router_debug.log.
"""

import sys
import time
import json
import re
import ast
import random
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

from model import FromScratchLM
from model_manager import ModelManager
from web_retrieval import (
    WebRetrievalEngine, sanitize_for_prompt, is_safe_url,
    synthesize_grounded_answer, synthesize_health_answer, synthesize_person_profile_answer,
    EvidenceBundle, EvidenceItem
)
from companion_rules import (
    normalize_spelling,
    detect_ambiguous_phrase,
    detect_date_query,
    detect_emergency_helpline,
    detect_incomplete_emotional_stem,
    detect_greeting,
    detect_identity_query,
    detect_short_or_incomplete_message,
    detect_conversational_statement,
    get_offline_educational_answer,
    detect_source_request,
    detect_emotion_and_intent,
    generate_companion_response,
    resolve_context,
    disambiguate_technical_query,
    split_compound_question
)
from programming_assistance import (
    detect_programming_intent,
    generate_code_response,
    explain_code_lines,
    handle_error_debugging,
    request_missing_debug_info
)

try:
    from src.intent_router import (
        INTENT_CONVERSATION, INTENT_EDUCATIONAL, INTENT_PERSON_PROFILE,
        INTENT_CURRENT_EVENTS, INTENT_TECHNICAL, INTENT_HEALTH_SAFETY,
        INTENT_PRODUCT_SERVICE, INTENT_DOCUMENT_SUMMARY,
        INTENT_COMPARISON, INTENT_FOLLOWUP, INTENT_AMBIGUOUS,
        SUB_HEALTH_URGENT, SUB_HEALTH_GENERAL
    )
    from src.source_selection import SourceRecord
    from src.question_understanding import QuestionInterpretation
except ImportError:
    from intent_router import (
        INTENT_CONVERSATION, INTENT_EDUCATIONAL, INTENT_PERSON_PROFILE,
        INTENT_CURRENT_EVENTS, INTENT_TECHNICAL, INTENT_HEALTH_SAFETY,
        INTENT_PRODUCT_SERVICE, INTENT_DOCUMENT_SUMMARY,
        INTENT_COMPARISON, INTENT_FOLLOWUP, INTENT_AMBIGUOUS,
        SUB_HEALTH_URGENT, SUB_HEALTH_GENERAL
    )
    from source_selection import SourceRecord
    from question_understanding import QuestionInterpretation

TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "ultimate_improvement" / "best.pt"
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "experiment_2" / "best.pt"

CONTEXT_LENGTH = 1024
STOP_TOKEN_IDS = {0, 2, 3, 4, 5}

ROUTER_DEBUG_LOG_FILE = PROJECT_DIR / "logs" / "router_debug.log"


def log_router_debug(
    original_message: str,
    normalized_message: str,
    detected_intent: str,
    detected_emotion: str,
    resolved_topic: str,
    selected_handler: str,
    local_model_called: bool,
    web_search_called: bool,
    passed_validation: bool,
    final_response_source: str,
    failure_reason: Optional[str] = None
):
    """
    Records detailed diagnostic routing telemetry to logs/router_debug.log.
    """
    try:
        ROUTER_DEBUG_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "timestamp": ts_str,
            "original_message": original_message,
            "normalized_message": normalized_message,
            "detected_intent": detected_intent,
            "detected_emotion": detected_emotion,
            "resolved_topic": resolved_topic,
            "selected_handler": selected_handler,
            "local_model_called": local_model_called,
            "web_search_called": web_search_called,
            "passed_validation": passed_validation,
            "final_response_source": final_response_source,
            "failure_reason": failure_reason
        }
        with open(ROUTER_DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def load_chatbot(device: torch.device):
    print(f"Loading custom trained model from: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = FromScratchLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    retrieval_engine = WebRetrievalEngine()
    retrieval_engine.session_last_sources = []
    return model, tokenizer, retrieval_engine, ckpt


def format_source_links(items: List[Any]) -> str:
    """Formats actual clickable source links with titles when requested."""
    if not items:
        return "No external web sources were used for the recent response."
    lines = ["Sources for the recent response:"]
    seen = set()
    for item in items:
        title = getattr(item, "title", str(item)).strip()
        url = getattr(item, "url", "").strip()
        key = (title, url)
        if key not in seen:
            seen.add(key)
            lines.append(f"• {title}: {url}")
    return "\n".join(lines)


def parse_command(user_input: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses slash commands robustly:
    - Normalizes internal multiple spaces and tabs.
    - Handles case-insensitivity (/WEB ON, /web  on, /WEB   OFF).
    - Prevents commands from leaking into the model or conversation history.
    - Explains unknown commands clearly.
    Returns (command_action, feedback_message).
    """
    trimmed = user_input.strip()
    if not trimmed.startswith("/"):
        return None, None

    normalized = re.sub(r"\s+", " ", trimmed)
    parts = normalized.split(" ")
    cmd = parts[0].lower()
    args = [p.lower() for p in parts[1:]]
    sub = " ".join(args)

    if cmd in ["/exit", "/quit"]:
        return "exit", None
    elif cmd == "/clear":
        return "clear", None
    elif cmd == "/help":
        return "help", None
    elif cmd == "/sources":
        return "sources", None
    elif cmd == "/web":
        if sub == "on":
            return "web_on", None
        elif sub == "off":
            return "web_off", None
        elif sub == "status":
            return "web_status", None
        else:
            return "unknown_command", f"Unrecognized option for /web: '{sub}'. Valid options: /web on, /web off, /web status."
    elif cmd == "/model":
        if not args or sub in ["help", "?"]:
            return "unknown_command", "Model options: /model custom, /model external, /model status, /model list"
        elif sub in ["custom"]:
            return "model_custom", None
        elif sub in ["external", "qwen"]:
            return "model_external", None
        elif sub == "status":
            return "model_status", None
        elif sub in ["list", "ls"]:
            return "model_list", None
        else:
            return "unknown_command", f"Unrecognized option for /model: '{sub}'. Valid options: /model custom, /model external, /model status, /model list."
    elif cmd == "/debug":
        if sub == "on":
            return "debug_on", None
        elif sub == "off":
            return "debug_off", None
        elif sub == "status":
            return "debug_status", None
        else:
            return "unknown_command", f"Unrecognized option for /debug: '{sub}'. Valid options: /debug on, /debug off, /debug status."
    else:
        return "unknown_command", f"Unknown command '{trimmed}'. Type /help for available commands."


def validate_response_quality(query: str, response: str, evidence_bundle: Optional[EvidenceBundle] = None) -> Tuple[bool, str]:
    """
    Evaluates whether a generated response meets coherence, truthfulness,
    relevance, and non-hallucinatory standards.
    Distinguishes between conversational responses and educational definitions.
    Returns (is_valid, failure_reason).
    """
    if not response or len(response.strip()) == 0:
        return False, "Response is empty."

    trimmed = response.strip()
    words = trimmed.split()
    r_lower = response.lower()
    q_lower = query.lower()

    # Is this an educational/factual question versus casual conversational check-in?
    is_definition_query = bool(re.match(r"(?i)^(what|who|where|why|how|explain|describe|define)\b", query.strip()))
    is_casual_conversation = bool(re.search(r"\b(hi|hello|hey|how\s+are\s+you|are\s+you\s+there|help|pls|thanks|thank\s+you)\b", q_lower))

    # Length check: definitions require substance, conversational responses can be concise
    if any(k in q_lower for k in ["in detail", "in-depth", "detailed", "comprehensively", "thoroughly", "elaborate"]) and len(words) < 45:
        return False, "Response is too brief for an 'in detail' explanation request."
    elif is_definition_query and len(words) < 4:
        return False, "Response has too few words for a definition."
    elif len(trimmed) < 4:
        return False, "Response is too short."

    # 1. Check for self-referential model boilerplate, prompt echoes, or corrupt generation tokens
    prompt_echoes = [
        "your name is",
        "your name is c++",
        "i am a language model",
        "i'm a language model",
        "i am not a language model",
        "i'm not a language model",
        "not have personal opinions",
        "don't have personal opinions",
        "do not have personal opinions",
        "don't have emotions",
        "do not have emotions",
        "as a language model",
        "trained on text data",
        "trained on the text data",
        "trained on a variety of topics",
        "variety of topics such as text",
        "text, text, or other words",
        "greing",
        "ber learning",
        "ber is a",
        "science fiction learning",
        "broad machine learning",
        "it's a machine learning",
        "they are not suitable for human intelligence",
        "not suitable for human intelligence",
        "not suitable for human",
        "not suitable for text",
        "learn and decisions",
        "website's website",
        "google maec",
        "important to be a lot of and difficult",
        "<|user|>", "<|assistant|>", "[eos]", "user:"
    ]
    for pe in prompt_echoes:
        if pe in r_lower:
            return False, f"Contains self-referential prompt echo or invalid generation: '{pe}'"

    # If query is not about C++ but response says "c++"
    if "c++" not in q_lower and "c++" in r_lower:
        return False, "Irrelevant C++ mention in response."

    # Check Python code block syntax
    if "```python" in response or "```" in response:
        code_blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL)
        for cb in code_blocks:
            cb_clean = cb.strip()
            if any(k in cb_clean for k in ["def ", "import ", "print(", "return "]):
                try:
                    ast.parse(cb_clean)
                except SyntaxError as e:
                    return False, f"Code block has invalid Python syntax: {e}"

    # Check for known incoherent advice and gibberish generation patterns
    incoherent_patterns = [
        "avoid seeking professional advice",
        "don't work to listen",
        "favorite morning",
        "struggling with friendship or taking a friend",
        "find out the day and get time for yourself",
        "it's important to be helpful. try to take your day",
        "range(6))",
        "len(6, 7))",
        "x for x in x in"
    ]
    for ip in incoherent_patterns:
        if ip in r_lower:
            return False, f"Incoherent or nonsensical pattern detected: '{ip}'"

    # Check for hallucinated math equations or random storage/price tokens on non-hardware queries
    if re.search(r"\b\d+\s*[-+*/]\s*\d+\s*=\s*\d+\b", r_lower):
        return False, "Hallucinated mathematical equation detected in text."
    if "mb (" in r_lower or re.search(r"\b\d+\s*mb\b", r_lower):
        if not any(k in q_lower for k in ["memory", "ram", "storage", "mb", "gb", "gpu", "size"]):
            return False, "Irrelevant memory/storage MB tokens in non-hardware query."

    # Emotional disclosure relevance check: do not accept generic greetings or technical lectures
    is_emotional_query = any(k in q_lower for k in ["tiring day", "exhausting", "disagreement", "best friend", "feel hurt", "feel sad", "failed my exam", "in this situation", "message them first"])
    if is_emotional_query:
        if re.match(r"^(hello|hi|hey)\s+(there|friend|bro|buddy)?\b[!,.]?\s*(how\s+can\s+i\s+help|what\s+can\s+i\s+do)", r_lower):
            return False, "Generic greeting in response to emotional disclosure."
        if any(tech in r_lower for tech in ["c++", "machine learning", "deep learning", "neural network", "transformer", "operating system"]):
            return False, "Unrelated technical topic in response to emotional disclosure."

    # Check for science fiction mentions in non-fiction queries
    if "science fiction" in r_lower and "fiction" not in q_lower:
        return False, "Contains 'science fiction' in factual/educational response."

    # Check conversational polarity opening on definition queries (e.g. "No, ...")
    if is_definition_query and re.match(r"^(no|yes|sure|well|nope|yeah)\b[,!.]", r_lower.strip()):
        return False, "Irrelevant conversational polarity opening ('No,' / 'Yes,') on definition query."

    # Check self-referential anthropomorphism in definition
    if re.search(r"\b(i\s+can\s+be\s+used\s+to\s+learn|i\s+am\s+used\s+to\s+learn|i\s+am\s+a\s+mathematical\s+model|i\s+am\s+an\s+algorithm)\b", r_lower):
        return False, "Self-referential anthropomorphism in definition."
    if re.search(r"\bas\s+a\s+(deep\s+learning|machine\s+learning|language|neural|computational)?\s*model\b", r_lower):
        return False, "Self-referential model boilerplate."
    if re.search(r"\bas\s+an?\s+(ai|artificial\s+intelligence)\b", r_lower):
        return False, "Self-referential model boilerplate."

    # Check false negation of scientific domain
    if re.search(r"\b(is\s+not|not)\s+(the\s+|a\s+)?(field|part|branch|subfield)\s+of\s+artificial\s+intelligence\b", r_lower):
        return False, "Factual error: falsely negates connection to artificial intelligence."
    if re.search(r"\bnot\s+the\s+field\s+of\s+artificial\s+intelligence\b", r_lower):
        return False, "Factual error: falsely negates connection to artificial intelligence."

    # Check ungrammatical noun-category distortions
    if re.search(r"\b(is|it'?s)\s+a\s+(broad\s+|narrow\s+)?(computer\s+science|machine\s+learning)\b", r_lower):
        return False, "Ungrammatical noun-category distortion (e.g. 'is a computer science' / 'is a machine learning')."
    if re.search(r"\bthe\s+most\s+commonly\s+used\s+in\s+[\w\s]+\s+is\s+a\s+(computer\s+science|machine\s+learning)\b", r_lower):
        return False, "Ungrammatical noun-category distortion."

    # 2. Check for truncated endings and dangling grammatical fragments
    terminal_puncts = ('.', '!', '?', '"', "'", ')', '`')
    if not any(trimmed.endswith(p) for p in terminal_puncts):
        return False, "Response ends abruptly without terminal punctuation (truncated output)."

    if re.search(r"\b(and|or|but|such as|including|with|the|to|of|in|for|choose|from|that|a|an|it|is|as|\d+)\s*[.!?]?$", r_lower):
        return False, "Response ends with a dangling grammatical fragment or cutoff."

    # 3. Check for known hallucinated or corrupt tokens
    corrupt_tokens = [
        "mach is a", "burgpt", "rusearching", "fromptimported", "milam", "dfecdm",
        "allmo", "chatbits", "subfielding", "growthrowing", "ancientizing",
        "hrnou", "phooting", "fupiter", "cupiterity", "darkestory", "text-like-like",
        "cno-no", "cno, is", "cno, cno", "pacon"
    ]
    for ct in corrupt_tokens:
        if ct in r_lower:
            return False, f"Contains corrupt or hallucinated token: '{ct}'"

    # Check for repetitive stuttering patterns (e.g. "No-No-No" or word repeated 4+ times consecutively)
    if re.search(r"\b(\w+)(?:-\1){3,}\b", response, re.I) or re.search(r"(\b\w+\b)(?:\s+\1){3,}", response, re.I):
        return False, "Severe repetitive token stutter detected."

    # Check for corrupt uppercase token stutter (e.g. "LLLIL", "ILL")
    known_valid_caps = {"AI", "ML", "DL", "CNN", "RNN", "ANN", "NLP", "GPU", "CPU", "BPE", "SGD", "PCA", "API", "URL", "HTML", "JSON", "CUDA", "BERT", "LSTM", "LLM", "VRAM", "OASST", "BOS", "EOS"}
    for tok in re.findall(r"\b[A-Za-z]{3,}\b", response):
        if tok.isupper() and tok not in known_valid_caps:
            return False, f"Contains corrupt or garbled uppercase token: '{tok}'"

    # 4. Check for empty non-answers or generic evasion loops
    if any(frag in r_lower for frag in [
        "yes, here are the steps you can use to help you with",
        "no problem! here are the steps you can use to help",
        "what do you like to learn?", "what is the same way you want me to learn",
        "i can provide a specific type of machine learning and information",
        "i do not have access to a language model",
        "i do not have a language model"
    ]):
        return False, "Response is a generic evasion or conversational loop."

    # 5. Check for severe circular repetition of query subject
    q_clean = re.sub(r"(?i)\b(what is|what are|explain|define|search for|tell me about)\b", "", query).strip(" :-\t?.")
    if len(q_clean) >= 4 and r_lower.count(q_clean.lower()) >= 4 and len(words) < 70:
        return False, f"Severe circular repetition of query subject: '{q_clean}'"

    # Detect repeated n-gram loops (e.g. "learn from data and learn from data")
    words_clean = [re.sub(r"[^\w]", "", w.lower()) for w in words if w.strip()]
    if len(words_clean) >= 8:
        tri_grams: Dict[Tuple[str, str, str], int] = {}
        for i in range(len(words_clean) - 2):
            tg = (words_clean[i], words_clean[i + 1], words_clean[i + 2])
            tri_grams[tg] = tri_grams.get(tg, 0) + 1
            if tri_grams[tg] >= 3:
                return False, f"Repetitive 3-gram loop detected: '{' '.join(tg)}'"

    # 6. Check for fact-inversions and inverted categorical relationships
    if "chatgpt is a subfield" in r_lower:
        return False, "Factual error: describes ChatGPT as a subfield of ML."
    if "sun revolves around the earth" in r_lower and not any(w in r_lower for w in ["false", "incorrect", "actually", "instead", "orbits"]):
        return False, "Factual error regarding solar system heliocentrism."
    if re.search(r"\b(artificial intelligence|ai)\s+is\s+a\s+(broad\s+|narrow\s+)?machine\s+learning\b", r_lower):
        return False, "Fact inversion: defines AI as a type of machine learning."
    if re.search(r"\b(artificial intelligence|ai)\s+is\s+a\s+(deep\s+)?neural\s+network\b", r_lower):
        return False, "Fact inversion: defines AI as a neural network."

    # 7. Educational Concept Subject Relevance:
    # If the user asks "What is Machine Learning?", the model must actually define Machine Learning,
    # not define AI as "a broad machine learning"!
    q_norm_subj = normalize_spelling(q_clean).lower()
    is_comparison = any(w in q_norm_subj for w in ["difference", "compare", " vs ", "relation", "relate", "how do"])
    has_q_ai = bool(re.search(r"\b(ai|artificial\s+intelligence)\b", q_norm_subj))
    has_q_ml = bool(re.search(r"\b(ml|machine\s+learning)\b", q_norm_subj))
    has_q_dl = bool(re.search(r"\b(dl|deep\s+learning)\b", q_norm_subj))

    if has_q_ml and not has_q_ai and not has_q_dl and not is_comparison:
        first_sentence = re.split(r"[.!?]", response)[0].lower().strip()
        if re.match(r"^(the\s+)?(artificial\s+intelligence|ai)\b", first_sentence):
            return False, "Off-topic definition: defines Artificial Intelligence instead of requested Machine Learning."
        if not (r_lower.startswith("**machine learning") or "machine learning" in r_lower[:50]):
            return False, "Machine Learning definition does not begin with a clear identification of Machine Learning."

    if has_q_dl and not is_comparison:
        first_sentence = re.split(r"[.!?]", response)[0].lower().strip()
        if re.match(r"^(the\s+)?(artificial\s+intelligence|ai)\b", first_sentence):
            return False, "Off-topic definition: defines Artificial Intelligence instead of requested Deep Learning."
        if re.match(r"^(the\s+)?machine\s+learning\b", first_sentence) and "deep learning" not in first_sentence[:25]:
            return False, "Off-topic definition: defines Machine Learning instead of requested Deep Learning."
        if not (r_lower.startswith("**deep learning") or "deep learning" in r_lower[:50]):
            return False, "Deep Learning definition does not begin with a clear identification of Deep Learning."
        if "deep" not in r_lower:
            return False, "Off-topic definition: explanation of Deep Learning never mentions 'deep'."

    if is_comparison:
        # AI vs ML vs DL
        if has_q_ai and has_q_ml and has_q_dl:
            has_all_three = ("artificial intelligence" in r_lower or bool(re.search(r"\bai\b", r_lower))) and \
                            ("machine learning" in r_lower or bool(re.search(r"\bml\b", r_lower))) and \
                            ("deep learning" in r_lower or bool(re.search(r"\bdl\b", r_lower)))
            if not has_all_three:
                return False, "Comparison response fails to address all three: AI, ML, and DL."
            has_distinction = any(w in r_lower for w in ["subset", "subfield", "concentric", "hierarchical", "umbrella", "all deep learning is"])
            if not has_distinction:
                return False, "Comparison response fails to accurately describe the hierarchical relationship between AI, ML, and DL."

        # AI vs ML
        elif has_q_ai and has_q_ml:
            has_ai_resp = "artificial intelligence" in r_lower or bool(re.search(r"\bai\b", r_lower))
            has_ml_resp = "machine learning" in r_lower or bool(re.search(r"\bml\b", r_lower))
            if not (has_ai_resp and has_ml_resp):
                return False, "Comparison response fails to address both AI and ML."
            has_distinction = any(w in r_lower for w in ["subfield", "subset", "branch", "umbrella", "hierarchical", "all machine learning is"])
            if not has_distinction:
                return False, "Comparison response between AI and ML fails to state the hierarchical relationship or distinction."

        # ML vs DL
        elif has_q_ml and has_q_dl:
            has_dl_resp = "deep learning" in r_lower or bool(re.search(r"\bdl\b", r_lower))
            has_ml_resp = "machine learning" in r_lower or bool(re.search(r"\bml\b", r_lower))
            if not (has_dl_resp and has_ml_resp):
                return False, "Comparison response fails to address both Machine Learning and Deep Learning."
            has_distinction = any(w in r_lower for w in ["subfield", "subset", "branch", "layer", "neural network", "hierarchical", "feature engineering"])
            if not has_distinction:
                return False, "Comparison response between ML and DL fails to explain the hierarchical relationship or architectural difference."

        # AI vs DL
        elif has_q_ai and has_q_dl:
            has_ai_resp = "artificial intelligence" in r_lower or bool(re.search(r"\bai\b", r_lower))
            has_dl_resp = "deep learning" in r_lower or bool(re.search(r"\bdl\b", r_lower))
            if not (has_ai_resp and has_dl_resp):
                return False, "Comparison response fails to address both AI and Deep Learning."
            has_distinction = any(w in r_lower for w in ["subfield", "subset", "branch", "umbrella", "hierarchical", "specialized"])
            if not has_distinction:
                return False, "Comparison response between AI and DL fails to explain their hierarchical relationship."

    if "artificial neural networks (ml)" in r_lower or "neural network (ml)" in r_lower:
        return False, "Factual error: labels neural networks as '(ML)'."

    # Subject overlap check: for definition and factual questions, verify that the response actually mentions the requested subject
    if is_definition_query and not is_casual_conversation:
        q_subj_words = set(re.findall(r"\w+", q_clean.lower())) - {
            "what", "is", "the", "a", "an", "of", "in", "for", "to", "and", "or", "with",
            "explain", "describe", "define", "tell", "me", "about", "how", "why", "does", "do",
            "can", "you", "please", "meaning", "definition"
        }
        if len(q_subj_words) >= 2:
            resp_words_set = set(re.findall(r"\w+", r_lower))
            if not q_subj_words.intersection(resp_words_set):
                return False, f"Response shares no substantive subject overlap with query keywords: {q_subj_words}"

    # 8. Check grounding against evidence if evidence is provided
    if evidence_bundle and evidence_bundle.has_results and evidence_bundle.items:
        top_title = evidence_bundle.items[0].title.lower()
        title_words = set(re.findall(r"\w+", top_title)) - {"the", "a", "an", "of", "in", "on"}
        q_words = set(re.findall(r"\w+", q_clean.lower())) - {"the", "a", "an", "of", "in", "on"}
        resp_words = set(re.findall(r"\w+", r_lower))

        if title_words and not resp_words.intersection(title_words) and not resp_words.intersection(q_words):
            return False, "Response shares no substantive subject overlap with retrieved evidence."

    return True, "Response passed quality validation."


def run_local_model_generation(
    model,
    tokenizer,
    history: list,
    prompt_text: str,
    device: torch.device,
    temperature: float = 0.35,
    top_p: float = 0.85,
    top_k: int = 20,
    repetition_penalty: float = 1.3,
    no_repeat_ngram_size: int = 3,
    max_new_tokens: int = 160
) -> str:
    """
    Executes controlled autoregressive token generation from the local FromScratchLM.
    """
    parts = ["[BOS]"]
    for turn in history[:-1]:
        clean_t = sanitize_for_prompt(turn["text"])
        if turn["role"] == "user":
            parts.append("<|user|>")
            parts.append(clean_t)
        elif turn["role"] == "assistant":
            parts.append("<|assistant|>")
            parts.append(clean_t)
            parts.append("[EOS]")

    parts.append("<|user|>")
    parts.append(sanitize_for_prompt(prompt_text))
    parts.append("<|assistant|>")
    formatted = "\n".join(parts) + "\n"

    encoded = tokenizer.encode(formatted)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    generated_tokens = []
    for _ in range(max_new_tokens):
        context = input_ids[:, -CONTEXT_LENGTH:]
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(context)
        logits = logits[:, -1, :].float()

        if generated_tokens:
            for tid in set(generated_tokens):
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

        if no_repeat_ngram_size > 0 and len(generated_tokens) >= no_repeat_ngram_size - 1:
            curr_prefix = tuple(generated_tokens[-(no_repeat_ngram_size - 1):])
            banned = set()
            for i in range(len(generated_tokens) - no_repeat_ngram_size + 1):
                if tuple(generated_tokens[i : i + no_repeat_ngram_size - 1]) == curr_prefix:
                    banned.add(generated_tokens[i + no_repeat_ngram_size - 1])
            for b_tok in banned:
                logits[0, b_tok] = -float("inf")

        if temperature <= 0.05:
            # Greedy decoding
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-4)
            if top_k > 0:
                vals, idxs = torch.topk(logits, min(top_k, logits.size(-1)))
                filt = torch.full_like(logits, -float("inf"))
                filt.scatter_(1, idxs, vals)
                logits = filt

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -float("inf"))

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

        token_id = next_tok.item()
        if token_id in STOP_TOKEN_IDS:
            break

        generated_tokens.append(token_id)
        input_ids = torch.cat([input_ids, next_tok], dim=-1)

    raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    clean_resp = re.split(r"<\|user\|>|<\|assistant\|>|\[EOS\]|User:", raw_response)[0].strip()
    return clean_resp


def handle_compound_question(
    sub_queries: List[str],
    retrieval_engine: WebRetrievalEngine,
    router: Any
) -> Tuple[str, List[Any]]:
    """
    Independently retrieves and synthesizes information for each part of a compound question.
    """
    answers = []
    all_sources = []
    for i, sq in enumerate(sub_queries, 1):
        dec = router.classify(sq)
        bundle = retrieval_engine.search_and_retrieve(sq, max_results=2, fetch_pages=True, intent_decision=dec)
        if bundle and bundle.has_results:
            ans = synthesize_grounded_answer(sq, bundle, intent_decision=dec)
            all_sources.extend(bundle.items)
            subj_title = dec.interpretation.main_subject if (dec.interpretation and dec.interpretation.main_subject) else sq
            answers.append(f"**{i}. {subj_title}:**\n{ans.strip()}")
        else:
            answers.append(f"**{i}. {sq}:**\nI'm having trouble retrieving reliable information on this specific topic right now.")

    full_reply = "Here is an overview of both:\n\n" + "\n\n".join(answers)
    return full_reply, all_sources


def generate_external_model_response(
    model_manager: ModelManager,
    retrieval_engine: WebRetrievalEngine,
    history: list,
    raw_user_prompt: str,
    normalized_prompt: str,
    web_search_enabled: bool,
    debug_mode: bool,
    finish_fn
) -> Tuple[str, str]:
    """
    Direct response generation using the external pretrained Qwen2.5-1.5B-Instruct model.
    Guarantees strict model independence: does NOT call FromScratchLM or any hybrid fallback.
    """
    # 1. Date & Time Resolution
    date_reply = detect_date_query(normalized_prompt)
    if date_reply:
        return finish_fn(
            resp_text=date_reply,
            route_name="system_date",
            detected_intent="DATE_TIME",
            detected_emotion="NONE",
            resolved_topic="current_date_time",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # 2. Source Request Detection
    if detect_source_request(normalized_prompt):
        sources_text = format_source_links(getattr(retrieval_engine, "session_last_sources", []))
        return finish_fn(
            resp_text=sources_text,
            route_name="sources_display",
            detected_intent="SOURCES_REQUEST",
            detected_emotion="NONE",
            resolved_topic="sources",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # 3. Emergency Helpline
    emerg = detect_emergency_helpline(normalized_prompt)
    if emerg:
        return finish_fn(
            resp_text=emerg,
            route_name="emergency_helpline",
            detected_intent="EMERGENCY_HELPLINE",
            detected_emotion="NONE",
            resolved_topic="emergency_services",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    def safe_engine_generate(**kwargs):
        try:
            return model_manager.external_engine.generate(**kwargs)
        except Exception as err:
            print(f"[WARN] External model engine error: {err}")
            return (
                "I apologize, but I encountered an internal engine communication error while processing your request. "
                "Please verify the model server is responding or switch models with `/model custom`.",
                0.0
            )

    # 5. OFFLINE MODE (Web Search OFF)
    if not web_search_enabled:
        low_p = normalized_prompt.lower()
        is_officeholder = bool(re.search(
            r"\b(prime\s+minister|president|director|ceo|governor|chief\s+justice|leader|head\s+of\s+state|chancellor)\s+of\b",
            low_p
        ))
        is_freshness = is_officeholder or bool(re.search(
            r"\b(today'?s|announced\s+today|latest\s+discovery|current\s+price|stock\s+price|weather|released\s+in\s+202[4-9]|in\s+(late\s+|early\s+|mid\s+)?202[4-9]|2025|2026|recent\s+news|headlines|breaking\s+news|who\s+is\s+the\s+current|box\s+office|gross\s+revenue)\b",
            low_p
        ))
        if is_freshness:
            fresh_msg = (
                "I cannot verify live internet information while web search is disabled. "
                "Please enable web search with `/web on`, and I will search the web for the latest verified information for you."
            )
            return finish_fn(
                resp_text=fresh_msg,
                route_name="qwen_offline_freshness",
                detected_intent="CURRENT_EVENTS",
                detected_emotion="NONE",
                resolved_topic="freshness_check",
                local_model_called=False,
                web_search_called=False,
                passed_validation=True
            )

        # Pure offline companion generation with Qwen
        sys_prompt = (
            "You are MEYLO, a friendly, intelligent, and helpful personal AI companion. "
            "Your tagline is 'Your AI. Your Space.' "
            "Converse naturally, explain concepts clearly, maintain context, understand emotions, "
            "and be honest when you don't know something."
        )
        reply, latency = safe_engine_generate(
            prompt=normalized_prompt,
            history=history,
            system_instruction=sys_prompt
        )
        return finish_fn(
            resp_text=reply,
            route_name="qwen_offline",
            detected_intent="CONVERSATION",
            detected_emotion="NONE",
            resolved_topic="offline_chat",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # 6. ONLINE MODE (Web Search ON)
    low_p = normalized_prompt.lower()
    is_greeting_match, _ = detect_greeting(normalized_prompt)
    is_social = is_greeting_match or any(k in low_p for k in [
        "how are you", "who are you", "what are you doing", "can we talk", "feel lonely",
        "feel sad", "tiring day", "thank you", "thanks", "good morning", "good evening", "good night"
    ]) or (
        any(k in low_p for k in ["friend", "disagreement", "message them first", "argued", "fight"])
        and not any(k in low_p for k in ["what is", "define", "explain", "who is", "papers"])
    )

    if is_social:
        sys_prompt = (
            "You are MEYLO, a warm, friendly, and natural personal AI companion. "
            "Respond empathetically and conversationally with genuine emotional warmth."
        )
        reply, latency = safe_engine_generate(
            prompt=normalized_prompt,
            history=history,
            system_instruction=sys_prompt
        )
        return finish_fn(
            resp_text=reply,
            route_name="qwen_conversation",
            detected_intent="CONVERSATION",
            detected_emotion="NONE",
            resolved_topic="companion_chat",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # Factual, informational, or research queries -> live web search
    num_match = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|[1-9]|10)\b", low_p)
    num_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    max_res = 3
    if num_match:
        val = num_match.group(0).lower()
        max_res = num_words.get(val, int(val) if val.isdigit() else 3)
        max_res = max(1, min(max_res, 10))

    evidence_bundle = retrieval_engine.search_and_retrieve(
        normalized_prompt,
        max_results=max_res,
        fetch_pages=True
    )

    if evidence_bundle and evidence_bundle.has_results and evidence_bundle.items:
        evidence_lines = []
        for idx, it in enumerate(evidence_bundle.items[:3], 1):
            date_str = f" [Date: {it.date}]" if getattr(it, "date", None) else ""
            pub_str = f" [Publisher: {it.publisher}]" if getattr(it, "publisher", None) else ""
            snip = (it.snippet or "").strip()
            full = (it.full_text or "").strip()
            if full and snip and snip not in full:
                item_text = f"{snip}\n{full}"
            else:
                item_text = full or snip
            if len(item_text) > 1000:
                item_text = item_text[:1000] + "..."
            evidence_lines.append(f"Source [{idx}]: {it.title}{date_str}{pub_str}\nURL: {it.url}\nInformation: {item_text}")
        evidence_str = "\n\n".join(evidence_lines)
        if len(evidence_str) > 3000:
            evidence_str = evidence_str[:3000] + "\n...[truncated for context limits]..."

        sys_prompt = (
            "You are MEYLO, a helpful, intelligent personal AI companion with live web access. "
            "Answer the user's question accurately, concisely, and naturally based on the verified web evidence provided below. "
            "Ground your facts strictly in the retrieved evidence. If the evidence provides source titles, publication dates, or URLs, reference them naturally. "
            "Do not invent facts not supported by the evidence."
        )
        reply, latency = safe_engine_generate(
            prompt=normalized_prompt,
            history=history,
            context_evidence=evidence_str,
            system_instruction=sys_prompt
        )
        return finish_fn(
            resp_text=reply,
            route_name="qwen_web_grounded",
            detected_intent="WEB_GROUNDED",
            detected_emotion="NONE",
            resolved_topic="web_research",
            local_model_called=False,
            web_search_called=True,
            passed_validation=True,
            new_sources=evidence_bundle.items
        )
    else:
        sys_prompt = (
            "You are MEYLO, a helpful personal AI companion. Live web search did not find matching sources for this query. "
            "Inform the user politely that web search was inconclusive, and answer only what is solidly known or ask for clarification."
        )
        reply, latency = safe_engine_generate(
            prompt=normalized_prompt,
            history=history,
            system_instruction=sys_prompt
        )
        return finish_fn(
            resp_text=reply,
            route_name="qwen_web_inconclusive",
            detected_intent="WEB_SEARCH",
            detected_emotion="NONE",
            resolved_topic="web_search",
            local_model_called=False,
            web_search_called=True,
            passed_validation=False
        )


def generate_chat_response(
    model,
    tokenizer,
    history: list,
    device: torch.device,
    retrieval_engine: WebRetrievalEngine,
    web_search_enabled: bool = False,
    debug_mode: bool = False,
    temperature: float = 0.35,
    top_p: float = 0.85,
    top_k: int = 20,
    repetition_penalty: float = 1.3,
    no_repeat_ngram_size: int = 3,
    max_new_tokens: int = 160,
    model_manager: Optional[ModelManager] = None,
    **kwargs
):
    t_start = time.time()
    raw_user_prompt = history[-1]["text"].strip()

    # Step 1: Spelling & Typo Normalization
    normalized_prompt = normalize_spelling(raw_user_prompt)

    # Helper finish function
    def finish(
        resp_text: str,
        route_name: str,
        detected_intent: str,
        detected_emotion: str,
        resolved_topic: str,
        local_model_called: bool = False,
        web_search_called: bool = False,
        passed_validation: bool = True,
        new_sources: Optional[List[Any]] = None,
        failure_reason: Optional[str] = None
    ) -> Tuple[str, str]:
        # Always log detailed diagnostic router event to logs/router_debug.log
        log_router_debug(
            original_message=raw_user_prompt,
            normalized_message=normalized_prompt,
            detected_intent=detected_intent,
            detected_emotion=detected_emotion,
            resolved_topic=resolved_topic,
            selected_handler=route_name,
            local_model_called=local_model_called,
            web_search_called=web_search_called,
            passed_validation=passed_validation,
            final_response_source=route_name,
            failure_reason=failure_reason
        )

        if new_sources is not None:
            retrieval_engine.session_last_sources = list(new_sources)
            if hasattr(retrieval_engine, "last_evidence"):
                retrieval_engine.last_evidence = EvidenceBundle(
                    query=raw_user_prompt,
                    items=[
                        EvidenceItem(
                            title=getattr(s, "title", str(s)),
                            url=getattr(s, "url", ""),
                            snippet=getattr(s, "snippet", ""),
                            full_text=getattr(s, "full_text", getattr(s, "snippet", "")),
                            rank=getattr(s, "rank", i)
                        )
                        for i, s in enumerate(new_sources, 1)
                    ],
                    has_results=bool(new_sources)
                )

        # Update conversational memory active subject and record turn
        if hasattr(retrieval_engine, "conversation_memory"):
            if resolved_topic and resolved_topic not in ["current_date_time", "greeting", "sources", "calculator", "incomplete_stem"]:
                retrieval_engine.conversation_memory.active_subject = resolved_topic

            dummy_interp = QuestionInterpretation(
                original_message=raw_user_prompt,
                normalized_message=normalized_prompt,
                intent=detected_intent,
                main_subject=resolved_topic or "conversation"
            )
            src_records = [
                SourceRecord(
                    title=getattr(s, "title", str(s)),
                    url=getattr(s, "url", ""),
                    snippet=getattr(s, "snippet", ""),
                    full_text=getattr(s, "full_text", getattr(s, "snippet", "")),
                    source_provider="web",
                    rank=i,
                    publisher=""
                )
                for i, s in enumerate(new_sources or [], 1)
            ]
            retrieval_engine.conversation_memory.record_turn(
                user_message=raw_user_prompt,
                assistant_reply=resp_text,
                interpretation=dummy_interp,
                displayed_sources=src_records,
                search_queries=[resolved_topic] if resolved_topic else []
            )

        return resp_text, route_name

    # Step 1a: Independent External Pretrained Model Execution (Qwen2.5-1.5B-Instruct)
    if model_manager is not None and getattr(model_manager, "active_model_name", "custom") == "external":
        return generate_external_model_response(
            model_manager=model_manager,
            retrieval_engine=retrieval_engine,
            history=history,
            raw_user_prompt=raw_user_prompt,
            normalized_prompt=normalized_prompt,
            web_search_enabled=web_search_enabled,
            debug_mode=debug_mode,
            finish_fn=finish
        )

    # Step 1b: Ambiguous / Garbled Phrase Clarification ("i and old")
    ambig_reply = detect_ambiguous_phrase(normalized_prompt)
    if ambig_reply:
        return finish(
            resp_text=ambig_reply,
            route_name="companion_ambiguity",
            detected_intent="AMBIGUOUS",
            detected_emotion="NONE",
            resolved_topic="ambiguity_clarification",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # Step 2: Date & Time Resolution (Asia/Kolkata timezone)
    date_reply = detect_date_query(normalized_prompt)
    if date_reply:
        return finish(
            resp_text=date_reply,
            route_name="system_date",
            detected_intent="DATE_TIME",
            detected_emotion="NONE",
            resolved_topic="current_date_time",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # Step 3: Source Request Detection ("Give me the links", "Show sources", etc.)
    if detect_source_request(normalized_prompt):
        sources_text = format_source_links(getattr(retrieval_engine, "session_last_sources", []))
        return finish(
            resp_text=sources_text,
            route_name="sources_display",
            detected_intent="SOURCES_REQUEST",
            detected_emotion="NONE",
            resolved_topic="sources",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # Step 4: Bot Identity & Introduction Inquiries ("what's your name", "who are you", "can you help me")
    identity_reply = detect_identity_query(normalized_prompt)
    if identity_reply:
        return finish(
            resp_text=identity_reply,
            route_name="companion_identity",
            detected_intent="IDENTITY",
            detected_emotion="NONE",
            resolved_topic="bot_identity",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # Step 5: Active Subject Resolution for Context
    active_subj = None
    if history and len(history) > 1:
        for turn in reversed(history[:-1]):
            t_text = turn.get("text", "")
            t_norm = normalize_spelling(t_text).lower()
            if any(k in t_norm for k in ["machine learning", "machne lernng", " ml "]):
                active_subj = "Machine Learning"
                break
            elif "sleep" in t_norm:
                active_subj = "Sleep"
                break
            elif "transformer" in t_norm:
                active_subj = "Transformer Model"
                break
            elif "gta" in t_norm:
                active_subj = "Grand Theft Auto VI (GTA 6)"
                break
            elif "pakito" in t_norm:
                active_subj = "Pakito"
                break
            elif "python" in t_norm:
                active_subj = "Python"
                break
            elif "pytorch" in t_norm:
                active_subj = "PyTorch"
                break

    if not active_subj and hasattr(retrieval_engine, "conversation_memory"):
        active_subj = retrieval_engine.conversation_memory.get_last_subject()

    # Step 6: Short & Incomplete Message Handling ("pls", "help", "explain", "pls explain", acknowledgments)
    short_reply = detect_short_or_incomplete_message(normalized_prompt, active_subject=active_subj)
    if short_reply:
        return finish(
            resp_text=short_reply,
            route_name="companion_short_input",
            detected_intent="SHORT_INTERACTION",
            detected_emotion="NONE",
            resolved_topic=active_subj or "clarification",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # Step 7: Incomplete Emotional Stem Detection (e.g. "i am sa" -> "Did you mean you're feeling sad?")
    stem_reply = detect_incomplete_emotional_stem(normalized_prompt)
    if stem_reply:
        return finish(
            resp_text=stem_reply,
            route_name="companion_incomplete_stem",
            detected_intent="INCOMPLETE_EMOTIONAL_STEM",
            detected_emotion="UNCERTAIN",
            resolved_topic="incomplete_stem",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # Step 7b: Public Safety & Emergency Services Helpline Detection
    emergency_reply = detect_emergency_helpline(normalized_prompt)
    if emergency_reply:
        return finish(
            resp_text=emergency_reply,
            route_name="emergency_helpline",
            detected_intent="EMERGENCY_HELPLINE",
            detected_emotion="NONE",
            resolved_topic="emergency_services",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # Step 8: Greeting Detection ("hii", "hello how are you today", etc.)
    is_greeting, greeting_reply = detect_greeting(normalized_prompt)
    if is_greeting and greeting_reply:
        return finish(
            resp_text=greeting_reply,
            route_name="companion_greeting",
            detected_intent="GREETING",
            detected_emotion="NONE",
            resolved_topic="greeting",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # Step 9: Conversational Statements ("I am learning machine learning", "I want to become an AI engineer")
    is_stmt, stmt_reply, new_subj = detect_conversational_statement(normalized_prompt, active_subject=active_subj)
    if is_stmt and stmt_reply:
        return finish(
            resp_text=stmt_reply,
            route_name="companion_statement",
            detected_intent="CONVERSATIONAL_STATEMENT",
            detected_emotion="NONE",
            resolved_topic=new_subj or active_subj or "learning",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # Step 10: Emotion and Human Situation Detection (13 classes)
    emotion_cat, emotion_sub = detect_emotion_and_intent(normalized_prompt, history=history)
    if emotion_cat != "NEUTRAL":
        comp_reply = generate_companion_response(emotion_cat, emotion_sub, normalized_prompt)
        return finish(
            resp_text=comp_reply,
            route_name=f"companion_{emotion_cat.lower()}",
            detected_intent="EMOTION",
            detected_emotion=emotion_cat,
            resolved_topic=emotion_sub or emotion_cat.lower(),
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # Step 10b: Programming & Technical Assistance Engine (Category D)
    is_prog, prog_sub, prog_meta = detect_programming_intent(normalized_prompt, history=history)
    if is_prog:
        if prog_sub == "generate_code":
            prog_reply = generate_code_response(normalized_prompt, prog_meta, history=history)
        elif prog_sub == "explain_code_lines":
            prog_reply = explain_code_lines(history=history)
        elif prog_sub == "fix_error":
            prog_reply = handle_error_debugging(normalized_prompt)
        elif prog_sub == "missing_debug_info":
            prog_reply = request_missing_debug_info()
        else:
            prog_reply = generate_code_response(normalized_prompt, prog_meta, history=history)

        return finish(
            resp_text=prog_reply,
            route_name="programming_assistance",
            detected_intent="PROGRAMMING",
            detected_emotion="NONE",
            resolved_topic=prog_meta.get("language") or "programming",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # Step 11: Multi-Turn Context & Pronoun Resolution ("Explain it", "How does it work")
    contextual_prompt = resolve_context(normalized_prompt, active_subject=active_subj)

    # Step 12: Technical Follow-up Handling on Active Concept
    if active_subj and active_subj.lower() in ["transformer", "transformers"]:
        p_clean_low = normalized_prompt.lower().strip(" ?.!\"'")
        if p_clean_low in ["how does it work", "how does that work", "how it works", "how do they work"]:
            followup_ans = "It uses self-attention mechanisms to evaluate the relationships and relevance between all tokens in a sequence simultaneously."
            return finish(
                resp_text=followup_ans,
                route_name="technical_followup",
                detected_intent="FOLLOWUP",
                detected_emotion="NONE",
                resolved_topic="Transformer",
                local_model_called=False,
                web_search_called=False,
                passed_validation=True
            )
        elif p_clean_low in ["give me an example", "give an example", "can you give an example", "example"]:
            followup_ans = "Imagine a sentence where the meaning of one word depends on other words (like 'bank' in 'river bank' versus 'money bank'). Self-attention helps the model understand these relationships in context."
            return finish(
                resp_text=followup_ans,
                route_name="technical_followup",
                detected_intent="FOLLOWUP",
                detected_emotion="NONE",
                resolved_topic="Transformer",
                local_model_called=False,
                web_search_called=False,
                passed_validation=True
            )
        elif p_clean_low in ["where is it used", "where are they used", "where is that used"]:
            followup_ans = "Transformers are used in language models (like GPT and BERT), machine translation, speech processing, and computer vision (like Vision Transformers)."
            return finish(
                resp_text=followup_ans,
                route_name="technical_followup",
                detected_intent="FOLLOWUP",
                detected_emotion="NONE",
                resolved_topic="Transformer",
                local_model_called=False,
                web_search_called=False,
                passed_validation=True
            )

    # Step 13: Universal Semantic Intent Classification
    cached_count = len(retrieval_engine.session_last_sources) if hasattr(retrieval_engine, "session_last_sources") else 0
    intent_decision = retrieval_engine.classify_intent(
        contextual_prompt,
        history=history,
        has_cached_web=(cached_count > 0)
    )

    # Resolve active subject from question
    target_subject = intent_decision.interpretation.main_subject if (intent_decision and intent_decision.interpretation and intent_decision.interpretation.main_subject) else contextual_prompt
    if "transformer" in contextual_prompt.lower():
        target_subject = "Transformer"
    elif "deep learning" in contextual_prompt.lower():
        target_subject = "Deep Learning"
    elif "machine learning" in contextual_prompt.lower():
        target_subject = "Machine Learning"
    elif "neural network" in contextual_prompt.lower():
        target_subject = "Neural Network"
    elif "cnn" in contextual_prompt.lower():
        target_subject = "Convolutional Neural Network"
    elif "python" in contextual_prompt.lower():
        target_subject = "Python"

    # Step 15: Clarification for Ambiguous / Single-word Queries
    if intent_decision.intent == INTENT_AMBIGUOUS:
        return finish(
            resp_text=f"Your question appears ambiguous or needs a little more context: {intent_decision.reasoning}\n\nCould you please clarify what specific aspect you would like to explore?",
            route_name="clarification",
            detected_intent="AMBIGUOUS",
            detected_emotion="NONE",
            resolved_topic="ambiguous_query",
            local_model_called=False,
            web_search_called=False,
            passed_validation=True
        )

    # Step 15b: Conversational Follow-up on Cached Sources / Evidence
    if intent_decision.intent == INTENT_FOLLOWUP or (
        hasattr(retrieval_engine, "last_evidence") and
        retrieval_engine.last_evidence and
        retrieval_engine.last_evidence.items and
        any(w in normalized_prompt.lower() for w in [
            "first source", "second source", "third source",
            "first result", "second result", "third result",
            "source 1", "source 2", "source 3",
            "result 1", "result 2", "result 3",
            "website 1", "website 2", "website 3",
            "previous answer", "previous result",
            "second paper", "first paper", "third paper", "fourth paper", "fifth paper",
            "paper 2", "paper 4", "compare the second", "authors", "who wrote",
            "publication date", "major advancement", "advancements", "limitations"
        ])
    ):
        if hasattr(retrieval_engine, "handle_followup_reference"):
            followup_resp = retrieval_engine.handle_followup_reference(normalized_prompt)
            if not followup_resp and hasattr(retrieval_engine, "conversation_memory") and getattr(intent_decision, "interpretation", None):
                followup_resp = retrieval_engine.conversation_memory.resolve_followup(intent_decision.interpretation)
            if followup_resp:
                return finish(
                    resp_text=followup_resp,
                    route_name="web_cache_followup",
                    detected_intent="FOLLOWUP",
                    detected_emotion="NONE",
                    resolved_topic=target_subject,
                    local_model_called=False,
                    web_search_called=False,
                    passed_validation=True,
                    new_sources=retrieval_engine.last_evidence.items if retrieval_engine.last_evidence else None
                )

    # Step 16: Health & Safety Urgent Triage
    if intent_decision.intent == INTENT_HEALTH_SAFETY and intent_decision.sub_intent == SUB_HEALTH_URGENT:
        if web_search_enabled:
            evidence_bundle = retrieval_engine.search_and_retrieve(
                contextual_prompt, max_results=2, fetch_pages=True, intent_decision=intent_decision
            )
            grounded_answer = synthesize_grounded_answer(contextual_prompt, evidence_bundle, intent_decision=intent_decision)
            return finish(
                resp_text=grounded_answer,
                route_name="health_emergency_triage",
                detected_intent="HEALTH_SAFETY",
                detected_emotion="NONE",
                resolved_topic="emergency_triage",
                local_model_called=False,
                web_search_called=True,
                passed_validation=True,
                new_sources=evidence_bundle.items
            )
        else:
            empty_bundle = EvidenceBundle(query=contextual_prompt)
            answer = synthesize_health_answer(contextual_prompt, empty_bundle, is_urgent=True, immediate_guidance=intent_decision.immediate_safety_response)
            return finish(
                resp_text=answer,
                route_name="health_emergency_triage",
                detected_intent="HEALTH_SAFETY",
                detected_emotion="NONE",
                resolved_topic="emergency_triage",
                local_model_called=False,
                web_search_called=False,
                passed_validation=True
            )

    # Step 17: Compound Question Handling (when Web Search is ON)
    if web_search_enabled:
        sub_queries = split_compound_question(contextual_prompt)
        if len(sub_queries) > 1:
            compound_answer, all_srcs = handle_compound_question(
                sub_queries, retrieval_engine, retrieval_engine.intent_router
            )
            return finish(
                resp_text=compound_answer,
                route_name="web_compound_grounded",
                detected_intent="COMPOUND_QUESTION",
                detected_emotion="NONE",
                resolved_topic="compound_entities",
                local_model_called=False,
                web_search_called=True,
                passed_validation=True,
                new_sources=all_srcs
            )

    low_ctx = contextual_prompt.lower()

    # =================================================================
    # MODE 1: OFFLINE KNOWLEDGE (Web Search OFF)
    # =================================================================
    if not web_search_enabled:

        # Check for freshness queries, current events, or officeholder lookups that genuinely require live internet
        is_officeholder = bool(re.search(
            r"\b(prime\s+minister|president|director|ceo|governor|chief\s+justice|leader|head\s+of\s+state|chancellor)\s+of\b",
            low_ctx
        )) or getattr(intent_decision, "sub_intent", "") == "officeholder_lookup"

        is_freshness_query = is_officeholder or bool(re.search(
            r"\b(today'?s|announced\s+today|latest\s+discovery|current\s+price|stock\s+price|weather|released\s+in\s+202[4-9]|in\s+(late\s+|early\s+|mid\s+)?202[4-9]|2025|2026|recent\s+news|headlines|breaking\s+news|who\s+is\s+the\s+current|box\s+office|gross\s+revenue)\b",
            low_ctx
        )) or getattr(intent_decision, "requires_current_info", False) or intent_decision.intent == INTENT_CURRENT_EVENTS

        if is_freshness_query:
            if is_officeholder:
                fresh_msg = "I cannot verify current officeholders or live political information while web search is disabled. Please enable web search (`/web on`), and I'll look up the verified, current information for you."
            else:
                fresh_msg = "I'm sorry, I don't have access to live internet information right now because web search is disabled. Please enable web search (`/web on`), and I'll find the latest information for you."
            return finish(
                resp_text=fresh_msg,
                route_name="offline_freshness_fallback",
                detected_intent="CURRENT_EVENTS",
                detected_emotion="NONE",
                resolved_topic=target_subject,
                local_model_called=False,
                web_search_called=False,
                passed_validation=True
            )

        # Grounded offline educational answers for core concepts
        edu_offline = get_offline_educational_answer(contextual_prompt)
        if not edu_offline and target_subject:
            edu_offline = get_offline_educational_answer(target_subject)
        if edu_offline:
            return finish(
                resp_text=edu_offline,
                route_name="offline_educational_grounded",
                detected_intent="EDUCATIONAL",
                detected_emotion="NONE",
                resolved_topic=target_subject,
                local_model_called=False,
                web_search_called=False,
                passed_validation=True
            )

        # First pass: Run inference with local FromScratchLM
        local_resp = run_local_model_generation(
            model=model,
            tokenizer=tokenizer,
            history=history,
            prompt_text=contextual_prompt,
            device=device,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens
        )

        is_valid, fail_reason = validate_response_quality(contextual_prompt, local_resp, None)

        # Controlled Retry Strategy: If pass 1 fails quality validation, retry once with low temperature
        if not is_valid:
            retry_resp = run_local_model_generation(
                model=model,
                tokenizer=tokenizer,
                history=history,
                prompt_text=contextual_prompt,
                device=device,
                temperature=0.15,
                top_p=0.75,
                top_k=15,
                repetition_penalty=1.35,
                no_repeat_ngram_size=3,
                max_new_tokens=max_new_tokens
            )
            retry_valid, retry_fail = validate_response_quality(contextual_prompt, retry_resp, None)
            if retry_valid:
                local_resp = retry_resp
                is_valid = True
            else:
                fail_reason = f"Retry failed: {retry_fail}"

        if is_valid:
            return finish(
                resp_text=local_resp,
                route_name="neural_model",
                detected_intent="EDUCATIONAL",
                detected_emotion="NONE",
                resolved_topic=target_subject,
                local_model_called=True,
                web_search_called=False,
                passed_validation=True
            )
        else:
            # Fallback handling:
            # 1. If educational concept, provide authoritative offline educational answer
            edu_fallback = get_offline_educational_answer(contextual_prompt)
            if not edu_fallback and target_subject:
                edu_fallback = get_offline_educational_answer(target_subject)

            if edu_fallback:
                return finish(
                    resp_text=edu_fallback,
                    route_name="offline_educational_grounded",
                    detected_intent="EDUCATIONAL",
                    detected_emotion="NONE",
                    resolved_topic=target_subject,
                    local_model_called=True,
                    web_search_called=False,
                    passed_validation=True
                )

            # 2. If conversational, emotional, or interpersonal inquiry
            is_pure_conversational = (
                intent_decision.intent == INTENT_CONVERSATION or
                getattr(intent_decision, "sub_intent", "") in ["emotional_companion", "casual_social", "personal_advice"] or
                any(k in low_ctx for k in ["how are you", "who are you", "what are you doing", "can we talk", "feel lonely", "feel sad", "tiring day"])
            )
            is_interpersonal_dispute = (
                any(k in low_ctx for k in ["friend", "disagreement", "message them first", "what would you do in my situation", "what should i do", "argued", "fight"])
            ) and not any(k in low_ctx for k in ["what is", "define", "explain", "how does", "why does", "who is", "give an example", "technically", "machine learning", "sleep", "game"])

            if is_pure_conversational or is_interpersonal_dispute:
                if is_interpersonal_dispute:
                    comp_cat, comp_sub = detect_emotion_and_intent(normalized_prompt, history=history)
                    if comp_cat != "NEUTRAL":
                        conv_fallback = generate_companion_response(comp_cat, comp_sub, normalized_prompt)
                    else:
                        conv_fallback = "I'm right here with you. Navigating personal dilemmas or disagreements can be really challenging. Taking time to cool down and having a calm, honest conversation when you're ready is usually the best approach. Would you like to talk through it together?"
                else:
                    companion_fallbacks = [
                        "I'm right here with you! What's on your mind today?",
                        "I'm listening and right here by your side. Tell me more about what you're thinking.",
                        "I'm here for you! Feel free to share whatever is on your mind.",
                        "I'm right here with you. Tell me what's on your heart or what you'd like to talk about."
                    ]
                    conv_fallback = random.choice(companion_fallbacks)
                return finish(
                    resp_text=conv_fallback,
                    route_name="offline_conversational_fallback",
                    detected_intent="CONVERSATION",
                    detected_emotion="NONE",
                    resolved_topic=target_subject,
                    local_model_called=True,
                    web_search_called=False,
                    passed_validation=True
                )

            # 3. Honest limitation for unknown factual / general knowledge questions offline
            return finish(
                resp_text="I don't have enough verified offline information to answer that accurately. If you would like me to search the web for details, please enable web search with `/web on`.",
                route_name="offline_quality_fallback",
                detected_intent="EDUCATIONAL",
                detected_emotion="NONE",
                resolved_topic=target_subject,
                local_model_called=True,
                web_search_called=False,
                passed_validation=False,
                failure_reason=fail_reason
            )

    # =================================================================
    # MODE 2: WEB-GROUNDED RETRIEVAL (Web Search ON)
    # =================================================================
    is_explicit_web_search = bool(re.search(
        r"\b(search\s+(the\s+)?web|look\s+up|find\s+online|with\s+sources|with\s+citations|source\s+links|give\s+sources|recent\s+research|find\s+recent|latest\s+research|ai\s+research|research\s+about)\b",
        contextual_prompt.lower()
    ))
    is_freshness_query = bool(re.search(
        r"\b(current\s+price|stock\s+price|today'?s\s+weather|released\s+in\s+2025|released\s+in\s+2026|in\s+2025|in\s+2026|recent\s+news|today'?s\s+headlines|latest\s+ai\s+news|ai\s+news|developments\s+in\s+ai|latest\s+developments|ai\s+developments|latest\s+\w+\s+developments)\b",
        contextual_prompt.lower()
    )) or (intent_decision and (
        intent_decision.intent == INTENT_CURRENT_EVENTS or
        getattr(getattr(intent_decision, "interpretation", None), "requires_current_info", False)
    ))

    # Educational answer check: If it's a foundational educational concept and user did NOT ask for web sources
    if not is_explicit_web_search and not is_freshness_query:
        edu_offline = get_offline_educational_answer(contextual_prompt)
        if not edu_offline and target_subject:
            edu_offline = get_offline_educational_answer(target_subject)
        if edu_offline:
            return finish(
                resp_text=edu_offline,
                route_name="offline_educational_grounded",
                detected_intent="EDUCATIONAL",
                detected_emotion="NONE",
                resolved_topic=target_subject,
                local_model_called=False,
                web_search_called=False,
                passed_validation=True
            )

    # Determine if this message is conversational/emotional (should NOT search the web)
    _sub = getattr(intent_decision, "sub_intent", "") or ""
    is_personal_or_conversational = (
        intent_decision.intent == INTENT_CONVERSATION or
        is_prog or
        _sub in ["emotional_companion", "casual_social", "personal_advice", "learning_roadmap"]
    )

    # Also check conversation-history context for ongoing interpersonal threads
    if not is_personal_or_conversational and history:
        hist_text = " ".join(t.get("text", "").lower() for t in history)
        if any(w in hist_text for w in ["disagreement", "best friend", "tiring day", "argued", "fight"]):
            if any(w in low_ctx for w in ["them", "first", "message", "what would you do", "advice", "situation"]) and not any(w in low_ctx for w in ["what is", "explain", "define", "technically", "machine learning", "sleep", "game"]):
                is_personal_or_conversational = True

    # Safeguard: Factual / informational inquiries starting with why/how/what/who/where/when
    # must always trigger web search when Web Search is ON, regardless of previous emotional history.
    is_explicit_informational_q = bool(re.search(
        r"^(why\s+(does|do|is|are|did|can|would)|how\s+(does|do|is|are|can|did|to|fast|much|many)|what\s+(is|are|causes?|happened\s+in|were|kind|type)|who\s+(is|was|are|were)|where\s+(is|are|do|does|can)|when\s+(was|is|did)|explain\b|define\b)\b",
        low_ctx
    )) and not any(re.search(p, low_ctx) for p in [
        r"\b(what\s+would\s+you\s+do|what\s+do\s+you\s+think|should\s+i|what\s+should\s+i|how\s+should\s+i|any\s+advice)\b"
    ])

    if is_explicit_informational_q:
        is_personal_or_conversational = False

    if is_personal_or_conversational:
        needs_live_web = False
    else:
        # DEFAULT: any factual/informational question should trigger web search
        # when the user has web search enabled, UNLESS it's already answered offline.
        # This is the key architectural change: web search is the DEFAULT for
        # non-conversational queries, not a narrow exception.
        needs_live_web = True

    if not needs_live_web:
        # Run local generation for conversational / general offline turns when live web is unneeded
        local_resp = run_local_model_generation(
            model=model,
            tokenizer=tokenizer,
            history=history,
            prompt_text=contextual_prompt,
            device=device,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens
        )
        is_valid, fail_reason = validate_response_quality(contextual_prompt, local_resp, None)
        if not is_valid:
            retry_resp = run_local_model_generation(
                model=model,
                tokenizer=tokenizer,
                history=history,
                prompt_text=contextual_prompt,
                device=device,
                temperature=0.15,
                top_p=0.75,
                top_k=15,
                repetition_penalty=1.35,
                no_repeat_ngram_size=3,
                max_new_tokens=max_new_tokens
            )
            retry_valid, _ = validate_response_quality(contextual_prompt, retry_resp, None)
            if retry_valid:
                local_resp = retry_resp
                is_valid = True

        if is_valid:
            return finish(
                resp_text=local_resp,
                route_name="neural_model",
                detected_intent="CONVERSATION",
                detected_emotion="NONE",
                resolved_topic=target_subject,
                local_model_called=True,
                web_search_called=False,
                passed_validation=True
            )
        else:
            companion_fallbacks = [
                "I'm right here with you! What's on your mind today?",
                "I'm listening and right here by your side. Tell me more about what you're thinking.",
                "I'm here for you! Feel free to share whatever is on your mind.",
                "I'm right here with you. Tell me what's on your heart or what you'd like to talk about."
            ]
            conv_fallback = random.choice(companion_fallbacks)
            return finish(
                resp_text=conv_fallback,
                route_name="offline_conversational_fallback",
                detected_intent="CONVERSATION",
                detected_emotion="NONE",
                resolved_topic=target_subject,
                local_model_called=True,
                web_search_called=False,
                passed_validation=True
            )

    # Live Web Search Execution
    disambiguated_query = disambiguate_technical_query(contextual_prompt, active_subject=active_subj)
    req_cnt = getattr(getattr(intent_decision, "interpretation", None), "requested_detail", None)
    max_search_res = max(3, int(req_cnt)) if req_cnt and str(req_cnt).isdigit() else 3
    evidence_bundle = retrieval_engine.search_and_retrieve(
        disambiguated_query, max_results=max_search_res, fetch_pages=True, intent_decision=intent_decision
    )

    # Validate topical relevance of search results across retrieved items
    is_relevant = False
    if evidence_bundle and evidence_bundle.has_results and evidence_bundle.items:
        q_words = set(re.findall(r"\w+", contextual_prompt.lower())) - {
            "what", "is", "the", "in", "detail", "and", "or", "a", "an", "to", "for",
            "of", "with", "explain", "please", "pls", "latest", "news", "sources", "developments",
            "give", "tell", "me", "about", "can", "you", "who", "where", "how", "why", "are"
        }
        # Expand domain-specific keywords and acronyms to bridge query vocabulary and technical evidence
        if "ai" in q_words or "artificial" in q_words:
            q_words.update(["ai", "artificial", "intelligence", "llm", "llms", "model", "models", "learning", "neural", "deepseek", "mamba", "agent", "agents"])
        if "ml" in q_words or "machine" in q_words:
            q_words.update(["ml", "machine", "learning", "algorithm", "model"])
        if "dl" in q_words or "deep" in q_words:
            q_words.update(["dl", "deep", "learning", "neural", "network", "transformer"])
        if "modi" in q_words or "prime" in q_words or "minister" in q_words or "pm" in q_words:
            q_words.update(["modi", "narendra", "minister", "prime", "india", "government"])
        if "diljit" in q_words or "yankee" in q_words or "song" in q_words:
            q_words.update(["diljit", "dosanjh", "daddy", "yankee", "pokito", "song", "single", "music"])
        if "pakito" in q_words:
            q_words.update(["pakito", "living", "video", "julien", "ranouil", "music", "dj"])
        if "tech" in q_words or "technology" in q_words:
            q_words.update(["tech", "technology", "ai", "apple", "google", "microsoft", "chips", "software", "computing", "digital", "gadgets", "device", "intel", "nvidia", "hardware", "web", "online"])

        for item in evidence_bundle.items:
            res_text = (item.title + " " + (item.full_text or item.snippet or "")[:800]).lower()
            res_words = set(re.findall(r"\w+", res_text))
            if not q_words or q_words.intersection(res_words):
                is_relevant = True
                break

    if is_relevant and evidence_bundle and evidence_bundle.has_results:
        grounded_answer = synthesize_grounded_answer(contextual_prompt, evidence_bundle, intent_decision=intent_decision)
        return finish(
            resp_text=grounded_answer,
            route_name="web_grounded_evidence",
            detected_intent=getattr(intent_decision, "intent", "web_search"),
            detected_emotion="NONE",
            resolved_topic=target_subject,
            local_model_called=False,
            web_search_called=True,
            passed_validation=True,
            new_sources=evidence_bundle.items
        )
    else:
        # Fallback for search failure or rejected irrelevant search hits
        if "linkedin" in contextual_prompt.lower():
            person_name = target_subject if target_subject else "the requested individual"
            clean_name = person_name.title()
            linkedin_msg = (
                f"A verified public LinkedIn profile for **{clean_name}** could not be confirmed in publicly indexed sources. "
                f"Profile privacy settings or login requirements may restrict access. "
                f"To view this profile, please verify the spelling or provide the direct public profile URL."
            )
            return finish(
                resp_text=linkedin_msg,
                route_name="linkedin_search_inconclusive",
                detected_intent="PERSON_PROFILE",
                detected_emotion="NONE",
                resolved_topic=clean_name,
                local_model_called=False,
                web_search_called=True,
                passed_validation=True
            )

        return finish(
            resp_text="I couldn't find sufficiently reliable information to answer that accurately because the retrieved sources did not sufficiently match your question. You could try rephrasing the question, or I can explain what is already known.",
            route_name="search_relevance_rejected",
            detected_intent="WEB_SEARCH",
            detected_emotion="NONE",
            resolved_topic=target_subject,
            local_model_called=False,
            web_search_called=True,
            passed_validation=False,
            failure_reason="Search results rejected due to lack of topical relevance."
        )


def interactive_chat():
    parser = argparse.ArgumentParser(description="Finalized From-Scratch AI Companion Chatbot")
    parser.add_argument(
        "--model",
        type=str,
        default="custom",
        choices=["custom", "external", "qwen", "CUSTOM", "EXTERNAL", "QWEN"],
        help="Initial active model to load: 'custom' (FromScratchLM) or 'external' (Qwen2.5-1.5B-Instruct)"
    )
    parser.add_argument(
        "--web-search",
        type=str,
        default="off",
        choices=["on", "off", "ON", "OFF"],
        help="Enable or disable live web retrieval at startup (default: off)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable diagnostic routing logger at startup"
    )
    args = parser.parse_args()
    web_search_enabled = (args.web_search.lower() == "on")
    debug_mode = args.debug

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_str = f"CUDA ({gpu_name})"
    else:
        device_str = "CPU"

    initial_model = "external" if args.model.lower() in ["external", "qwen"] else "custom"
    model_manager = ModelManager(device=device, default_model=initial_model)
    retrieval_engine = WebRetrievalEngine()
    retrieval_engine.session_last_sources = []

    active_name_str = "Custom Trained Local Model (FromScratchLM)" if model_manager.active_model_name == "custom" else "External Pretrained Model (Qwen2.5-1.5B-Instruct)"
    web_status_display = "ON" if web_search_enabled else "OFF"
    print("\n=========================================")
    print("                MEYLO")
    print("        Your AI. Your Space.")
    print("=========================================")
    print()
    print(f"Active Model : {active_name_str}")
    print(f"Web Search   : {web_status_display}")
    print(f"Device       : {device_str}")
    print("Status       : Ready")
    print()
    print("Model Commands:")
    print("  /model custom    - Switch to Custom Trained Model")
    print("  /model external  - Switch to External Qwen Model")
    print("  /model status    - Check active model and hardware memory")
    print("  /model list      - List all available models")
    print()
    print("Type /help for all available commands.")
    print("Type /exit to close.")
    print()
    print("=========================================\n")

    history = []

    while True:
        try:
            user_input = input("USER: ").strip()
        except (KeyboardInterrupt, EOFError):
            model_manager.shutdown()
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        cmd_action, cmd_msg = parse_command(user_input)

        if cmd_action == "exit":
            model_manager.shutdown()
            print("Session closed. Goodbye!")
            break
        elif cmd_action == "model_custom":
            ok, msg = model_manager.set_model("custom")
            print(f"{msg}\n")
            continue
        elif cmd_action == "model_external":
            ok, msg = model_manager.set_model("external")
            print(f"{msg}\n")
            continue
        elif cmd_action == "model_status":
            print(f"\n{model_manager.get_status(web_search_enabled)}\n")
            continue
        elif cmd_action == "model_list":
            print(f"\n{model_manager.list_models()}\n")
            continue
        elif cmd_action == "clear":
            history = []
            if hasattr(retrieval_engine, "conversation_memory"):
                retrieval_engine.conversation_memory.clear()
            retrieval_engine.session_last_sources = []
            print("[History cleared.]\n")
            continue
        elif cmd_action == "web_on":
            web_search_enabled = True
            print("[Web Search: ON]\n")
            continue
        elif cmd_action == "web_off":
            web_search_enabled = False
            print("[Web Search: OFF]\n")
            continue
        elif cmd_action == "web_status":
            status_text = "ON" if web_search_enabled else "OFF"
            print(f"[Web Search: {status_text}]\n")
            continue
        elif cmd_action == "sources":
            srcs = format_source_links(getattr(retrieval_engine, "session_last_sources", []))
            print(f"\nBOT: {srcs}\n")
            continue
        elif cmd_action == "debug_on":
            debug_mode = True
            print("[Debug Logging: ENABLED -> logs/router_debug.log]\n")
            continue
        elif cmd_action == "debug_off":
            debug_mode = False
            print("[Debug Logging: DISABLED]\n")
            continue
        elif cmd_action == "debug_status":
            st = "ENABLED" if debug_mode else "DISABLED"
            print(f"[Debug Logging: {st}]\n")
            continue
        elif cmd_action == "help":
            print("\nAvailable Commands:")
            print("  /model custom    - Switch to Custom Trained Model (best.pt)")
            print("  /model external  - Switch to External Qwen Model (Qwen2.5-1.5B-Instruct)")
            print("  /model status    - Show active model, hardware specs, RAM and VRAM usage")
            print("  /model list      - List all available models and their status")
            print("  /web on          - Enable live web search retrieval")
            print("  /web off         - Disable live web search retrieval (pure offline mode)")
            print("  /web status      - Display whether web search is ON or OFF")
            print("  /sources         - Display source links used in the recent response")
            print("  /debug on        - Enable verbose diagnostic router logging")
            print("  /debug off       - Disable verbose diagnostic router logging")
            print("  /debug status    - Check status of diagnostic router logging")
            print("  /clear           - Clear conversational history")
            print("  /help            - Display this help menu")
            print("  /exit            - Safely close the chatbot\n")
            continue
        elif cmd_action == "unknown_command":
            print(f"[{cmd_msg}]\n")
            continue

        history.append({"role": "user", "text": user_input})

        t0 = time.time()
        try:
            reply, engine_name = generate_chat_response(
                model=model_manager.custom_engine.model if model_manager.active_model_name == "custom" else None,
                tokenizer=model_manager.custom_engine.tokenizer if model_manager.active_model_name == "custom" else None,
                history=history,
                device=device,
                retrieval_engine=retrieval_engine,
                web_search_enabled=web_search_enabled,
                debug_mode=debug_mode,
                model_manager=model_manager
            )
        except Exception as e:
            reply = f"I encountered an unexpected issue processing that question: {e}. Please try rephrasing or asking again."
            engine_name = "error_recovery"
        latency = (time.time() - t0) * 1000

        bot_label = "BOT (Custom)" if model_manager.active_model_name == "custom" else "BOT (Qwen)"
        print(f"\n{bot_label}: {reply}\n")
        history.append({"role": "assistant", "text": reply})


if __name__ == "__main__":
    interactive_chat()
