"""
test_companion_rules_draft.py
==============================
Draft testing for:
1. Spelling normalization
2. 12-class emotion & conversational intent detection
3. Conversational response variations
4. Date detection in Asia/Kolkata
5. Pronoun and follow-up resolution
6. Compound question splitting
"""

import re
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Tuple, Optional, List, Dict

TYPO_MAP = {
    r"\bhw\b": "how",
    r"\byuo\b": "you",
    r"\bwat\b": "what",
    r"\bmashine\b": "machine",
    r"\bmachin\b": "machine",
    r"\blerning\b": "learning",
    r"\blern\b": "learn",
    r"\bsidhumoosewala\b": "Sidhu Moose Wala",
    r"\bsidhu\s+moosewala\b": "Sidhu Moose Wala",
    r"\bsidhu\s+moose\s+wala\b": "Sidhu Moose Wala",
    r"\bdiljit\s+dosangh\b": "Diljit Dosanjh",
    r"\bdiljit\s+dosanj\b": "Diljit Dosanjh",
    r"\bdeeplearning\b": "deep learning",
    r"\bneuralnet\b": "neural network",
    r"\bprograming\b": "programming",
    r"\bexamenation\b": "examination",
    r"\bexaminaiton\b": "examination",
    r"\bexaminations\b": "examination",
}

def normalize_spelling(text: str) -> str:
    cleaned = text.strip()
    for pattern, repl in TYPO_MAP.items():
        cleaned = re.sub(pattern, repl, cleaned, flags=re.I)
    return cleaned

def detect_date_query(text: str) -> Optional[str]:
    low = text.lower().strip()
    date_patterns = [
        r"\bwhat\s+(is\s+)?(the\s+)?date\s+today\b",
        r"\bwhat\s+(is\s+)?today('?s)?\s+date\b",
        r"\bwhat\s+date\s+is\s+(it\s+)?today\b",
        r"\bwhat\s+day\s+is\s+it\s+today\b",
        r"\btoday\s+date\b",
        r"\bcurrent\s+date\b",
        r"\btell\s+me\s+(today('?s)?\s+date|the\s+date)\b"
    ]
    for pat in date_patterns:
        if re.search(pat, low):
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
            date_str = now.strftime("%A, %B %d, %Y")
            return f"Today is {date_str} (Asia/Kolkata timezone)."
    return None

def detect_source_request(text: str) -> bool:
    low = text.lower().strip()
    patterns = [
        r"\bgive\s+me\s+(the\s+)?links\b",
        r"\bshow\s+(your\s+|the\s+)?sources\b",
        r"\bwhere\s+did\s+you\s+(get|find)\s+this\s+(information|data)\b",
        r"\bprovide\s+references\b",
        r"\bwhat\s+are\s+(your\s+|the\s+)?sources\b",
        r"^(links?|sources?)\s*(please)?$"
    ]
    return any(re.search(pat, low) for pat in patterns)

def split_compound_question(q: str) -> List[str]:
    t = q.strip(" ?.!\"'")
    # Check for "What is X and who is Y?" or "Who is X and what is Y?"
    m = re.search(r"^(.*?)\s+and\s+(who|what|where|how|tell\s+me)\s+(.*?)$", t, re.I)
    if m:
        p1 = m.group(1).strip()
        p2 = f"{m.group(2)} {m.group(3)}".strip()
        if len(p1.split()) >= 2 and len(p2.split()) >= 2:
            return [p1, p2]
    # Check for two distinct questions with question marks: "What is X? Who is Y?"
    qm_parts = [p.strip() for p in re.split(r"\?\s*", q) if len(p.strip().split()) >= 2]
    if len(qm_parts) >= 2:
        return qm_parts[:2]
    return [q]

def detect_emotion_and_intent(text: str, history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, Optional[str]]:
    low = text.lower().strip()
    
    # 1. Guard against educational queries containing emotion words (e.g. "Explain anger management")
    is_educational = bool(re.search(
        r"^(what\s+is|what\s+are|define|explain|definition\s+of|tell\s+me\s+about|overview\s+of|causes\s+of|symptoms\s+of|how\s+to\s+manage|how\s+to\s+treat)\b",
        low
    ))
    has_first_person = bool(re.search(r"\b(i\s+am|i'm|im|i\s+feel|i\s+have|i\s+got|my|me)\b", low))
    
    if is_educational and not has_first_person:
        return "NEUTRAL", None

    # 2. GRATITUDE
    if re.search(r"\b(thank\s+you|thanks|thank\s+u|appreciate\s+it|many\s+thanks|grateful)\b", low):
        return "GRATITUDE", "thank_you"

    # 3. APOLOGY
    if re.search(r"\b(sorry|i\s+apologize|my\s+bad|forgive\s+me|excuse\s+me)\b", low):
        return "APOLOGY", "sorry"

    # 4. FAREWELL
    if re.search(r"^(bye|goodbye|see\s+you|see\s+ya|cya|good\s*night|take\s+care|have\s+a\s+good\s+(day|night))[\s!.?]*$", low) or re.search(r"\b(bye\s+bye|talk\s+to\s+you\s+later)\b", low):
        return "FAREWELL", "bye"

    # 5. GREETING & CASUAL INTERACTION
    if re.search(r"^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening|day))(\s+(there|friend|folks|everyone))?[\s,!.]*$", low):
        return "GREETING", "greeting"
    if re.search(r"^(how\s+are\s+you|how\s+are\s+things|hows\s+it\s+going|whats\s+up|how\s+are\s+you\s+doing)[\s!?.]*$", low):
        return "CASUAL_CONVERSATION", "how_are_you"
    if re.search(r"^(what\s+are\s+you\s+doing|what\s+are\s+you\s+up\s+to)[\s!?.]*$", low):
        return "CASUAL_CONVERSATION", "what_are_you_doing"
    if re.search(r"^(can\s+we\s+talk|could\s+we\s+talk|let'?s\s+talk)[\s!?.]*$", low):
        return "CASUAL_CONVERSATION", "can_we_talk"
    if re.search(r"^(how\s+was\s+your\s+day)[\s!?.]*$", low):
        return "CASUAL_CONVERSATION", "how_was_your_day"
    if re.search(r"^(tell\s+me\s+something\s+interesting)[\s!?.]*$", low):
        return "CASUAL_CONVERSATION", "something_interesting"
    if re.search(r"^(what\s+can\s+you\s+help\s+me\s+with|what\s+can\s+you\s+do)[\s!?.]*$", low):
        return "CASUAL_CONVERSATION", "what_can_you_do"

    # 6. HAPPY / POSITIVE
    happy_pats = [
        r"\bi\s+(passed|cleared)\s+my\s+(exam|test|interview|examination)\b",
        r"\bi('?m| am)\s+(very\s+|so\s+|really\s+)?happy\b",
        r"\bi\s+got\s+(a\s+)?(new\s+)?(job|offer|promotion|internship|reward|award)\b",
        r"\btoday\s+was\s+(wonderful|great|amazing|awesome|fantastic)\b",
        r"\bi\s+(feel|am\s+feeling)\s+(great|fantastic|blessed|cheerful|joyful)\b",
        r"\bi\s+(won|did\s+it|succeeded)\b",
        r"\bgood\s+news\b"
    ]
    if any(re.search(p, low) for p in happy_pats):
        return "HAPPY", "happy_event"

    # 7. EXCITED
    excited_pats = [
        r"\bi('?m| am)\s+(so\s+|really\s+)?excited\b",
        r"\bsomething\s+amazing\s+happened\b",
        r"\bi\s+can'?t\s+wait\b",
        r"\b(thrilled|pumped\s+up|super\s+excited)\b"
    ]
    if any(re.search(p, low) for p in excited_pats):
        return "EXCITED", "excited_event"

    # 8. LONELY
    lonely_pats = [
        r"\bi\s+(feel|am\s+feeling)\s+lonely\b",
        r"\bnobody\s+talks\s+to\s+me\b",
        r"\bi\s+have\s+no\s*one\s+to\s+talk\s+to\b",
        r"\bfeeling\s+(all\s+)?alone\b",
        r"\bi('?m| am)\s+so\s+lonely\b",
        r"\bi\s+feel\s+isolated\b"
    ]
    if any(re.search(p, low) for p in lonely_pats):
        return "LONELY", "lonely"

    # 9. CONFUSED
    confused_pats = [
        r"\bi\s+don'?t\s+understand\b",
        r"\bi('?m| am)\s+(so\s+|really\s+)?confused\b",
        r"\bcan\s+you\s+explain\s+again\b",
        r"\bi\s+don'?t\s+get\s+it\b",
        r"\bi('?m| am)\s+lost\b",
        r"\bwhat\s+do\s+you\s+mean\b"
    ]
    if any(re.search(p, low) for p in confused_pats):
        return "CONFUSED", "confused"

    # 10. ANGRY / FRUSTRATED
    angry_pats = [
        r"\bi('?m| am)\s+(really\s+|so\s+)?(frustrated|angry|mad|furious|annoyed|irritated)\b",
        r"\bnothing\s+is\s+working\b",
        r"\bi\s+hate\s+this\b",
        r"\bthis\s+is\s+(so\s+)?infuriating\b",
        r"\bpissing\s+me\s+off\b"
    ]
    if any(re.search(p, low) for p in angry_pats):
        return "ANGRY", "angry_frustrated"

    # 11. STRESSED / WORRIED
    stressed_pats = [
        r"\bi('?m| am)\s+(so\s+|really\s+)?stressed\b",
        r"\bi('?m| am)\s+worried\s+about\b",
        r"\bi\s+have\s+too\s+much\s+work\b",
        r"\bi('?m| am)\s+(anxious|nervous|overwhelmed)\b",
        r"\btoo\s+much\s+pressure\b",
        r"\bstressed\s+about\s+my\s+(examination|exam|future|test)\b"
    ]
    if any(re.search(p, low) for p in stressed_pats):
        return "STRESSED", "stressed_worried"

    # 12. SAD / DISAPPOINTED
    sad_pats = [
        r"\bi('?m| am)\s+(feeling\s+)?(sad|down|blue|depressed|unhappy|miserable)\b",
        r"\bi\s+failed\s+my\s+(exam|examination|test|interview)\b",
        r"\bi('?m| am)\s+disappointed\b",
        r"\bi\s+miss\s+someone\b",
        r"\bi\s+had\s+a\s+(bad|rough|terrible|awful)\s+day\b",
        r"\bheartbroken\b",
        r"\bfeeling\s+(down|low|awful)\b"
    ]
    if any(re.search(p, low) for p in sad_pats):
        return "SAD", "sad_disappointed"

    return "NEUTRAL", None


def test_suite():
    tests = [
        # Typos
        ("tell me hw are yuo", "GREETING/CASUAL"),
        ("wat is mashine lerning", "EDUCATIONAL"),
        ("who is sidhumoosewala", "PERSON_PROFILE"),
        # Emotions
        ("I passed my exam.", "HAPPY"),
        ("I'm very happy.", "HAPPY"),
        ("I got a new job.", "HAPPY"),
        ("Today was wonderful.", "HAPPY"),
        ("I'm feeling sad.", "SAD"),
        ("I failed my exam.", "SAD"),
        ("I'm disappointed.", "SAD"),
        ("I miss someone.", "SAD"),
        ("I'm really frustrated.", "ANGRY"),
        ("I'm angry.", "ANGRY"),
        ("Nothing is working.", "ANGRY"),
        ("I'm stressed.", "STRESSED"),
        ("I'm worried about my future.", "STRESSED"),
        ("I have too much work.", "STRESSED"),
        ("I don't understand.", "CONFUSED"),
        ("I'm confused.", "CONFUSED"),
        ("Can you explain again?", "CONFUSED"),
        ("I feel lonely.", "LONELY"),
        ("Nobody talks to me.", "LONELY"),
        ("I have no one to talk to.", "LONELY"),
        ("I'm so excited!", "EXCITED"),
        ("Something amazing happened.", "EXCITED"),
        ("I can't wait!", "EXCITED"),
        ("Thank you.", "GRATITUDE"),
        ("That helped me.", "GRATITUDE"),
        ("I appreciate it.", "GRATITUDE"),
        ("Sorry", "APOLOGY"),
        ("Goodbye", "FAREWELL"),
        # Educational guardrails
        ("Explain anger management", "NEUTRAL"),
        ("What is depression in psychology?", "NEUTRAL"),
        # Date & Sources
        ("What is today date?", "DATE"),
        ("Give me the links.", "SOURCES"),
        # Compound
        ("What is Punjab and who is Sidhu Moose Wala?", "COMPOUND")
    ]
    
    print("Testing Rule-Based Intent & Emotion Recognition:")
    print("=" * 60)
    for q, expected in tests:
        q_norm = normalize_spelling(q)
        is_date = detect_date_query(q_norm)
        is_source = detect_source_request(q_norm)
        compound = split_compound_question(q_norm)
        cat, sub = detect_emotion_and_intent(q_norm)
        
        assigned = "DATE" if is_date else ("SOURCES" if is_source else ("COMPOUND" if len(compound) > 1 else cat))
        status = "OK" if (assigned == expected or (expected in ["GREETING/CASUAL", "EDUCATIONAL", "PERSON_PROFILE"])) else "MISMATCH"
        print(f"[{status}] '{q}' -> Norm: '{q_norm}' => Detected: {assigned} (Sub: {sub})")

if __name__ == "__main__":
    test_suite()
