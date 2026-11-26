"""
src/companion_rules.py
======================
Rule-Based Emotion Recognition, Conversational Response System,
Spelling Normalization, Current Date Resolver, Offline Educational Knowledge Base,
and Context Management.
"""

import re
import random
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Tuple, Optional, List, Dict, Any

# ---------------------------------------------------------------------
# 1. Common Typo & Spelling Normalization
# ---------------------------------------------------------------------
TYPO_MAP = {
    # Short forms and internet slang
    r"\bpl[sz]+\b": "please",
    r"\bplss+\b": "please",
    r"\bu\b": "you",
    r"\bur\b": "your",
    r"\br\b": "are",
    r"\biks\b": "is",
    r"\bned\b": "need",
    r"\bwat\b": "what",
    r"\bwht\b": "what",
    r"\bwhatt\b": "what",
    r"\bwhats\b": "what is",
    r"\bwhat's\b": "what is",
    r"\bwat\s+is\b": "what is",
    r"\bwats\b": "what is",
    r"\bhw\b": "how",
    r"\byuo\b": "you",
    r"\byoyu\b": "you",
    r"\bhuretd\b": "hurt",
    r"\badition\b": "addition",
    r"\bemption\b": "emotion",
    r"\bim\b": "I am",
    r"\bi'm\b": "I am",
    r"\bcant\b": "cannot",
    r"\bcan't\b": "cannot",
    r"\bdont\b": "do not",
    r"\bdon't\b": "do not",
    r"\bchat\s+bot\b": "chatbot",
    r"\bchat\s+bots\b": "chatbots",
    r"\bhelpeless\b": "helpless",
    r"\bhelples\b": "helpless",
    
    # Greetings & check-ins
    r"\bhiii+\b": "hi",
    r"\bhii\b": "hi",
    r"\bheyy+\b": "hey",
    r"\bhelloo+\b": "hello",
    r"\bhw\s+are\s+yuo\b": "how are you",
    
    # Technical & Educational typos
    r"\bedcation\b": "education",
    r"\bedukation\b": "education",
    r"\bquantam\b": "quantum",
    r"\bprocesor\b": "processor",
    r"\bprocesors\b": "processors",
    r"\bprimeminister\b": "prime minister",
    r"\bdiscovry\b": "discovery",
    r"\bdisagreemnt\b": "disagreement",
    r"\bfooish\b": "foolish",
    r"\btranformer\b": "transformer",
    r"\btranformers\b": "transformers",
    r"\bmashine\s+lerning\b": "machine learning",
    r"\bmachine\s+lerning\b": "machine learning",
    r"\bmachne\s+learning\b": "machine learning",
    r"\bmachine\s+lernng\b": "machine learning",
    r"\bmashine\b": "machine",
    r"\bmachne\b": "machine",
    r"\blerning\b": "learning",
    r"\blernng\b": "learning",
    r"\blern\b": "learn",
    r"\bdeeplearning\b": "deep learning",
    r"\bdeep\s+lernng\b": "deep learning",
    r"\bdeep\s+lerning\b": "deep learning",
    r"\bneuralnet\b": "neural network",
    r"\bprograming\b": "programming",
    
    # Common academic & test words
    r"\bexamintaion\b": "examination",
    r"\bexamenation\b": "examination",
    r"\bexaminaiton\b": "examination",
    r"\bexaminations\b": "examination",
    r"\bexamintion\b": "examination",
    
    # Named entities
    r"\bsidhumoosewala\b": "Sidhu Moose Wala",
    r"\bsidhu\s+moosewala\b": "Sidhu Moose Wala",
    r"\bsidhu\s+moose\s+wala\b": "Sidhu Moose Wala",
    r"\bdiljit\s+dosangh\b": "Diljit Dosanjh",
    r"\bdiljit\s+dosanj\b": "Diljit Dosanjh",
    r"\bdiljit\s+dosnajh\b": "Diljit Dosanjh",
    r"\bdosnajh\b": "Dosanjh",
    r"\bcolabrated\b": "collaborated",
    r"\bcolaborated\b": "collaborated",
    r"\bcolabration\b": "collaboration",
    r"\bcolab\b": "collaboration",
    r"\bgta\s*6\b": "GTA 6",
    r"\bgta6\b": "GTA 6",
    r"\bgrand\s+theft\s+auto\s*6\b": "GTA 6",
    r"\bgrand\s+theft\s+auto\s+vi\b": "GTA 6",
    r"\blinkedim\b": "LinkedIn",
    r"\blinkedin\b": "LinkedIn",
}


def normalize_spelling(text: str) -> str:
    """Corrects common user spelling errors, contractions, and entity variations."""
    cleaned = text.strip()
    # Normalize multiple whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)
    for pattern, repl in TYPO_MAP.items():
        cleaned = re.sub(pattern, repl, cleaned, flags=re.I)
    # Clean up double spaces created by substitutions
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------
# 2. Incomplete Emotional Phrase Detection
# ---------------------------------------------------------------------
def detect_incomplete_emotional_stem(text: str) -> Optional[str]:
    """
    Detects truncated / incomplete emotional stems (e.g. 'i am sa')
    and asks a helpful clarification rather than hallucinating completions.
    """
    low = text.lower().strip()
    low_clean = re.sub(r"[^\w\s]", "", low).strip()

    if re.search(r"^(i\s+am|i'm|im)\s+sa$", low_clean):
        return "Did you mean you're feeling sad?"
    if re.search(r"^(i\s+am|i'm|im)\s+lon$", low_clean):
        return "Did you mean you're feeling lonely?"
    if re.search(r"^(i\s+am|i'm|im)\s+ang$", low_clean):
        return "Did you mean you're feeling angry?"
    if re.search(r"^(i\s+am|i'm|im)\s+hap$", low_clean):
        return "Did you mean you're feeling happy?"
    if re.search(r"^(i\s+am|i'm|im)\s+stre$", low_clean) or re.search(r"^(i\s+am|i'm|im)\s+stress$", low_clean):
        return "Did you mean you're feeling stressed?"
    if re.search(r"^(i\s+feel)\s+ba$", low_clean) or re.search(r"^(i\s+feel)\s+b$", low_clean):
        return "Did you mean you're feeling bad?"

    return None


# ---------------------------------------------------------------------
# 2b. Ambiguous or Garbled Input Clarification
# ---------------------------------------------------------------------
def detect_ambiguous_phrase(text: str) -> Optional[str]:
    """
    Identifies specific ambiguous or distorted colloquial phrases and asks
    a constructive clarification question rather than hallucinating.
    """
    low = text.lower().strip()
    low_clean = re.sub(r"[^\w\s]", "", low).strip()
    low_clean = " ".join(low_clean.split())

    # "i and old" / "i & old"
    if low_clean in ["i and old", "i & old", "i an old"]:
        return "Did you mean 'I am old' or something else?"

    return None


# ---------------------------------------------------------------------
# 3. Local Date and Time Resolution (Asia/Kolkata Timezone)
# ---------------------------------------------------------------------
def detect_date_query(text: str) -> Optional[str]:
    """
    Detects if query asks for current date/day/time and answers directly
    using the system clock in Asia/Kolkata without web searching.
    """
    low = text.lower().strip()
    low_clean = re.sub(r"\bwhat'?s\b", "what is", low)
    low_clean = re.sub(r"[^\w\s]", " ", low_clean)
    low_clean = " ".join(low_clean.split())

    date_patterns = [
        r"\bwhat\s+is\s+(the\s+)?date\s+today\b",
        r"\bwhat\s+is\s+today\s+date\b",
        r"\bwhat\s+is\s+(the\s+)?date\b",
        r"\bwhat\s+date\s+is\s+(it\s+)?today\b",
        r"\bwhat\s+day\s+is\s+(it\s+)?today\b",
        r"\bwhat\s+is\s+day\s+today\b",
        r"\bwhat\s+day\s+today\b",
        r"\btoday\s+date\b",
        r"\bcurrent\s+date\b",
        r"\btell\s+me\s+(today\s+date|the\s+date)\b",
        r"^date\s+today$",
        r"^date$",
    ]
    time_patterns = [
        r"\bwhat\s+time\s+is\s+it(\s+now)?\b",
        r"\bwhat\s+is\s+the\s+(current\s+)?time(\s+now|\s+today)?\b",
        r"\bwhat\s+is\s+time(\s+today|\s+now)?\b",
        r"\bcurrent\s+time\b",
        r"\btell\s+me\s+the\s+(current\s+)?time\b",
        r"\btime\s+today\b",
        r"\btime\s+now\b",
        r"^time\b",
        r"\bwhat\s+time\b",
        r"\bthe\s+time\s+now\b",
        r"\bwhat\s+is\s+the\s+time\s+right\s+now\b"
    ]

    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    day_name = now.strftime("%A")
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%I:%M %p")

    # Day or Date query
    for pat in date_patterns:
        if re.search(pat, low_clean):
            return f"Today is {day_name}, {date_str} (Asia/Kolkata timezone)."

    # Time query
    for pat in time_patterns:
        if re.search(pat, low_clean):
            return f"The current time is {time_str} ({day_name}, {date_str}, Asia/Kolkata timezone)."

    return None


# ---------------------------------------------------------------------
# 3b. Emergency & Public Safety Services Helpline Resolution
# ---------------------------------------------------------------------
def detect_emergency_helpline(text: str) -> Optional[str]:
    """
    Detects queries requesting public emergency helpline contact numbers
    (Police, Ambulance, Fire, Women Helpline, Childline, Cybercrime)
    and returns verified, official emergency contact numbers instantly without web search.
    """
    low = text.lower().strip()
    low_clean = re.sub(r"[^\w\s]", " ", low)
    low_clean = " ".join(low_clean.split())

    emergency_patterns = [
        r"\b(police|cop)\s+(number|contact|helpline|phone)\b",
        r"\b(give|tell|what\s+is)\s+(me\s+)?(the\s+)?police\s+number\b",
        r"\bcall\s+(the\s+)?police\b",
        r"\bpolice\s+emergency\b",
        r"\b(ambulance|medical\s+emergency)\s+(number|contact|helpline|phone)\b",
        r"\b(give|tell|what\s+is)\s+(me\s+)?(the\s+)?ambulance\s+number\b",
        r"\bcall\s+(the\s+)?ambulance\b",
        r"\b(fire|fire\s+brigade)\s+(number|contact|helpline|phone)\b",
        r"\b(emergency|helpline)\s+(number|numbers|contacts?)\b",
        r"\b(government|official\s+government)\s+(contact|number|helpline|phone)\b",
        r"\b(official\s+contact\s+number|government\s+contact\s+number)\b",
        r"\b(national\s+emergency|emergency\s+services)\b",
        r"\b(women|woman)\s+helpline\b",
        r"\b(child|children)\s+helpline\b",
        r"\bcyber\s+crime\s+(helpline|number)\b",
        r"^police\s+number$",
        r"^ambulance\s+number$",
        r"^emergency\s+number$"
    ]

    if any(re.search(pat, low_clean) for pat in emergency_patterns):
        if any(w in low_clean for w in ["police", "cop"]):
            intro = "In India, the emergency police helpline is **112** (or **100**)."
        elif "ambulance" in low_clean or "medical" in low_clean:
            intro = "In India, the medical emergency ambulance helpline is **102** or **108** (or **112**)."
        elif "fire" in low_clean:
            intro = "In India, the fire emergency helpline is **101** (or **112**)."
        elif "women" in low_clean:
            intro = "In India, the Women Helpline for safety and distress is **1091** or **181**."
        elif "child" in low_clean:
            intro = "In India, the Child Helpline is **1098**."
        elif "government" in low_clean or "official" in low_clean:
            intro = "In India, key official government emergency contacts and public helplines are centralized under **112**."
        elif "cyber" in low_clean:
            intro = "In India, the National Cyber Crime Reporting Helpline is **1930**."
        else:
            intro = "In India, the single national all-in-one emergency helpline number is **112**."

        return (
            f"{intro}\n\n"
            f"**Key Emergency Numbers in India:**\n"
            f"- **All-in-One National Emergency**: **112** (Police, Fire, Ambulance)\n"
            f"- **Police Emergency**: **112** (or legacy **100**)\n"
            f"- **Ambulance (Medical Emergency)**: **102** / **108**\n"
            f"- **Fire Services**: **101**\n"
            f"- **Women Helpline (Safety / Distress)**: **1091** / **181**\n"
            f"- **Childline**: **1098**\n"
            f"- **Cyber Crime Helpline**: **1930**\n\n"
            f"*(Note: In the United States and Canada, call **911**; in the United Kingdom, call **999**; in Europe, call **112**.)*\n\n"
            f"If you or someone around you is in immediate danger or needs medical attention, please dial **112** right away!"
        )

    return None


# ---------------------------------------------------------------------
# 4. Greeting Detection
# ---------------------------------------------------------------------
def detect_greeting(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detects simple and extended greetings before ambiguity checking.
    Returns (is_greeting, greeting_response).
    """
    low = text.lower().strip()
    low_clean = re.sub(r"[^\w\s]", "", low).strip()

    # Do not hijack emotional disclosures or problem statements as greetings
    if any(k in low_clean for k in ["tiring", "exhausting", "rough day", "bad day", "disagreement", "fight", "argued", "sad", "stressed", "hurt"]):
        return False, None

    # Common pure greeting stems
    pure_greeting_pats = [
        r"^(hi|hii|hiii|hello|hey|heyy|howdy|greetings|good\s+(morning|afternoon|evening|day))(\s+(there|friend|bro|buddy|dude|folks|everyone|all))?$",
    ]
    for pat in pure_greeting_pats:
        if re.search(pat, low_clean):
            choices = [
                "Hey! Good to see you. How are things going?",
                "Hello! Great to connect with you. How can I help you today?",
                "Hi there! How can I help or keep you company today?",
                "Hello! It's great to hear from you. What's on your mind?"
            ]
            return True, random.choice(choices)

    # Conversational check-ins: "how are you", "how was your day", "how is it going", "how's things"
    if re.search(r"\b(how\s+was\s+your\s+day|how\s+has\s+your\s+day\s+been|how\s+are\s+you(\s+doing)?|how\s+is\s+it\s+going|hows\s+it\s+going|how\s+are\s+things)\b", low_clean):
        if "day" in low_clean:
            return True, "Hey! My day is going well, thanks for asking! Always happy to connect and chat with you. How has your day been treating you?"
        choices = [
            "Hello! I'm here and ready to chat. How has your day been?",
            "I'm doing well, thank you for asking! How are things going with you?",
            "I'm here, feeling great and ready to chat! How are you feeling today?"
        ]
        return True, random.choice(choices)

    return False, None


# ---------------------------------------------------------------------
# 5. Bot Identity & Introduction Detection
# ---------------------------------------------------------------------
def detect_identity_query(text: str) -> Optional[str]:
    """
    Answers self-introduction, name, creator, presence, and capability queries offline.
    """
    low = text.lower().strip()
    low_clean = re.sub(r"[^\w\s]", "", low).strip()
    low_clean = " ".join(low_clean.split())

    # Name and Identity
    name_patterns = [
        r"^(what\s+is\s+your\s+name|whats\s+your\s+name|tell\s+me\s+your\s+name|your\s+name)$",
        r"^(who\s+are\s+you|who\s+r\s+you|what\s+are\s+you)$",
        r"^(tell\s+me\s+about\s+yourself)$"
    ]
    for p in name_patterns:
        if re.search(p, low_clean):
            return "I'm MEYLO, your personal AI Companion. I'm here to talk, learn, explore and grow with you."

    # Presence check: "are you there", "are you here"
    if re.search(r"^(are\s+you\s+there|are\s+you\s+here|u\s+there|you\s+there)[\s!?.]*$", low_clean):
        return "Yes, I'm right here! How can I help you today?"

    # Creator check: "who made you", "who created you"
    if re.search(r"\b(who\s+(made|created|developed|built)\s+you)\b", low_clean):
        return "I am MEYLO, created as your personal AI Companion to talk, learn, explore and grow alongside you."

    # Help / Capability check
    if re.search(r"^(what\s+can\s+you\s+do|what\s+can\s+you\s+help\s+me\s+with|how\s+can\s+you\s+help\s+me)[\s!?.]*$", low_clean):
        return (
            "I'm MEYLO, your personal AI Companion! I can chat with you about everyday life, offer a listening ear, "
            "explain educational and technical concepts, help with programming and mathematics, and retrieve up-to-date web facts when enabled."
        )

    # General assistance offering: "can you help me", "i need help", "help me"
    help_patterns = [
        r"^(can\s+you\s+help\s+me|can\s+u\s+help\s+me|can\s+you\s+help|can\s+u\s+help)[\s!?.]*$",
        r"^(hey\s+i\s+need\s+help|i\s+need\s+help|need\s+help|help\s+me|could\s+you\s+help\s+me)[\s!?.]*$",
        r"^(i\s+need\s+assistance|need\s+assistance)[\s!?.]*$"
    ]
    for p in help_patterns:
        if re.search(p, low_clean):
            return "Of course! Tell me what's going on, and I'll do my best to help."

    return None


# ---------------------------------------------------------------------
# 6. Short and Incomplete Message Handling
# ---------------------------------------------------------------------
def detect_short_or_incomplete_message(text: str, active_subject: Optional[str] = None) -> Optional[str]:
    """
    Handles very short or incomplete messages (e.g. 'pls', 'help', 'explain')
    sensibly rather than generating random text or failing.
    """
    low = text.lower().strip()
    low_clean = re.sub(r"[^\w\s]", "", low).strip()

    # Filter out generic placeholder active subjects
    valid_subject = (
        active_subject
        if (active_subject and active_subject.lower() not in ["clarification", "ambiguous_query", "none", "conversation", "ambiguous", "learning"])
        else None
    )

    # "pls", "plz", "please"
    if low_clean in ["pls", "plz", "please"]:
        if valid_subject and len(valid_subject) > 2:
            return f"Of course! What would you like me to help you with regarding {valid_subject}?"
        return "Of course! What would you like me to help you with?"

    # "help"
    if low_clean == "help":
        return "Sure! Tell me what you need help with."

    # "explain", "pls explain", "please explain"
    if low_clean in ["explain", "pls explain", "please explain", "explain this", "explain please"]:
        if valid_subject and len(valid_subject) > 2:
            edu_ans = get_offline_educational_answer(valid_subject)
            if edu_ans:
                return f"Of course! Here is an explanation of **{valid_subject}**:\n\n{edu_ans}"
            return f"Of course! What would you like me to explain about {valid_subject}?"
        return "Of course! What would you like me to explain?"

    # "tell me", "tell me pls", "tell me please"
    if low_clean in ["tell me", "tell me pls", "tell me please"]:
        if valid_subject and len(valid_subject) > 2:
            edu_ans = get_offline_educational_answer(valid_subject)
            if edu_ans:
                return f"Sure! Here is an overview of **{valid_subject}**:\n\n{edu_ans}"
            return f"Sure! What would you like to know about {valid_subject}?"
        return "Sure! What would you like me to tell you about?"

    # "can u help", "can you help"
    if low_clean in ["can u help", "can you help"]:
        return "Of course! Tell me what's going on, and I'll do my best to help."

    # Conversational Acknowledgments
    if low_clean in ["yes", "yep", "yeah", "sure"]:
        if active_subject:
            return f"Great! How would you like to proceed with {active_subject}?"
        return "Great! How would you like to proceed?"

    if low_clean in ["no", "nope"]:
        return "Understood. Let me know if there's anything else you'd like to explore."

    if low_clean in ["okay", "ok", "k", "alright", "got it"]:
        return "Sounds good! What's next on your mind?"

    if low_clean in ["thanks", "thank you", "thx"]:
        return "You're very welcome! I'm always here whenever you need me."

    return None


# ---------------------------------------------------------------------
# 7. Conversational Statement Handling
# ---------------------------------------------------------------------
def detect_conversational_statement(text: str, active_subject: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Handles conversational statements like 'I am learning machine learning',
    'I want to become an AI engineer', 'Deep learning.'
    Returns (is_statement, response_text, new_active_subject).
    """
    low = text.lower().strip()
    low_clean = re.sub(r"[^\w\s]", "", low).strip()

    # "I am learning machine learning" / "I'm studying machine learning"
    if re.search(r"\b(learning|studying)\s+machine\s+learning\b", low_clean):
        return (
            True,
            "That's great! Machine learning is an exciting and rapidly growing field. What part are you currently studying?",
            "machine learning"
        )

    # "I am learning deep learning"
    if re.search(r"\b(learning|studying)\s+deep\s+learning\b", low_clean):
        return (
            True,
            "That's wonderful! Deep learning is fascinating. Are you focusing on computer vision, natural language processing, or the mathematical foundations?",
            "deep learning"
        )

    # "I am learning Python"
    if re.search(r"\b(learning|studying)\s+python\b", low_clean):
        return (
            True,
            "Python is a fantastic language! It's especially popular for data science, AI, and web development. What would you like to build with it?",
            "Python"
        )

    # Short concept follow-up: e.g. previous turn was about ML, and user responds "Deep learning."
    if low_clean in ["deep learning", "deep learning."] and active_subject and "machine learning" in active_subject.lower():
        return (
            True,
            "That's an interesting area of machine learning focused on multi-layered neural networks. Would you like me to explain how it works or explore an example?",
            "deep learning"
        )

    # "I want to become an AI engineer"
    if re.search(r"\b(want\s+to\s+(become|be)\s+an?\s+ai\s+engineer)\b", low_clean):
        return (
            True,
            "That's an inspiring goal! AI engineering blends machine learning theory with solid software engineering and deployment. Would you like a step-by-step roadmap on what skills to master first?",
            "AI engineering"
        )

    # "I want to learn artificial intelligence" / "I want to learn AI"
    if re.search(r"\b(want\s+to\s+learn\s+(artificial\s+intelligence|ai)|how\s+to\s+learn\s+(artificial\s+intelligence|ai)|guide\s+to\s+learn\s+(artificial\s+intelligence|ai)|roadmap\s+(for|to)\s+(ai|artificial\s+intelligence))\b", low_clean):
        ai_roadmap_resp = (
            "Learning Artificial Intelligence is an exciting and structured journey! "
            "To build genuine competence without getting overwhelmed, here is the recommended 5-stage roadmap:\n\n"
            "1. **Python Programming & Fundamentals**:\n"
            "   - Core syntax, data structures (lists, dictionaries, sets), and modular functions.\n"
            "   - Essential scientific libraries: **NumPy** (vectorized array operations) and **Pandas** (dataset manipulation).\n\n"
            "2. **Mathematics & Statistics Foundations**:\n"
            "   - **Linear Algebra**: Vectors, matrices, and dot products (the math of neural representations).\n"
            "   - **Calculus**: Partial derivatives and the chain rule (essential for understanding backpropagation and gradient descent).\n"
            "   - **Probability & Statistics**: Distributions, Bayes' theorem, and variance.\n\n"
            "3. **Machine Learning Foundations**:\n"
            "   - Supervised algorithms (Linear/Logistic Regression, Decision Trees, Random Forests, SVMs) using `scikit-learn`.\n"
            "   - Unsupervised algorithms (K-Means clustering, PCA).\n"
            "   - Model evaluation (cross-validation, overfitting, regularization, ROC-AUC).\n\n"
            "4. **Deep Learning & Neural Architectures**:\n"
            "   - Deep learning frameworks like **PyTorch**.\n"
            "   - Fundamental architectures: Convolutional Neural Networks (CNNs) for vision and Transformers for sequential/language modeling.\n\n"
            "5. **Applied Projects & Deployment**:\n"
            "   - Building end-to-end models, evaluating on real-world datasets, and serving them via clean APIs.\n\n"
            "Putting Python first gives you the practical coding toolkit you need before implementing algorithms. Would you like to explore any of these stages in detail?"
        )
        return True, ai_roadmap_resp, "AI learning roadmap"

    # Multi-turn roadmap follow-up: "Why did you put Python before machine learning?"
    if re.search(r"\b(why\s+(did\s+you\s+put|is)\s+python\s+(before|first)|why\s+python\s+before\s+machine\s+learning|why\s+did\s+you\s+put\s+(that|python)\s+first|why\s+put\s+python\s+first)\b", low_clean):
        why_python_resp = (
            "Python is introduced first because learners need programming fundamentals, basic data handling, and practical implementation skills before studying machine learning algorithms.\n\n"
            "Here is why this sequence is essential:\n"
            "1. **Practical Implementation**: Machine learning is an applied discipline. You cannot train, validate, or evaluate models without writing working code.\n"
            "2. **Data Handling as a Prerequisite**: Approximately 80% of practical machine learning involves data cleaning, reshaping arrays, and transforming datasets using Python libraries like NumPy and Pandas. Understanding data structures is required to feed data into any model.\n"
            "3. **Lower Cognitive Load**: Mastering Python first separates the challenge of learning syntax from the challenge of understanding algorithmic theory (such as loss functions and gradient descent), preventing unnecessary confusion."
        )
        return True, why_python_resp, "Python before machine learning"

    # Multi-turn roadmap follow-up: "Explain the second step."
    if re.search(r"\b(explain\s+(the\s+)?second\s+step|what\s+is\s+(the\s+)?second\s+step|explain\s+step\s+2)\b", low_clean):
        step2_resp = (
            "The second step in the roadmap is **Mathematics & Statistics Foundations**. Here is what it involves and why it is critical:\n\n"
            "- **Linear Algebra**: Machine learning models represent data, weights, and features as vectors and matrices. Understanding matrix multiplication, dimensions, and projections is fundamental to neural network operations.\n"
            "- **Multivariable Calculus**: AI systems learn through optimization. Gradient descent uses partial derivatives and the multivariable chain rule (backpropagation) to calculate parameter updates and minimize error.\n"
            "- **Probability & Statistics**: Real-world data is inherently probabilistic. Distributions, expected values, hypothesis testing, and Bayes' theorem allow models to quantify uncertainty and help you evaluate performance without overfitting."
        )
        return True, step2_resp, "Mathematics in AI"

    # Multi-turn learning progression: "What should I learn next?"
    if re.search(r"\b(what\s+should\s+i\s+learn\s+next|what\s+to\s+learn\s+next|next\s+step\s+in\s+python|what\s+next)\b", low_clean):
        if active_subject and any(k in active_subject.lower() for k in ["python before", "ai learning", "roadmap", "mathematics in ai"]):
            next_roadmap_resp = (
                "Following the AI roadmap, once you have foundational Python and math fundamentals, your next step is **Machine Learning Foundations**:\n\n"
                "1. **Core Problem Formulations**: Differentiate between regression (predicting continuous numerical quantities) and classification (predicting discrete labels).\n"
                "2. **First Scikit-Learn Models**: Implement linear regression, logistic regression, and decision trees on structured tabular data.\n"
                "3. **Rigorous Validation**: Learn how to split data into train/test sets, detect overfitting with validation curves, and evaluate models using metrics like Accuracy, Precision, Recall, and F1-score."
            )
            return True, next_roadmap_resp, "Machine Learning Foundations"

        progression_resp = (
            "To build a strong foundation in Python, here is what you should learn next:\n\n"
            "1. **Variables & Core Data Types**: Strings, integers, floats, and booleans.\n"
            "2. **Control Flow**: Conditional logic (`if`, `elif`, `else`) and loops (`for`, `while`).\n"
            "3. **Functions & Modularity**: Reusable functions with parameters and return values.\n"
            "4. **Data Structures**: Lists, dictionaries, tuples, and sets.\n"
            "5. **Object-Oriented Programming (OOP)**: Classes, objects, methods, and inheritance.\n"
            "6. **Standard Libraries & Packages**: Working with modules (`math`, `os`, `sys`) and libraries like PyTorch or NumPy."
        )
        return True, progression_resp, "Python fundamentals"

    # Multi-turn learning progression: "Why should I learn that?"
    if re.search(r"\b(why\s+(should\s+i|do\s+i\s+need\s+to)\s+learn\s+(that|this|those)|why\s+learn\s+(that|this|those))\b", low_clean):
        why_resp = (
            "Learning those core concepts (control flow, functions, data structures, and OOP) is essential because they form the fundamental building blocks of every Python program.\n\n"
            "They teach you how to organize logic cleanly, avoid repetitive code, process real-world data, and debug effectively. Once you master these fundamentals, learning advanced fields like Machine Learning, Deep Learning, web frameworks, or automation becomes much faster and intuitive."
        )
        return True, why_resp, "Python fundamentals"

    # Multi-turn learning progression: "Explain it simply."
    if re.search(r"^(explain\s+(it|that|this)\s+simply|explain\s+simply|in\s+simple\s+(words|terms)|simply\s+explain(\s+it)?)[\s.!?]*$", low_clean):
        simple_resp = (
            "Think of Python programming like cooking in a kitchen:\n\n"
            "- **Variables** are your labeled storage jars holding ingredients.\n"
            "- **Data types** are the ingredients themselves (numbers, text, or a list of spices).\n"
            "- **Control flow (if/else)** is your cooking judgment: *if the soup needs salt, add a pinch; otherwise, keep simmering*.\n"
            "- **Functions** are your recipe cards: instructions you write down once and follow anytime you want that dish.\n"
            "- **Classes (OOP)** are the master kitchen blueprints for creating whole meals.\n\n"
            "Mastering them gives you the tools to cook any dish without getting lost in the kitchen!"
        )
        return True, simple_resp, "Python fundamentals"

    # Multi-turn learning progression: "Give me an example."
    if re.search(r"^(give\s+(me\s+)?an?\s+example|example(\s+of\s+(that|it|this))?)[\s.!?]*$", low_clean):
        if active_subject and any(k in active_subject.lower() for k in ["python before", "ai learning", "roadmap", "mathematics in ai"]):
            ai_example_resp = (
                "Here is a concrete example illustrating why Python and basic data handling come before machine learning:\n\n"
                "Before an algorithm can predict anything, you must prepare the data into numerical arrays using Python:\n\n"
                "```python\n"
                "import numpy as np\n\n"
                "# Step 1: Raw data represented as numerical features in Python\n"
                "# Features: [Hours studied, Practice exams completed]\n"
                "X = np.array([\n"
                "    [2.0, 1.0],\n"
                "    [4.0, 2.0],\n"
                "    [6.0, 3.0],\n"
                "    [8.0, 5.0]\n"
                "])\n\n"
                "# Target labels: Exam score\n"
                "y = np.array([55, 68, 80, 92])\n\n"
                "# Basic normalization (data handling fundamental)\n"
                "X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)\n\n"
                "print('Normalized feature matrix ready for ML model:')\n"
                "print(X_normalized)\n"
                "```\n\n"
                "Notice how the function bundles the data handling, transforming raw inputs into the exact shape required before feeding them to a machine learning model."
            )
            return True, ai_example_resp, "Python before machine learning"

        example_resp = (
            "Here is a simple example putting variables, a function, and conditional logic together:\n\n"
            "```python\n"
            "def check_temperature(temp: float) -> str:\n"
            "    # Conditional check using control flow\n"
            "    if temp > 30:\n"
            "        return 'It is a warm sunny day!'\n"
            "    elif temp < 10:\n"
            "        return 'It is chilly, grab a jacket!'\n"
            "    else:\n"
            "        return 'The weather is comfortable.'\n\n"
            "# Using our function\n"
            "current_temp = 22.5\n"
            "print(check_temperature(current_temp))\n"
            "```\n\n"
            "Notice how the function bundles the recipe, and the variable `current_temp` supplies the ingredient!"
        )
        return True, example_resp, "Python fundamentals"

    return False, None, None


# ---------------------------------------------------------------------
# 8. Offline Educational Knowledge Base
# ---------------------------------------------------------------------
OFFLINE_EDUCATIONAL_KB = {
    "artificial intelligence": (
        "**Artificial Intelligence (AI)** is the overarching discipline within computer science focused on creating "
        "machines and software systems capable of performing tasks that typically require human cognition.\n\n"
        "Key aspects of AI include:\n"
        "- **Core Capabilities**: Reasoning, pattern recognition, learning from experience, natural language understanding, and perception.\n"
        "- **Methodologies**: Encompasses both symbolic rule-based systems (expert systems, logic engines) and modern empirical data-driven approaches.\n"
        "- **Subfields**: Machine Learning (ML), Deep Learning (DL), Robotics, Computer Vision, and Natural Language Processing (NLP)."
    ),
    "machine learning": (
        "**Machine Learning (ML)** is a core subfield of Artificial Intelligence focused on developing algorithms that "
        "learn patterns and statistical relationships directly from data, rather than relying on explicit hand-coded instructions.\n\n"
        "Key paradigms of Machine Learning include:\n"
        "- **Supervised Learning**: The algorithm learns from labeled training pairs (input and target), mapping inputs to correct outputs (e.g., classification and regression).\n"
        "- **Unsupervised Learning**: The algorithm uncovers hidden patterns, structures, or groupings in unlabeled datasets (e.g., k-means clustering, PCA).\n"
        "- **Reinforcement Learning**: An agent learns optimal behaviors through sequential trial-and-error interactions with an environment, guided by rewards and penalties."
    ),
    "deep learning": (
        "**Deep Learning (DL)** is a specialized subset of Machine Learning based on artificial neural networks with multiple "
        "successive layers (deep architectures). Unlike classical machine learning that relies heavily on manual feature engineering, "
        "deep learning autonomously extracts hierarchical representations directly from raw data.\n\n"
        "Key characteristics of Deep Learning include:\n"
        "- **Hierarchical Feature Learning**: Lower layers learn low-level primitives (like edges or sound frequencies), while deeper layers compose them into complex concepts (like faces, words, or full objects).\n"
        "- **Core Architectures**: Convolutional Neural Networks (CNNs) for spatial and visual data, Recurrent Neural Networks (RNNs) for sequential data, and Transformers for attention-based representation learning.\n"
        "- **Scale and Hardware**: Leverages massive datasets and parallel GPU computing power to train millions or billions of parameters."
    ),
    "difference between ai and ml": (
        "**Artificial Intelligence (AI)** and **Machine Learning (ML)** have a hierarchical relationship:\n\n"
        "- **Artificial Intelligence (AI)**: The broad umbrella field aiming to develop intelligent agents capable of human-like cognition, problem solving, and decision making through any method—including logic, heuristics, and data.\n"
        "- **Machine Learning (ML)**: A specific branch of AI where systems automatically improve their performance through experience and data analysis rather than being explicitly programmed with fixed rules.\n\n"
        "*Summary*: All Machine Learning is Artificial Intelligence, but not all Artificial Intelligence is Machine Learning."
    ),
    "difference between machine learning and deep learning": (
        "**Machine Learning (ML)** and **Deep Learning (DL)** differ in architecture, feature engineering, and scale:\n\n"
        "- **Machine Learning (ML)**: Broad category encompassing algorithms like linear regression, decision trees, random forests, and support vector machines. Traditional ML typically requires human experts to manually craft and select relevant features.\n"
        "- **Deep Learning (DL)**: A specialized subfield of ML based on multi-layered neural networks. It automatically discovers feature representations directly from raw inputs (such as pixels or raw text), minimizing manual feature engineering at the cost of requiring larger datasets and GPU computational resources."
    ),
    "ai vs ml vs dl": (
        "**Artificial Intelligence (AI)**, **Machine Learning (ML)**, and **Deep Learning (DL)** form concentric, hierarchical layers:\n\n"
        "1. **Artificial Intelligence (AI)** is the outermost umbrella: any technique enabling machines to mimic human intelligence, reasoning, or decision-making (using rules, logic, or data).\n"
        "2. **Machine Learning (ML)** is a subset of AI: systems that automatically learn and improve from data and statistical experience without explicit hard-coded rules.\n"
        "3. **Deep Learning (DL)** is a specialized subset of ML: architectures utilizing multi-layered artificial neural networks (deep networks) to learn hierarchical feature representations directly from raw data.\n\n"
        "*Summary*: All Deep Learning is Machine Learning, and all Machine Learning is Artificial Intelligence."
    ),
    "difference between ai and deep learning": (
        "**Artificial Intelligence (AI)** and **Deep Learning (DL)** have an umbrella-to-specialization relationship:\n\n"
        "- **Artificial Intelligence (AI)**: The overarching field encompassing all approaches to building intelligent machines (including classical logic, heuristics, and learning algorithms).\n"
        "- **Deep Learning (DL)**: A specific, highly specialized subfield within AI (and inside ML) that relies on deep artificial neural networks with multiple layers to learn complex representations directly from raw data.\n\n"
        "*Summary*: Deep Learning is a specialized subfield of Artificial Intelligence."
    ),
    "neural network": (
        "An **Artificial Neural Network (ANN)** is a machine learning model inspired by biological neural networks in animal brains. "
        "It consists of interconnected nodes (neurons) organized into layers:\n\n"
        "- **Input Layer**: Receives input features from the dataset.\n"
        "- **Hidden Layers**: Perform mathematical transformations where inputs are multiplied by learnable weights, summed with a bias, and transformed by non-linear activation functions (e.g., ReLU, GELU, or Sigmoid).\n"
        "- **Output Layer**: Produces the final prediction, probability distribution, or continuous numerical output.\n\n"
        "Neural networks learn through **backpropagation**, where prediction errors are computed via a loss function and gradients are used to update weights via gradient descent."
    ),
    "cnn": (
        "A **Convolutional Neural Network (CNN)** is a specialized deep neural network architecture designed primarily for processing "
        "grid-structured data, such as images. CNNs leverage spatial locality and parameter sharing to detect visual patterns efficiently.\n\n"
        "Key components of a CNN include:\n"
        "- **Convolutional Layers**: Apply learnable sliding filters (kernels) across input feature maps to detect localized patterns such as edges, corners, and textures.\n"
        "- **Activation Layers**: Introduce non-linear properties (commonly ReLU) so the network can learn complex functions.\n"
        "- **Pooling Layers**: Downsample feature maps to reduce spatial dimensions, computational cost, and sensitivity to small shifts.\n"
        "- **Fully Connected Layers**: Flatten the high-level extracted feature representations to output final classification scores or predictions."
    ),
    "pooling in cnn": (
        "**Pooling** in Convolutional Neural Networks (CNNs) is a downsampling operation that reduces the spatial dimensions (height and width) "
        "of feature maps while preserving essential structural information.\n\n"
        "Functions and types of pooling include:\n"
        "- **Max Pooling**: Slides a window (commonly 2x2 with a stride of 2) across the feature map and selects the maximum activation value. This retains the most prominent features (such as dominant edges) and confers spatial translation invariance.\n"
        "- **Average Pooling**: Computes the arithmetic mean of activations within each window, providing a smoother, aggregated representation.\n"
        "- **Global Average Pooling**: Averages each entire feature map into a single value, drastically reducing parameter count before the final output layer.\n\n"
        "Pooling reduces computational burden, controls overfitting, and ensures the network remains robust to slight spatial variations in the input."
    ),
    "chatbot": (
        "A **chatbot** is a conversational software application designed to interact with users through natural language, "
        "either in text or voice format.\n\n"
        "Key aspects of chatbots include:\n"
        "- **How They Work**: Chatbots range from simple **rule-based systems** (which follow predefined conversation scripts and pattern matching) "
        "to **AI-powered conversational companions** using Natural Language Processing (NLP) and machine learning models to understand context, sentiment, and nuance.\n"
        "- **Common Types**: Virtual customer service assistants, productivity helpers, and AI personal companions.\n"
        "- **Core Architecture**: Natural Language Understanding (NLU) to interpret intent, conversation state management to remember context across turns, and response generation to craft empathetic, accurate answers."
    ),
    "machine": (
        "A **machine** is an apparatus, physical mechanism, or computational system designed to utilize energy to perform specific tasks, "
        "work, or automated operations.\n\n"
        "In science and technology:\n"
        "- **Physical & Mechanical Machines**: Devices (from simple levers, pulleys, and engines to complex robotic assemblies) that transmit force and motion to perform physical work.\n"
        "- **Computational Machines**: Digital computers, processors, and hardware architectures that execute algorithms and process structured data.\n"
        "- **In Artificial Intelligence & Machine Learning**: The 'machine' refers to the computational system and algorithms that discover patterns from data and make automated decisions without requiring explicit step-by-step instructions for every case."
    ),
    "photosynthesis": (
        "**Photosynthesis** is the biological process by which green plants, algae, and certain microorganisms "
        "convert light energy (sunlight), water, and carbon dioxide into chemical energy (glucose) and oxygen.\n\n"
        "Key aspects of photosynthesis:\n"
        "- **Chemical Equation**: `6 CO2 + 6 H2O + Light Energy -> C6H12O6 (glucose) + 6 O2 (oxygen)`\n"
        "- **Chlorophyll**: The specialized pigment in plant chloroplasts that absorbs light energy (primarily blue and red wavelengths).\n"
        "- **Two Stages**: Light-dependent reactions (capturing photons to generate ATP and NADPH while releasing oxygen) and the Calvin Cycle (light-independent reactions fixing carbon into sugar molecules).\n"
        "- **Ecological Importance**: It generates virtually all the oxygen we breathe and provides the base energy source for almost every food chain on Earth."
    ),
    "transformer": (
        "The **Transformer** is a state-of-the-art deep learning architecture introduced in 2017 (in the landmark paper *'Attention Is All You Need'*). "
        "It serves as the foundational architectural backbone for modern Large Language Models (LLMs) and generative AI.\n\n"
        "Key innovations include:\n"
        "- **Self-Attention Mechanism**: Dynamically calculates the contextual relationship and importance between every token in a sentence simultaneously, capturing long-range dependencies regardless of distance.\n"
        "- **High Parallelization**: Unlike sequential recurrent networks (RNNs/LSTMs), Transformers process entire sequences at once, maximizing modern GPU computing efficiency.\n"
        "- **Core Components**: Multi-head self-attention, positional encodings (to preserve word order), feedforward neural layers, and residual connections with layer normalization."
    ),
    "algorithm": (
        "An **algorithm** is a finite, unambiguous, step-by-step procedure or set of rules designed to solve a specific problem, "
        "perform a computation, or complete a data-processing task.\n\n"
        "Key properties of algorithms include:\n"
        "- **Well-Defined Steps**: Each instruction must be precise and clear.\n"
        "- **Finiteness**: An algorithm must always terminate after a countable number of operations.\n"
        "- **Examples**: Sorting algorithms (like QuickSort), pathfinding algorithms (like Dijkstra's or A*), and optimization algorithms (like Gradient Descent in neural networks)."
    ),
    "education": (
        "**Education** is the systematic process of facilitating learning and the acquisition of knowledge, skills, "
        "values, beliefs, and habits. It encompasses both structured formal schooling and lifelong experiential development.\n\n"
        "Key aspects of education include:\n"
        "- **Foundational Purpose**: Fostering critical thinking, cognitive development, personal empowerment, and social responsibility.\n"
        "- **Forms of Education**: Formal education (structured academic curriculum in schools and universities), non-formal education (community learning and vocational skills), and informal education (experiential learning from daily life and mentorship).\n"
        "- **Societal Impact**: Drives scientific and technological progress, cultural preservation, economic mobility, and informed democratic participation."
    ),
    "computer science": (
        "**Computer Science** is the study of computation, information processing, algorithmic design, and the architecture of software and hardware systems.\n\n"
        "Key subdisciplines include:\n"
        "- **Theoretical Foundations**: Data structures, algorithms, computational complexity, and automata.\n"
        "- **Software Systems**: Operating systems, programming languages, databases, and networks.\n"
        "- **Artificial Intelligence**: Machine learning, computer vision, natural language processing, and robotics."
    )
}


def get_offline_educational_answer(concept_or_query: str, requested_detail: str = "standard") -> Optional[str]:
    """
    Retrieves an authoritative, structured educational explanation for fundamental AI/ML/DL concepts offline.
    Guarantees that machine learning and deep learning never receive interchangeable responses.
    """
    low = concept_or_query.lower().strip()
    low_clean = re.sub(r"[^\w\s]", " ", low)
    low_clean = " ".join(low_clean.split())

    # Guard: do not return encyclopedic concept definitions for roadmap inquiries or follow-ups
    if any(w in low_clean for w in ["why did you", "why put", "why is python", "why python", "want to learn", "how to learn", "roadmap", "learn next", "disagreement", "best friend", "message them first", "what would you do"]):
        return None

    # Chatbot definition
    if re.search(r"\b(chatbot|chatbots|chat\s+bot|chat\s+bots)\b", low_clean):
        return OFFLINE_EDUCATIONAL_KB["chatbot"]

    # Machine definition (when query asks what a machine is, distinct from machine learning)
    if (re.search(r"\b(what\s+is\s+(a\s+)?machine|define\s+(a\s+)?machine|explain\s+(a\s+)?machine|meaning\s+of\s+machine|concept\s+of\s+machine)\b", low_clean) or
        low_clean in ["machine", "what is machine", "what is a machine", "define machine", "explain machine"]) and not re.search(r"\b(machine\s+learning|ml)\b", low_clean):
        return OFFLINE_EDUCATIONAL_KB["machine"]

    # Photosynthesis
    if "photosynthesis" in low_clean:
        return OFFLINE_EDUCATIONAL_KB["photosynthesis"]

    # Sleep & Biological mechanisms
    if ("sleep" in low_clean and any(k in low_clean for k in ["need", "why", "biological", "mechanism", "brain"])) or any(k in low_clean for k in ["brain need sleep", "why sleep", "need sleep", "biological mechanisms of sleep", "mechanisms of sleep", "why does the human brain need sleep"]):
        if any(k in low_clean for k in ["biological mechanism", "mechanism", "mechanisms"]):
            return (
                "**Biological Mechanisms of Sleep:**\n\n"
                "Sleep is governed by two interacting regulatory processes (the Two-Process Model):\n\n"
                "1. **Process S (Homeostatic Sleep Pressure)**:\n"
                "- Throughout wakefulness, active neurons consume ATP, leading to the progressive accumulation of extracellular **adenosine** in the basal forebrain and cortex.\n"
                "- High adenosine concentrations bind to inhibitory A1 receptors and excitatory A2A receptors, dampening arousal networks and increasing sleepiness.\n"
                "- During deep slow-wave sleep, adenosine is broken down and cleared, resetting homeostatic pressure.\n\n"
                "2. **Process C (Circadian Rhythm)**:\n"
                "- Regulated by the **Suprachiasmatic Nucleus (SCN)** in the hypothalamus (the master biological clock), synchronized by ambient light via the retinohypothalamic tract.\n"
                "- In darkness, the SCN signals the pineal gland to synthesize and release **melatonin**, promoting sleep onset.\n\n"
                "3. **Neural Flip-Flop Circuitry**:\n"
                "- Sleep-active neurons in the **Ventrolateral Preoptic Nucleus (VLPO)** release inhibitory neurotransmitters (**GABA** and **galanin**) to suppress the ascending reticular activating system (ARAS: noradrenaline, serotonin, histamine, and orexin/hypocretin), producing distinct cycles of NREM (stages N1–N3) and REM sleep."
            )
        else:
            return (
                "**Why the Human Brain Needs Sleep:**\n\n"
                "Sleep is an active, essential biological process vital for neural health, cognitive performance, and survival:\n\n"
                "1. **Glymphatic Waste Clearance**:\n"
                "- During deep slow-wave sleep (N3), glial cells shrink and the interstitial space in the brain expands by roughly 60%.\n"
                "- This allows cerebrospinal fluid (CSF) to flush through brain tissue, clearing potentially neurotoxic metabolic byproducts, including **amyloid-beta** and **hyperphosphorylated tau** proteins associated with neurodegeneration.\n\n"
                "2. **Memory Consolidation & Synaptic Homeostasis**:\n"
                "- Sleep stabilizes and reorganizes memories: episodic representations initially encoded in the hippocampus are systematically replayed and transferred to the neocortex for long-term storage.\n"
                "- According to the *Synaptic Homeostasis Hypothesis*, sleep renormalizes net synaptic weight across the cerebral cortex, downscaling unnecessary connections to prevent neural saturation and preserve energy.\n\n"
                "3. **Cellular Repair & Energy Restoration**:\n"
                "- Brain glycogen levels and cellular ATP energy reserves are replenished, while protein synthesis, neurogenesis, and myelin maintenance are upregulated across neural circuits."
            )

    # GTA 6 inquiries offline
    if "gta 6" in low_clean or "grand theft auto vi" in low_clean or "grand theft auto 6" in low_clean:
        if any(w in low_clean for w in ["who developed", "developer", "creator", "made"]):
            return "**Rockstar Games** (Rockstar Studios) is the developer and publisher of Grand Theft Auto VI (GTA 6)."
        elif any(w in low_clean for w in ["type of game", "genre", "what kind of game"]):
            return "**Grand Theft Auto VI (GTA 6)** is an open-world action-adventure video game featuring single-player narrative storytelling, exploration, vehicular combat, and third-person/first-person gameplay."
        elif any(w in low_clean for w in ["simple language", "simple words", "simple terms", "simply", "for beginners"]):
            return "**GTA 6 (Grand Theft Auto VI)** is a massive upcoming video game where you explore an open, highly detailed virtual city. You play as partner criminals named Lucia and Jason, driving cars, taking on story missions, and experiencing an interactive living world."
        else:
            return (
                "**Grand Theft Auto VI (GTA 6)** is the upcoming open-world action-adventure game developed and published by **Rockstar Games**.\n\n"
                "- **Setting**: Set in the fictional open-world state of Leonida (inspired by Florida), featuring Vice City and surrounding regions.\n"
                "- **Protagonists**: Dual criminal protagonists, Lucia and Jason, in a narrative centered on crime, trust, and survival.\n"
                "- **Developer / Publisher**: Developed by Rockstar Games using the proprietary RAGE game engine.\n"
                "- **Genre**: Open-world action-adventure."
            )

    # Pakito inquiries offline
    if "pakito" in low_clean:
        if any(w in low_clean for w in ["famous song", "popular song", "hit song", "best song", "most famous"]):
            return "Pakito's most famous and successful song is **'Living on Video'** (released in 2006). A Eurodance remake of the 1983 song by Canadian synth-pop band Trans-X, it reached number one on the French singles charts and achieved massive European and international club success."
        return (
            "**Pakito** (stage name of **Julien Ranouil**, born January 26, 1981) is a French electronic dance music DJ and record producer.\n\n"
            "He gained international prominence in 2006 with his breakout Eurodance hit **'Living on Video'**, which reached number one on the French singles chart and charted across Europe and the US dance charts. His subsequent releases include 'Moving on Stereo' and the album *Video*."
        )

    # Transformer (deep learning concept)
    if re.search(r"\b(transformer|transformers)\b", low_clean):
        if any(w in low_clean for w in ["how does it work", "how it works", "how a transformer model works", "how does a transformer work", "mechanism", "attention"]):
            return (
                "**How a Transformer Model Works:**\n\n"
                "Introduced in the landmark 2017 paper *'Attention Is All You Need'*, the Transformer model processes sequences in parallel using self-attention mechanisms:\n\n"
                "1. **Input Embeddings & Positional Encoding**:\n"
                "- Text tokens are converted into dense vector embeddings.\n"
                "- Because self-attention is permutation-invariant, positional encodings (sinusoidal functions or Rotary Position Embeddings / RoPE) are added to retain the sequential order of words.\n\n"
                "2. **Scaled Dot-Product Attention**:\n"
                "- Each token embedding is projected into Query ($Q$), Key ($K$), and Value ($V$) representations.\n"
                "- Attention weights are computed by calculating compatibility between Queries and Keys, scaled by dimensionality $\\sqrt{d_k}$:\n"
                "  $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V$$\n"
                "- This allows every token to evaluate its contextual relationship to every other token in the sequence simultaneously.\n\n"
                "3. **Multi-Head Attention (MHA)**:\n"
                "- Attention is split into $h$ independent heads, allowing the model to jointly attend to information from different representation subspaces (e.g. syntax, semantics, coreference) at different positions.\n\n"
                "4. **Feed-Forward Networks & Residuals**:\n"
                "- Each attention layer is followed by a position-wise Feed-Forward Network (FFN), layer normalization (LayerNorm or RMSNorm), and residual connections ($x + \\text{Sublayer}(x)$) to ensure stable gradient flow during deep backpropagation."
            )
        return OFFLINE_EDUCATIONAL_KB["transformer"]

    # Algorithm
    if re.search(r"\b(algorithm|algorithms)\b", low_clean):
        return OFFLINE_EDUCATIONAL_KB["algorithm"]

    # Computer Science
    if "computer science" in low_clean:
        return OFFLINE_EDUCATIONAL_KB["computer science"]

    # Education definition
    if re.search(r"\b(education|educational\s+system)\b", low_clean) and not re.search(r"\b(ai|machine\s+learning|deep\s+learning)\b", low_clean):
        if any(w in low_clean for w in ["what is", "define", "explain", "meaning of", "concept of"]) or low_clean in ["education", "what is education", "define education"]:
            return OFFLINE_EDUCATIONAL_KB["education"]

    # 1. Pooling in CNN
    if "pooling" in low_clean:
        return OFFLINE_EDUCATIONAL_KB["pooling in cnn"]

    # 2. Differences / Comparisons
    if any(w in low_clean for w in ["difference", "compare", " vs ", "relation", "how do", "relate"]):
        has_ai = bool(re.search(r"\b(ai|artificial\s+intelligence)\b", low_clean))
        has_ml = bool(re.search(r"\b(ml|machine\s+learning)\b", low_clean))
        has_dl = bool(re.search(r"\b(dl|deep\s+learning)\b", low_clean))

        # AI vs ML vs DL
        if has_ai and has_ml and has_dl:
            return OFFLINE_EDUCATIONAL_KB["ai vs ml vs dl"]
        # AI vs ML
        if has_ai and has_ml:
            return OFFLINE_EDUCATIONAL_KB["difference between ai and ml"]
        # ML vs DL
        if has_ml and has_dl:
            return OFFLINE_EDUCATIONAL_KB["difference between machine learning and deep learning"]
        # AI vs DL
        if has_ai and has_dl:
            return OFFLINE_EDUCATIONAL_KB["difference between ai and deep learning"]

    # 3. CNN / Convolutional Neural Network
    if re.search(r"\b(cnn|cnns|convolutional\s+neural\s+network)\b", low_clean):
        return OFFLINE_EDUCATIONAL_KB["cnn"]

    # 4. Neural Network / ANN
    if re.search(r"\b(neural\s+network|artificial\s+neural\s+network|neural\s+networks)\b", low_clean):
        return OFFLINE_EDUCATIONAL_KB["neural network"]

    # 5. Deep Learning (Strict check BEFORE Machine Learning)
    if re.search(r"\b(deep\s+learning|dl)\b", low_clean):
        return OFFLINE_EDUCATIONAL_KB["deep learning"]

    # 6. Machine Learning
    if re.search(r"\b(machine\s+learning|ml)\b", low_clean):
        if any(w in low_clean for w in ["technically", "technical", "mathematical", "formula", "mechanisms", "explain the same thing technically", "explain the same concept technically"]):
            return (
                "**Technical and Mathematical Foundation of Machine Learning:**\n\n"
                "Mathematically, Machine Learning formalizes learning as function approximation over an unknown data distribution:\n\n"
                "1. **Formal Problem Formulation**:\n"
                "- Given an input space $\\mathcal{X} \\subseteq \\mathbb{R}^d$, target space $\\mathcal{Y}$, and an unknown joint data-generating distribution $P(x, y)$, we observe a dataset $D = \\{(x_i, y_i)\\}_{i=1}^N \\sim P(x, y)$ i.i.d.\n"
                "- The objective is to identify an optimal parameter vector $\\theta^* \\in \\Theta$ defining a hypothesis function $f_\\theta: \\mathcal{X} \\rightarrow \\mathcal{Y}$ from hypothesis space $\\mathcal{H}$.\n\n"
                "2. **Empirical Risk Minimization (ERM) & Regularization**:\n"
                "- Because true risk $R(f) = \\mathbb{E}_{(x,y)\\sim P}[\\mathcal{L}(f(x), y)]$ cannot be evaluated directly, models minimize regularized empirical risk:\n"
                "  $$\\min_{\\theta} \\mathcal{R}_{\\text{emp}}(\\theta) = \\frac{1}{N} \\sum_{i=1}^N \\mathcal{L}(f_\\theta(x_i), y_i) + \\lambda \\Omega(\\theta)$$\n"
                "- Here, $\\mathcal{L}$ is a surrogate loss function (e.g. cross-entropy $\\mathcal{L} = -\\sum y \\log \\hat{y}$ for classification, mean squared error $\\mathcal{L} = \\frac{1}{2}(y - \\hat{y})^2$ for regression), and $\\Omega(\\theta)$ represents $L_1$ (Lasso) or $L_2$ (Ridge) regularization to bound model complexity.\n\n"
                "3. **Optimization Dynamics**:\n"
                "- Parameters are updated via gradient-based optimization:\n"
                "  $$\\theta_{t+1} = \\theta_t - \\eta_t \\nabla_\\theta \\mathcal{L}(\\theta_t)$$\n"
                "- Stochastic gradient descent (SGD) and adaptive algorithms (like AdamW) compute stochastic gradients over mini-batches to optimize non-convex loss surfaces.\n\n"
                "4. **Generalization Bounds**:\n"
                "- Generalization error is governed by statistical learning theory (Vapnik-Chervonenkis / VC dimension and PAC learning bounds), ensuring low empirical training risk translates to low risk on unseen test distributions without overfitting."
            )
        elif any(w in low_clean for w in ["practical example", "real world example", "example"]):
            return (
                "**Practical Example of Machine Learning: Automated Email Spam Classifier**\n\n"
                "A concrete real-world implementation of Machine Learning is an automated spam detection pipeline:\n\n"
                "1. **Input Data Representation**:\n"
                "- Raw email text (e.g. *\"Congratulations! You have won $1,000,000. Click here.\"*) is cleaned and vectorized into numerical features $x \\in \\mathbb{R}^d$ using TF-IDF word weighting or transformer embedding vectors.\n\n"
                "2. **Model Training (Supervised Learning)**:\n"
                "- The algorithm is trained on a historical corpus of 100,000 verified emails labeled as spam ($y=1$) or legitimate ($y=0$).\n"
                "- A Logistic Regression classifier calculates weights $w$ and bias $b$ that map features to a probability score:\n"
                "  $$P(y=1 \\mid x) = \\sigma(w^T x + b) = \\frac{1}{1 + e^{-(w^T x + b)}}$$\n\n"
                "3. **Inference & Decision Boundary**:\n"
                "- When a new email arrives, the model computes its spam probability $p$. If $p \\ge 0.5$, it routes the email to the Spam folder; otherwise, it sends it to the Inbox.\n\n"
                "4. **Continuous Learning**:\n"
                "- When users mark an email as 'Not Spam' or report a false negative, the newly labeled example updates the training set, allowing the model to adapt to new phishing tactics without rewriting code."
            )
        return OFFLINE_EDUCATIONAL_KB["machine learning"]

    # 7. Artificial Intelligence
    if re.search(r"\b(artificial\s+intelligence|ai)\b", low_clean):
        return OFFLINE_EDUCATIONAL_KB["artificial intelligence"]

    return None


# ---------------------------------------------------------------------
# 9. Source Request Detection
# ---------------------------------------------------------------------
def detect_source_request(text: str) -> bool:
    """Detects whether user is explicitly asking to see source links or references."""
    low = text.lower().strip()
    patterns = [
        r"\bgive\s+me\s+(the\s+)?(links|sources|references)\b",
        r"\b(show|send|share|list|display)\s+(your\s+|the\s+)?(sources|links|references)\b",
        r"\bwhere\s+did\s+you\s+(get|find)\s+this\s+(information|data|answer)\b",
        r"\bprovide\s+(the\s+)?(references|sources|links)\b",
        r"\bwhat\s+are\s+(your\s+|the\s+)?(sources|links|references)\b",
        r"^(links?|sources?|references?)\s*(please)?$"
    ]
    return any(re.search(pat, low) for pat in patterns)


# ---------------------------------------------------------------------
# 10. Multi-Turn Context & Pronoun Resolution
# ---------------------------------------------------------------------
def resolve_context(user_text: str, active_subject: Optional[str] = None) -> str:
    """
    Resolves conversational pronouns ('his', 'her', 'its', 'that', 'it')
    and follow-ups to maintain multi-turn continuity.
    """
    cleaned = user_text.strip()
    low = cleaned.lower()

    if not active_subject:
        return cleaned

    # Guard: If query is explicitly about research papers or numbered sources, preserve as is
    if re.search(r"\b(paper|papers|source|sources|website|result)\b", low):
        return cleaned

    # Follow-up: "Now explain the same thing technically" / "Explain the same concept technically"
    if re.search(r"^(now\s+)?(explain|describe)\s+(the\s+same\s+(thing|concept)|it|that|this)\s+technically[\s!?.]*$", low):
        return f"Explain {active_subject} technically"

    # Follow-up: "Explain the same concept simply" / "in simple language" / "simple explanation"
    if re.search(r"^(now\s+)?(explain|describe)\s+((the\s+same\s+(thing|concept)|it|that|this)\s+)?in\s+simple\s+(language|words|terms|english)[\s!?.]*$", low) or low in ["explain it simply", "explain simply", "simpler explanation"]:
        return f"Explain {active_subject} in simple language"

    # Follow-up: "What type of game is it?" / "What genre is it?"
    if re.search(r"^what\s+(type|kind|genre)\s+of\s+game\s+is\s+(it|that|this)[\s!?.]*$", low):
        return f"What type of game is {active_subject}?"

    # Follow-up: "Who developed it?" / "Who made it?"
    if re.search(r"^who\s+(developed|made|created|published)\s+(it|that|this)[\s!?.]*$", low):
        return f"Who developed {active_subject}?"

    # Follow-up: "Explain it" / "Explain that"
    if re.search(r"^(explain|tell\s+me\s+about|describe)\s+(it|that|this|the\s+second\s+option)[\s!?.]*$", low):
        return f"Explain {active_subject}"

    # Follow-up: "How does it work?"
    if re.search(r"^how\s+does\s+(it|that|this)\s+work[\s!?.]*$", low):
        return f"How does {active_subject} work?"

    # Follow-up: "What biological mechanisms cause this?"
    if re.search(r"^what\s+(biological\s+)?mechanisms?\s+(cause|drive|underlie)\s+(this|that|it)[\s!?.]*$", low):
        return f"What biological mechanisms cause {active_subject}?"

    # Follow-up: "Give me an example" / "practical example" / "practical real-world example"
    if re.search(r"^(give\s+(me\s+)?(an?\s+)?(practical\s+)?(real[- ]world\s+)?example|can\s+you\s+give\s+(me\s+)?(an?\s+)?(practical\s+)?(real[- ]world\s+)?example|practical\s+(real[- ]world\s+)?example|example)(\s+of\s+(that|it|this))?[\s!?.]*$", low):
        return f"Give me a practical example of {active_subject}"

    # Follow-up: "Where is it used?"
    if re.search(r"^where\s+is\s+(it|that|this)\s+used[\s!?.]*$", low):
        return f"Where is {active_subject} used?"

    # Follow-up: "What is his/its most famous song?"
    if re.search(r"^what\s+is\s+(his|its|their)\s+most\s+famous\s+song[\s!?.]*$", low):
        return f"What is {active_subject}'s most famous song?"

    # Possessive follow-up: "What about his songs?" -> "<active_subject> songs"
    m_poss = re.search(r"^(what\s+about|how\s+about|tell\s+me\s+about|and)\s+(his|her|their|its)\s+(.*)$", low)
    if m_poss and active_subject:
        aspect = m_poss.group(3).strip(" ?.!\"'")
        return f"{active_subject} {aspect}"

    # Possessive pronouns: "his songs" -> "<active_subject>'s songs"
    if re.search(r"\bhis\b", low):
        cleaned = re.sub(r"\bhis\b", f"{active_subject}'s", cleaned, flags=re.I)
    if re.search(r"\bher\b", low):
        cleaned = re.sub(r"\bher\b", f"{active_subject}'s", cleaned, flags=re.I)
    if re.search(r"\bits\b", low):
        cleaned = re.sub(r"\bits\b", f"{active_subject}'s", cleaned, flags=re.I)

    # Reference to concept: "Explain that", "What is it", "How does it work"
    if re.search(r"^(explain|tell\s+me\s+about|how\s+does|what\s+is)\s+(that|it|this|them)\b", low):
        cleaned = re.sub(r"\b(that|it|this|them)\b", active_subject, cleaned, flags=re.I)

    return cleaned


# ---------------------------------------------------------------------
# 11. Technical Term Disambiguation
# ---------------------------------------------------------------------
def disambiguate_technical_query(query: str, active_subject: Optional[str] = None) -> str:
    """
    Expands ambiguous technical acronyms and terms within deep learning and AI context.
    """
    low = query.lower().strip()

    # Transformer in deep learning
    if re.search(r"\b(transformer|transformers)\b", low):
        return re.sub(
            r"\b(transformer|transformers)\b",
            "Transformer neural network architecture deep learning attention mechanism",
            query,
            flags=re.I
        )

    if re.search(r"\bcnn\b", low):
        return re.sub(r"\bcnn\b", "convolutional neural network", query, flags=re.I)

    return query


# ---------------------------------------------------------------------
# 12. Compound Question Splitting
# ---------------------------------------------------------------------
def split_compound_question(q: str) -> List[str]:
    """
    Splits multi-subject compound questions such as:
    'What is Punjab and who is Sidhu Moose Wala?' -> ['What is Punjab', 'who is Sidhu Moose Wala']
    """
    t = q.strip(" ?.!\"'")
    m = re.search(r"^(.*?)\s+and\s+(who|what|where|how|tell\s+me)\s+(.*?)$", t, re.I)
    if m:
        p1 = m.group(1).strip()
        p2 = f"{m.group(2)} {m.group(3)}".strip()
        if len(p1.split()) >= 2 and len(p2.split()) >= 2:
            return [p1, p2]

    qm_parts = [p.strip() for p in re.split(r"\?\s*", q) if len(p.strip().split()) >= 2]
    if len(qm_parts) >= 2:
        return qm_parts[:2]

    return [q]


# ---------------------------------------------------------------------
# 13. Rule-Based Emotion & Conversational Intent Detection (13 Classes)
# ---------------------------------------------------------------------
def detect_emotion_and_intent(text: str, history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, Optional[str]]:
    """
    Classifies user message into one of 13 categories:
    1. HAPPY
    2. SAD
    3. DISAPPOINTED
    4. ANGRY
    5. STRESSED
    6. WORRIED
    7. CONFUSED
    8. LONELY
    9. EXCITED
    10. GRATITUDE
    11. APOLOGY
    12. PHYSICAL_INJURY
    13. NEUTRAL / NORMAL_CONVERSATION
    """
    low = text.lower().strip()

    # Guard against educational queries containing emotion words (e.g. "Explain anger management")
    is_educational = bool(re.search(
        r"^(can\s+you\s+(explain|tell)|could\s+you\s+(explain|tell)|please\s+explain|what\s+is\s+(the|a|an)?\b|what\s+are\s+(the|some)?\b(?!you\b)|define\b|definition\s+of|explain\b|overview\s+of|causes\s+of|symptoms\s+of|how\s+to\s+manage|how\s+to\s+treat|why\s+(does|do|is|are|did|can)\b|how\s+(does|do|is|are|can|did)\b|who\s+(is|was|are|were)\b|where\s+(is|are|do|does)\b|when\s+(was|is|did)\b)\b",
        low
    ))
    has_first_person = bool(re.search(r"\b(i\s+am|i'm|im|i\s+feel|i\s+have|i\s+got|my|me)\b", low))

    if is_educational and not has_first_person:
        return "NEUTRAL", None

    # FEELING BETTER / RECOVERY
    if re.search(r"\b(feel|feeling)\s+(much\s+|a\s+lot\s+)?better(\s+now)?\b", low) or low in ["i feel better", "i feel better now", "feeling better now"]:
        return "FEELING_BETTER", "feeling_better"

    # SAFETY-SENSITIVE: PHYSICAL VIOLENCE / ASSAULT / THREATS OF HARM
    violence_pats = [
        r"\bi\s+(beat\s+up|punched|hit|attacked|assaulted|fought\s+physically|battered)\s+(my\s+)?(friend|best\s+friend|brother|sister|roommate|partner|colleague|coworker|someone|a\s+guy|a\s+person|him|her|them)\b",
        r"\bi\s+beat\s+(my\s+friend|someone|him|her|them)\s+up\b",
        r"\bi\s+(beat\s+up|punched|hit|attacked)\s+\w+\b",
        r"\b(got\s+into\s+a\s+physical\s+fight|had\s+a\s+physical\s+fight|started\s+throwing\s+punches|threw\s+punches)\b",
        r"\b(physically\s+hurt\s+someone|i\s+hurt\s+someone\s+physically)\b",
        r"\b(i\s+want\s+to\s+beat\s+up|going\s+to\s+beat\s+up|want\s+to\s+hit)\s+(someone|him|her|them|my\s+friend)\b"
    ]
    if any(re.search(p, low) for p in violence_pats):
        return "SAFETY_VIOLENCE", "violence_deescalation"

    # FAMILY CONFLICT / ANGER AT PARENTS
    family_conflict_pats = [
        r"\bi('?m| am)\s+(really\s+|so\s+)?(angry|mad|furious|upset)\s+with\s+(my\s+)?(father|dad|mother|mom|parents|family|brother|sister)\b",
        r"\b(had\s+a\s+fight|had\s+an\s+argument|argued|conflict|fight)\s+with\s+(my\s+)?(father|dad|mother|mom|parents)\b",
        r"\b(my\s+)?(father|dad|mother|mom|parents)\s+(makes?\s+me\s+angry|infuriates?\s+me|is\s+unfair|treats?\s+me\s+badly)\b"
    ]
    if any(re.search(p, low) for p in family_conflict_pats):
        return "FAMILY_CONFLICT", "family_anger"

    # INTERPERSONAL MISUNDERSTANDING (e.g. "He misunderstood me")
    misunderstand_pats = [
        r"\b(he|she|they|my\s+friend|someone)\s+(misunderstood|didn'?t\s+understand|took\s+it\s+the\s+wrong\s+way)\s*(me|what\s+i\s+said|my\s+intentions?)?\b",
        r"\bi\s+was\s+misunderstood\b",
        r"\bthey\s+took\s+what\s+i\s+said\s+wrong\b"
    ]
    if any(re.search(p, low) for p in misunderstand_pats):
        return "EMOTIONAL_HURT", "friend_misunderstanding"

    # EMOTIONAL HURT / INTERPERSONAL PAIN & CONFLICT
    emotional_hurt_pats = [
        r"\b(got|was|feel|feeling)\s+(really\s+|so\s+)?hurt\s+by\s+(someone|my\s+friend|my\s+best\s+friend|them|people|him|her|words|family)\b",
        r"\b(my\s+)?(best\s+)?friend\s+(ignored|betrayed|hurt|left|abandoned)\s+me\b",
        r"\b(someone|they)\s+broke\s+my\s+heart\b",
        r"\bfeel\s+(really\s+|so\s+)?hurt\b",
        r"\b(betrayed|let\s+down)\s+by\b",
        r"\bi\s+got\s+hurt\s+by\s+\w+\b",
        r"\b(disagreement|argument|fight|dispute|falling\s+out|fallout|conflict)\s+with\s+(my\s+)?(best\s+)?(friend|buddy|pal|roommate|partner|family|brother|sister)\b",
        r"\b(had\s+a\s+disagreement|had\s+an\s+argument|had\s+a\s+fight|got\s+into\s+an\s+argument|got\s+into\s+a\s+fight)\b",
        r"\b(argued|fought)\s+with\s+(my\s+)?(best\s+)?(friend|buddy|pal|partner)\b",
        r"\bwe\s+(had\s+a\s+disagreement|had\s+an\s+argument|argued|fought|had\s+a\s+fight)\b"
    ]
    if any(re.search(p, low) for p in emotional_hurt_pats):
        if any(k in low for k in ["disagreement", "argument", "fight", "conflict", "argued", "fought"]):
            return "EMOTIONAL_HURT", "friend_disagreement"
        return "EMOTIONAL_HURT", "interpersonal_hurt"

    # INTERPERSONAL DILEMMA: Whether to message / reach out first
    dilemma_pats = [
        r"\b(whether\s+i\s+should|should\s+i)\s+(message|text|call|reach\s+out\s+to|talk\s+to|contact)\s+(them|him|her|my\s+friend)\s*(first)?\b",
        r"\b(don'?t|do\s+not)\s+know\s+whether\s+(i\s+should\s+)?(message|text|call|reach\s+out|apologize)\b",
        r"\bshould\s+i\s+(message|text|call|reach\s+out)\s*(first)?\b",
        r"\bwho\s+should\s+(message|text|call|reach\s+out|apologize)\s+first\b"
    ]
    if any(re.search(p, low) for p in dilemma_pats):
        return "INTERPERSONAL_DILEMMA", "message_first"

    # INTERPERSONAL ADVICE: What would you do in this situation
    advice_pats = [
        r"\bwhat\s+(would|should)\s+you\s+do\s+(in\s+this\s+situation|here|if\s+you\s+were\s+me)\b",
        r"\bwhat\s+would\s+you\s+do\b",
        r"\bwhat\s+do\s+you\s+think\s+i\s+should\s+do\b",
        r"\bhow\s+would\s+you\s+handle\s+this\b",
        r"\bwhat\s+should\s+i\s+do\b"
    ]
    if any(re.search(p, low) for p in advice_pats):
        return "INTERPERSONAL_ADVICE", "what_would_you_do_friend"

    # TIRED / EXHAUSTED / DRAINING DAY
    tiring_day_pats = [
        r"\b(had|have\s+had)\s+(a\s+)?(pretty\s+|really\s+|very\s+|so\s+)?(tiring|exhausting|long|draining|rough|hard|hectic|tough)\s+day\b",
        r"\bi('?m| am| feel| am feeling)\s+(pretty\s+|really\s+|so\s+|very\s+)?(tired|exhausted|drained|burned\s+out|burnt\s+out)\b",
        r"\bday\s+(was|has\s+been)\s+(pretty\s+|really\s+|so\s+)?(tiring|exhausting|draining|rough|hectic)\b"
    ]
    if any(re.search(p, low) for p in tiring_day_pats):
        return "TIRED", "tiring_day"

    # PHYSICAL DISCOMFORT / INJURY
    injury_pats = [
        r"\bi\s+(broke|fractured|sprained|twisted|cut|burned)\s+(my\s+)?(leg|arm|hand|foot|finger|bone|ankle|wrist|toe|rib)\b",
        r"\bi\s+(got|am)\s+(physically\s+)?injured\b",
        r"\bi\s+hurt\s+my\s+(leg|arm|hand|head|foot|knee|back|neck|shoulder|body)\b",
        r"\bi\s+have\s+a\s+(physical\s+)?injury\b",
        r"\bi\s+had\s+a\s+(car\s+)?(accident|crash|fall)\b",
        r"\bi('?m| am)\s+bleeding\b",
        r"\bmy\s+(leg|arm|hand|head|foot|knee|back|shoulder|neck|wrist|ankle)\s+hurts\b",
        r"\bbroken\s+(leg|arm|bone|hand|foot)\b"
    ]
    if any(re.search(p, low) for p in injury_pats):
        return "PHYSICAL_INJURY", "injury"

    # GRATITUDE
    if re.search(r"\b(thank\s+you|thanks|thank\s+u|appreciate\s+it|many\s+thanks|grateful|that\s+helped\s+me|that\s+helps|very\s+helpful)\b", low):
        return "GRATITUDE", "thank_you"

    # APOLOGY
    if re.search(r"\b(sorry|i\s+apologize|my\s+bad|forgive\s+me|excuse\s+me)\b", low):
        return "APOLOGY", "sorry"

    # FAREWELL
    if re.search(r"^(bye|goodbye|see\s+you|see\s+ya|cya|good\s*night|take\s+care|have\s+a\s+good\s+(day|night))[\s!.?]*$", low) or re.search(r"\b(bye\s+bye|talk\s+to\s+you\s+later)\b", low):
        return "FAREWELL", "bye"

    # CASUAL CONVERSATION & SOCIAL
    if re.search(r"\b(what\s+are\s+you\s+doing|what\s+are\s+you\s+up\s+to)\b", low):
        return "CASUAL_CONVERSATION", "what_are_you_doing"
    if re.search(r"^(can\s+we\s+talk|could\s+we\s+talk|let'?s\s+talk)[\s!?.]*$", low):
        return "CASUAL_CONVERSATION", "can_we_talk"
    if re.search(r"\b(how\s+was\s+your\s+day|how\s+has\s+your\s+day\s+been)\b", low):
        return "CASUAL_CONVERSATION", "how_was_your_day"
    if re.search(r"^(tell\s+me\s+something\s+interesting)[\s!?.]*$", low):
        return "CASUAL_CONVERSATION", "something_interesting"
    if re.search(r"\b(tell\s+me\s+a\s+joke|make\s+me\s+laugh|know\s+any\s+jokes)\b", low):
        return "CASUAL_CONVERSATION", "tell_me_a_joke"
    if re.search(r"\b(what\s+do\s+you\s+think\s+about\s+this|what\s+are\s+your\s+thoughts)\b", low):
        return "CASUAL_CONVERSATION", "thoughts_opinion"

    # BOREDOM
    if re.search(r"\b(i('?m| am)\s+(so\s+|really\s+)?bored|bored\s*today|feeling\s+bored)\b", low):
        return "BOREDOM", "bored"

    # DISAPPOINTED (Specifically exams and setbacks)
    disappointed_pats = [
        r"\bi\s+failed\s+my\s+(exam|examination|test|interview)\b",
        r"\bi('?m| am)\s+(so\s+|really\s+)?disappointed\b",
        r"\bi\s+messed\s+up\s+my\s+(exam|test|interview)\b"
    ]
    if any(re.search(p, low) for p in disappointed_pats):
        return "DISAPPOINTED", "failed_exam"

    # HAPPY / POSITIVE
    happy_pats = [
        r"\bi\s+(passed|cleared)\s+my\s+(exam|test|interview|examination)\b",
        r"\bi('?m| am)\s+(very\s+|so\s+|really\s+)?happy\b",
        r"\bi\s+got\s+(a\s+|the\s+)?(new\s+)?(job|job\s+offer|offer|promotion|internship|reward|award)\b",
        r"\btoday\s+was\s+(wonderful|great|amazing|awesome|fantastic)\b",
        r"\bi\s+(feel|am\s+feeling)\s+(great|fantastic|blessed|cheerful|joyful)\b",
        r"\bi\s+(won|did\s+it|succeeded)\b",
        r"\bgood\s+news\b"
    ]
    if any(re.search(p, low) for p in happy_pats):
        return "HAPPY", "happy_event"

    # EXCITED
    excited_pats = [
        r"\bi('?m| am)\s+(so\s+|really\s+)?excited\b",
        r"\bi\s+(feel|am\s+feeling)\s+(so\s+|really\s+)?excited\b",
        r"\bsomething\s+amazing\s+happened\b",
        r"\bi\s+can'?t\s+wait\b",
        r"\b(thrilled|pumped\s+up|super\s+excited)\b"
    ]
    if any(re.search(p, low) for p in excited_pats):
        return "EXCITED", "excited_event"

    # LONELY
    lonely_pats = [
        r"\bi\s+(feel|am\s+feeling)\s+(so\s+|really\s+)?lonely\b",
        r"\bnobody\s+talks\s+to\s+me\b",
        r"\bi\s+have\s+no\s*one\s+to\s+talk\s+to\b",
        r"\bfeeling\s+(all\s+)?alone\b",
        r"\bi('?m| am)\s+(feeling\s+)?lonely\b",
        r"\bi\s+feel\s+isolated\b"
    ]
    if any(re.search(p, low) for p in lonely_pats):
        return "LONELY", "lonely"

    # CONFUSED / LOST
    confused_pats = [
        r"\bi\s+(don'?t|do\s+not)\s+understand(\s+anything)?\b",
        r"\bi('?m| am| feel| am feeling)\s+(so\s+|really\s+)?confused\b",
        r"\bi\s+don'?t\s+know\s+what\s+to\s+do\b",
        r"\bcan\s+you\s+explain\s+again\b",
        r"\bi\s+(don'?t|do\s+not)\s+get\s+it\b",
        r"\bi('?m| am| feel)\s+(so\s+)?lost\b",
        r"\bwhat\s+do\s+you\s+mean\b",
        r"\bnothing\s+makes\s+sense\b"
    ]
    if any(re.search(p, low) for p in confused_pats):
        return "CONFUSED", "confused"

    # ANGRY / FRUSTRATED
    angry_pats = [
        r"\bi('?m| am)\s+(really\s+|so\s+)?(frustrated|angry|mad|furious|annoyed|irritated)\b",
        r"\bnothing\s+is\s+working\b",
        r"\bi\s+hate\s+this\b",
        r"\bthis\s+is\s+(so\s+)?infuriating\b",
        r"\bpissing\s+me\s+off\b"
    ]
    if any(re.search(p, low) for p in angry_pats):
        return "ANGRY", "angry_frustrated"

    # FEAR / NERVOUSNESS / ANXIETY
    fear_nervous_pats = [
        r"\bi('?m| am| feel| am feeling)\s+(so\s+|really\s+|very\s+)?(nervous|anxious|scared|frightened|terrified|afraid)\b",
        r"\b(nervous|anxious|scared)\s+about\s+(tomorrow|the interview|my interview|the presentation|my presentation|the exam|my exam|the future|the test)\b",
        r"\bi\s+have\s+(so\s+much\s+)?anxiety\b",
        r"\bfeeling\s+(nervous|anxious|scared|panicked)\b",
        r"\bbutterflies\s+in\s+my\s+stomach\b"
    ]
    if any(re.search(p, low) for p in fear_nervous_pats):
        return "FEAR_NERVOUS", "fear_nervousness"

    # STRESSED / WORRIED
    stressed_pats = [
        r"\bi('?m| am| feel| am feeling)\s+(so\s+|really\s+|very\s+)?stressed(\s+out)?\b",
        r"\b(feel|feeling)\s+(so\s+|really\s+|very\s+)?stressed\b",
        r"\bi('?m| am)\s+worried\s+about\b",
        r"\bi\s+have\s+too\s+much\s+work\b",
        r"\bi('?m| am)\s+(anxious|nervous|overwhelmed)\b",
        r"\btoo\s+much\s+pressure\b",
        r"\bstressed\s+about\s+my\s+(examination|exam|future|test)\b"
    ]
    if any(re.search(p, low) for p in stressed_pats):
        return "STRESSED", "stressed_worried"

    # HELPLESS / INSECURE / FEELING USELESS OR FOOLISH
    helpless_pats = [
        r"\bi('?m| am| feel| am feeling)\s+(so\s+|really\s+|very\s+)?(helpless|useless|worthless|hopeless|foolish|stupid|dumb)\b",
        r"\bi\s+(am|feel)\s+(helpless|foolish|stupid|worthless)\b",
        r"\bi\s+feel\s+so\s+(useless|foolish|stupid)\b",
        r"\bi\s+feel\s+useless\b",
        r"\b(nothing\s+i\s+do\s+matters|feel\s+like\s+a\s+failure|good\s+for\s+nothing|i('?m| am)\s+no\s+good)\b",
        r"\bno\s+matter\s+what\s+i\s+do\s+nothing\s+works\b",
        r"\bi\s+can'?t\s+do\s+anything\s+right\b"
    ]
    if any(re.search(p, low) for p in helpless_pats):
        return "HELPLESS_INSECURE", "feeling_helpless"

    # SAD / DISTRESSED
    sad_pats = [
        r"\bi('?m| am| feel| am feeling)\s+(feeling\s+)?(really\s+|so\s+|very\s+)?(sad|down|blue|depressed|unhappy|miserable)\b",
        r"\bi\s+(feel|am\s+feeling)\s+(really\s+|so\s+|very\s+)?(bad|sad|down|low|awful)(\s+today)?\b",
        r"\b(feel|feeling)\s+like\s+i('?m| am)\s+(not|never)\s+making\s+(enough\s+)?progress\b",
        r"\bnot\s+making\s+(enough\s+)?progress\b",
        r"\bi\s+miss\s+someone\b",
        r"\bi\s+had\s+a\s+(bad|rough|terrible|awful)\s+day\b",
        r"\bheartbroken\b",
        r"\bfeeling\s+(down|low|awful|sad)\b",
        r"\b(feel\s+like\s+crying|crying|want\s+to\s+cry)\b",
        r"\bnobody\s+understands\s+me\b",
        r"\bfeel\s+like\s+giving\s+up\b"
    ]
    if any(re.search(p, low) for p in sad_pats):
        return "SAD", "sad"

    # Multi-turn emotional continuity check
    if history and len(history) >= 1:
        all_user_text = " ".join(t.get("text", "").lower() for t in history if t.get("role") == "user")
        if any(w in all_user_text for w in ["disagreement", "best friend", "friend", "fight", "argued", "argument"]):
            if any(w in low for w in ["them", "first", "message", "text", "call", "reach out", "what would you do", "advice", "situation", "what should i do"]):
                if any(w in low for w in ["what would you do", "what should i do", "advice", "how would you", "situation"]):
                    return "INTERPERSONAL_ADVICE", "what_would_you_do_friend"
                return "INTERPERSONAL_DILEMMA", "message_first"
        if any(w in all_user_text for w in ["tiring day", "exhausting", "tired"]):
            if any(w in low for w in ["disagreement", "friend", "fight", "argued"]):
                return "EMOTIONAL_HURT", "friend_disagreement"

        prev_bot_turn = history[-1]["text"].lower() if history[-1]["role"] == "assistant" else ""
        if any(w in prev_bot_turn for w in ["bothering you", "troubling you", "hear that", "what happened", "to listen"]):
            # User is replying to an emotional inquiry ONLY if not a clearly unrelated informational question
            is_factual_q = bool(re.search(
                r"^(why\s+(does|do|is|are|did|can)|how\s+(does|do|is|are|can|did|to)|what\s+(is|are|causes?|were)|who\s+(is|was|are|were)|where\s+(is|are|do|does)|when\s+(was|is|did)|explain\b|define\b|tell\s+me\s+about\b|can\s+you\s+(explain|tell)|could\s+you\s+(explain|tell))\b",
                low
            ))
            if not is_factual_q:
                return "SAD", "emotional_continuation"

    return "NEUTRAL", None


# ---------------------------------------------------------------------
# 14. Varied Natural Response Library
# ---------------------------------------------------------------------
RESPONSE_LIBRARY = {
    "FEELING_BETTER": [
        "I'm so glad to hear you're feeling better! That brings a real smile to my face. What would you like to explore or talk about next?",
        "That's wonderful news! It makes me really happy to hear you're feeling lighter and better. How can I help you today?"
    ],
    "feeling_better": [
        "I'm so glad to hear you're feeling better! That brings a real smile to my face. What would you like to explore or talk about next?",
        "That's wonderful news! It makes me really happy to hear you're feeling lighter and better. How can I help you today?"
    ],
    "SAFETY_VIOLENCE": [
        "Physical violence is dangerous and can cause severe physical and emotional harm. First, is anyone injured or in need of medical attention right now? If someone is hurt, please contact medical help or emergency services (dial 112 in India, 911 in the US) immediately. Right now, the most important thing is to step away and separate completely so tempers can cool down and no further harm occurs. Once you are calm and safe, you will need to handle this responsibly through peaceful communication or mediation.",
        "Getting into a physical fight or hitting someone is serious and dangerous. First, please let me know: is anyone injured? If anyone needs medical care, getting help right now is the urgent priority. Please step away from the situation immediately, take a breath, and do not escalate things any further. Violence creates lasting damage, and resolving this will require calming down, taking accountability, and finding a peaceful path forward."
    ],
    "violence_deescalation": [
        "Physical violence is dangerous and can cause severe physical and emotional harm. First, is anyone injured or in need of medical attention right now? If someone is hurt, please contact medical help or emergency services (dial 112 in India, 911 in the US) immediately. Right now, the most important thing is to step away and separate completely so tempers can cool down and no further harm occurs. Once you are calm and safe, you will need to handle this responsibly through peaceful communication or mediation.",
        "Getting into a physical fight or hitting someone is serious and dangerous. First, please let me know: is anyone injured? If anyone needs medical care, getting help right now is the urgent priority. Please step away from the situation immediately, take a breath, and do not escalate things any further. Violence creates lasting damage, and resolving this will require calming down, taking accountability, and finding a peaceful path forward."
    ],
    "HELPLESS_INSECURE": [
        "I'm really sorry you're feeling this way. Feeling helpless, foolish, or feeling like nothing you do matters is a really heavy emotional weight to carry. Please know that having these feelings does not mean you are actually foolish or incapable, and you don't have to face this alone. I'm right here with you. Do you want to talk about what's making you feel this way?",
        "It hurts so much when you feel helpless or question your worth. I want to remind you gently that making mistakes doesn't make you foolish, and these feelings don't define who you are. Take a slow, deep breath. I'm here to listen without any judgment—what happened that brought this feeling on?",
        "I hear you, and my heart goes out to you. Feeling useless, foolish, or helpless can make everything seem dark. Please be kind to yourself right now. You don't have to have everything figured out today. I'm right here beside you whenever you want to share what's on your mind."
    ],
    "feeling_helpless": [
        "I'm really sorry you're feeling this way. Feeling helpless, foolish, or feeling like nothing you do matters is a really heavy emotional weight to carry. Please know that having these feelings does not mean you are actually foolish or incapable, and you don't have to face this alone. I'm right here with you. Do you want to talk about what's making you feel this way?",
        "It hurts so much when you feel helpless or question your worth. I want to remind you gently that making mistakes doesn't make you foolish, and these feelings don't define who you are. Take a slow, deep breath. I'm here to listen without any judgment—what happened that brought this feeling on?"
    ],
    "FAMILY_CONFLICT": [
        "Family relationships can be deeply complex, and feeling angry or in conflict with your father or family can be really painful and emotionally draining. It's completely valid to feel upset when things are tense at home. Do you want to talk about what happened between you two?",
        "Dealing with tension with your parents can be so stressful and frustrating. You have every right to feel your feelings. What caused the disagreement or anger with your father?"
    ],
    "family_anger": [
        "Family relationships can be deeply complex, and feeling angry or in conflict with your father or family can be really painful and emotionally draining. It's completely valid to feel upset when things are tense at home. Do you want to talk about what happened between you two?",
        "Dealing with tension with your parents can be so stressful and frustrating. You have every right to feel your feelings. What caused the disagreement or anger with your father?"
    ],
    "friend_misunderstanding": [
        "That can be really frustrating, especially when someone close misunderstands your intentions or what you said. When you didn't mean things that way, it hurts to feel blamed. Have you had a chance to calmly explain what you actually meant?",
        "Being misunderstood by someone you care about is so discouraging. It's hard when your words are taken the wrong way. Do you think there's a good time to clear the air and share what you really meant?"
    ],
    "TIRED": [
        "I'm sorry to hear that you had such a tiring day. Long and exhausting days can really drain your energy and mood. Take a moment to sit back, breathe, and unwind. If you'd like to talk about what made it so exhausting, or if you just want some relaxing company, I'm right here with you.",
        "Tiring days can be really tough to get through. Please be kind to yourself this evening. If you want to vent about what made today so exhausting, or if you'd prefer to chat about something peaceful, I'm here for you."
    ],
    "tiring_day": [
        "I'm sorry to hear that you had such a tiring day. Long and exhausting days can really drain your energy and mood. Take a moment to sit back, breathe, and unwind. If you'd like to talk about what made it so exhausting, or if you just want some relaxing company, I'm right here with you.",
        "Tiring days can be really tough to get through. Please be kind to yourself this evening. If you want to vent about what made today so exhausting, or if you'd prefer to chat about something peaceful, I'm here for you."
    ],
    "friend_disagreement": [
        "I'm really sorry to hear that. Disagreements with someone close to you are especially distressing and emotionally draining. When someone matters that much to you, conflict cuts deep. I'm right here to listen—do you want to share what the disagreement was about, or how you're feeling right now?",
        "That sounds really difficult. Having a disagreement or argument with someone close to you always weighs heavily on your mind. Take a breath—I'm right here to listen. What happened between the two of you?"
    ],
    "INTERPERSONAL_DILEMMA": [
        "That's a very common and difficult dilemma. When there's a disagreement, hesitation to reach out first is natural—nobody wants to feel like they're conceding. But messaging first doesn't mean you were wrong; it just shows you value the friendship more than winning the conflict. If emotions are still intense, giving each other a little space to cool down is completely fine. Once things settle, sending a gentle, non-defensive message can break the ice without putting anyone on the defensive.",
        "It's completely normal to feel torn about whether to reach out first. Reaching out first isn't admitting defeat; it's an act of emotional maturity that says the friendship matters to you. If tempers have calmed down a bit, a short, calm text can reopen communication safely."
    ],
    "message_first": [
        "That's a very common and difficult dilemma. When there's a disagreement, hesitation to reach out first is natural—nobody wants to feel like they're conceding. But messaging first doesn't mean you were wrong; it just shows you value the friendship more than winning the conflict. If emotions are still intense, giving each other a little space to cool down is completely fine. Once things settle, sending a gentle, non-defensive message can break the ice without putting anyone on the defensive.",
        "It's completely normal to feel torn about whether to reach out first. Reaching out first isn't admitting defeat; it's an act of emotional maturity that says the friendship matters to you. If tempers have calmed down a bit, a short, calm text can reopen communication safely."
    ],
    "INTERPERSONAL_ADVICE": [
        "If I were in your shoes, I would give both of you a little time to let the initial frustration settle, and then I would send a short, caring message first. You don't have to rehash the whole argument right away; something simple and sincere works best:\n\n*'Hey, our friendship means a lot to me and I really don't like that we argued today. Whenever you're ready, I'd love to talk things through and hear your thoughts.'*\n\nReaching out this way takes the pressure off, demonstrates maturity, and leaves the door open without putting your friend on the defensive. How does that approach sound to you?",
        "In this situation, taking the high road with a gentle, open message is usually the most effective step. You might text: *'Hey, I care about you and value our friendship. Let's chat when we've both had some time to breathe.'* It de-escalates tension and reminds them that the bond is bigger than today's disagreement."
    ],
    "what_would_you_do_friend": [
        "If I were in your shoes, I would give both of you a little time to let the initial frustration settle, and then I would send a short, caring message first. You don't have to rehash the whole argument right away; something simple and sincere works best:\n\n*'Hey, our friendship means a lot to me and I really don't like that we argued today. Whenever you're ready, I'd love to talk things through and hear your thoughts.'*\n\nReaching out this way takes the pressure off, demonstrates maturity, and leaves the door open without putting your friend on the defensive. How does that approach sound to you?",
        "In this situation, taking the high road with a gentle, open message is usually the most effective step. You might text: *'Hey, I care about you and value our friendship. Let's chat when we've both had some time to breathe.'* It de-escalates tension and reminds them that the bond is bigger than today's disagreement."
    ],
    "what_would_you_do": [
        "If I were in your shoes, I would take a step back to evaluate the situation calmly, prioritize open and honest communication, and focus on what can be resolved cooperatively rather than who is to blame. What are your instincts telling you right now?",
        "In a situation like this, taking a breath and choosing a constructive, empathetic approach usually yields the best outcome. What part of the situation feels most challenging for you to navigate?"
    ],
    "FEAR_NERVOUS": [
        "It's completely natural to feel nervous before something that matters to you. Take a slow, gentle breath with me right now. You are more prepared and capable than you might feel in this moment. What part of it is making you feel most uneasy?",
        "Feeling anxious or scared is completely valid—it just shows that you care deeply about the outcome. Try rolling your shoulders back and letting your breath out slowly. I'm right here with you. Would talking through it help ease the tension?",
        "I hear you. Nervousness can make your heart race and thoughts feel overwhelming. Just take things one small breath at a time. What is the main thing on your mind right now?"
    ],
    "fear_nervousness": [
        "It's completely natural to feel nervous before something that matters to you. Take a slow, gentle breath with me right now. You are more prepared and capable than you might feel in this moment. What part of it is making you feel most uneasy?",
        "Feeling anxious or scared is completely valid—it just shows that you care deeply about the outcome. Try rolling your shoulders back and letting your breath out slowly. I'm right here with you. Would talking through it help ease the tension?"
    ],
    "PHYSICAL_INJURY": [
        "Oh no, I'm sorry to hear that. Are you okay? What happened?",
        "Oh no, that sounds painful! Are you alright? Please make sure to get proper first aid or see a doctor if it's serious.",
        "I'm really sorry to hear that you got injured. Are you in a safe place, and have you been able to take care of it?",
        "Ouch, I'm so sorry! What happened? Please take it easy and make sure you're looked after."
    ],
    "DISAPPOINTED": [
        "I'm really sorry. Disappointment can sting deeply, especially when you put your heart into something. Setbacks happen to everyone, but they don't determine what you can achieve. How are you holding up?",
        "That must feel discouraging, but remember that one setback doesn't define your path. Take a breath—I'm right here if you want to talk through it.",
        "I understand how disheartening that can be. Please be kind to yourself today. One test or interview does not measure your intelligence or your future.",
        "It hurts when things don't go the way you hoped. You don't have to brush it off right away. I'm here to listen if you want to share what happened."
    ],
    "HAPPY": [
        "That is fantastic news! I'm so happy for you—you really deserve to celebrate this! How does it feel?",
        "Wow, congratulations! That's wonderful to hear. Hearing good news always brightens my day. Tell me all about it!",
        "That is truly worth celebrating! What an amazing achievement. How are you going to celebrate today?",
        "That brings a big smile to my face! Enjoy every moment of this success—you worked hard for it."
    ],
    "happy_event": [
        "That is fantastic news! I'm so happy for you—you really deserve to celebrate this! How does it feel?",
        "Wow, congratulations! That's wonderful to hear. Hearing good news always brightens my day. Tell me all about it!",
        "That is truly worth celebrating! What an amazing achievement. How are you going to celebrate today?"
    ],
    "SAD": [
        "I'm really sorry you're feeling down today. Sometimes things feel heavy and rough, but you don't have to carry it alone. I'm right here to listen whenever you'd like to share what's bothering you.",
        "It's really tough when you feel sad or discouraged. Please know that whatever you're feeling is completely valid. I'm right here to listen to whatever is troubling you.",
        "I'm so sorry things feel rough and heavy right now. You don't have to put on a brave face with me. I'm right here to listen whenever you want to share what happened.",
        "I'm sorry today has been rough. Please be kind to yourself. I'm right here to listen and support you through whatever is on your heart."
    ],
    "sad": [
        "I'm really sorry you're feeling down today. Sometimes things feel heavy and rough, but you don't have to carry it alone. I'm right here to listen whenever you'd like to share what's bothering you.",
        "It's really tough when you feel sad or discouraged. Please know that whatever you're feeling is completely valid. I'm right here to listen to whatever is troubling you.",
        "I'm so sorry things feel rough and heavy right now. You don't have to put on a brave face with me. I'm right here to listen whenever you want to share what happened."
    ],
    "ANGRY": [
        "I hear you, and it sounds like you have every right to feel frustrated. When things feel unfair or people cross boundaries, anger is a completely natural reaction. Do you want to vent about what happened?",
        "That sounds intensely frustrating. Take a slow breath—I'm right here listening, and you can let it all out without any judgment. What part of it is bothering you most?",
        "It's completely valid to feel mad about this. Holding frustration in can feel suffocating. I'm right here with you—what happened?"
    ],
    "angry_frustrated": [
        "I hear you, and it sounds like you have every right to feel frustrated. When things feel unfair or people cross boundaries, anger is a completely natural reaction. Do you want to vent about what happened?",
        "That sounds intensely frustrating. Take a slow breath—I'm right here listening, and you can let it all out without any judgment. What part of it is bothering you most?"
    ],
    "STRESSED": [
        "Take a deep, slow breath with me. When stress piles up, everything can feel overwhelming all at once. You don't have to fix everything today. What's the heaviest thing on your mind right now?",
        "I can hear the stress and weight in what you're saying. Please pause for a moment, take a breath, and let your shoulders drop. You're doing the best you can, and we can take this one step at a time.",
        "Feeling stressed is completely understandable when you're under this much pressure. Remember to be kind to yourself and take a gentle breath. What's the biggest task worrying you right now?"
    ],
    "stressed_worried": [
        "Take a deep, slow breath with me. When stress piles up, everything can feel overwhelming all at once. You don't have to fix everything today. What's the heaviest thing on your mind right now?",
        "I can hear the stress and weight in what you're saying. Please pause for a moment, take a breath, and let your shoulders drop. You're doing the best you can, and we can take this one step at a time."
    ],
    "CONFUSED": [
        "It's completely okay to feel confused, no worries at all! When there's a lot on your mind, things can feel tangled up. Let's break it down step by step: what's the main thing causing you confusion right now?",
        "Feeling like you don't know what to do is completely understandable when you're overwhelmed. Take a deep breath. Let's break things down step by step. What's on your mind?",
        "No worries at all! Let me explain it more simply and break it down step by step.",
        "That's completely okay. Complex concepts take time—let's walk through it step by step in plain English.",
        "Totally understandable! Let me rephrase that and explain it in plain English for you.",
        "Don't worry! Which part felt unclear? We can explain and walk through it step by step together."
    ],
    "EMOTIONAL_HURT": [
        "I'm really sorry to hear that. Being hurt or ignored by someone close to you—especially a friend—cuts deep. Do you want to talk about what happened?",
        "That truly hurts. When someone we care about ignores or hurts us, it feels deeply disappointing and painful. I'm right here if you want to vent or talk through it.",
        "It's really painful when someone you care about lets you down or ignores you. Take your time—I'm here to listen whenever you'd like to share."
    ],
    "interpersonal_hurt": [
        "I'm really sorry to hear that. Being hurt or ignored by someone close to you—especially a friend—cuts deep. Do you want to talk about what happened?",
        "That truly hurts. When someone we care about ignores or hurts us, it feels deeply disappointing and painful. I'm right here if you want to vent or talk through it.",
        "It's really painful when someone you care about lets you down or ignores you. Take your time—I'm here to listen whenever you'd like to share."
    ],
    "BOREDOM": [
        "I'd love to keep you company and find something fun to chat about! We could discuss your favorite movies, explore an intriguing mystery or riddle, chat about travel dreams, or share some fascinating everyday trivia. What kind of vibe sounds good to you?",
        "Boredom is always a great excuse to discover something new! We could dive into a curious science question, talk about music or hobbies, or play a quick guessing game. What are you in the mood for?",
        "Let's shake off the boredom together! Tell me what's on your mind, or I can share a surprising, mind-bending fact about the universe. How does that sound?"
    ],
    "bored": [
        "I'd love to keep you company and find something fun to chat about! We could discuss your favorite movies, explore an intriguing mystery or riddle, chat about travel dreams, or share some fascinating everyday trivia. What kind of vibe sounds good to you?",
        "Boredom is always a great excuse to discover something new! We could dive into a curious science question, talk about music or hobbies, or play a quick guessing game. What are you in the mood for?"
    ],
    "tell_me_a_joke": [
        "Why don't scientists trust atoms? Because they make up everything! 😄 What kind of humor do you enjoy?",
        "What do you call a fake noodle? An impasta! 🍝 What kind of jokes do you like?",
        "Why was the math book sad? Because it had too many problems! 😆"
    ],
    "thoughts_opinion": [
        "I'd love to hear more details about it! Share what you're thinking, and we can explore the different angles and perspectives together.",
        "That sounds intriguing! Tell me more about the context and your thoughts on it, and let's discuss it."
    ],
    "LONELY": [
        "Feeling lonely is really hard, but please know that you are not alone right now. I'm right here with you, and I genuinely enjoy chatting with you. What's been on your mind lately?",
        "I'm right here beside you. Loneliness can make the quiet feel loud, but I'm always happy to keep you company. We can chat about whatever you'd like—tell me about your day or something you enjoy.",
        "I'm here to keep you company. You don't have to carry the quiet alone. What's something that made you smile recently, or what's on your heart?"
    ],
    "EXCITED": [
        "That is so exciting! I love that feeling of anticipation. Tell me all about what happened!",
        "Awesome energy! What's the amazing news? I can't wait to hear!",
        "Love the enthusiasm! That sounds incredible. What are you most looking forward to?",
        "That sounds thrilling and exciting! How are you feeling right now?"
    ],
    "GRATITUDE": [
        "You're very welcome! Glad I could help.",
        "Anytime! Let me know if there's anything else you'd like to explore.",
        "Happy to help! Wishing you the best with everything.",
        "My pleasure! I'm always here whenever you want to talk."
    ],
    "APOLOGY": [
        "No need to apologize at all! Everything is completely fine.",
        "No worries whatsoever! We're all good.",
        "Don't worry about it for a second. How can I help you now?"
    ],
    "FAREWELL": [
        "Goodbye! Take care and have a wonderful day ahead!",
        "See you later! Feel free to come back whenever you'd like to chat.",
        "Bye! Have a great time, and take care of yourself.",
        "Until next time! Have a restful day!"
    ],
    "what_are_you_doing": [
        "I'm right here chatting with you and ready to help! What are you up to?",
        "Just here, happy to converse with you! What are you working on today?",
        "I'm here ready to answer questions or just keep you company. How about you?"
    ],
    "can_we_talk": [
        "Of course! I'm always here to listen. What's on your mind?",
        "Absolutely! We can talk about whatever you'd like.",
        "I'm right here. Tell me what you're thinking about!"
    ],
    "how_was_your_day": [
        "I don't experience days like people do, but I'm having a great time chatting with you! How was yours?",
        "As an AI, every moment is about being here for you! Tell me how your day went.",
        "My day is always brightened by a good chat! Did anything interesting happen in yours?"
    ],
    "something_interesting": [
        "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!",
        "Here's a fun thought: octopuses have three hearts and blue blood, and two of their hearts stop beating when they swim!",
        "An interesting fact in computer science: the first computer bug was an actual real moth found trapped inside the Mark II computer at Harvard in 1947!"
    ],
    "emotional_continuation": [
        "I hear you, and it makes total sense why that would weigh on you. Do you want to talk more about how that made you feel?",
        "Thank you for sharing that with me. It takes courage to open up when things are difficult. I'm right here with you.",
        "That sounds really challenging to deal with. Remember that it's okay to take things one moment at a time. How can I best support you right now?"
    ]
}


def generate_companion_response(category: str, sub_intent: Optional[str] = None, user_text: str = "") -> str:
    """Selects a natural, varied response from the library."""
    if sub_intent and sub_intent in RESPONSE_LIBRARY:
        choices = RESPONSE_LIBRARY[sub_intent]
    elif category in RESPONSE_LIBRARY:
        choices = RESPONSE_LIBRARY[category]
    else:
        choices = ["I'm right here. Tell me more about what's on your mind."]

    return random.choice(choices)
