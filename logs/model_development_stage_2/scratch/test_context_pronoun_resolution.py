"""
test_context_pronoun_resolution.py
===================================
Tests multi-turn pronoun and topic resolution:
1. "Who is Diljit Dosanjh?" -> "What about his songs?"
2. "What is CNN?" -> "Explain pooling" -> "Give me an example"
"""

import re

def resolve_context(user_text: str, active_subject: str = None) -> str:
    cleaned = user_text.strip()
    low = cleaned.lower()
    
    # 1. Possessive pronouns ("his songs", "her albums", "its features")
    if active_subject:
        # "his <noun>" -> "<active_subject>'s <noun>"
        if re.search(r"\bhis\b", low):
            resolved = re.sub(r"\bhis\b", f"{active_subject}'s", cleaned, flags=re.I)
            return resolved
        if re.search(r"\bher\b", low):
            resolved = re.sub(r"\bher\b", f"{active_subject}'s", cleaned, flags=re.I)
            return resolved
        if re.search(r"\bits\b", low):
            resolved = re.sub(r"\bits\b", f"{active_subject}'s", cleaned, flags=re.I)
            return resolved
        # Pronoun "that" / "it" / "this" / "them"
        if re.search(r"^(explain|tell me about|how does|what is)\s+(that|it|this|them)\b", low):
            resolved = re.sub(r"\b(that|it|this|them)\b", active_subject, cleaned, flags=re.I)
            return resolved
        # Follow-up "Give me an example" / "example of that"
        if re.search(r"^(give\s+me\s+an?\s+example|can\s+you\s+give\s+an?\s+example|example)(\s+of\s+(that|it|this))?[\s!?.]*$", low):
            return f"Give me an example of {active_subject}"
        # "How does it work?"
        if re.search(r"^how\s+does\s+(it|that|this)\s+work[\s!?.]*$", low):
            return f"How does {active_subject} work?"
        # "Where is it used?"
        if re.search(r"^where\s+is\s+(it|that|this)\s+used[\s!?.]*$", low):
            return f"Where is {active_subject} used?"
            
    return cleaned

t1 = resolve_context("What about his songs?", active_subject="Diljit Dosanjh")
print("Resolved 1:", t1)

t2 = resolve_context("Give me an example.", active_subject="pooling in CNN")
print("Resolved 2:", t2)

t3 = resolve_context("How does it work?", active_subject="Transformer architecture")
print("Resolved 3:", t3)

t4 = resolve_context("Where is it used?", active_subject="Transformer architecture")
print("Resolved 4:", t4)
