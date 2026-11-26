import sys
import re
import html
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from web_retrieval import WebRetrievalEngine, EvidenceBundle, EvidenceItem

engine = WebRetrievalEngine()

def synthesize_grounded_answer(query: str, bundle: EvidenceBundle) -> str:
    """
    Synthesizes an authoritative, clear, and structured answer directly from verified web evidence.
    Extracts core definitions, explanatory context, and relevant applications.
    """
    if not bundle.has_results or not bundle.items:
        return f"No verified web sources were found to answer: '{query}'."

    q_lower = query.lower()
    top_item = bundle.items[0]
    
    # 1. Gather all substantive sentences from top items
    all_sentences = []
    for item in bundle.items[:3]:
        text_source = item.full_text if item.full_text else item.snippet
        # Clean out Wikipedia navigation boilerplate
        clean_text = re.sub(r"(From Wikipedia, the free encyclopedia|This article is about.*?\.|\{\{.*?\}\}|\[\d+\])", " ", text_source)
        # Split into clean sentences
        raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if len(s.strip()) > 25]
        for s in raw_sents:
            # Skip boilerplate headers
            if any(bp in s.lower() for bp in ["wikipedia", "redirects here", "for the tv series", "see also", "external links", "main article"]):
                continue
            if s not in all_sentences:
                all_sentences.append(s)

    # 2. Extract best definition sentence
    # Look for sentences defining the query topic (e.g. "X is a ...", "X refers to ...")
    clean_q = engine.extract_search_query(query).lower()
    q_keywords = [w for w in re.findall(r"\w+", clean_q) if w not in {"what", "is", "the", "explain", "in", "simple", "words", "difference", "between"}]
    
    definition_sentences = []
    elaboration_sentences = []
    application_sentences = []

    for s in all_sentences:
        s_lower = s.lower()
        if any(term in s_lower for term in [" is a ", " is an ", " are a ", " refers to ", " is defined as ", " focuses on "]):
            # Check if it mentions at least one keyword
            if any(kw in s_lower for kw in q_keywords):
                definition_sentences.append(s)
            else:
                elaboration_sentences.append(s)
        elif any(term in s_lower for term in ["applied to", "applications", "used in", "such as", "including", "examples"]):
            application_sentences.append(s)
        else:
            elaboration_sentences.append(s)

    # 3. Assemble structured answer
    parts = []
    
    # Primary definition
    if definition_sentences:
        primary_def = definition_sentences[0]
        # Clean any trailing punctuation or artifacts
        primary_def = primary_def.strip()
        parts.append(primary_def)
    elif all_sentences:
        parts.append(all_sentences[0])

    # Key details / elaboration
    added = set()
    elabs = []
    for s in (definition_sentences[1:] + elaboration_sentences):
        if len(elabs) >= 2:
            break
        if s not in parts and s not in added:
            added.add(s)
            elabs.append(s)
    if elabs:
        parts.append(" ".join(elabs))

    # Applications / Examples section if applicable
    if "application" in q_lower or "example" in q_lower:
        apps = []
        for s in application_sentences:
            if len(apps) >= 2:
                break
            if s not in parts and s not in added:
                added.add(s)
                apps.append(s)
        if apps:
            parts.append("Applications and Use Cases:\n" + "\n".join(f"- {a}" for a in apps))

    # Difference / comparison queries
    if "difference" in q_lower or "between" in q_lower:
        comp_items = []
        for item in bundle.items[:3]:
            lead = item.snippet
            if item.full_text:
                first_sent = [s.strip() for s in item.full_text.split(".") if len(s.strip()) > 20]
                if first_sent:
                    lead = first_sent[0]
            comp_items.append(f"- **{item.title}**: {lead}")
        if comp_items:
            parts.append("Key Distinctions:\n" + "\n".join(comp_items))

    final_text = "\n\n".join(parts)
    return final_text


queries_to_test = [
    "What is deep learning?",
    "What is ChatGPT?",
    "Explain the difference between AI, machine learning, and deep learning.",
    "Give three real-world applications of deep learning."
]

for q in queries_to_test:
    b = engine.search_and_retrieve(q, max_results=3, fetch_pages=True)
    ans = synthesize_grounded_answer(q, b)
    print("=" * 70)
    print("QUESTION:", q)
    print("SYNTHESIZED GROUNDED ANSWER:\n", ans)
    print("\nSOURCES:")
    for i, it in enumerate(b.items, 1):
        print(f"[{i}] {it.title}: {it.url}")
