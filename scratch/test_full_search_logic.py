import re
import urllib.request
import urllib.parse
import json

headers = {'User-Agent': 'DeepLearningEducationalBot/1.0 (https://github.com/example/dl-project; contact: student@example.edu) Python-urllib/3.11'}

def clean_concept(q: str) -> str:
    # 1. Strip explicit command prefixes
    q = re.sub(r"(?i)^(/search|search\s+for|search\s+the\s+web\s+for|search\s+the\s+web|search|look\s+up|please\s+search\s+for|retrieve\s+and\s+summarize)\s*", "", q.strip())
    # 2. Strip polite framing
    q = re.sub(r"(?i)^(can\s+you\s+please|can\s+you|could\s+you|please)\s+", "", q)
    q = re.sub(r"(?i)^(summarize\s+an\s+accessible\s+public\s+educational\s+source\s+about|summarize\s+a\s+public\s+educational\s+source\s+about|summarize\s+an\s+educational\s+source\s+about|summarize)\s+", "", q)
    q = re.sub(r"(?i)^(explain\s+the\s+difference\s+between|difference\s+between|differences\s+between)\s+", "difference between ", q)
    q = re.sub(r"(?i)^(explain|define|describe|tell\s+me\s+about|what\s+is\s+the\s+definition\s+of|what\s+is|what\s+are|who\s+is|who\s+was|where\s+is|when\s+was)\s+", "", q)
    q = re.sub(r"(?i)^(give\s+three\s+real-world\s+applications\s+of|give\s+3\s+real-world\s+applications\s+of|give\s+real-world\s+applications\s+of|give\s+three\s+applications\s+of|give\s+3\s+applications\s+of|give\s+applications\s+of)\s+", "applications of ", q)
    q = re.sub(r"(?i)\b(in\s+simple\s+words|in\s+simple\s+terms|in\s+layman\'s\s+terms|for\s+beginners|simply\s+explained|step\s+by\s+step)\b", "", q)
    q = re.sub(r"(?i)^(a|an|the)\s+", "", q)
    q = re.sub(r"[\?\.!\"]+$", "", q).strip(" ?.!\"'")
    return q.strip()

def check_concept_match(target_concept: str, title: str, snippet: str = "") -> float:
    t_low = title.strip().lower()
    c_low = target_concept.strip().lower()

    if t_low == c_low:
        return 1000.0

    t_clean = re.sub(r"\s*\([^)]*\)", "", t_low).strip()
    if t_clean == c_low:
        return 900.0

    # Check for redirect / alias mentioned in title or lead snippet
    if c_low in t_low or (snippet and c_low in snippet[:150].lower()):
        # e.g. "Weak supervision (also known as semi-supervised learning)"
        return 850.0

    ml_modifiers = ["self-", "self ", "semi-", "semi ", "un-", "unsupervised", "weak ", "weak-", "reinforcement"]
    
    target_has_mod = set()
    for mod in ml_modifiers:
        if mod in c_low:
            target_has_mod.add("un" if mod == "unsupervised" else mod.strip(" -"))
            
    title_has_mod = set()
    for mod in ml_modifiers:
        if mod in t_low:
            title_has_mod.add("un" if mod == "unsupervised" else mod.strip(" -"))

    unwanted_mods = title_has_mod - target_has_mod
    if unwanted_mods:
        return -200.0

    missing_mods = target_has_mod - title_has_mod
    if missing_mods:
        return -200.0

    if re.search(r"\b" + re.escape(c_low) + r"\b", t_low):
        return 300.0

    return 0.0

def fetch_wikipedia_concept(concept: str):
    t_quoted = urllib.parse.quote(concept.strip())
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles={t_quoted}&redirects=1&format=json"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            d = json.loads(resp.read().decode())
            for pid, p in d.get("query", {}).get("pages", {}).items():
                if "missing" not in p and p.get("title") and p.get("extract"):
                    return {
                        "title": p["title"],
                        "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(p['title'].replace(' ', '_'))}",
                        "snippet": p["extract"][:300].strip(),
                        "full_text": p["extract"].strip()
                    }
    except Exception:
        pass
    return None

test_queries = [
    "What is supervised learning?",
    "/search What is supervised learning?",
    "What is self-supervised learning?",
    "What is unsupervised learning?",
    "What is semi-supervised learning?",
    "What is reinforcement learning?",
    "What is machine learning?",
    "What is deep learning?",
    "What is a neural network?",
    "What is ChatGPT?"
]

print("=== VERIFYING RE-RANKED RESULTS ===")
for q in test_queries:
    concept = clean_concept(q)
    exact_res = fetch_wikipedia_concept(concept)
    if exact_res:
        score = check_concept_match(concept, exact_res["title"], exact_res["snippet"])
        print(f"Query: {q:38} -> Extracted: {concept:25} -> Primary: {exact_res['title']:30} [Score: {score}]")
    else:
        print(f"Query: {q:38} -> FAILED TO FETCH")
