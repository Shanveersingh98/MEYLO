import urllib.request, urllib.parse, json, re

headers = {'User-Agent': 'DeepLearningEducationalBot/1.0 (https://github.com/example/dl-project; contact: student@example.edu) Python-urllib/3.11'}

def extract_concept_query(user_prompt: str) -> str:
    q = user_prompt.strip()
    # Remove commands: /search, search, look up
    q = re.sub(r"(?i)^(/search|search\s+for|search\s+the\s+web\s+for|search\s+the\s+web|search|look\s+up|please\s+search\s+for|retrieve\s+and\s+summarize)\s*", "", q)
    # Remove question openers
    q = re.sub(r"(?i)^(can\s+you\s+please|can\s+you|could\s+you|please)\s+", "", q)
    q = re.sub(r"(?i)^(summarize\s+an\s+accessible\s+public\s+educational\s+source\s+about|summarize\s+a\s+public\s+educational\s+source\s+about|summarize\s+an\s+educational\s+source\s+about|summarize)\s+", "", q)
    q = re.sub(r"(?i)^(explain\s+the\s+difference\s+between|difference\s+between|differences\s+between)\s+", "difference between ", q)
    q = re.sub(r"(?i)^(explain|define|describe|tell\s+me\s+about|what\s+is\s+the\s+definition\s+of|what\s+is|what\s+are|who\s+is|who\s+was|where\s+is|when\s+was)\s+", "", q)
    q = re.sub(r"(?i)^(give\s+three\s+real-world\s+applications\s+of|give\s+3\s+real-world\s+applications\s+of|give\s+real-world\s+applications\s+of|give\s+three\s+applications\s+of|give\s+3\s+applications\s+of|give\s+applications\s+of)\s+", "applications of ", q)
    q = re.sub(r"(?i)\b(in\s+simple\s+words|in\s+simple\s+terms|in\s+layman\'s\s+terms|for\s+beginners|simply\s+explained|step\s+by\s+step)\b", "", q)
    q = re.sub(r"(?i)^(a|an|the)\s+", "", q)
    q = re.sub(r"[\?\.!\"]+$", "", q).strip(" ?.!\"'")
    return q.strip()

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
    "What is ChatGPT?",
    "Explain the difference between supervised and self-supervised learning."
]

print("=== CONCEPT EXTRACTION TEST ===")
for q in test_queries:
    cq = extract_concept_query(q)
    print(f"{q:65} -> {cq!r}")
