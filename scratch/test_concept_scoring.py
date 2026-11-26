import re

def check_concept_match(target_concept: str, title: str) -> float:
    t_low = title.strip().lower()
    c_low = target_concept.strip().lower()

    if t_low == c_low:
        return 1000.0

    t_clean = re.sub(r"\s*\([^)]*\)", "", t_low).strip()
    if t_clean == c_low:
        return 900.0

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

test_cases = [
    ("supervised learning", ["Self-supervised learning", "Supervised learning", "Reinforcement learning", "Weak supervision", "Machine learning"]),
    ("self-supervised learning", ["Supervised learning", "Self-supervised learning", "Weak supervision", "Representation learning"]),
    ("unsupervised learning", ["Supervised learning", "Unsupervised learning", "Machine learning", "Representation learning"]),
    ("reinforcement learning", ["Reinforcement learning", "Supervised learning", "Deep reinforcement learning", "Machine learning"]),
    ("neural network", ["Neural network", "Neural network (machine learning)", "Convolutional neural network", "Biological neural network"]),
    ("deep learning", ["Deep learning", "Deep reinforcement learning", "Comparison of deep learning software", "Machine learning"])
]

print("=== CONCEPT DISCRIMINATION SCORING TEST ===")
for concept, candidates in test_cases:
    print(f"\nTarget Concept: '{concept}'")
    scored = [(cand, check_concept_match(concept, cand)) for cand in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    for rank, (cand, score) in enumerate(scored, 1):
        print(f"  Rank {rank}: {cand:35} -> Score: {score}")
