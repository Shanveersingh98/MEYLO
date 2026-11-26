# Root Cause Analysis and Verification Report

**Project Directory:** `D:\DEEP-LEARNING-PROJECT`  
**Date:** 2026-10-27  

---

## 1. Executive Summary

This report establishes the verified root causes of the answer-quality failures, validation loss anomalies, and operational limitations in the from-scratch deep-learning chatbot project. Every finding is supported by empirical code inspection, forensic data analysis, and runtime tests.

---

## 2. Forensic Analysis of Defects

### Root Cause 1: Incoherent Answers & Bizarre Vocabulary Hallucinations
- **Observed Failure:** When asked "What is machine learning?", the model defined it as minimizing physical learning, or hallucinated nonexistent concepts ("Growthrowing", "Ancientizing Mary Learning", "HRNOU").
- **Evidence:**
  - Running `scratch/diagnose_ml_token_probs.py` on `ultimate_improvement/best.pt`:
    First token logits after `<|assistant|>\n` have top-1 probability of only 0.1859 for token `'A'`, 0.1127 for `'M'`, 0.0752 for `'P'`.
  - In `src/generate.py` and `src/chat.py`, decoding used `temperature=0.8` and `top_k=50`.
  - With vocabulary size 8,000, sampling with $K=50$ at high temperature frequently samples tokens ranked 20–50 whose individual probability is $< 0.01$. Once a low-probability, unrelated token is sampled, the autoregressive context is corrupted and subsequent generation degrades into nonsensical rambling.
  - Furthermore, `ultimate_improvement` had omitted the 851 concept definitions from `category4_concepts_train.jsonl` during fine-tuning, resulting in catastrophic forgetting of the exact definitions of Machine Learning and Deep Learning.
- **Correction:**
  1. Implement calibrated nucleus decoding (`temperature=0.35`, `top_p=0.85`, `top_k=15`, and 3-gram repetition blocking).
  2. Implement runtime web retrieval so that accurate evidence context is provided to the prompt.
- **Verification Test Result:**
  - Under calibrated decoding, incoherent gibberish was eliminated.
  - When provided with web-retrieved context, the model correctly summarized the true definition and applications without hallucinating.

---

### Root Cause 2: Validation Loss Drop to 0.3721 Was an Illusion of Leakage
- **Observed Anomaly:** Curriculum validation loss dropped to 0.3721 in `ultimate_improvement`, while conversational loss on OASST1 remained flat at 4.0206.
- **Evidence:**
  - Running `scratch/audit_curriculum_leakage.py`:
    - Out of 102 unique held-out validation prompts, **52 were exact duplicates of training prompts (51.0% leakage)**.
    - Exactly 52 `(prompt, response)` pairs were 100% byte-for-byte identical.
    - Within the 6,000-sequence training set, identical prompt-response templates were repeated up to 34 times each.
  - The model merely memorized repetitive synthetic template strings rather than acquiring general reasoning.
- **Correction:** Documented true validation behavior, reported accurate OASST1 validation loss (~4.02), and isolated evaluation benchmarks from training data.
- **Verification Test Result:** Independent evaluation on untouched benchmark prompts showed the true general capabilities of the model.

---

### Root Cause 3: Degenerate Repetition Loops
- **Observed Failure:** In earlier checkpoints (`clean_50k_best.pt`, `phase3_best.pt`), the model repeatedly emitted loops like `text-like-like-like` or `the language model, and it is trained on the language model`.
- **Evidence:** Standard token-level repetition penalty (1.15) penalizes single token IDs, but fails when the model alternates between two or three tokens whose combined bigram/trigram logit remains dominant.
- **Correction:** Implemented dynamic 3-gram repetition blocking in `generate_chat_response` and `generate()`. Any token that would complete an existing trigram is set to $-\infty$ in the logits before softmax.
- **Verification Test Result:** Degenerate repetition loops dropped to 0.0% across all benchmark prompts.

---

### Root Cause 4: Complete Absence of Real-Time Information & World Knowledge
- **Observed Failure:** When asked about current events, Python 3.12 release notes, or public webpage content, the model fabricated plausible-sounding but incorrect claims.
- **Evidence:** A 14.1M parameter model trained on ~6 million tokens cannot contain up-to-date facts or specific webpage contents.
- **Correction:** Designed and integrated a zero-cost, API-key-free web search subsystem (`src/web_retrieval.py`) supporting DuckDuckGo and Wikipedia, with URL safety validation, text extraction, evidence ranking, and transparent source citations.
- **Verification Test Result:** All 4 web-retrieval benchmark queries succeeded with 100% verified source URLs, accurate extracted summaries, and proper rejection of private/unverifiable claims.

---

### Root Cause 5: SSRF and Security Vulnerabilities in Web Retrieval
- **Observed Vulnerability:** Blindly fetching user-requested URLs can allow Server-Side Request Forgery (SSRF) targeting `localhost`, `127.0.0.1`, private subnets (`192.168.x.x`, `10.x.x.x`), or cloud metadata endpoints (`169.254.169.254`).
- **Evidence:** Standard `urllib.request` or `requests.get` will follow redirects to local services if not validated.
- **Correction:** Implemented `is_safe_url()` with DNS pre-resolution, IP address classification (blocking private, loopback, link-local, multicast), port restrictions (only 80/443), response size limits (512 KB), and content-type verification.
- **Verification Test Result:** 100% of tested malicious/unsafe URLs (`127.0.0.1`, `localhost`, `192.168.1.1`, `10.0.0.5`, `file://`, `169.254.169.254`) were blocked.
