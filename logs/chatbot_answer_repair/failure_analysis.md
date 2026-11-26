# FAILURE ANALYSIS: INVESTIGATION & REPRODUCTION OF CHATBOT DEFECTS

**Project Root:** `D:\DEEP-LEARNING-PROJECT`  
**Date:** 2026-10-27  
**Model:** `FromScratchLM` (14.1M parameters, checkpoint: `checkpoints/ultimate_improvement/best.pt`)  
**Environment:** Windows 11 | RTX 3050 Laptop GPU (6 GB VRAM) | PyTorch 2.14.0+cu130 (CUDA 13.0)

---

## 1. Reproduction of User-Reported Failures

The application was loaded on GPU and evaluated using the exact queries reported by the user.

### Case 1: "what is deep learning"
* **Routing Decision:** `should_search_web = False` (`"Standard conversational query handled locally."`)
* **Engine Used:** `neural_model` (standalone local Transformer)
* **Observed Response:**
  > *"Mach is a deep learning paradigm that involves the deep learning of machine learning and artificial neural networks. It is an important part of artificial neural network, which are trained on many tasks such as neural networks, speech recognition, natural language translation, and machine learning."*
* **Defects Confirmed:**
  1. Hallucinated token *"Mach"* as a subject noun.
  2. Circular tautology (*"deep learning paradigm that involves the deep learning of machine learning..."*).
  3. Fails to define the core concept: multilayered neural networks learning hierarchical feature representations.
  4. Misrouted: despite web retrieval being enabled (`/web on`), the question was classified as not requiring external evidence.

### Case 2: "what is chatgpt"
* **Routing Decision:** `should_search_web = False` (`"Standard conversational query handled locally."`)
* **Engine Used:** `neural_model`
* **Observed Response:**
  > *"Burgpt is a chat-source chat assistant that allows users to interact with the way, and respond to the chatbits of text."*
* **Defects Confirmed:**
  1. Invented non-existent words (*"Burgpt"*, *"chatbits"*).
  2. Factually devoid of meaning: does not mention OpenAI, GPT large language models, conversational fine-tuning, or generative AI.
  3. Misrouted: treated as an ordinary local conversation turn rather than an encyclopedic factual inquiry.

---

## 2. Reproduction of Explicit Search Commands (`/search`)

To investigate whether web retrieval was properly connected to the generation workflow, the explicit search commands were tested through the chat application interface.

### Case 3: `/search what is deep learning`
* **Routing Decision:** `should_search_web = True` (`"Explicit user search request."`)
* **Engine Used:** `web_retrieval` (latency: 5,691.6 ms)
* **Retrieved Sources:**
  1. *Deep learning*: `https://en.wikipedia.org/wiki/Deep_learning`
  2. *Deep reinforcement learning*: `https://en.wikipedia.org/wiki/Deep_reinforcement_learning`
  3. *Artificial intelligence*: `https://en.wikipedia.org/wiki/Artificial_intelligence`
* **Observed Response Produced by Model:**
  > *"No problem! Here are the steps you can use to learn from a machine learning model:*
  > 
  > *1. \*\*RUsearching a machine that is designed to be used to generate text data, such as image recognition, speech recognition, and text-based images.*
  > *2. \*\*Fromptimported:\*\* A deep learning algorithm that uses a variety of applications to perform tasks, including data analysis, and data structures.*
  > *3. \*\*MILAM) \*\*ALLMO\*\* - This technique involves training a wide range of techniques, including image processing, and image recognition.*
  > *4. \*\*DFECDM)\*\*: The choice between data is based on a specific type of text data. It"*
* **Defects Confirmed:**
  * While the web search provider successfully retrieved authoritative Wikipedia sources, **the local 14.1M parameter neural model completely broke down when attempting to ingest the retrieved context**.
  * The model output was corrupted with bizarre synthetic hallucinated tokens (*"RUsearching"*, *"Fromptimported"*, *"MILAM"*, *"ALLMO"*, *"DFECDM"*).
  * The system blindly presented this hallucinated text alongside genuine Wikipedia citations, directly violating truthfulness and grounding requirements.

### Case 4: `/search what is ChatGPT`
* **Routing Decision:** `should_search_web = True`
* **Engine Used:** `web_retrieval` (latency: 3,558.6 ms)
* **Retrieved Sources:**
  1. *ChatGPT*: `https://en.wikipedia.org/wiki/ChatGPT`
  2. *ChatGPT Atlas*: `https://en.wikipedia.org/wiki/ChatGPT_Atlas`
  3. *GPT-4*: `https://en.wikipedia.org/wiki/GPT-4`
* **Observed Response Produced by Model:**
  > *"Yes, here are the steps you can use to help you with?*
  > 
  > *Sources:*
  > *[1] ChatGPT: https://en.wikipedia.org/wiki/ChatGPT"*
* **Defects Confirmed:**
  * Empty, truncated non-answer.
  * The model failed to articulate any explanation of ChatGPT despite having retrieved the authoritative Wikipedia article.

---

## 3. Root Cause Analysis

Three distinct software and architectural root causes were proven:

### Root Cause 1: Overly Restrictive Decision Policy in `should_search_web`
The decision function in `src/web_retrieval.py` only triggered search on:
- Explicit directives (`/search`, `search the web`)
- Raw URLs (`http://`, `https://`)
- Temporal keywords (`"latest release"`, `"current version"`, `"today"`, `"yesterday"`)
- Narrow informational prefixes (`"what is the latest"`, `"who is the current"`)

It completely omitted foundational knowledge-seeking question patterns:
- `"what is"`, `"what are"`, `"who is"`, `"define"`, `"explain"`, `"difference between"`
- Technical and scientific terms: `"deep learning"`, `"chatgpt"`, `"neural network"`, `"supervised learning"`.

Consequently, queries like `"what is deep learning"` and `"what is chatgpt"` were unconditionally classified as `False` and routed directly to the local model.

### Root Cause 2: In-Context Prompt Distribution Mismatch on 14.1M LM
The `FromScratchLM` model has only 14,120,768 parameters ($d_{\text{model}}=384, 6\text{ layers}$). Its pretraining corpus consisted of dialogue turns, synthetic math pairs, and Python code. It was **never trained on reading comprehension or retrieval-augmented generation (RAG)** prompts containing hundreds of characters of raw text prepended before the question.

When raw scraped text was prepended inside `<|user|>`, the model's self-attention layers degraded, causing the model to emit noise tokens (*"RUsearching"*, *"Fromptimported"*, *"MILAM"*).

### Root Cause 3: Missing Output Quality Validator & Evidence Fallback
The chat application had no mechanism to evaluate whether a generated response was coherent, truthful, or grounded in the retrieved sources. It treated whatever tokens the neural model sampled as final, attaching real URLs even when the model's text was pure gibberish.

---

## 4. Remediation Plan

1. **Routing Overhaul:** Expand `should_search_web` to route all knowledge-seeking, educational, scientific, and technical questions to live web retrieval by default when `/web on` is active, reserving local routing strictly for greetings, personal chit-chat, and math tools.
2. **Authoritative Evidence Extraction:** Upgrade `WikipediaProvider` to use the official Wikimedia Lead Extract API (`prop=extracts&exintro=1&explaintext=1`), ensuring pristine, encyclopedic, zero-boilerplate definitions without HTML noise or sidebar navigation links.
3. **Evidence-Grounded Answer Synthesis:** Implement `synthesize_grounded_answer()` to deterministically produce clean, structured, and factual answers directly from verified sources.
4. **Response Quality Validator:** Implement `validate_response_quality()` in `chat.py`. If neural generation contains hallucinated tokens, circular tautologies, or factual inversions, reject it immediately and present the verified evidence synthesis labeled as `web_grounded_evidence`.
