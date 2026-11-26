# WEB ROUTING AUDIT: INTENT CLASSIFICATION & DECISION POLICY SPECIFICATION

**Project Root:** `D:\DEEP-LEARNING-PROJECT`  
**Module Audited:** `src\web_retrieval.py::WebRetrievalEngine.should_search_web`  
**Date:** 2026-10-27

---

## 1. Audit of the Previous Routing Implementation

The legacy routing function `should_search_web(user_prompt, web_mode_enabled)` used a narrow heuristic structure:

```python
# Legacy Implementation (Defective)
def should_search_web(self, user_prompt: str, web_mode_enabled: bool = True) -> Tuple[bool, str]:
    if not web_mode_enabled:
        return False, "Web search disabled by configuration."

    p_lower = user_prompt.lower().strip()

    # 1. Explicit search commands
    if ("/search" in p_lower or "search the web" in p_lower or ...):
        return True, "Explicit user search request."

    # 2. Direct URL present
    if "http://" in p_lower or "https://" in p_lower:
        return True, "Contains explicit web URL to retrieve."

    # 3. Real-time / version-specific keywords
    temporal_keywords = ["latest release", "current version", "today", "yesterday", "weather", ...]
    if any(kw in p_lower for kw in temporal_keywords):
        return True, "Contains time-sensitive or version-specific indicators."

    # 4. Factual / informational keywords
    informational_keywords = ["what is the latest", "who is the current", "summarize this webpage", ...]
    if any(kw in p_lower for kw in informational_keywords):
        return True, "Informational query requiring external evidence."

    return False, "Standard conversational query handled locally."
```

### Critical Flaws in the Legacy Decision Boundary
1. **Omission of Core Question Stems:** It tested for `"what is the latest"`, but completely ignored standard inquiries such as `"what is"`, `"what are"`, `"who is"`, `"define"`, `"explain"`, `"how does"`, or `"why is"`.
2. **Ignorance of Domain Entities:** Queries naming major technical subjects (`"deep learning"`, `"chatgpt"`, `"neural network"`, `"machine learning"`) failed all keyword filters and dropped through to the local fallback.
3. **Overconfidence in Small Local Model:** It assumed that any query not explicitly demanding "latest" or "weather" could be answered accurately by a 14.1M parameter model, which is fundamentally incapable of memorizing broad factual knowledge.

---

## 2. The Corrected Decision Policy & Taxonomy

To ensure reliable, predictable, and verifiable routing, the routing logic is structured as an explicit multi-tier intent classifier:

```
                          [Incoming User Prompt]
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        Web Search Disabled? (/web off)       Direct Math Formula?
         ──► ROUTE: LOCAL MODEL               ──► ROUTE: SAFE CALCULATOR
                  │
                  ▼
       Follow-up on Cached Search?
       ("what did the second website say?")
         ──► ROUTE: SEARCH CACHE HANDLER
                  │
                  ▼
       Pure Greeting / Chit-Chat?
       ("hello", "how are you", "who are you", "thanks")
         ──► ROUTE: LOCAL MODEL (Fast Conversational Dialogue)
                  │
                  ▼
       Default Knowledge-Seeking Inquiries
       ("what is", "explain", "deep learning", "chatgpt", "why", "how", etc.)
         ──► ROUTE: LIVE WEB RETRIEVAL (Verified Evidence Grounding)
```

### Tier 1: System Controls & Math Tools
* **Web Toggle (`/web off`):** When the user disables web retrieval, all routing immediately directs to local processing with transparent quality disclaimers.
* **Deterministic Calculations:** Arithmetic formulas (e.g. `"What is 45 * 12?"`, `"compute 1024 / 8"`) are intercepted by `safe_calculator` with 100% precision and zero latency.

### Tier 2: Conversational Cache Resolution
* Follow-ups referencing previous searches (`"what did the second source say"`, `"summarize the first website"`, `"compare source 1 and source 2"`) bypass re-searching and immediately load the cached evidence bundle from `retrieval_engine.last_evidence`.

### Tier 3: Pure Social Dialogue & Chit-Chat (Local Model)
Questions matching standard conversational greetings and identity questions are routed directly to the local model:
* **Greetings:** `"hi"`, `"hello"`, `"hey"`, `"good morning"`, `"greetings"`, `"how are you"`, `"what's up"`.
* **Bot Persona:** `"who are you"`, `"what is your name"`, `"what can you do"`, `"help me"`.
* **Politeness & Closures:** `"thank you"`, `"thanks"`, `"goodbye"`, `"bye"`, `"see you"`.
* **Creative Prompts:** `"tell me a joke"`, `"say something nice"`, `"write a short poem"`.

### Tier 4: Knowledge-Seeking, Educational, and Factual Queries (Live Web Retrieval)
When `/web on` is active (the default), any query seeking definitions, explanations, scientific principles, technology overviews, or real-world facts is automatically routed to live web search:
* **Question Patterns:** `"what is"`, `"what are"`, `"who is"`, `"who was"`, `"where is"`, `"when was"`, `"which"`, `"why"`, `"how does"`, `"how do"`, `"how is"`, `"define"`, `"definition"`, `"explain"`, `"tell me about"`, `"describe"`, `"difference between"`, `"compare"`, `"applications of"`, `"examples of"`, `"history of"`, `"pros and cons"`.
* **Technical & Scientific Entities:** `"deep learning"`, `"chatgpt"`, `"machine learning"`, `"neural network"`, `"supervised"`, `"unsupervised"`, `"reinforcement learning"`, `"transformers"`, `"python"`, `"pytorch"`, `"artificial intelligence"`, `"photosynthesis"`, `"gravity"`, `"algorithm"`.
* **Direct URLs:** Any input containing `http://` or `https://`.
* **Explicit Search Directives:** `/search`, `search the web`, `look up`, `find info`.

---

## 3. Decision Trace Examples

| Query | Classified Intent | Routing Target | Reason Logged |
| :--- | :--- | :--- | :--- |
| `"Hello! How are you doing today?"` | Social Greeting | `neural_model` | Standard greeting or conversational chit-chat handled locally. |
| `"What is 45 * 12?"` | Math Calculation | `calculator` | Deterministic arithmetic expression. |
| `"What is deep learning?"` | Educational Definition | `web_retrieval` | Knowledge-seeking or educational query routed to web search for verified evidence. |
| `"What is ChatGPT?"` | Technology Overview | `web_retrieval` | Knowledge-seeking or educational query routed to web search for verified evidence. |
| `"Explain the difference between AI and ML"` | Concept Comparison | `web_retrieval` | Knowledge-seeking or educational query routed to web search for verified evidence. |
| `"What did the second website say?"` | Multi-turn Follow-up | `web_cache_followup` | Follow-up referencing cached search results. |
| `"What is deep learning?"` *(with `/web off`)* | User Overridden | `neural_model` | Web search disabled by configuration (/web off). |
