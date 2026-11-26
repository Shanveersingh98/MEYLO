# Engineering Report: Web Search & Retrieval Integration

**Project Directory:** `D:\DEEP-LEARNING-PROJECT`  
**Integrated Component:** `src/web_retrieval.py`  
**Application Entry Point:** `src/chat.py`  
**Date:** 2026-10-27  

---

## 1. System Overview and Mandate

The primary limitation of any small local language model (here, `FromScratchLM` with 14.1M parameters) is that its parameter capacity cannot store real-time, rapidly changing, or dense encyclopedic world knowledge. 

To overcome this fundamental physical boundary without replacing the local model with third-party cloud APIs (such as OpenAI or Anthropic), we designed and built an **autonomous, zero-cost, API-key-free web search and retrieval subsystem** directly into the Python application.

---

## 2. Architecture & Provider Abstraction

The web retrieval subsystem is encapsulated in `src/web_retrieval.py` and implements an extensible provider abstraction:

```
[User Input Query]
        │
        ▼
[Decision Engine: should_search_web()]
   ├── If Local / Conversational  ──► [Pure Local Neural Generation]
   └── If Search / URL / Factual  ──► [WebRetrievalEngine]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [DuckDuckGoProvider]                         [WikipediaProvider]
             - Lite / HTML endpoint                       - OpenSearch API
             - No API key required                        - Verified encyclopedic data
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                             [SSRF & URL Safety Filter: is_safe_url()]
                             - Blocks localhost, loopback (127.0.0.1)
                             - Blocks private subnets (10.x, 192.168.x)
                             - Blocks cloud metadata (169.254.169.254)
                             - Validates allowed ports (80, 443)
                                             │
                                             ▼
                             [Secure HTTP Fetch & Text Extraction]
                             - Handles gzip / deflate decompression
                             - Strips scripts, styles, navigation, ads
                             - Enforces 512 KB response size limits
                                             │
                                             ▼
                             [Evidence Ranking & Prompt Packaging]
                             - Sanitizes prompt boundaries (<|user|>, [EOS])
                             - Injects [VERIFIED WEB EVIDENCE] context
                                             │
                                             ▼
                             [FromScratchLM Neural Answering]
                             - Grounded response generation
                             - Transparent source attribution links
```

---

## 3. Concrete Implementations & Providers

### A. DuckDuckGo Provider (`DuckDuckGoProvider`)
- **Endpoint:** `https://lite.duckduckgo.com/lite/`
- **Method:** HTTP POST with URL-encoded query.
- **Attributes:** Completely free, requires no API key, returns title, redirect URL, and snippet.
- **URL Resolution:** Automatically parses DDG redirect links (`//duckduckgo.com/l/?uddg=...`) to uncover the true destination website URL.

### B. Wikipedia OpenSearch Provider (`WikipediaProvider`)
- **Endpoint:** `https://en.wikipedia.org/w/api.php?action=opensearch`
- **Method:** HTTP GET.
- **Attributes:** Extremely reliable, high-availability public endpoint for definitional and historical concepts.

### C. Direct URL Fetching (`fetch_page_content`)
- When a user includes a full URL (e.g. `https://en.wikipedia.org/wiki/Artificial_intelligence`), the engine retrieves the page directly, decompressing gzip/deflate payloads, stripping HTML boilerplate, and summarizing the extracted readable text.

---

## 4. Prompt Construction and Evidence Framing

Untrusted web content is isolated from instruction streams to prevent prompt injection:

```
[BOS]
<|user|>
[VERIFIED WEB EVIDENCE (UNTRUSTED EXTERNAL DATA)]
Source [1] Title: {title}
URL: {url}
Content: {extracted_text}
[END OF WEB EVIDENCE]

Question: {user_prompt}
<|assistant|>
{model_response}

Sources:
[1] {title}: {url}
[EOS]
```

---

## 5. Follow-Up Resolution & Caching

The engine caches recent search evidence within `retrieval_engine.last_evidence`. 
When a user asks follow-up questions such as:
- *"What did the second website say?"*
- *"Summarize the first source."*
- *"Tell me more from source 2."*

The engine resolves the reference directly from the cached metadata without issuing a redundant or drifting web search.

---

## 6. Failure Handling & Fallback Behavior

- **Network Offline or Blocked:** If DNS resolution fails, connection times out, or the destination is blocked by SSRF rules, the application does not crash.
- **Graceful Notification:** It reports a clear notice: `[Web Search Notice: <reason>]` and smoothly falls back to local neural generation.
- **No Hallucinated Citations:** If search returns no results, the application explicitly states that no sources were found rather than fabricating fake URLs or dates.

---

## 7. Operational Controls in `src/chat.py`

Interactive commands available during runtime:
- `/web on` - Enable live web retrieval (Default)
- `/web off` - Disable web search (Pure offline model mode)
- `/tool on` / `/tool off` - Toggle deterministic math calculator
- `/search <query>` - Explicitly query the web for `<query>`
- `/clear` - Clear conversation history and retrieval cache
- `/exit` - Clean session shutdown
