# PROJECT CLEANUP, REPAIR & VERIFICATION REPORT
**Local AI Companion Architecture & Maintenance Report**  
**Project Directory:** `D:\DEEP-LEARNING-PROJECT`  
**Date:** November 20, 2026  
**Status:** COMPLETED & VERIFIED (12/12 System Checks Passing)

---

## 1. Qwen HTTP 400 Bad Request Crash — Root Cause & Permanent Repair

### 1.1 The Incident
During the user conversation:
```
USER: name of current cm of punjab india
```
The application encountered an uncaught runtime exception terminating the chatbot session:
```
urllib.error.HTTPError: HTTP Error 400: Bad Request
  File "src/chat.py", line 1951, in <module>
  File "src/chat.py", line 1932, in interactive_chat
  File "src/chat.py", line 994, in generate_chat_response
  File "src/chat.py", line 851, in generate_external_model_response
  File "src/model_manager.py", line 383, in generate
    urllib.request.urlopen(req, timeout=45)
```

### 1.2 Actual Root Cause
1. **Context Window Constraint:** The background `llama-server.exe` was launched with a default context size of `-c 2048`.
2. **Context Overflow:** When full web pages were retrieved by the web search engine, the unbudgeted evidence string grew to 17,467 characters (3,886 tokens).
3. **Server Rejection:** `llama.cpp` server strictly rejected the payload with `HTTP 400 Bad Request: request (3886 tokens) exceeds the available context size (2048 tokens)`.
4. **Unhandled Exception:** The HTTP request had no fallback/catch logic for context overflow, which allowed the exception to bubble up and crash the interactive chat process.

### 1.3 Permanent Solution Implemented
1. **Context Size Doubled:** Updated `src/model_manager.py` to start `llama-server.exe` with `-c 4096`, doubling the available context window.
2. **Strict Character & Token Budgeting:**
   - In `src/chat.py`, bounded per-source evidence to 1,000 characters and total `evidence_str` to $\le 3,000$ characters.
   - In `src/model_manager.py`, added safety bounding: `context_evidence[:3000]` and conversational history capped to the last 4 turns (max 400 chars each).
3. **Graceful Exception Recovery & Auto-Truncation:**
   - Wrapped `urllib.request.urlopen` in `ExternalModelEngine.generate()` with detailed exception handling for `HTTPError`, `URLError`, `TimeoutError`, and `JSONDecodeError`.
   - On an HTTP 400 context overflow, the engine automatically catches the error, compresses `context_evidence` to a compact 1,200 characters, and retries the request safely.
4. **Crash-Proof `safe_engine_generate` Wrapper:**
   - In `src/chat.py`, all calls to `external_engine.generate` are routed through `safe_engine_generate()`. If the server fails or throws an uncaught error, a polite fallback message is returned and the chatbot session remains alive.

---

## 2. RobustContentExtractor HTML Parser Bug & Fix

### 2.1 The Issue
During testing of the Punjab CM query, Wikipedia URLs (e.g., `https://en.wikipedia.org/wiki/Bhagwant_Mann`) returned only 68 characters of metadata ("Bhagwant Mann - Wikipedia Indian politician and comedian (born 1973)") instead of the article content, completely omitting his tenure as Chief Minister.

### 2.2 Root Causes Identified
1. **Root Tag Regex Match:** Wikipedia's Vector 2022 skin contains `vector-feature-language-in-main-menu-disabled` on the root `<html>` tag. Because `menu` matched `boilerplate_re`, `self.ignored_depth` was incremented to 1 at `<html>`, causing the parser to ignore the entire document body.
2. **Void Tag Depth Desynchronization:** HTML void tags (`<img>`, `<input>`, `<br>`, `<hr>`, `<meta>`, `<link>`) do not have closing tags in HTML5. When `handle_starttag` encountered a void tag inside an ignored block, it incremented `self.ignored_depth += 1`. Because `handle_endtag` is never called for void tags, `self.ignored_depth` was permanently corrupted and could never return to 0.

### 2.3 The Solution
1. **Exempted Root Containers:** `html` and `body` tags are explicitly exempted from boilerplate checks.
2. **Void Tag Handling:** Defined `VOID_HTML_TAGS`. Void tags never increment `ignored_depth` and never get pushed to the tag stack.
3. **Snippet & Full-Text Fusion:** In `src/chat.py`, if the search engine snippet contains unique facts not present in the extracted lead paragraph, both the search snippet and the article text are preserved.

---

## 3. Removal of Function Tool (Calculator)

### 3.1 Complete Deletion of Tool Overhead
As instructed, all function-calling and tool-assisted mechanisms have been eradicated from the chatbot:
- Removed `/tool on` and `/tool off` commands from `src/chat.py` and `help.bat`.
- Removed `from evaluate_tool_assisted import safe_calculator` from `src/chat.py`.
- Removed regex tool interceptors in `src/chat.py`, `src/intent_router.py`, and `src/question_understanding.py`.
- Eliminated `[Tool: Calculator]` prefixes.

### 3.2 Preservation of Pure Neural Mathematical Reasoning
Mathematical queries are now classified as `intent=INTENT_EDUCATIONAL` and answered through natural language generation by the active model:
- **Calculus:** Qwen explains derivatives using the quotient rule ($\frac{d}{dx}[\frac{\sin x}{\cos x}] = \sec^2 x$).
- **Arithmetic:** Qwen calculates $45 \times 12 = 540$ with step-by-step arithmetic.
- **Percentages:** Qwen calculates $15\%$ of $800 = 120$.
- **Word Problems:** Qwen calculates monthly and annual savings ($50,000 \times 20\% = 10,000$; $10,000 \times 12 = 120,000$).
- **Custom Model:** Generates conversational math or gracefully admits offline boundaries.

---

## 4. Search Retrieval Quality & Disambiguation

### 4.1 Geographic Disambiguation (Punjab India vs Pakistan)
- When queries contain `india` or `indian`, a heavy penalty (-800.0) is applied to results mentioning `pakistan` without `india`.
- Incumbent boosting (+400.0) is applied to officeholder pages containing `"currently serving"`, `"incumbent"`, or `"assumed office"`.
- Provider search limit expanded to `max(8, max_results * 2)` to prevent shallow cutoffs.
- **Result:** Query `"name of current cm of punjab india"` unequivocally returns **Bhagwant Mann** (serving since March 16, 2022) with zero confusion with Pakistan (Maryam Nawaz).

### 4.2 Cross-Topic Anti-Contamination
- **Biomedical / Cancer Queries:** Enforced filtering that blocks academic AI-agent papers from invading cancer biology queries.
- **Financial / Stock Queries:** Enforced filtering that returns live market quotes and ticker data rather than Python programming snippets.
- **Educational Concepts:** Safeguarded machine learning and deep learning queries from generic artificial intelligence article usurpation.

---

## 5. Project Directory Audit & Cleanup

### 5.1 Safe File Deletions (Zero Collateral Damage)
Audited all files across `D:\DEEP-LEARNING-PROJECT`. Only verified 0-byte temporary test files were removed:
- `D:\DEEP-LEARNING-PROJECT\test_flamingo.tmp` (0 bytes) $\rightarrow$ DELETED
- `D:\DEEP-LEARNING-PROJECT\MAX_STEPS` (0 bytes) $\rightarrow$ DELETED

### 5.2 Preserved Critical Assets
- **Checkpoints:** `checkpoints/ultimate_improvement/best.pt` preserved intact (SHA-256: `CECBA4946A8B5A0404E42E9E1790C0D780228DA45E07B1910D190D89CBADF2FC`).
- **Database:** `data/companion_project.db` (SQLite integrity: `ok`, 10 tables, 84 datasets).
- **External Model:** `models/external/qwen2.5-1.5b-instruct/model.gguf` (940.3 MB).
- **Runtime & Environment:** `runtime/llama_cpp/llama-server.exe`, `.venv/`, and `Python312/`.
- **Tokenizers & Configs:** `tokenizer/` and `configs/`.

---

## 6. Comprehensive Verification Results (12/12 PASS)

| Test ID | Test Name | Target Module | Result | Details |
| :---: | :--- | :--- | :---: | :--- |
| **01** | `checkpoint_sha256` | `best.pt` | **PASS** | Matches expected SHA-256 hash |
| **02** | `database_integrity` | SQLite DB | **PASS** | `PRAGMA integrity_check` = `ok` |
| **03** | `model_manager_init` | `ModelManager` | **PASS** | Starts clean, reports CPU/GPU specs |
| **04** | `qwen_offline_calculus` | Neural Math | **PASS** | Quotient rule $\sec^2(x)$, zero tools |
| **05** | `qwen_offline_arithmetic` | Neural Math | **PASS** | $45 \times 12 = 540$, zero tools |
| **06** | `qwen_web_punjab_cm` | Web Grounding | **PASS** | Returns Bhagwant Mann, no Pakistan |
| **07** | `cancer_disambiguation` | Web Retrieval | **PASS** | AACR / Breast cancer; 0 AI papers |
| **08** | `stock_retrieval` | Web Retrieval | **PASS** | Apple NASDAQ AAPL market quotes |
| **09** | `model_switch_custom` | Command Router | **PASS** | `/model custom` switches cleanly |
| **10** | `custom_model_conversational`| Custom PyTorch| **PASS** | Empathy & offline honesty preserved |
| **11** | `model_switch_external`| Command Router | **PASS** | `/model external` switches cleanly |
| **12** | `crash_resilience` | Exception Guard | **PASS** | HTTP 400 error handled gracefully |
