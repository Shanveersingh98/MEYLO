# Technical Audit: Universal Question Answering & Intelligent Web Retrieval
**Project**: From-Scratch Deep-Learning Chatbot (`D:\DEEP-LEARNING-PROJECT`)  
**Date**: October 2026  
**Environment**: Python 3.12.10 (AMD64, Windows 11), PyTorch 2.14.0+cu130, 14.1M parameter Transformer model (`checkpoints/ultimate_improvement/best.pt`)

---

## 1. Executive Summary & Diagnostic Findings

Recent real-world evaluation exposed two critical failures in the from-scratch chatbot:
1. **Failure A (Public Profile Search)**:
   - *Query*: `"tell me who is the shanveer singh khosa on the linkedin"`
   - *Symptom*: Produced an educational concept failure message: `[Web Search Notice: Could not locate verified educational sources specifically defining 'tell me who is the shanveer singh khosa on the linkedin'. Exact concept verification was unavailable.]`
2. **Failure B (Health & Urgent Symptom Query)**:
   - *Query*: `"i have pain in my heart"`
   - *Symptom*: Dumped webpage navigation and category links from a scraped chest-pain article, with inadequate urgent medical safety guidance.

### Root-Cause Analysis

| Component | Verified Defect | Underlying Technical Cause |
| :--- | :--- | :--- |
| **Intent Routing** | Treated profile query as educational concept | Unrecognized conversational phrasing was defaulted to `INTENT_EDUCATIONAL` with the full verbatim prompt passed to Wikipedia exact-title search. |
| **Search Provider** | Network timeout on DuckDuckGo | DuckDuckGo (`html.duckduckgo.com`, `lite.duckduckgo.com`, `api.duckduckgo.com`) resolves to Azure IPs (`20.204.244.192`) which time out at the TCP socket layer on this network/ISP. The application had no working general-entity fallback provider. |
| **Content Extraction** | Scraped navigation menus & category links | Regex `<div id="content">(.*?)</div>` is non-greedy and terminates at the first nested `</div>` inside `<div class="breadcrumbs">...</div>` or `<div class="category">...</div>`, dropping the actual article body and preserving only navigation/breadcrumbs. |
| **LinkedIn Handling** | Zero-length extraction on profile URLs | Stripped `<head>` and `<script>`, losing all metadata (`<meta name="description">`, `og:description`, JSON-LD schema) which LinkedIn uses to deliver public profile content. |
| **Health & Safety** | Delayed triage & raw excerpt dumping | Urgent symptom reports did not prioritize immediate, concise triage guidance, and lacked a structured clinical overview with explicit red-flag warnings and non-diagnostic disclaimers. |

---

## 2. Reproduction Evidence

### Reproduction Case A: Public Profile Query
- **Prompt**: `"tell me who is the shanveer singh khosa on the linkedin"`
- **Observed Route**: In prior testing, routed to Wikipedia concept resolver with query `"tell me who is the shanveer singh khosa on the linkedin"`.
- **Observed Result**: Failed exact Wikipedia title search; generated educational error notice.
- **Provider Status**: DuckDuckGo TCP handshake timed out after 10 seconds.

### Reproduction Case B: Acute Symptom Query
- **Prompt**: `"i have pain in my heart"`
- **Observed Issue**: HTML extraction on Healthline/article sites truncated at the first inner `</div>`, returning breadcrumb navigation text: `"Health > Heart Disease > Categories"`.
- **Urgent Triage**: Did not immediately present emergency numbers and danger signs before search execution.

---

## 3. Targeted Architecture & Repair Strategy

1. **Robust Multi-Provider Web Retrieval Subsystem**:
   - Integrate **Wikidata Entity API** (`wbsearchentities`): Resolves public figures, executives, researchers, companies, and organizations in < 0.7s with zero API keys.
   - Integrate **Wikipedia OpenSearch & Fulltext Extract API**: Retrieves authoritative encyclopedic articles with clean lead extracts in < 0.8s.
   - Integrate **PubMed / NCBI Clinical Search API**: Retrieves authoritative medical and health literature for clinical topics in < 1.5s.
   - Integrate **OpenAlex Academic API**: Resolves researchers, authors, and scientific concepts in < 1.2s.
   - Preserve **Direct Public Webpage Retrieval** with full SSRF protection, strict timeouts, and byte limits.

2. **Universal Intent Understanding & Routing**:
   - 12 comprehensive intent classes:
     1. `general_conversation`
     2. `educational_concept`
     3. `person_profile`
     4. `current_events`
     5. `technical_programming`
     6. `health_safety` (`urgent_emergency` vs `general_health_info`)
     7. `product_service`
     8. `deterministic_tool`
     9. `document_summary`
     10. `comparison`
     11. `followup_reference`
     12. `ambiguous_unspecified`
   - Structured JSON logging to `logs/universal_question_answering/routing_runtime.log`.

3. **DOM-Aware Content Extraction (`RobustContentExtractor`)**:
   - Built with Python's standard library `html.parser.HTMLParser` using a stateful tag stack.
   - Completely skips `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, `<aside>`, `<noscript>`, and boilerplate classes (`breadcrumb`, `advert`, `newsletter`, `social`, `categories`).
   - Automatically harvests `<title>`, `<meta name="description">`, `og:description`, and schema.org JSON-LD blocks for JavaScript-rendered public profiles (e.g., LinkedIn).
   - Cleans raw LaTeX mathematical markup (`\displaystyle`, `\times`, `\mid`, `\frac`) into clean readable text.

4. **Public Profile Search Workflow**:
   - Normalizes conversational requests into targeted entity queries (`"Shanveer Singh Khosa LinkedIn"`, `"Shanveer Singh Khosa"`).
   - Searches across Wikidata, public profiles, and authoritative databases.
   - Honest uncertainty: if an individual cannot be verified from public sources, returns a helpful notice explaining potential privacy settings and requesting distinguishing details, rather than an educational error.
   - Identity ambiguity disclosure: identifies when multiple individuals share similar names.

5. **Safe, Structured Health Triage**:
   - Immediate triage advice displayed first for acute symptom reports (`"i have pain in my heart"`).
   - Directs user to local emergency services without delaying for network calls.
   - Clear distinction between urgent symptoms and informational queries.
   - Includes red-flag signs, structured potential causes, and non-diagnostic disclaimers.
