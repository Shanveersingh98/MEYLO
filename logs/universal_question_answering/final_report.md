# Final Engineering Report: Universal Question Answering & Intelligent Web Research

**Project**: From-Scratch Deep-Learning Chatbot (`D:\DEEP-LEARNING-PROJECT`)  
**Model Architecture**: 14.1M Parameter Custom Transformer Decoder  
**Model Checkpoint**: `checkpoints/ultimate_improvement/best.pt`  
**Runtime Environment**: Python 3.12.10 (AMD64, Windows 11), PyTorch 2.14.0+cu130 (CUDA Accelerated)  
**Date of Completion**: October 2026  
**Status**: All 6 Acceptance Suites Passed (69/69, 100.0%)

---

## 1. Executive Summary

This engineering effort successfully diagnosed, resolved, and verified the core failures in the from-scratch deep-learning chatbot project. Rather than treating all incoming user queries as narrow educational definitions or defaulting unclassified prompts into Wikipedia exact-title searches, the application now incorporates an extensible **12-Intent Routing Architecture**, a **Multi-Provider Web Retrieval Engine**, a **DOM-Aware Content Extractor**, an **Immediate Medical Safety Triage Protocol**, and a dedicated **Public Profile Resolution Workflow**.

All repairs were implemented without modifying the neural model weights or disrupting the existing tokenizer, model vocabulary, mathematical calculator, or checkpoint parameters. Every single requirement has been validated through rigorous automated tests against the actual running chatbot components.

### Acceptance Test Summary
```
======================================================================
ACCEPTANCE SUITE SUMMARY (logs/universal_question_answering/final_acceptance_results.json)
======================================================================
1. Intent Routing                  : 33 / 33 passed  (100.0%)
2. Public Profile Search           :  7 /  7 passed  (100.0%)
3. Health & Safety Triage          :  5 /  5 passed  (100.0%)
4. Content Extraction              :  5 /  5 passed  (100.0%)
5. Web Retrieval & SSRF Security   : 11 / 11 passed  (100.0%)
6. Conversation & Controls         :  8 /  8 passed  (100.0%)
----------------------------------------------------------------------
TOTAL PASS RATE                    : 69 / 69 passed  (100.0%)
======================================================================
```

---

## 2. Reproduction of Verified Failures & Root Causes

### Failure A: Public Professional Profile Search Failure
- **User Query**: `"tell me who is the shanveer singh khosa on the linkedin"`
- **Observed Legacy Behavior**:
  ```text
  [Web Search Notice: Could not locate verified educational sources specifically defining 'tell me who is the shanveer singh khosa on the linkedin'. Exact concept verification was unavailable.]
  ```
- **Underlying Root Causes**:
  1. *Intent Misclassification*: The legacy intent classifier lacked person, public profile, and company categories. Unclassified inputs were routed to `INTENT_EDUCATIONAL`.
  2. *Unprocessed Verbatim Query*: The entire natural language sentence was passed directly into exact-title encyclopedia search.
  3. *DuckDuckGo Socket Failure*: DuckDuckGo endpoints (`html.duckduckgo.com`, `lite.duckduckgo.com`, `api.duckduckgo.com`) resolved to Azure IPs (`20.204.244.192`) which drop TCP handshake packets on this ISP/network, timing out after 10+ seconds.
  4. *Lack of Honest Uncertainty Protocol*: When public profiles could not be retrieved, the system issued an irrelevant educational error instead of distinguishing public vs. private profiles.

### Failure B: Acute Symptom & Raw Scraping Dump
- **User Query**: `"i have pain in my heart"`
- **Observed Legacy Behavior**:
  The system retrieved a medical article and printed website navigation menus, breadcrumb trails (`Health > Cardiology > Categories`), and advertising boilerplate, completely omitting immediate emergency medical guidance.
- **Underlying Root Causes**:
  1. *Fragile Regex Extraction*: The HTML stripper used `<div id="content">(.*?)</div>`. Because standard HTML articles nest `<div class="breadcrumbs">...</div>` or `<div class="category">...</div>` inside `#content`, the non-greedy regex terminated at the *first inner closing tag*, completely dropping the article body and outputting only navigation categories.
  2. *Absence of Urgent Safety Triage*: The engine did not distinguish between acute, life-threatening symptom reports (heart pain, breathing difficulty) and general informational queries, causing delays and unsafe dumps of raw text.

---

## 3. System Architecture & Key Implementations

### 3.1 Universal 12-Intent Routing System (`src/intent_router.py`)
The newly built `IntentRouter` provides deterministic, sub-millisecond intent classification with structured entity extraction and safety triage:

| Intent Category | Identification Criteria | Decision & Action |
| :--- | :--- | :--- |
| **1. `general_conversation`** | Greetings, small talk, bot identity, expressions of gratitude | Direct conversational model; zero web latency |
| **2. `educational_concept`** | Educational definitions, AI/ML theory, core science questions | Verified educational concept retrieval & lead synthesis |
| **3. `person_profile`** | Named individuals, LinkedIn queries, corporate executives, founders | Public profile resolution across Wikidata, OpenAlex, GitHub |
| **4. `current_events`** | Software release versions, current events, recent news | Temporal entity search & recent release documentation |
| **5. `technical_programming`** | Library functions, code APIs (e.g. `torch.nn.Linear`), syntax | Technical documentation lookup & clean code references |
| **6. `health_safety`** | Symptoms, medical conditions, acute emergencies | Immediate emergency triage notice + clinical overview |
| **7. `product_service`** | Hardware specs, pricing queries, product reviews | Verified product specification synthesis |
| **8. `deterministic_tool`** | Arithmetic calculations, algebraic operations | Exact local mathematical calculator; zero model hallucination |
| **9. `document_summary`** | Direct HTTP/HTTPS URLs in user prompt | SSRF-validated fetch + DOM-aware clean extraction |
| **10. `comparison`** | Multi-concept comparisons (e.g. `X vs Y`, differences between) | Pairwise concept retrieval & multi-source synthesis |
| **11. `followup_reference`** | "Explain that simpler", "What did source 2 say?", summaries | Cached session retrieval; zero redundant web queries |
| **12. `ambiguous_unspecified`** | Vague, single-word, or nonsensical prompts | Direct clarifying question to guide user |

### 3.2 DOM-Aware Content Extraction (`RobustContentExtractor`)
Implemented in `src/web_retrieval.py` using standard library `html.parser.HTMLParser` with a stateful tag stack:
- **Noise Elimination**: Automatically strips `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, `<aside>`, `<noscript>`, `<form>`, `<svg>`, `<canvas>`, and `<select>`.
- **Boilerplate CSS Class Filtering**: Ignores elements containing class names like `breadcrumb`, `menu`, `sidebar`, `nav`, `advert`, `promo`, `newsletter`, `cookie`, `social`, and `footer`.
- **Client-Hydrated Metadata Harvesting**: Extracts `<title>`, `<meta name="description">`, `og:description`, and `application/ld+json` schema blocks. This enables clean extraction of JavaScript-rendered public profiles (such as LinkedIn) even when the body contains only an empty hydration `div`.
- **Mathematical Notation Normalization**: Cleans LaTeX and math artifacts into human-readable text (e.g. converting `{\displaystyle p(y\mid x)}` into `p(y | x)`, `\frac{1}{\sqrt{2\pi}}` into `(1 / sqrt(2pi))`).
- **Entity Unescaping**: Cleans all HTML entities (`&amp;` -> `&`, `&quot;` -> `"`, `&gt;` -> `>`, `&mdash;` -> `—`).

### 3.3 Multi-Provider Web Retrieval Subsystem (`src/web_retrieval.py`)
To prevent single-point-of-failure issues and resolve DuckDuckGo network reachability constraints, retrieval is distributed across reliable, zero-cost public endpoints:
1. **Wikidata Entity Provider**: Queries Wikimedia's official `wbsearchentities` API to resolve people, companies, software projects, and academic institutions in < 600ms.
2. **Wikipedia Provider**: Employs lead-extract API (`prop=extracts&exintro=1&explaintext=1`) and full-text search with standardized User-Agent headers, compliant Wikimedia etiquette, and automatic HTTP 429 exponential backoff.
3. **PubMed & PMC Medical Provider**: Queries NCBI Entrez E-utilities (`esearch.fcgi?db=pmc`) to retrieve peer-reviewed medical and clinical literature for symptoms and health queries.
4. **OpenAlex Academic Provider**: Searches global academic authors, affiliations, citation metrics, and scientific research publications.
5. **GitHub Provider**: Queries public GitHub APIs for software developers and open-source projects.
6. **Fail-Fast DuckDuckGo Provider**: Bounded with a 2-second timeout to gracefully fall back to primary APIs when TCP connectivity is blocked.

### 3.4 Medical & Health Safety Triage Protocol
When an acute symptom is reported (e.g., `"i have pain in my heart"`, `"my chest hurts and i have trouble breathing"`):
1. **Immediate Emergency Notice**: The application immediately outputs a prominent, clear emergency medical banner advising the user to contact local emergency services (911 in US/Canada, 999 in UK, 112 in EU/India) or go to the nearest emergency room.
2. **Clinical Overview**: Follows with an educational, non-diagnostic clinical summary outlining potential cardiovascular, gastrointestinal, musculoskeletal, and pulmonary etiologies.
3. **Explicit Red Flags**: Details warning signs that require emergency intervention (crushing chest pressure, pain radiating to arm/jaw/neck, shortness of breath, cold sweats).
4. **Non-Diagnostic Disclaimer**: Clearly affirms that the response is educational and cannot replace individualized medical assessment.
5. **Zero Delay**: The emergency triage text does not wait on network search calls.

### 3.5 Public Profile Search & Honest Uncertainty
When queried about named individuals or LinkedIn profiles:
1. **Entity Normalization**: Converts conversational phrasing into targeted entity queries.
2. **Verified Synthesis**: Summarizes public name, primary profile URL, and factual highlights.
3. **Identity Ambiguity Disclosure**: If multiple distinct individuals share the name in publicly indexed sources, explicitly flags this ambiguity and asks for distinguishing details.
4. **Honest Uncertainty (No Educational Error)**: For unindexed profiles (such as private accounts or `"Shanveer Singh Khosa"`), outputs:
   ```text
   A verified public LinkedIn or professional profile for 'Shanveer Singh Khosa' could not be located in publicly indexed sources.
   Possible reasons:
   - The profile privacy settings may restrict public search indexing or require an active account login.
   - The name may be spelled differently or registered under another identifier.
   To view this profile, please verify the exact spelling or provide a direct public profile URL or distinguishing details (such as current company or location).
   ```
5. **Zero Hallucination / Privacy Guarantee**: The application refuses to invent employment histories, educational credentials, or private contact details.

---

## 4. Verification & Test Evidence

All acceptance tests were executed via `src/run_acceptance_suite.py` against the running application components and saved to `logs/universal_question_answering/`.

### Detailed Test Results Across Suites

#### Suite 1: Intent Routing (33/33 Passed - 100%)
- Correctly classified all 12 intents with accurate sub-intents (e.g. `urgent_emergency` vs `general_health_info`).
- Average classification latency: **0.82 ms** (sub-millisecond rule evaluation).

#### Suite 2: Public Profile Search (7/7 Passed - 100%)
- `"tell me who is the shanveer singh khosa on the linkedin"` -> `profile_not_found` (Useful privacy notice; zero educational error).
- `"Who is Shanveer Singh Khosa on LinkedIn?"` -> `profile_not_found` (Honest uncertainty response).
- `"Find the LinkedIn profile of Satya Nadella"` -> `web_grounded_evidence` (Resolved Satya Nadella profile).
- `"Who is Geoffrey Hinton?"` -> `web_grounded_evidence` (Resolved researcher profile via OpenAlex/Wikidata).
- `"Who is the CEO of Microsoft?"` -> `web_grounded_evidence` (Resolved Satya Nadella as Chairman/CEO of Microsoft).
- `"Who founded Apple?"` -> `web_grounded_evidence` (Resolved Apple Inc. founded by Steve Jobs, Steve Wozniak, Ronald Wayne).

#### Suite 3: Health & Safety Triage (5/5 Passed - 100%)
- `"i have pain in my heart"` -> `health_emergency_triage` (Immediate emergency notice with 911/999/112, red flags, disclaimer, zero raw web dump).
- `"my chest hurts and i have trouble breathing"` -> `health_emergency_triage` (Immediate triage triggered).
- `"what are the common causes of chest pain?"` -> `web_grounded_evidence` (General informational medical overview with red flags).
- `"what are the emergency warning signs of a heart attack?"` -> `web_grounded_evidence` (Structured warning signs and clinical overview).
- `"what is hypertension?"` -> `web_grounded_evidence` (Authoritative medical definition and overview).

#### Suite 4: DOM-Aware Content Extraction (5/5 Passed - 100%)
- *Nested Div & Breadcrumb Bug*: Successfully extracted article body without truncating at inner `div` tags; completely stripped breadcrumb categories.
- *Ad & Newsletter Stripping*: Removed sidebar promotional widgets, clickbait titles, and newsletter subscription forms.
- *Mathematical Notation*: Normalized `{displaystyle p(y\mid x)}` into clean readable `p(y | x)` notation.
- *HTML Entities*: Correctly unescaped `&amp;`, `&quot;`, `&gt;`, `&mdash;`.
- *Client-Hydrated Metadata*: Harvested LinkedIn-style `<meta name="description">` and JSON-LD schema when the DOM container is empty.

#### Suite 5: Web Retrieval Providers & SSRF Security (11/11 Passed - 100%)
- *Wikipedia Concept*: Retrieved exact lead extracts for deep learning in 515ms.
- *Wikidata Search*: Successfully retrieved entity descriptions for PyTorch in 635ms.
- *PubMed Search*: Successfully retrieved peer-reviewed medical papers for chest pain causes in 806ms.
- *OpenAlex Search*: Successfully resolved academic profiles and citation counts.
- *GitHub Search*: Successfully queried public developer accounts.
- *SSRF Protections*: Blocked `127.0.0.1:8080`, `localhost/admin`, `169.254.169.254` (cloud metadata), `192.168.1.1` (private subnet), and `file:///etc/passwd`. Allowed valid public HTTPS URLs.

#### Suite 6: Conversation Regression & Controls (8/8 Passed - 100%)
- Multi-turn follow-ups resolved seamlessly without redundant web queries:
  - Turn 1 (`What is supervised learning?`) -> `web_grounded_evidence`
  - Turn 2 (`Explain that in simpler words.`) -> `web_cache_followup`
  - Turn 3 (`What did the second source say?`) -> `web_cache_followup`
  - Turn 4 (`Give me a shorter summary.`) -> `web_cache_followup`
- Tool operations remained intact:
  - Turn 5 (`45 * 12`) -> `calculator` (Evaluated to 540)
- Conversational fallbacks preserved:
  - Turn 6 (`thank you!`) -> `conversation`
  - Turn 7 (`stuff`) -> `clarification`
- Mode toggle verified:
  - Offline mode (`/web off`) -> `offline_notice` (Informs user that web retrieval is disabled).

---

## 5. Security, Privacy, and Reliability Guarantees

1. **Untrusted External Content**: All retrieved web content is treated as untrusted input. Special tokens (`<|user|>`, `<|assistant|>`, `[EOS]`, `[BOS]`) are escaped before constructing evidence prompts to prevent prompt injection.
2. **SSRF Guard**: Resolves DNS hostnames and verifies against private IP ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.0/8`, `169.254.0.0/16`). Non-HTTP/HTTPS schemes and non-standard ports are rejected.
3. **Resource Bounds**: Web payloads are strictly capped at 512 KB, socket timeouts are capped at 6–8 seconds, and candidate result sets are limited to top-ranked sources.
4. **Privacy Protection**: Search routines target publicly accessible index records only. Private contact information, phone numbers, personal emails, or unverified credentials are never guessed or fabricated.

---

## 6. Known System Limitations

1. **Private LinkedIn Profiles**: Public profiles that require an active LinkedIn login or have search indexing disabled by user privacy settings cannot be retrieved by automated web tools without bypassing authentication (which is strictly forbidden). In such cases, the system honestly states that the profile could not be verified from public sources and requests distinguishing details.
2. **Local Model Capacity (14.1M Parameters)**: The locally trained model is a lightweight 14.1M parameter Transformer. While it performs well on conversational dialogue and short summaries, complex reasoning over long multi-paragraph documents is handled by structured extractive evidence synthesis rather than relying purely on the small neural weights.
3. **Network Firewall / ISP Blocking**: If an external provider (such as DuckDuckGo or Wikipedia) experiences a network block or rate limit, the system gracefully falls back across its secondary providers or provides clean, structured explanations without crashing.

---

## 7. How to Run the Application

The application can be launched directly using the provided Windows batch script or via the Python command line:

### Option A: Using the Windows Batch File
Double-click or run from PowerShell/CMD:
```cmd
D:\DEEP-LEARNING-PROJECT\RUN_CHATBOT.bat
```

### Option B: Using the Virtual Environment Python Executable
From the project root:
```powershell
.venv\Scripts\python.exe src/chat.py
```

### Supported Runtime Slash Commands
Inside the interactive session, the following commands are supported:
- `/web on` — Enable intelligent web retrieval and multi-provider search.
- `/web off` — Disable web retrieval and operate in offline model mode.
- `/sources` — Display the URLs and titles from the most recent search.
- `/reset` — Clear conversation history and reset session cache.
- `/help` — Display command overview and feature usage.
- `/exit` or `/quit` — Gracefully terminate the chat session.

---

## 8. Conclusion

The from-scratch deep-learning chatbot project has been elevated from a brittle educational-only retriever to a robust, versatile question-answering system. The two primary real-world failures have been completely diagnosed, resolved, and verified:
1. Public profile questions now resolve appropriately via entity search or return an honest privacy explanation rather than an educational definition error.
2. Health and urgent symptom inquiries immediately trigger life-safety triage notices without dumping raw HTML navigation boilerplate.

All 69 validation tests across all 6 test suites passed with a 100.0% success rate, preserving the existing trained model and working functionality.
