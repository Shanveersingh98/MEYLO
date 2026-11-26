import urllib.request
import urllib.parse
import json

# Test 1: DDG Instant Answer API
try:
    url = "https://api.duckduckgo.com/?q=" + urllib.parse.quote("Python programming language") + "&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("DDG Instant Answer Heading:", data.get("Heading"))
        print("DDG Abstract:", data.get("Abstract")[:100] if data.get("Abstract") else "No abstract")
        print("DDG RelatedTopics count:", len(data.get("RelatedTopics", [])))
except Exception as e:
    print("DDG Instant Answer error:", e)

# Test 2: Wikipedia Full-Text Search API
try:
    url = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=" + urllib.parse.quote("Python 3.12 release") + "&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "FromScratchChatbot/1.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        hits = data.get("query", {}).get("search", [])
        print("Wikipedia full text hits:", len(hits))
        for h in hits[:2]:
            print("  ", h.get("title"), "->", h.get("snippet")[:80])
except Exception as e:
    print("Wikipedia full text error:", e)

# Test 3: Mojeek or other public web search engine
try:
    url = "https://www.mojeek.com/search?q=" + urllib.parse.quote("Python 3.12 release")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        content = resp.read().decode("utf-8", errors="ignore")
        print("Mojeek length:", len(content), "Results class?:", "results-standard" in content or "title" in content)
except Exception as e:
    print("Mojeek error:", e)
