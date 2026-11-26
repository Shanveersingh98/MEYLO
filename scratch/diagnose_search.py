import urllib.request
import urllib.parse
import re
import html
import json

q = "Shanveer Singh Khosa LinkedIn"
print("=== Diagnosing Public Web Search ===")

# Test 1: DuckDuckGo HTML
print("\n--- Test 1: html.duckduckgo.com ---")
try:
    data = urllib.parse.urlencode({"q": q, "b": "", "kl": "us-en"}).encode("utf-8")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://duckduckgo.com/",
        "Origin": "https://duckduckgo.com",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    req = urllib.request.Request("https://html.duckduckgo.com/html/", data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
        print("Status:", resp.status, "Length:", len(body))
        if "anomaly" in body.lower() or "challenge" in body.lower():
            print("Detected bot challenge / CAPTCHA block in html.duckduckgo.com!")
        items = re.findall(r'<a\s+[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', body, re.DOTALL)
        print("Matches found:", len(items))
        if len(items) == 0:
            print("Preview first 500 chars:", body[:500])
except Exception as e:
    print("Exception 1:", e)

# Test 2: DuckDuckGo Lite
print("\n--- Test 2: lite.duckduckgo.com ---")
try:
    data = urllib.parse.urlencode({"q": q}).encode("utf-8")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://lite.duckduckgo.com/",
    }
    req = urllib.request.Request("https://lite.duckduckgo.com/lite/", data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        content = resp.read().decode("utf-8", errors="ignore")
        print("Status:", resp.status, "Length:", len(content))
        if "anomaly" in content.lower() or "challenge" in content.lower() or "robot" in content.lower():
            print("Detected bot challenge in lite.duckduckgo.com!")
        print("Preview first 500 chars:", content[:500])
except Exception as e:
    print("Exception 2:", e)

# Test 3: DuckDuckGo Instant Answer API
print("\n--- Test 3: api.duckduckgo.com ---")
try:
    enc_q = urllib.parse.quote(q)
    api_url = f"https://api.duckduckgo.com/?q={enc_q}&format=json&no_html=1"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        d = json.loads(resp.read().decode("utf-8"))
        print("Abstract:", d.get("AbstractText", ""))
        print("Heading:", d.get("Heading", ""))
        print("Related count:", len(d.get("RelatedTopics", [])))
except Exception as e:
    print("Exception 3:", e)
