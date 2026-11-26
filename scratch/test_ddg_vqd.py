import urllib.request
import urllib.parse
import re
import json

def ddg_vqd_search(query: str):
    q_enc = urllib.parse.quote(query)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    # Step 1: Get VQD token from duckduckgo.com
    url1 = f"https://duckduckgo.com/?q={q_enc}"
    req1 = urllib.request.Request(url1, headers=headers)
    try:
        with urllib.request.urlopen(req1, timeout=8) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Step 1 failed: {e}"

    vqd_match = re.search(r'vqd=([0-9_\-]+)', text) or re.search(r'vqd="([0-9_\-]+)"', text) or re.search(r'vqd:\s*"([0-9_\-]+)"', text)
    if not vqd_match:
        return f"Could not find VQD token in page (len={len(text)})"
    vqd = vqd_match.group(1)
    print(f"Got VQD token: {vqd}")

    # Step 2: Query links.duckduckgo.com/d.js
    url2 = f"https://links.duckduckgo.com/d.js?q={q_enc}&vqd={vqd}&p=1"
    req2 = urllib.request.Request(url2, headers=headers)
    try:
        with urllib.request.urlopen(req2, timeout=8) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
        print(f"Step 2 response len={len(data)}")
        # Check if contains results
        if "results" in data or "d" in data:
            print("Step 2 output preview:", data[:300])
        return "SUCCESS"
    except Exception as e:
        return f"Step 2 failed: {e}"

print(ddg_vqd_search("Shanveer Singh Khosa LinkedIn"))
