import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request
import urllib.parse
import re
import html
import base64

def check_bing(q):
    print(f"\n--- Checking Bing for: {q} ---")
    url = f"https://www.bing.com/search?q={urllib.parse.quote(q)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    print("Body length:", len(body))
    print("'No results found':", "No results found" in body)
    print("'There are no results':", "There are no results" in body)
    
    # Let's find all <h2> tags
    h2s = re.findall(r'<h2[^>]*>(.*?)</h2>', body, re.DOTALL)
    print(f"Total H2 tags: {len(h2s)}")
    for i, h in enumerate(h2s[:5]):
        txt = re.sub(r'<[^>]+>', '', h).strip()
        print(f"  H2 #{i+1}: {txt}")

check_bing("Satya Nadella LinkedIn")
check_bing("causes of chest pain")
check_bing("deep learning")
