import urllib.request
import urllib.parse
import re

queries = ["causes of chest pain", "Shanveer Singh Khosa LinkedIn", "torchvision library"]

for q in queries:
    print(f"\n--- Testing query: {q} ---")
    
    # 1. Test html.duckduckgo.com/html/
    try:
        html_url = "https://html.duckduckgo.com/html/"
        data = urllib.parse.urlencode({"q": q}).encode("utf-8")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        req = urllib.request.Request(html_url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
        has_results = "result__snippet" in content or "result__title" in content
        print(f"html.duckduckgo.com: len={len(content)}, has_results={has_results}")
    except Exception as e:
        print(f"html.duckduckgo.com error: {e}")

    # 2. Test lite.duckduckgo.com/lite/
    try:
        lite_url = "https://lite.duckduckgo.com/lite/"
        data = urllib.parse.urlencode({"q": q}).encode("utf-8")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        req = urllib.request.Request(lite_url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
        has_results = "result-link" in content
        print(f"lite.duckduckgo.com: len={len(content)}, has_results={has_results}")
    except Exception as e:
        print(f"lite.duckduckgo.com error: {e}")
