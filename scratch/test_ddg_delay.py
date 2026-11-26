import time
import urllib.request
import urllib.parse
import re
import html

def ddg_search(query: str):
    lite_url = "https://lite.duckduckgo.com/lite/"
    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    req = urllib.request.Request(lite_url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error: {e}"

    if "anomaly" in content or "challenge" in content:
        return "CHALLENGE"

    link_pattern = re.compile(
        r"<a\s+[^>]*href=['\"]([^'\"]+)['\"][^>]*class=['\"]result-link['\"][^>]*>(.*?)</a>",
        re.DOTALL | re.IGNORECASE,
    )
    snippets = list(link_pattern.finditer(content))
    return f"SUCCESS: {len(snippets)} results"

print("Waiting 3 seconds before query...")
time.sleep(3)
res = ddg_search("Shanveer Singh Khosa LinkedIn")
print("Result:", res)
