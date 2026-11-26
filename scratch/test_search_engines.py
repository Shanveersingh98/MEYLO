import urllib.request
import urllib.parse
import re

query = "Shanveer Singh Khosa LinkedIn"
q_enc = urllib.parse.quote(query)

test_endpoints = [
    ("Bing", f"https://www.bing.com/search?q={q_enc}"),
    ("Yahoo", f"https://search.yahoo.com/search?p={q_enc}"),
    ("Mojeek", f"https://www.mojeek.com/search?q={q_enc}"),
    ("DDG_HTML_GET", f"https://html.duckduckgo.com/html/?q={q_enc}"),
    ("DDG_Lite_GET", f"https://lite.duckduckgo.com/lite/?q={q_enc}"),
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

for name, url in test_endpoints:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        print(f"[{name}] status={resp.status}, length={len(body)}")
        # Check if Shanveer or linkedin is in body
        has_target = "shanveer" in body.lower() or "linkedin" in body.lower()
        print(f"  Target found: {has_target}")
        if has_target:
            # Find links
            links = re.findall(r'href=[\'"](https?://[^\'"]+)[\'"]', body)
            li_links = [l for l in links if "linkedin" in l]
            print(f"  Found {len(li_links)} LinkedIn links: {li_links[:2]}")
    except Exception as e:
        print(f"[{name}] failed: {e}")
