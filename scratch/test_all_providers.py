import urllib.request
import urllib.parse
import json
import re

query = "Shanveer Singh Khosa LinkedIn"
q_enc = urllib.parse.quote(query)

providers = [
    ("Mojeek", f"https://www.mojeek.com/search?q={q_enc}"),
    ("Qwant", f"https://api.qwant.com/v3/search/web?q={q_enc}&count=5&locale=en_US"),
    ("Startpage", f"https://www.startpage.com/sp/search?query={q_enc}"),
    ("Google_HTML", f"https://www.google.com/search?q={q_enc}&hl=en"),
    ("Brave", f"https://search.brave.com/search?q={q_enc}"),
    ("Wikipedia", f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q_enc}&format=json&srlimit=3"),
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

for name, url in providers:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read()
            text = data.decode("utf-8", errors="ignore")
            print(f"[{name}] HTTP {resp.status}, length: {len(text)}")
            if "shanveer" in text.lower():
                print(f"  -> Found 'shanveer' in {name} output!")
            if "linkedin" in text.lower():
                print(f"  -> Found 'linkedin' in {name} output!")
    except Exception as e:
        print(f"[{name}] failed: {e}")
