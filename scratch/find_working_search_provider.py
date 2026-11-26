import urllib.request, urllib.parse, json

queries = ["Shanveer Singh Khosa LinkedIn", "chest pain causes", "what is supervised learning"]

# List of public endpoints to test
endpoints = [
    ("SearXNG searx.be", "https://searx.be/search?q={q}&format=json"),
    ("SearXNG search.ononoki.org", "https://search.ononoki.org/search?q={q}&format=json"),
    ("SearXNG baresearch.org", "https://baresearch.org/search?q={q}&format=json"),
    ("SearXNG searx.space instance 1", "https://northboot.xyz/searxng/search?q={q}&format=json"),
    ("SearXNG priv.au", "https://priv.au/search?q={q}&format=json"),
    ("DuckDuckGo Instant Answer", "https://api.duckduckgo.com/?q={q}&format=json&no_html=1"),
    ("Wikidata API", "https://www.wikidata.org/w/api.php?action=wbsearchentities&search={q}&language=en&format=json"),
]

q = "Shanveer Singh Khosa LinkedIn"
enc_q = urllib.parse.quote(q)

for name, template in endpoints:
    url = template.format(q=enc_q)
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        })
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            print(f"[{name}] HTTP {resp.status}, Len: {len(data)}")
            try:
                js = json.loads(data)
                if "results" in js:
                    print(f"   Found {len(js['results'])} results in SearXNG JSON!")
                    for item in js['results'][:2]:
                        print("     -", item.get("title"), item.get("url"))
            except Exception:
                pass
    except Exception as e:
        print(f"[{name}] Failed: {e}")
