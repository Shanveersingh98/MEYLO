import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, json

def search_wikipedia(query):
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit=5"
    req = urllib.request.Request(url, headers={"User-Agent": "FromScratchBot/1.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("query", {}).get("search", [])

for q in ["Satya Nadella", "Shanveer Singh Khosa", "Geoffrey Hinton"]:
    hits = search_wikipedia(q)
    print(f"\nWikipedia hits for '{q}': {len(hits)}")
    for h in hits[:2]:
        print(f"  {h['title']}: {h['snippet'][:120]}")
