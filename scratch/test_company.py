import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, json

def search_company(comp):
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles={urllib.parse.quote(comp)}&redirects=1&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "FromScratchBot/1.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for pid, p in pages.items():
            if "extract" in p:
                return p["title"], p["extract"][:400]
    return None, None

print("Microsoft:", search_company("Microsoft"))
print("Apple Inc.:", search_company("Apple Inc."))
