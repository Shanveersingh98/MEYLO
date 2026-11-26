import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, json

def search_wikidata(query):
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={urllib.parse.quote(query)}&language=en&format=json&limit=5"
    req = urllib.request.Request(url, headers={"User-Agent": "FromScratchBot/1.0 (Educational AI Assistant)"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    results = []
    for item in data.get("search", []):
        results.append({
            "id": item.get("id"),
            "label": item.get("label"),
            "description": item.get("description", ""),
            "url": item.get("concepturi")
        })
    return results

for q in ["Satya Nadella", "Shanveer Singh Khosa", "Geoffrey Hinton", "Microsoft", "PyTorch"]:
    res = search_wikidata(q)
    print(f"\nWikidata search for '{q}': {len(res)} results")
    for r in res:
        print(f"  [{r['id']}] {r['label']}: {r['description']} ({r['url']})")
