import urllib.request, urllib.parse, json

url = "https://searx.be/search?q=" + urllib.parse.quote("Shanveer Singh Khosa LinkedIn") + "&format=json"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=5) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    print("Number of results:", len(data.get("results", [])))
    for r in data.get("results", [])[:5]:
        print("  Title:", r.get("title"))
        print("  URL:", r.get("url"))
        print("  Content:", r.get("content")[:150])
