import urllib.request
import urllib.parse
import json
import http.cookiejar

# Test 1: DDG with cookiejar and realistic browser headers
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

headers = [
    ("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    ("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"),
    ("Accept-Language", "en-US,en;q=0.9"),
    ("Referer", "https://duckduckgo.com/"),
]
opener.addheaders = headers

print("--- Testing DDG Lite with CookieJar ---")
try:
    data = urllib.parse.urlencode({"q": "causes of chest pain"}).encode("utf-8")
    req = urllib.request.Request("https://lite.duckduckgo.com/lite/", data=data)
    with opener.open(req, timeout=6) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    print(f"DDG Lite with CookieJar: len={len(html)}, has 'result-link': {'result-link' in html}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: DDG API
print("\n--- Testing DDG Instant Answer API ---")
try:
    api_url = "https://api.duckduckgo.com/?q=causes+of+chest+pain&format=json&no_html=1"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    print(f"DDG API heading: '{data.get('Heading')}', abstract len: {len(data.get('AbstractText', ''))}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Public SearXNG / HTML search endpoints
print("\n--- Testing Public SearXNG JSON API ---")
searx_instances = [
    "https://search.ononoki.org",
    "https://searx.be",
    "https://search.sapti.me",
    "https://priv.au"
]
for inst in searx_instances:
    try:
        url = f"{inst}/search?q=causes+of+chest+pain&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            print(f"SearXNG ({inst}): Got {len(results)} results! First: '{results[0].get('title')}'")
            break
    except Exception as e:
        print(f"SearXNG ({inst}) failed: {e}")

# Test 4: Wikipedia Search on 'Chest pain'
print("\n--- Testing Wikipedia API on 'Chest pain' ---")
try:
    wiki_url = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=causes%20of%20chest%20pain&format=json&srlimit=3"
    req = urllib.request.Request(wiki_url, headers={"User-Agent": "FromScratchChatbot/1.0"})
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        hits = data.get("query", {}).get("search", [])
        print(f"Wikipedia: Got {len(hits)} hits! Titles: {[h['title'] for h in hits]}")
except Exception as e:
    print(f"Error: {e}")
