import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, json, time

instances = [
    "https://search.ononoki.org",
    "https://searx.be",
    "https://priv.au",
    "https://search.sapti.me",
    "https://searx.si",
    "https://baresearch.org"
]

print("=== Testing SearXNG Instances ===")
for base in instances:
    url = f"{base}/search?q=Shanveer+Singh+Khosa&format=json"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            dur = time.time() - t0
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            print(f"[{base}] SUCCESS ({dur:.2f}s): {len(results)} results")
            for r in results[:2]:
                print(f"   {r.get('title')} -> {r.get('url')}")
    except Exception as e:
        dur = time.time() - t0
        print(f"[{base}] FAILED ({dur:.2f}s): {e}")
