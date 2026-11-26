import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, time

test_urls = [
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://pytorch.org",
    "https://docs.python.org/3/",
    "https://github.com",
    "https://www.mayoclinic.org/diseases-conditions/chest-pain/symptoms-causes/syc-20370838",
    "https://www.healthline.com/health/chest-pain",
    "https://www.linkedin.com/in/satyanadella"
]

print("=== Testing Direct Public Webpage Fetching ===")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

for url in test_urls:
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            dur = time.time() - t0
            body = resp.read()
            print(f"[SUCCESS] {dur:.2f}s | HTTP {resp.status} | {len(body)} bytes | {url[:60]}")
    except Exception as e:
        dur = time.time() - t0
        print(f"[FAILED] {dur:.2f}s | {e} | {url[:60]}")
