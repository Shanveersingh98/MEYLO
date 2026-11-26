import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, time, json, re

engines = [
    ("Ecosia", "https://www.ecosia.org/search?q=Satya+Nadella"),
    ("Ask", "https://www.ask.com/web?q=Satya+Nadella"),
    ("Dogpile", "https://www.dogpile.com/serp?q=Satya+Nadella"),
    ("Marginalia", "https://api.marginalia.nu/public/search/Satya+Nadella"),
    ("Wikidata", "https://www.wikidata.org/w/api.php?action=wbsearchentities&search=Satya+Nadella&language=en&format=json")
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9"
}

for name, url in engines:
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            dur = time.time() - t0
            body = resp.read().decode("utf-8", errors="ignore")
            print(f"[{name}] SUCCESS in {dur:.2f}s | {resp.status} | len={len(body)}")
            # check if title/snippet appears
            if "Satya" in body:
                print(f"   Contains 'Satya'!")
    except Exception as e:
        dur = time.time() - t0
        print(f"[{name}] FAILED in {dur:.2f}s | {e}")
