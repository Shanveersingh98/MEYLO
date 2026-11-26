import urllib.request
import urllib.parse
import json
import time

def test_endpoint(name, url, method="GET", data=None, headers=None):
    if headers is None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*"
        }
    t0 = time.time()
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=5) as resp:
            dur = time.time() - t0
            body = resp.read()
            print(f"[{name}] SUCCESS in {dur:.2f}s: HTTP {resp.status}, bytes={len(body)}")
            return True, body
    except Exception as e:
        dur = time.time() - t0
        print(f"[{name}] FAILED in {dur:.2f}s: {e}")
        return False, None

print("=== Testing Search Engine Reachability ===")
# 1. Wikipedia
test_endpoint("Wikipedia API", "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=python&format=json")

# 2. Bing
test_endpoint("Bing Search", "https://www.bing.com/search?q=Shanveer+Singh+Khosa")

# 3. Yahoo
test_endpoint("Yahoo Search", "https://search.yahoo.com/search?p=Shanveer+Singh+Khosa")

# 4. Mojeek
test_endpoint("Mojeek", "https://www.mojeek.com/search?q=Shanveer+Singh+Khosa")

# 5. Qwant
test_endpoint("Qwant", "https://api.qwant.com/v3/search/web?q=Shanveer+Singh+Khosa&count=5")

# 6. Brave
test_endpoint("Brave Search", "https://search.brave.com/search?q=Shanveer+Singh+Khosa")

# 7. Startpage
test_endpoint("Startpage", "https://www.startpage.com/sp/search?query=Shanveer+Singh+Khosa")

# 8. GitHub API (useful for developer/researcher public profiles)
test_endpoint("GitHub Search API", "https://api.github.com/search/users?q=Shanveer+Singh+Khosa", headers={"User-Agent": "FromScratchBot/1.0"})
