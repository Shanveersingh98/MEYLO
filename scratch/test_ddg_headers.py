import sys, urllib.request, urllib.parse, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# Test various header and query configurations on html.duckduckgo.com
configs = [
    {
        "name": "Standard browser with Cookie",
        "url": "https://html.duckduckgo.com/html/",
        "data": {"q": "Shanveer Singh Khosa LinkedIn", "b": "", "kl": "us-en"},
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Referer": "https://duckduckgo.com/",
            "Origin": "https://duckduckgo.com",
            "Cookie": "dcm=1; kl=us-en",
            "Content-Type": "application/x-www-form-urlencoded"
        }
    },
    {
        "name": "GET request with kl=us-en",
        "url": "https://html.duckduckgo.com/html/?q=Shanveer+Singh+Khosa+LinkedIn&kl=us-en",
        "data": None,
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Referer": "https://duckduckgo.com/",
            "Cookie": "dcm=1; kl=us-en"
        }
    },
    {
        "name": "Lite with Cookie",
        "url": "https://lite.duckduckgo.com/lite/",
        "data": {"q": "Shanveer Singh Khosa LinkedIn", "kl": "us-en"},
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://lite.duckduckgo.com/",
            "Cookie": "dcm=1; kl=us-en",
            "Content-Type": "application/x-www-form-urlencoded"
        }
    }
]

for cfg in configs:
    try:
        if cfg["data"]:
            data_bytes = urllib.parse.urlencode(cfg["data"]).encode("utf-8")
            req = urllib.request.Request(cfg["url"], data=data_bytes, headers=cfg["headers"])
        else:
            req = urllib.request.Request(cfg["url"], headers=cfg["headers"])
        with urllib.request.urlopen(req, timeout=4) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
            print(f"[{cfg['name']}] Status: {resp.status}, Len: {len(text)}, has result__snippet: {'result__snippet' in text or 'result-snippet' in text}")
            if "result__snippet" in text or "result-snippet" in text:
                titles = re.findall(r'<a[^>]*class="[^"]*(?:result__a|result-link)[^"]*"[^>]*>(.*?)</a>', text)
                print(f"  Found {len(titles)} titles:")
                for t in titles[:3]:
                    print("   -", re.sub(r'<[^>]+>', '', t).strip())
    except Exception as e:
        print(f"[{cfg['name']}] Error: {e}")
