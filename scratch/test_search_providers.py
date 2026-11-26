import urllib.request, urllib.parse, json

def test_ddg_get(query):
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            print("DDG GET: status", resp.status, "len", len(data), "has result-snippet:", "result__snippet" in data)
    except Exception as e:
        print("DDG GET err:", e)

def test_mojeek(query):
    url = f"https://www.mojeek.com/search?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            print("Mojeek: status", resp.status, "len", len(data), "has title:", "<a class=\"title\"" in data or "class=\"snippet\"" in data)
    except Exception as e:
        print("Mojeek err:", e)

def test_qwant(query):
    url = f"https://api.qwant.com/v3/search/web?q={urllib.parse.quote(query)}&count=5&locale=en_US"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            print("Qwant: status", resp.status, "len", len(data))
    except Exception as e:
        print("Qwant err:", e)

def test_brave_web(query):
    url = f"https://search.brave.com/search?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            print("Brave: status", resp.status, "len", len(data), "has snippet:", "snippet" in data.lower())
    except Exception as e:
        print("Brave err:", e)

if __name__ == "__main__":
    q = "Shanveer Singh Khosa LinkedIn"
    print("Testing query:", q)
    test_ddg_get(q)
    test_mojeek(q)
    test_brave_web(q)
    test_qwant(q)
