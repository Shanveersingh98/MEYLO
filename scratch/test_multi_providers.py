import urllib.request, urllib.parse, re

def test_ecosia(query):
    url = f"https://www.ecosia.org/search?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            print("Ecosia status:", resp.status, "len:", len(data), "has result:", "result__title" in data or "mainline__result" in data or "class=\"result" in data)
            titles = re.findall(r'<a[^>]*class="[^"]*result-title[^"]*"[^>]*>(.*?)</a>', data)
            print("Ecosia titles:", len(titles))
    except Exception as e:
        print("Ecosia err:", e)

def test_startpage(query):
    url = "https://www.startpage.com/sp/search"
    data = urllib.parse.urlencode({"query": query}).encode("utf-8")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            print("Startpage status:", resp.status, "len:", len(data))
    except Exception as e:
        print("Startpage err:", e)

def test_bing(query):
    url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            print("Bing status:", resp.status, "len:", len(data), "has results:", "b_algo" in data)
            results = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', data, re.DOTALL)
            print("Bing b_algo count:", len(results))
            for r in results[:2]:
                title_m = re.search(r'<h2><a[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>', r)
                snip_m = re.search(r'<p[^>]*>(.*?)</p>', r)
                if title_m:
                    print("  Bing title:", re.sub(r'<[^>]+>', '', title_m.group(2)))
                    print("  Bing URL:", title_m.group(1))
                    if snip_m:
                        print("  Bing snippet:", re.sub(r'<[^>]+>', '', snip_m.group(1))[:150])
    except Exception as e:
        print("Bing err:", e)

def test_yahoo(query):
    url = f"https://search.yahoo.com/search?p={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            print("Yahoo status:", resp.status, "len:", len(data), "has algo:", "class=\"algo" in data or "class=\"title" in data)
    except Exception as e:
        print("Yahoo err:", e)

if __name__ == "__main__":
    q = "Shanveer Singh Khosa LinkedIn"
    print("Testing query:", q)
    test_bing(q)
    test_ecosia(q)
    test_startpage(q)
    test_yahoo(q)
