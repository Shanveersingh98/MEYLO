import sys
import urllib.request
import urllib.parse
import re
import html

sys.stdout.reconfigure(encoding="utf-8")

def search_ddg_html(query: str, limit: int = 5):
    url = "https://html.duckduckgo.com/html/"
    data = urllib.parse.urlencode({
        "q": query,
        "b": "",
        "kl": "us-en"
    }).encode("utf-8")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://duckduckgo.com/",
        "Origin": "https://duckduckgo.com",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=8) as resp:
        body = resp.read().decode("utf-8", errors="ignore")

    items = re.findall(r'<a\s+[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?<a\s+[^>]*class="result__snippet"[^>]*>(.*?)</a>', body, re.DOTALL)
    results = []
    for u, t, s in items[:limit]:
        clean_u = u
        if "uddg=" in u:
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(u).query)
            if "uddg" in parsed:
                clean_u = parsed["uddg"][0]
        clean_t = html.unescape(re.sub(r'<[^>]+>', '', t)).strip()
        clean_s = html.unescape(re.sub(r'<[^>]+>', '', s)).strip()
        results.append({"title": clean_t, "url": clean_u, "snippet": clean_s})
    return results

print("Testing 'causes of chest pain'...")
res = search_ddg_html("causes of chest pain", 3)
print(f"Got {len(res)} results:")
for r in res:
    print(f"- {r['title']}\n  URL: {r['url']}\n  Snippet: {r['snippet']}\n")
