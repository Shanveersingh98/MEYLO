import urllib.request
import urllib.parse
import re

def search_ddg_html(query: str, limit: int = 3):
    url = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote(query)
    req = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }
    )
    results = []
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            html = r.read().decode('utf-8', errors='ignore')
            # Extract result blocks
            blocks = re.findall(r'<div class="result__body[^"]*"[^>]*>(.*?)</div>\s*</div>', html, re.DOTALL)
            for b in blocks:
                # Title & URL
                t_match = re.search(r'<a class="result__url[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', b, re.DOTALL)
                title_match = re.search(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', b, re.DOTALL)
                # Or regular title link
                link_m = re.search(r'<a class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', b, re.DOTALL)
                snip_m = re.search(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', b, re.DOTALL)
                
                title = re.sub(r'<[^>]+>', '', link_m.group(2)).strip() if link_m else ""
                href = link_m.group(1).strip() if link_m else ""
                snippet = re.sub(r'<[^>]+>', '', snip_m.group(1)).strip() if snip_m else ""
                
                # Unescape DuckDuckGo redirect
                if "uddg=" in href:
                    m = re.search(r'uddg=([^&]+)', href)
                    if m:
                        href = urllib.parse.unquote(m.group(1))
                
                if title and (snippet or href):
                    results.append({"title": title, "url": href, "snippet": snippet})
                    if len(results) >= limit:
                        break
    except Exception as e:
        print("DDG HTML error:", e)
    return results

res = search_ddg_html("diljit dosanjh daddy yankee song")
print("Found results:", len(res))
for r in res:
    print("Title:", r["title"])
    print("URL:", r["url"])
    print("Snippet:", r["snippet"])
    print("-" * 40)
