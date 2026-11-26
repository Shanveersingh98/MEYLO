import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, re, html, base64

def decode_bing_url(bing_ck_url):
    try:
        parsed = urllib.parse.urlparse(bing_ck_url)
        qs = urllib.parse.parse_qs(parsed.query)
        u_val = qs.get("u", [""])[0]
        if u_val.startswith("a1"):
            raw_b64 = u_val[2:]
            rem = len(raw_b64) % 4
            if rem > 0:
                raw_b64 += "=" * (4 - rem)
            return base64.b64decode(raw_b64).decode("utf-8", errors="ignore")
    except Exception:
        pass
    return bing_ck_url

def parse_bing_results(query):
    url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8", errors="ignore")

    # Pattern for bing organic results: <li class="b_algo">
    # Within each b_algo, find <h2><a href="...">Title</a></h2>
    # and <p ...>Snippet</p> or <div class="b_caption">...
    items = re.findall(r'<li\s+class="b_algo"[^>]*>(.*?)</li>', body, re.DOTALL)
    results = []
    for item_html in items:
        # Title and url
        link_m = re.search(r'<h2><a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>', item_html, re.DOTALL)
        if not link_m:
            continue
        raw_url = link_m.group(1)
        raw_title = link_m.group(2)
        
        # Snippet
        snippet_m = re.search(r'<p[^>]*>(.*?)</p>', item_html, re.DOTALL)
        snippet = snippet_m.group(1) if snippet_m else ""
        
        clean_url = decode_bing_url(html.unescape(raw_url))
        clean_title = html.unescape(re.sub(r'<[^>]+>', '', raw_title)).strip()
        clean_snippet = html.unescape(re.sub(r'<[^>]+>', '', snippet)).strip()
        
        results.append({
            "url": clean_url,
            "title": clean_title,
            "snippet": clean_snippet
        })
    return results

print("=== Bing: Satya Nadella LinkedIn ===")
res1 = parse_bing_results("Satya Nadella LinkedIn")
print(f"Found {len(res1)} results:")
for r in res1[:5]:
    print("URL:", r["url"])
    print("TITLE:", r["title"])
    print("SNIPPET:", r["snippet"])
    print("---")

print("\n=== Bing: causes of chest pain ===")
res2 = parse_bing_results("causes of chest pain")
print(f"Found {len(res2)} results:")
for r in res2[:5]:
    print("URL:", r["url"])
    print("TITLE:", r["title"])
    print("SNIPPET:", r["snippet"])
    print("---")
