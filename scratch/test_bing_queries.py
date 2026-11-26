import sys, urllib.request, urllib.parse, re, base64
sys.stdout.reconfigure(encoding="utf-8")

def decode_bing_url(u_param):
    if not u_param:
        return ""
    if u_param.startswith("a1"):
        u_param = u_param[2:]
    u_param = u_param.replace("-", "+").replace("_", "/")
    rem = len(u_param) % 4
    if rem:
        u_param += "=" * (4 - rem)
    try:
        return base64.b64decode(u_param).decode("utf-8", errors="ignore")
    except Exception:
        return ""

def search_bing(q):
    url = "https://www.bing.com/search?q=" + urllib.parse.quote(q)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    req = urllib.request.Request(url, headers=headers)
    text = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", errors="ignore")
    results = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', text, re.DOTALL)
    print(f"Query: {q} -> Results: {len(results)}")
    for r in results[:3]:
        title_m = re.search(r'<h2><a[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>', r)
        snip_m = re.search(r'<p[^>]*>(.*?)</p>', r)
        if title_m:
            raw_url = title_m.group(1).replace("&amp;", "&")
            u_match = re.search(r'[?&]u=([^&]+)', raw_url)
            real_url = decode_bing_url(u_match.group(1)) if u_match else raw_url
            title = re.sub(r'<[^>]+>', '', title_m.group(2)).strip()
            print("  Title:", title)
            print("  URL:", real_url)
            if snip_m:
                print("  Snip:", re.sub(r'<[^>]+>', '', snip_m.group(1))[:100])

if __name__ == "__main__":
    search_bing('"Shanveer Singh Khosa"')
    search_bing('site:linkedin.com "Shanveer Singh Khosa"')
    search_bing('causes of chest pain emergency')
