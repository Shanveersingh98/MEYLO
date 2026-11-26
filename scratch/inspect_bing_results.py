import sys, urllib.request, urllib.parse, re, base64
sys.stdout.reconfigure(encoding="utf-8")

def decode_bing_url(u_param):
    if not u_param:
        return ""
    if u_param.startswith("a1"):
        u_param = u_param[2:]
    # Replace URL safe chars if any
    u_param = u_param.replace("-", "+").replace("_", "/")
    rem = len(u_param) % 4
    if rem:
        u_param += "=" * (4 - rem)
    try:
        return base64.b64decode(u_param).decode("utf-8", errors="ignore")
    except Exception:
        return ""

url = "https://www.bing.com/search?q=" + urllib.parse.quote("Shanveer Singh Khosa LinkedIn")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=8) as resp:
    data = resp.read().decode("utf-8", errors="ignore")

results = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', data, re.DOTALL)
print(f"Total Bing results: {len(results)}")
for i, r in enumerate(results, 1):
    title_m = re.search(r'<h2><a[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>', r)
    snip_m = re.search(r'<p[^>]*>(.*?)</p>', r)
    if title_m:
        raw_url = title_m.group(1).replace("&amp;", "&")
        u_match = re.search(r'[?&]u=([^&]+)', raw_url)
        real_url = decode_bing_url(u_match.group(1)) if u_match else raw_url
        title = re.sub(r'<[^>]+>', '', title_m.group(2)).strip()
        snippet = re.sub(r'<[^>]+>', '', snip_m.group(1)).strip() if snip_m else ""
        print(f"[{i}] {title}")
        print(f"    URL: {real_url}")
        print(f"    Snippet: {snippet[:150]}")
