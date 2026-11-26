import urllib.request, urllib.parse, http.cookiejar, time

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

# First, visit home page to establish session/cookies
home_req = urllib.request.Request(
    "https://duckduckgo.com/",
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
)
with opener.open(home_req, timeout=5) as resp:
    print("Home status:", resp.status, "Cookies:", len(cj))
    for c in cj:
        print("  Cookie:", c.name, c.value)

# Now do search
data = urllib.parse.urlencode({"q": "Shanveer Singh Khosa LinkedIn", "b": ""}).encode("utf-8")
search_req = urllib.request.Request(
    "https://html.duckduckgo.com/html/",
    data=data,
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://duckduckgo.com/",
        "Origin": "https://duckduckgo.com",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/x-www-form-urlencoded",
    }
)
with opener.open(search_req, timeout=5) as resp:
    content = resp.read().decode("utf-8", errors="ignore")
    print("Search status:", resp.status, "Length:", len(content))
    print("Has result snippet:", "result__snippet" in content)
    import re
    snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', content, re.DOTALL)
    print("Found snippets:", len(snippets))
    for s in snippets[:3]:
        print("  Snippet:", re.sub(r'<[^>]+>', '', s).strip())
