import urllib.request
import urllib.parse
import re
import html

url = "https://html.duckduckgo.com/html/"
data = urllib.parse.urlencode({
    "q": "Shanveer Singh Khosa LinkedIn",
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

# Parse results:
# In html.duckduckgo.com, each result is inside <div class="result results_links results_links_deep web-result ">
# or <h2 class="result__title"><a class="result__a" href="...">Title</a></h2>
# <a class="result__snippet" href="...">Snippet</a>
result_blocks = re.findall(r'<div[^>]*class="[^"]*result\s+results_links[^"]*"[^>]*>(.*?)</div>\s*</div>', body, re.DOTALL)
print(f"Result blocks found: {len(result_blocks)}")

# Alternative: find all result__a and result__snippet
items = re.findall(r'<a\s+[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?<a\s+[^>]*class="result__snippet"[^>]*>(.*?)</a>', body, re.DOTALL)
print(f"Parsed items: {len(items)}")
for u, t, s in items[:4]:
    clean_u = u
    if "uddg=" in u:
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(u).query)
        if "uddg" in parsed:
            clean_u = parsed["uddg"][0]
    clean_t = html.unescape(re.sub(r'<[^>]+>', '', t)).strip()
    clean_s = html.unescape(re.sub(r'<[^>]+>', '', s)).strip()
    print(f"Title: {clean_t}\nURL: {clean_u}\nSnippet: {clean_s}\n")
