import urllib.request
import urllib.parse
import re
import html

q = "Shanveer Singh Khosa LinkedIn"
url = f"https://www.bing.com/search?q={urllib.parse.quote(q)}"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9"
}

req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=10) as resp:
    body = resp.read().decode("utf-8", errors="ignore")

print("Bing response status:", resp.status, "Length:", len(body))

# Inspect result items in Bing
# Standard Bing results are <li class="b_algo"><h2><a href="...">Title</a></h2><div class="b_caption"><p>Snippet</p></div></li>
items = re.findall(r'<li\s+class="b_algo"[^>]*>.*?<h2><a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>.*?<p[^>]*>(.*?)</p>', body, re.DOTALL)
print(f"Found {len(items)} b_algo items with standard regex")

if len(items) == 0:
    # Try more lenient regex
    h2_items = re.findall(r'<h2><a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>', body, re.DOTALL)
    print(f"Found {len(h2_items)} h2 links:")
    for u, t in h2_items[:5]:
        clean_t = html.unescape(re.sub(r'<[^>]+>', '', t)).strip()
        print("  LINK:", u)
        print("  TITLE:", clean_t)
else:
    for u, t, p in items[:5]:
        clean_t = html.unescape(re.sub(r'<[^>]+>', '', t)).strip()
        clean_p = html.unescape(re.sub(r'<[^>]+>', '', p)).strip()
        print("---")
        print("URL:", u)
        print("TITLE:", clean_t)
        print("SNIPPET:", clean_p)
