import urllib.request
import urllib.parse
import re

url = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote("diljit dosanjh daddy yankee song")
req = urllib.request.Request(
    url,
    headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
)
with urllib.request.urlopen(req, timeout=5) as r:
    html = r.read().decode('utf-8', errors='ignore')

titles = re.findall(r'<h2 class="result__title">\s*<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
snippets = re.findall(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', html, re.DOTALL)

print(f"Found {len(titles)} titles and {len(snippets)} snippets")
for i in range(min(len(titles), len(snippets), 5)):
    raw_href, raw_title = titles[i]
    raw_snip = snippets[i]
    title = re.sub(r'<[^>]+>', '', raw_title).strip()
    snippet = re.sub(r'<[^>]+>', '', raw_snip).strip()
    
    # decode DDG uddg url
    href = raw_href
    if "uddg=" in raw_href:
        m = re.search(r'uddg=([^&]+)', raw_href)
        if m:
            href = urllib.parse.unquote(m.group(1))
    
    print(f"Result {i+1}:")
    print("  Title:", title)
    print("  URL:", href)
    print("  Snippet:", snippet)
