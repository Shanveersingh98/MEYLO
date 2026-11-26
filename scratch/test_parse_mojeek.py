import urllib.request
import urllib.parse
import re
import html

query = "Shanveer Singh Khosa LinkedIn"
q_enc = urllib.parse.quote(query)
url = f"https://www.mojeek.com/search?q={q_enc}"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=8) as resp:
    body = resp.read().decode("utf-8", errors="ignore")

print("Mojeek body length:", len(body))
# Look for results in Mojeek
# Mojeek uses <a class="ob" href="...">...</a> or <ul class="results"><li>...
res_list = re.findall(r'<a\s+[^>]*class=["\']ob["\'][^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', body, re.DOTALL)
print(f"Mojeek results found: {len(res_list)}")
for u, t in res_list[:5]:
    print(f"  Title: {html.unescape(re.sub(r'<[^>]+>', '', t))} | URL: {u}")

# Look for snippets in <p class="s">...</p>
snips = re.findall(r'<p\s+class=["\']s["\'][^>]*>(.*?)</p>', body, re.DOTALL)
print(f"Mojeek snippets found: {len(snips)}")
for s in snips[:3]:
    print(f"  Snippet: {html.unescape(re.sub(r'<[^>]+>', '', s))[:150]}")
