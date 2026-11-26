import urllib.request
import urllib.parse
import re
import html

query = "Shanveer Singh Khosa LinkedIn"
q_enc = urllib.parse.quote(query)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# 1. Parse DDG HTML GET
print("--- Parsing DDG HTML GET ---")
url = f"https://html.duckduckgo.com/html/?q={q_enc}"
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=8) as resp:
    body = resp.read().decode("utf-8", errors="ignore")

# Find result links and snippets in DDG HTML
titles = re.findall(r'<a\s+[^>]*class=["\']result__url["\'][^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', body, re.DOTALL)
if not titles:
    # Try another pattern
    links = re.findall(r'<a\s+[^>]*class=["\']result__snippet["\'][^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', body, re.DOTALL)
print(f"DDG HTML results found: {len(titles)}")
for href, txt in titles[:3]:
    clean_txt = re.sub(r'<[^>]+>', '', txt).strip()
    print(f"  {clean_txt} -> {href}")

# Let's inspect class names in DDG HTML
result_matches = re.findall(r'<h2\s+[^>]*class=["\']result__title["\'][^>]*>(.*?)</h2>', body, re.DOTALL)
print(f"DDG h2 titles found: {len(result_matches)}")
for rm in result_matches[:3]:
    # extract a href and text
    a_m = re.search(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', rm, re.DOTALL)
    if a_m:
        raw_u = a_m.group(1)
        raw_t = re.sub(r'<[^>]+>', '', a_m.group(2)).strip()
        print(f"  Title: {html.unescape(raw_t)} | Link: {raw_u}")

# 2. Parse Bing
print("\n--- Parsing Bing HTML ---")
url_bing = f"https://www.bing.com/search?q={q_enc}"
req_b = urllib.request.Request(url_bing, headers=headers)
with urllib.request.urlopen(req_b, timeout=8) as resp:
    body_b = resp.read().decode("utf-8", errors="ignore")

# Bing result titles are in <li class="b_algo"><h2><a href="...">...</a></h2>
bing_hits = re.findall(r'<li\s+class=["\']b_algo["\'][^>]*>.*?<h2><a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a></h2>.*?<div\s+class=["\']b_caption["\']>(?:.*?<p>(.*?)</p>)?', body_b, re.DOTALL)
print(f"Bing hits found: {len(bing_hits)}")
for href, t, snip in bing_hits[:3]:
    clean_t = re.sub(r'<[^>]+>', '', t).strip()
    clean_snip = re.sub(r'<[^>]+>', '', snip).strip() if snip else ""
    print(f"  Bing: {clean_t} | {href}")
    print(f"    Snippet: {clean_snip[:120]}")
