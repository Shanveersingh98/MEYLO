import urllib.request
import urllib.parse
import re
import html

query = "Shanveer Singh Khosa LinkedIn"
q_enc = urllib.parse.quote(query)
url = f"https://www.google.com/search?q={q_enc}&hl=en"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=8) as resp:
    body = resp.read().decode("utf-8", errors="ignore")

# Extract search results from Google HTML
# In desktop Google search, results are in <div class="g"> or <a href="/url?q=..." or direct <a href="https://..."><h3>...</h3>
# Let's inspect <h3> tags
h3_matches = re.findall(r'<h3[^>]*>(.*?)</h3>', body)
print(f"Total H3 headers found: {len(h3_matches)}")
for h in h3_matches[:5]:
    print(f"  H3: {html.unescape(re.sub(r'<[^>]+>', '', h))}")

# Look for links containing target
links = re.findall(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(?:<h3[^>]*>(.*?)</h3>)?', body, re.DOTALL)
print(f"Total anchor tags with href found: {len(links)}")
found_results = []
for href, h3 in links:
    if "linkedin" in href.lower() or (h3 and "linkedin" in h3.lower()):
        # Extract clean URL
        actual_url = href
        if "/url?q=" in href:
            actual_url = href.split("/url?q=")[1].split("&")[0]
        actual_url = urllib.parse.unquote(actual_url)
        clean_title = html.unescape(re.sub(r'<[^>]+>', '', h3)).strip() if h3 else actual_url
        found_results.append((clean_title, actual_url))

print(f"Found {len(found_results)} LinkedIn results:")
for t, u in found_results[:5]:
    print(f"  Title: {t}\n  URL: {u}\n")
