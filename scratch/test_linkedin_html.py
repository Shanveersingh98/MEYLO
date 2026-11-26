import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, re

url = "https://www.linkedin.com/in/satyanadella"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=10) as resp:
    raw_html = resp.read().decode("utf-8", errors="ignore")

title = re.search(r"<title>(.*?)</title>", raw_html)
print("Title:", title.group(1) if title else "No title")

# Check for JSON-LD / schema.org metadata in LinkedIn
json_ld = re.findall(r'<script\s+type="application/ld\+json"[^>]*>(.*?)</script>', raw_html, re.DOTALL)
print(f"Found {len(json_ld)} JSON-LD blocks:")
for i, block in enumerate(json_ld):
    print(f"Block #{i+1}:\n", block.strip()[:300])

# Check meta tags: description, og:description, etc.
metas = re.findall(r'<meta\s+[^>]*name="([^"]+)"[^>]*content="([^"]*)"', raw_html)
metas_og = re.findall(r'<meta\s+[^>]*property="([^"]+)"[^>]*content="([^"]*)"', raw_html)
print("\nMeta tags:")
for k, v in metas + metas_og:
    if any(term in k.lower() for term in ["desc", "title", "name", "author"]):
        print(f"  {k} = {v}")
