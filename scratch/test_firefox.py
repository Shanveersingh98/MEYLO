import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, re, html

url = 'https://www.google.com/search?q=' + urllib.parse.quote('Satya Nadella LinkedIn')
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=10) as resp:
    content = resp.read().decode('utf-8', errors='ignore')

print('Length:', len(content))
# Find links in Google
links = re.findall(r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', content, re.DOTALL)
print('Links count:', len(links))
for u, t in links:
    txt = html.unescape(re.sub(r'<[^>]+>', '', t)).strip()
    if txt and ("/url?q=" in u or "linkedin.com" in u or (u.startswith("http") and "google" not in u)):
        print(f"FOUND: {txt[:60]} -> {u[:100]}")
