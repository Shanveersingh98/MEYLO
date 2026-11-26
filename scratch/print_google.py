import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, re

url = 'https://www.google.com/search?q=' + urllib.parse.quote('Satya Nadella LinkedIn')
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
with urllib.request.urlopen(req, timeout=10) as resp:
    content = resp.read().decode('utf-8', errors='ignore')

title = re.search(r'<title>(.*?)</title>', content)
print("Title:", title.group(1) if title else "No title")
print("First 800 chars:\n", content[:800])
