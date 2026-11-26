import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, re, html

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

url = "https://www.google.com/search?q=" + urllib.parse.quote("Satya Nadella LinkedIn")
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=10) as resp:
    body = resp.read().decode("utf-8", errors="ignore")

print("Google body length:", len(body))

# Check for CAPTCHA
if "Our systems have detected unusual traffic" in body or "/sorry/" in body:
    print("Google CAPTCHA detected")
else:
    print("No CAPTCHA!")

# Look for links in Google: <a href="/url?q=... or <a jsname=... href="http...
links = re.findall(r'<a\s+[^>]*href="(/url\?q=[^"&]+|https://[^"]+)"[^>]*>(.*?)</a>', body, re.DOTALL)
print(f"Total matching a tags: {len(links)}")
for u, t in links[:15]:
    clean_t = re.sub(r'<[^>]+>', '', t).strip()
    if clean_t and not any(k in u for k in ["google.com", "gstatic.com"]):
        print(f"  {clean_t[:60]} -> {u}")
