import urllib.request
import urllib.parse
import re

url = "https://html.duckduckgo.com/html/"
data = urllib.parse.urlencode({
    "q": "causes of chest pain",
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
try:
    with urllib.request.urlopen(req, timeout=8) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    print("Response len:", len(body))
    # Look for results
    links = re.findall(r'<a[^>]*class="result__snippet"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', body)
    print("Found snippet links:", len(links))
    titles = re.findall(r'<a[^>]*class="result__url"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', body)
    print("Found url links:", len(titles))
    if not links and not titles:
        # Check what HTML contains
        print("Page title:", re.findall(r'<title>(.*?)</title>', body))
except Exception as e:
    print("Error:", e)
