import urllib.request, urllib.parse, re

url = "https://www.google.com/search?q=" + urllib.parse.quote("Shanveer Singh Khosa LinkedIn") + "&hl=en"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=5) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
        print("Google status:", resp.status, "Length:", len(text))
        # Look for search results
        links = re.findall(r'<a\s+href="(/url\?q=[^"]+|https://[^"]+linkedin\.com[^"]*)"', text)
        print("Found links:", len(links))
        for l in links[:5]:
            print("  Link:", l)
        
        # Look for titles
        titles = re.findall(r'<h3[^>]*>(.*?)</h3>', text)
        print("Found h3 titles:", len(titles))
        for t in titles[:5]:
            print("  Title:", re.sub(r'<[^>]+>', '', t))
except Exception as e:
    print("Google error:", e)
