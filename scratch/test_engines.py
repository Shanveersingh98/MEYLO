import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, re, html

def test_engine(name, url, headers):
    print(f"\n=== Testing {name} ===")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        print(f"Status: {resp.status}, Length: {len(body)}")
        
        # Look for search result links
        links = re.findall(r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', body, re.DOTALL)
        http_links = [(u, html.unescape(re.sub(r'<[^>]+>', '', t)).strip()) for u, t in links if u.startswith("http") and not any(d in u for d in ["startpage.com", "google.com/policies", "microsoft.com"])]
        print(f"Found {len(http_links)} external links:")
        for u, t in http_links[:5]:
            if t:
                print(f"  {t[:60]} -> {u}")
    except Exception as e:
        print("Error:", e)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Startpage
test_engine("Startpage", "https://www.startpage.com/sp/search?query=Satya+Nadella+LinkedIn", headers)
# Google
test_engine("Google", "https://www.google.com/search?q=Satya+Nadella+LinkedIn", headers)
# Wikipedia API
test_engine("Wikipedia API", "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=Satya+Nadella&format=json", headers)
