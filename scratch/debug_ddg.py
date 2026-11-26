import urllib.request
import urllib.parse
import re

url = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote('Python 3.12 release')
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=8) as resp:
        content = resp.read().decode('utf-8', errors='ignore')
        print("html.ddg length:", len(content), "anomaly?:", 'anomaly' in content)
        links = re.findall(r'<a\s+[^>]*class=["\']result__url["\'][^>]*href=["\']([^"\']+)["\']', content)
        print("Links count:", len(links))
        snippets = re.findall(r'<a\s+[^>]*class=["\']result__snippet["\'][^>]*>(.*?)</a>', content)
        print("Snippets count:", len(snippets))
        for s in snippets[:3]:
            print("  snippet:", re.sub(r'<[^>]+>', '', s)[:100])
except Exception as e:
    print("Error:", e)
