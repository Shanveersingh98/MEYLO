import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, re, html

uas = {
    "Android Mobile": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Old Chrome": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36",
    "iPhone Safari": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1",
}

for name, ua in uas.items():
    print(f"\n--- Testing {name} ---")
    url = 'https://www.google.com/search?q=' + urllib.parse.quote('Satya Nadella LinkedIn')
    req = urllib.request.Request(url, headers={'User-Agent': ua, 'Accept-Language': 'en-US,en;q=0.9'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
        print('Length:', len(content))
        links = re.findall(r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', content, re.DOTALL)
        ext_links = []
        for u, t in links:
            txt = html.unescape(re.sub(r'<[^>]+>', '', t)).strip()
            if '/url?q=' in u:
                m = re.search(r'/url\?q=([^&]+)', u)
                if m:
                    actual = urllib.parse.unquote(m.group(1))
                    if not any(k in actual for k in ['google.com', 'support.google']):
                        ext_links.append((actual, txt))
        print(f"Ext links found: {len(ext_links)}")
        for actual, txt in ext_links[:3]:
            print(f"  {txt[:50]} -> {actual}")
    except Exception as e:
        print("Error:", e)
