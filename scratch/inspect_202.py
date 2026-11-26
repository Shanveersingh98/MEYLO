import urllib.request, urllib.parse

url = "https://html.duckduckgo.com/html/"
data = urllib.parse.urlencode({"q": "Shanveer Singh Khosa", "b": ""}).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://duckduckgo.com/",
    "Origin": "https://duckduckgo.com",
})
with urllib.request.urlopen(req) as resp:
    text = resp.read().decode("utf-8", errors="ignore")
    print("Length:", len(text))
    # Look for form or script
    import re
    forms = re.findall(r'<form[^>]*>.*?</form>', text, re.DOTALL)
    print("Forms:", len(forms))
    for f in forms:
        print(f[:300])
