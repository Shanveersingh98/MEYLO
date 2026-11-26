import urllib.request, urllib.parse

url = "https://html.duckduckgo.com/html/"
data = urllib.parse.urlencode({"q": "Shanveer Singh Khosa LinkedIn", "b": "", "kl": "us-en"}).encode("utf-8")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://duckduckgo.com/",
    "Origin": "https://duckduckgo.com",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/x-www-form-urlencoded",
}

try:
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=8) as resp:
        content = resp.read().decode("utf-8", errors="ignore")
        print("Status code:", resp.status)
        print("Content length:", len(content))
        print("Preview:\n", content[:600])
except Exception as e:
    print("Error:", e)
