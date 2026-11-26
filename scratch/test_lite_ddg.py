import urllib.request, urllib.parse, re

data = urllib.parse.urlencode({"q": "Shanveer Singh Khosa LinkedIn"}).encode("utf-8")
req = urllib.request.Request(
    "https://lite.duckduckgo.com/lite/",
    data=data,
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://lite.duckduckgo.com/",
        "Origin": "https://lite.duckduckgo.com",
        "Content-Type": "application/x-www-form-urlencoded"
    }
)
try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
        print("Lite DDG status:", resp.status, "len:", len(text))
        print("Preview:\n", text[:300])
except Exception as e:
    print("Lite DDG error:", e)
