import urllib.request
import urllib.parse
import re

query = "causes of chest pain"
lite_url = "https://lite.duckduckgo.com/lite/"
data = urllib.parse.urlencode({"q": query}).encode("utf-8")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Content-Type": "application/x-www-form-urlencoded",
}
req = urllib.request.Request(lite_url, data=data, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=8) as resp:
        content = resp.read().decode("utf-8", errors="ignore")
    print(f"Content length: {len(content)}")
    print("Has result-link:", "result-link" in content)
    print("Has challenge/anomaly:", "anomaly" in content or "challenge" in content)
    if not ("result-link" in content):
        print("First 500 chars of content:")
        print(content[:500])
except Exception as e:
    print(f"Error: {e}")
