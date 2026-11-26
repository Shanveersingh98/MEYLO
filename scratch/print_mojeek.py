import urllib.request
import urllib.parse

query = "Shanveer Singh Khosa LinkedIn"
q_enc = urllib.parse.quote(query)
url = f"https://www.mojeek.com/search?q={q_enc}"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=8) as resp:
    body = resp.read().decode("utf-8", errors="ignore")

print(body)
