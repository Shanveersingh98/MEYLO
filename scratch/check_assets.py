"""
scratch/check_assets.py
"""
import urllib.request
import json

url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/tags/b12567"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=10) as r:
    rel = json.loads(r.read().decode())

print("Tag:", rel.get("tag_name"))
for a in rel.get("assets", []):
    name = a["name"]
    if "win" in name.lower() or "x64" in name.lower():
        size_mb = a["size"] / (1024**2)
        print(f"{name:50} | {size_mb:6.2f} MB | {a['browser_download_url']}")
