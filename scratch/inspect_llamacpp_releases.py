"""
scratch/inspect_llamacpp_releases.py
Check latest llama.cpp releases for Windows portable binaries.
"""
import urllib.request
import json

url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
headers = {"User-Agent": "Mozilla/5.0"}
try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    print("Latest Release Tag:", data.get("tag_name"))
    print("Release Name      :", data.get("name"))
    print("\nAssets for Windows:")
    for a in data.get("assets", []):
        name = a.get("name", "")
        if "win" in name.lower() or "windows" in name.lower():
            size_mb = a.get("size", 0) / (1024**2)
            print(f"  - {name} ({size_mb:.2f} MB)")
            print(f"    URL: {a.get('browser_download_url')}")
except Exception as e:
    print("Error fetching releases:", e)
