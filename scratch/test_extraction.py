import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request
sys.path.insert(0, "src")
from web_retrieval import extract_text_from_html

url = "https://www.healthline.com/health/chest-pain"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=10) as resp:
    raw_html = resp.read().decode("utf-8", errors="ignore")

print(f"Raw HTML length: {len(raw_html)}")
extracted = extract_text_from_html(raw_html)
print(f"Extracted text length: {len(extracted)}")
print("=" * 60)
print("EXTRACTED PREVIEW (first 1000 chars):")
print("=" * 60)
print(extracted[:1000])
print("=" * 60)
