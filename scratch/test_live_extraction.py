import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request
from test_html_parser import RobustContentExtractor

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

def test_url_extraction(name, url):
    print(f"\n=== Testing Extraction: {name} ===")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw_html = resp.read().decode("utf-8", errors="ignore")
    parser = RobustContentExtractor()
    parser.feed(raw_html)
    text = parser.get_extracted_text(max_chars=1000)
    print("Length:", len(text))
    print("Content preview:")
    print(text)

test_url_extraction("LinkedIn Satya Nadella", "https://www.linkedin.com/in/satyanadella")
test_url_extraction("Healthline Chest Pain", "https://www.healthline.com/health/chest-pain")
