import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

urls = [
    "https://export.arxiv.org/api/query?search_query=ti:agent+AND+cat:cs.AI&max_results=5&sortBy=submittedDate&sortOrder=descending",
    "http://export.arxiv.org/api/query?search_query=all:agent&max_results=3"
]

for url in urls:
    print("Trying:", url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read()
            print("Success! Bytes:", len(data))
    except Exception as e:
        print("Failed:", e)
