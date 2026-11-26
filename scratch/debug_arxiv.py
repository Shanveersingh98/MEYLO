import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

url = "https://export.arxiv.org/api/query?search_query=ti:%22AI+agent%22+OR+abs:%22AI+agent%22&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending"
headers = {"User-Agent": "DeepLearningResearchBot/1.2"}
print("Requesting:", url)
try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = resp.read()
        print("Read bytes:", len(data))
        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        print("Entries found:", len(entries))
        for e in entries:
            t = e.find("atom:title", ns)
            print("Title:", t.text.strip().replace("\n", " ") if t is not None else "")
except Exception as e:
    print("Error:", type(e), e)
