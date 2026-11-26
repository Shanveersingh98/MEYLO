import urllib.request, urllib.parse, json

services = [
    ("Google News RSS", "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"),
    ("Semantic Scholar Authors", "https://api.semanticscholar.org/graph/v1/author/search?query={q}&fields=name,url,affiliations,paperCount"),
    ("Crossref Works", "https://api.crossref.org/works?query={q}&rows=3"),
    ("OpenAlex Works", "https://api.openalex.org/works?search={q}"),
    ("Wikidata Entities", "https://www.wikidata.org/w/api.php?action=wbsearchentities&search={q}&language=en&format=json"),
    ("Wikipedia OpenSearch", "https://en.wikipedia.org/w/api.php?action=opensearch&search={q}&limit=3&namespace=0&format=json")
]

q = "Shanveer Singh Khosa"
for name, url_template in services:
    url = url_template.format(q=urllib.parse.quote(q))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FromScratchBot/1.0 (Educational)"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            print(f"[{name}] HTTP {resp.status}, Len: {len(data)}")
    except Exception as e:
        print(f"[{name}] Err: {e}")
