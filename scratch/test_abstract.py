import urllib.request
import urllib.parse
import json

def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    words_by_pos = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words_by_pos[pos] = word
    sorted_pos = sorted(words_by_pos.keys())
    return " ".join(words_by_pos[p] for p in sorted_pos)

url = "https://api.openalex.org/works?filter=title.search:agent,title.search:AI&per-page=5&sort=publication_date:desc"
headers = {"User-Agent": "DeepLearningResearchBot/1.2 (mailto:shanveer@example.com)"}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=8) as resp:
    data = json.loads(resp.read().decode("utf-8"))

for w in data.get("results", [])[:3]:
    title = w.get("title")
    authors = [a.get("author", {}).get("display_name") for a in w.get("authorships", [])]
    pub_date = w.get("publication_date")
    abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
    doi = w.get("doi") or (w.get("primary_location") or {}).get("landing_page_url")
    is_oa = w.get("open_access", {}).get("is_oa")
    
    print(f"Title: {title}")
    print(f"Authors: {', '.join(authors[:4])}")
    print(f"Date: {pub_date}")
    print(f"URL: {doi}")
    print(f"Abstract preview: {abstract[:200]}...")
    print("-" * 50)
