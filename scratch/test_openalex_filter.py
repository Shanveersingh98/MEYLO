import urllib.request
import urllib.parse
import json

queries = [
    'https://api.openalex.org/works?filter=title.search:agent,title.search:AI&per-page=5&sort=publication_date:desc',
    'https://api.openalex.org/works?search=%22AI%20agents%22&per-page=5&sort=publication_date:desc',
    'https://api.openalex.org/works?filter=title.search:LLM%20agents&per-page=5&sort=publication_date:desc'
]
headers = {"User-Agent": "DeepLearningResearchBot/1.2 (mailto:shanveer@example.com)"}

for q in queries:
    print("Testing query:", q)
    try:
        req = urllib.request.Request(q, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for w in data.get("results", [])[:3]:
                print(f" - {w.get('title')} ({w.get('publication_date')})")
                authors = [a.get("author", {}).get("display_name") for a in w.get("authorships", [])]
                print(f"   Authors: {', '.join(authors[:3])}")
                doi = w.get("doi") or (w.get("primary_location") or {}).get("landing_page_url")
                print(f"   URL: {doi}")
    except Exception as e:
        print("Error:", e)
    print("-" * 50)
