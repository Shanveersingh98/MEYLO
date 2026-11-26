import urllib.request
import urllib.parse
import json
import re
import html

def search_wikipedia_fulltext(query: str, limit: int = 5):
    clean_q = urllib.parse.quote(query.strip())
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={clean_q}&utf8=1&format=json&srlimit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "FromScratchChatbot/1.0 (Educational AI Project)"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = []
            for hit in data.get("query", {}).get("search", []):
                title = hit.get("title", "")
                snippet = hit.get("snippet", "")
                # clean HTML tags from snippet
                clean_snippet = re.sub(r"<[^>]+>", "", snippet)
                clean_snippet = html.unescape(clean_snippet).strip()
                page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                results.append({
                    "title": title,
                    "url": page_url,
                    "snippet": clean_snippet,
                    "provider": "wikipedia"
                })
            return results
    except Exception as e:
        print("Error:", e)
        return []

print("Results for 'Python 3.12 release notes':")
for r in search_wikipedia_fulltext("Python 3.12 release notes", 3):
    print(" ", r["title"], "->", r["url"], "\n   ", r["snippet"][:100])

print("\nResults for 'What is machine learning':")
for r in search_wikipedia_fulltext("What is machine learning", 3):
    print(" ", r["title"], "->", r["url"], "\n   ", r["snippet"][:100])
