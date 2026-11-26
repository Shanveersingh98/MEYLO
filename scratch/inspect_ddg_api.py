import urllib.request
import urllib.parse
import json

queries = [
    "What is machine learning",
    "Python 3.12 release notes updates",
    "artificial intelligence",
    "photosynthesis"
]

for q in queries:
    url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(q)}&format=json&no_html=1"
    req = urllib.request.Request(url, headers={"User-Agent": "FromScratchChatbot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            abstract = data.get("AbstractText", "")
            heading = data.get("Heading", "")
            url_abs = data.get("AbstractURL", "")
            topics = data.get("RelatedTopics", [])
            print(f"=== Query: {q} ===")
            print(f"  Heading: {heading}")
            print(f"  AbstractURL: {url_abs}")
            print(f"  AbstractText: {abstract[:120]}")
            print(f"  RelatedTopics count: {len(topics)}")
            for t in topics[:2]:
                if "Text" in t:
                    print(f"    Topic: {t.get('Text')[:80]} -> {t.get('FirstURL')}")
                elif "Topics" in t:
                    for sub in t["Topics"][:1]:
                        print(f"    SubTopic: {sub.get('Text')[:80]} -> {sub.get('FirstURL')}")
    except Exception as e:
        print(f"Query {q} error:", e)
