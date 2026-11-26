import urllib.request
import urllib.parse
import json

url = "https://api.openalex.org/works?search=" + urllib.parse.quote("AI agents") + "&per-page=5&sort=publication_date:desc"
headers = {"User-Agent": "DeepLearningResearchBot/1.2 (mailto:shanveer@example.com)"}
print("Requesting OpenAlex works:", url)
try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("Meta count:", data.get("meta", {}).get("count"))
        results = data.get("results", [])
        print("Results length:", len(results))
        for w in results:
            title = w.get("title")
            pub_date = w.get("publication_date")
            doi = w.get("doi") or (w.get("primary_location") or {}).get("landing_page_url")
            authors = [a.get("author", {}).get("display_name") for a in w.get("authorships", [])]
            is_oa = w.get("open_access", {}).get("is_oa")
            print(f" - {title} ({pub_date})")
            print(f"   Authors: {', '.join(authors[:3])}")
            print(f"   URL: {doi}")
            print(f"   Open Access: {is_oa}")
except Exception as e:
    print("Error:", type(e), e)
