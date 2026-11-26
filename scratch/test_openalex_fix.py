"""
scratch/test_openalex_fix.py
Test improved OpenAlex query generation, date filtering, and diverse topics.
"""
import urllib.request
import urllib.parse
import json
import re
from datetime import datetime

STANDARD_USER_AGENT = "DeepLearningResearchBot/1.2 (https://github.com/shanveer/deep-learning-project; contact: test@example.com) Python-urllib/3.12"

def test_query(raw_query, limit=3):
    today_str = datetime.now().strftime("%Y-%m-%d")
    clean_q = raw_query.strip()
    cleaned_topic = re.sub(
        r"(?i)^(find|search|get|show|give|list)?\s*(\d+|five|four|three|two)?\s*(recent|latest)?\s*(research\s+)?(papers?|studies?|preprints?)\s*(about|on|regarding|for|in)?\s*",
        "",
        clean_q
    ).strip(" ?.!\"'")
    if not cleaned_topic or len(cleaned_topic) < 2:
        cleaned_topic = clean_q

    enc_topic = urllib.parse.quote(cleaned_topic)
    api_url = f"https://api.openalex.org/works?search={enc_topic}&filter=to_publication_date:{today_str}&per-page={limit}&sort=publication_date:desc"
    
    print(f"\nQuery: '{raw_query}' -> Extracted Topic: '{cleaned_topic}'")
    print("API URL:", api_url)
    
    req = urllib.request.Request(api_url, headers={"User-Agent": STANDARD_USER_AGENT})
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    results = data.get("results", [])
    print(f"Results returned: {len(results)}")
    for i, w in enumerate(results, 1):
        title = w.get("title")
        pub_date = w.get("publication_date")
        print(f"  {i}. [{pub_date}] {title[:80]}...")
        if pub_date > today_str:
            print(f"     [ERROR] FUTURE DATE: {pub_date} > {today_str}")

test_query("Find five recent research papers about AI agents.")
test_query("recent papers on cancer research")
test_query("latest research from NVIDIA")
test_query("quantum computing preprints")
