import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, json, time

def test_json_api(name, url, headers=None):
    if headers is None:
        headers = {"User-Agent": "FromScratchBot/1.0 (Educational AI Assistant; https://example.org)"}
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            dur = time.time() - t0
            data = resp.read()
            print(f"[{name}] SUCCESS ({dur:.2f}s) - {len(data)} bytes")
            return True, data
    except Exception as e:
        dur = time.time() - t0
        print(f"[{name}] FAILED ({dur:.2f}s) - {e}")
        return False, None

print("=== Testing Authoritative Public Information APIs ===")

# 1. Wikidata (People, companies, entities)
test_json_api("Wikidata Entity Search", "https://www.wikidata.org/w/api.php?action=wbsearchentities&search=Shanveer+Singh+Khosa&language=en&format=json")
test_json_api("Wikidata Satya Nadella", "https://www.wikidata.org/w/api.php?action=wbsearchentities&search=Satya+Nadella&language=en&format=json")

# 2. OpenAlex (Global researchers & academics)
test_json_api("OpenAlex Author Search", "https://api.openalex.org/authors?search=Geoffrey+Hinton")

# 3. PubMed (Authoritative Health / Medical Studies)
test_json_api("PubMed Health Search", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term=chest+pain+causes&retmode=json&retmax=3")

# 4. CrossRef (Scientific & technical publications)
test_json_api("CrossRef Works", "https://api.crossref.org/works?query=deep+learning&rows=2")

# 5. GitHub Users (Public developer profiles)
test_json_api("GitHub Users", "https://api.github.com/search/users?q=Shanveer+Singh+Khosa")

# 6. Wikipedia Fulltext & Lead Extract
test_json_api("Wikipedia Search", "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=chest+pain&format=json")
