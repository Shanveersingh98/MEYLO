import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, json, re, time

print("=== Testing Wikidata Provider ===")
def wikidata_search(query, limit=5, timeout=5):
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={urllib.parse.quote(query)}&language=en&format=json&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "FromScratchBot/1.0 (Educational AI Assistant)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = []
            for item in data.get("search", []):
                results.append({
                    "title": item.get("label", ""),
                    "description": item.get("description", ""),
                    "url": item.get("concepturi", ""),
                    "id": item.get("id")
                })
            return results
    except Exception as e:
        print("Wikidata error:", e)
        return []

print("Wikidata Satya Nadella:", wikidata_search("Satya Nadella"))
print("Wikidata Shanveer Singh Khosa:", wikidata_search("Shanveer Singh Khosa"))
print("Wikidata PyTorch:", wikidata_search("PyTorch"))

print("\n=== Testing PubMed Health Provider ===")
def pubmed_search(query, limit=3, timeout=5):
    enc_q = urllib.parse.quote(query)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={enc_q}&retmode=json&retmax={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "FromScratchBot/1.0 (Medical Information)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            id_list = data.get("esearchresult", {}).get("idlist", [])
            print("PubMed PMC IDs:", id_list)
            return id_list
    except Exception as e:
        print("PubMed error:", e)
        return []

pubmed_search("causes of chest pain emergency signs")
