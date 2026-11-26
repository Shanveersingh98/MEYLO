import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
from web_retrieval import WikipediaProvider
import re

p = WikipediaProvider()

def find_role_person(role, company):
    comp_res = p.fetch_exact_concept(company)
    if not comp_res or not comp_res.full_text:
        return None
    text = comp_res.full_text
    
    # Check CEO
    if role.lower() == "ceo":
        # Search for pattern: [Name] (became|is|as) CEO
        m = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:became|is|was\s+appointed|serves\s+as)\s+CEO", text)
        if m:
            person = m.group(1).strip()
            print(f"Extracted CEO of {company}: {person}")
            person_res = p.fetch_exact_concept(person)
            return person_res, comp_res
    elif role.lower() in ["founder", "founded"]:
        # Search for pattern: founded ... by [Names]
        m = re.search(r"founded\s+in\s+\d+\s+.*?by\s+([^.]+)", text, re.I)
        if m:
            founders = m.group(1).strip()
            print(f"Extracted founders of {company}: {founders}")
            return comp_res, None
    return None

print("Microsoft:", find_role_person("CEO", "Microsoft"))
print("Apple Inc:", find_role_person("founder", "Apple Inc."))
