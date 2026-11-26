import sys
sys.path.insert(0, 'src')
from web_retrieval import ArxivProvider

p = ArxivProvider()
print("Searching raw prompt...")
r1 = p.search("Find five recent research papers about AI agents", limit=10)
print(f"Results for raw: {len(r1)}")
for r in r1[:5]:
    print(f"  {r.title} | {r.date} | {r.url}")

print("\nSearching keyword query 'AI agents'...")
r2 = p.search("AI agents", limit=10)
print(f"Results for 'AI agents': {len(r2)}")
for r in r2[:5]:
    print(f"  {r.title} | {r.date} | {r.url}")
