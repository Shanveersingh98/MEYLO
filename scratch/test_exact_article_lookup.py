import urllib.request, urllib.parse, json

headers = {'User-Agent': 'DeepLearningEducationalBot/1.0 (https://github.com/example/dl-project; contact: student@example.edu) Python-urllib/3.11'}

concepts = [
    'supervised learning',
    'self-supervised learning',
    'unsupervised learning',
    'semi-supervised learning',
    'reinforcement learning',
    'machine learning',
    'deep learning',
    'neural network',
    'ChatGPT'
]

print("=== EXACT WIKIPEDIA ARTICLE LOOKUP ===")
for c in concepts:
    t_quoted = urllib.parse.quote(c)
    title_url = f'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles={t_quoted}&redirects=1&format=json'
    req = urllib.request.Request(title_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            d = json.loads(resp.read().decode())
            pages = d.get('query', {}).get('pages', {})
            for pid, p in pages.items():
                is_missing = 'missing' in p
                title = p.get('title')
                extract = p.get('extract', '')[:80].replace('\n', ' ')
                print(f"Concept: {c:25} -> Found: {not is_missing} | Title: {title:30} | {extract}")
    except Exception as e:
        print(f"Concept: {c:25} -> Error: {e}")
