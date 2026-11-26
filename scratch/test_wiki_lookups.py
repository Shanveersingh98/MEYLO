import urllib.request, urllib.parse, json

headers = {'User-Agent': 'DeepLearningEducationalBot/1.0 (https://github.com/example/dl-project; contact: student@example.edu) Python-urllib/3.11'}

# Test 1: Direct title lookup
t_quoted = urllib.parse.quote('Supervised learning')
title_url = f'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles={t_quoted}&redirects=1&format=json'
req = urllib.request.Request(title_url, headers=headers)
with urllib.request.urlopen(req) as resp:
    d = json.loads(resp.read().decode())
    pages = d.get('query', {}).get('pages', {})
    for pid, p in pages.items():
        print('Direct Title:', p.get('title'), '| Missing?:', 'missing' in p)
        if 'extract' in p:
            print('  Extract start:', p['extract'][:120])

# Test 2: OpenSearch
opensearch_url = f'https://en.wikipedia.org/w/api.php?action=opensearch&search={t_quoted}&limit=5&format=json'
req2 = urllib.request.Request(opensearch_url, headers=headers)
with urllib.request.urlopen(req2) as resp:
    d2 = json.loads(resp.read().decode())
    print('OpenSearch titles:', d2[1])

# Test 3: Quoted search
q_str = '"supervised learning"'
q_quoted = urllib.parse.quote(q_str)
sr_url = f'https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q_quoted}&utf8=1&format=json&srlimit=5'
req3 = urllib.request.Request(sr_url, headers=headers)
with urllib.request.urlopen(req3) as resp:
    d3 = json.loads(resp.read().decode())
    print('Quoted search titles:', [h['title'] for h in d3.get('query', {}).get('search', [])])
