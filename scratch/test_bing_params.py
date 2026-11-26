import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.parse, re, html, base64

def decode_bing_url(bing_ck_url):
    try:
        parsed = urllib.parse.urlparse(bing_ck_url)
        qs = urllib.parse.parse_qs(parsed.query)
        u_val = qs.get("u", [""])[0]
        if u_val.startswith("a1"):
            raw_b64 = u_val[2:]
            rem = len(raw_b64) % 4
            if rem > 0:
                raw_b64 += "=" * (4 - rem)
            return base64.b64decode(raw_b64).decode("utf-8", errors="ignore")
    except Exception:
        pass
    return bing_ck_url

def test_bing_query(q_str):
    print(f"\n--- Bing Query: {q_str} ---")
    url = f"https://www.bing.com/search?q={urllib.parse.quote(q_str)}&rdr=1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8", errors="ignore")

    items = re.findall(r'<li\s+class="b_algo"[^>]*>(.*?)</li>', body, re.DOTALL)
    print(f"Results count: {len(items)}")
    for it in items[:4]:
        link_m = re.search(r'<h2><a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>', it, re.DOTALL)
        snippet_m = re.search(r'<p[^>]*>(.*?)</p>', it, re.DOTALL)
        if link_m:
            u = decode_bing_url(html.unescape(link_m.group(1)))
            t = html.unescape(re.sub(r'<[^>]+>', '', link_m.group(2))).strip()
            s = html.unescape(re.sub(r'<[^>]+>', '', snippet_m.group(1))).strip() if snippet_m else ""
            print(f"  Title: {t}")
            print(f"  URL: {u}")
            print(f"  Snippet: {s[:100]}")
            print("  ---")

test_bing_query('"Satya Nadella" linkedin')
test_bing_query('Satya Nadella CEO Microsoft')
test_bing_query('Shanveer Singh Khosa')
test_bing_query('"Geoffrey Hinton" university')
