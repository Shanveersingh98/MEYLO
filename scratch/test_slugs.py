import sys
sys.stdout.reconfigure(encoding="utf-8")
import urllib.request, urllib.error, re

variations = [
    "https://www.linkedin.com/in/shanveer-singh-khosa",
    "https://www.linkedin.com/in/shanveersinghkhosa",
    "https://www.linkedin.com/in/shanveer-singh",
    "https://www.linkedin.com/in/shanveerkhosa",
    "https://www.linkedin.com/in/shanveer"
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

print("=== Checking LinkedIn URL slugs for Shanveer Singh Khosa ===")
for u in variations:
    try:
        req = urllib.request.Request(u, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            title = re.search(r"<title>(.*?)</title>", body)
            desc = re.search(r'<meta\s+[^>]*name="description"[^>]*content="([^"]*)"', body)
            t_txt = title.group(1) if title else ""
            d_txt = desc.group(1) if desc else ""
            print(f"[FOUND] {u} -> Title: {t_txt} | Desc: {d_txt[:100]}")
    except urllib.error.HTTPError as e:
        print(f"[{e.code}] {u}")
    except Exception as e:
        print(f"[ERR] {u} -> {e}")
