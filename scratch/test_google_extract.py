import urllib.request, urllib.parse, re, html

url = "https://www.google.com/search?q=" + urllib.parse.quote("Shanveer Singh Khosa LinkedIn") + "&hl=en"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
        print("Status:", resp.status, "Length:", len(text))
        
        # Save HTML to inspect
        with open("scratch/google_output.html", "w", encoding="utf-8") as f:
            f.write(text)
        print("Saved to scratch/google_output.html")
except Exception as e:
    print("Error:", e)
