import urllib.request
import re
import html

def extract_text_clean(raw_html: str, max_chars: int = 3500) -> str:
    if not raw_html:
        return ""
    # 1. Strip non-content tags
    clean = re.sub(r'<!--.*?-->', ' ', raw_html, flags=re.DOTALL)
    clean = re.sub(r'<(script|style|nav|footer|header|aside|noscript|svg|form|select|button|iframe)[^>]*>.*?</\1>', ' ', clean, flags=re.DOTALL|re.IGNORECASE)
    
    # 2. Extract specific article or main container if present
    content = None
    for pattern in [
        r'<article\b[^>]*>(.*?)</article>',
        r'<main\b[^>]*>(.*?)</main>',
        r'<div\b[^>]*id=["\'](?:mw-content-text|bodyContent|content)["\'][^>]*>(.*?)</div>',
        r'<div\b[^>]*class=["\'][^"\']*(?:article-body|entry-content|post-content|article-content|main-content)[^"\']*["\'][^>]*>(.*?)</div>'
    ]:
        m = re.search(pattern, clean, re.DOTALL|re.IGNORECASE)
        if m and len(m.group(1)) > 300:
            content = m.group(1)
            break
            
    if not content:
        content = clean
        
    # 3. Strip boilerplate divs
    boilerplate = r'<(div|section|aside|ul|ol|table)[^>]*class=["\'][^"\']*(?:sidebar|breadcrumb|cookie|banner|advert|ad-|ad_|sponsor|popup|newsletter|promo|social|share|catlinks|category)["\'][^>]*>.*?</\1>'
    for _ in range(2):
        content = re.sub(boilerplate, ' ', content, flags=re.DOTALL|re.IGNORECASE)
        
    # 4. Convert block elements to newlines
    content = re.sub(r'</?(?:p|div|h[1-6]|li|blockquote|tr|article|section)[^>]*>', '\n', content, flags=re.IGNORECASE)
    content = re.sub(r'<[^>]+>', ' ', content)
    text = html.unescape(content)
    lines = [re.sub(r'\s+', ' ', l).strip() for l in text.split('\n') if len(re.sub(r'\s+', ' ', l).strip()) > 20]
    return '\n'.join(lines[:10])

if __name__ == "__main__":
    url = "https://www.medicalnewstoday.com/articles/321650"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw_html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")
    result = extract_text_clean(raw_html)
    print("Extracted from Medical News Today:\n" + result[:500])
