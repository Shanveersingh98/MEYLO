import urllib.request
import gzip
import re
import html
from html.parser import HTMLParser
import sys
sys.path.insert(0, 'src')
from web_retrieval import RobustContentExtractor, clean_mathematical_notation

VOID_HTML_TAGS = {
    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
    'link', 'meta', 'param', 'source', 'track', 'wbr'
}

headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request('https://en.wikipedia.org/wiki/Bhagwant_Mann', headers=headers)
with urllib.request.urlopen(req, timeout=5) as resp:
    data = resp.read(2000000)
    try:
        data = gzip.decompress(data)
    except:
        pass
    html_str = data.decode('utf-8', errors='ignore')

class TestExtractor(RobustContentExtractor):
    def handle_starttag(self, tag, attrs):
        attrs_dict = {k.lower(): (v or '') for k, v in attrs}
        if tag in ('html', 'body'):
            return
        if self.in_head:
            if tag == 'title':
                self.in_title = True
            elif tag == 'meta':
                name = attrs_dict.get('name', '').lower()
                prop = attrs_dict.get('property', '').lower()
                content = attrs_dict.get('content', '').strip()
                if name == 'description':
                    self.meta_description = content
                elif prop == 'og:title':
                    self.og_title = content
                elif prop == 'og:description':
                    self.og_description = content
            elif tag == 'script' and attrs_dict.get('type') == 'application/ld+json':
                self.in_json_ld = True
                self.current_json_ld = ''
            return

        if tag == 'head':
            self.in_head = True
            return

        if tag in VOID_HTML_TAGS:
            return

        self.tag_stack.append((tag, attrs_dict))

        if tag == 'script' and attrs_dict.get('type') == 'application/ld+json':
            self.in_json_ld = True
            self.current_json_ld = ''
            return

        if tag in self.blacklist_tags:
            self.ignored_depth += 1
            return

        cls_id = f"{attrs_dict.get('class', '')} {attrs_dict.get('id', '')} {attrs_dict.get('role', '')}"
        if self.boilerplate_re.search(cls_id):
            self.ignored_depth += 1
            return

        if self.ignored_depth > 0:
            self.ignored_depth += 1
            return

        if tag in {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr', 'article', 'section', 'blockquote'}:
            self._flush_line()

    def handle_endtag(self, tag):
        if tag in ('html', 'body') or tag in VOID_HTML_TAGS:
            return

        if tag == 'head':
            self.in_head = False
            return

        if self.in_head:
            if tag == 'title':
                self.in_title = False
            elif tag == 'script' and self.in_json_ld:
                self.in_json_ld = False
                if self.current_json_ld.strip():
                    self.json_ld_blocks.append(self.current_json_ld.strip())
            return

        if tag == 'script' and self.in_json_ld:
            self.in_json_ld = False
            if self.current_json_ld.strip():
                self.json_ld_blocks.append(self.current_json_ld.strip())
            return

        if self.ignored_depth > 0:
            self.ignored_depth -= 1
            if self.tag_stack:
                self.tag_stack.pop()
            return

        if self.tag_stack:
            self.tag_stack.pop()

        if tag in {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr', 'article', 'section', 'blockquote'}:
            self._flush_line()

p = TestExtractor()
p.feed(html_str)
txt = p.get_extracted_text(max_chars=1200)
print('Length:', len(txt))
print('Output:\n', txt[:500])
