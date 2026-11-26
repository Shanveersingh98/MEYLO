import sys
sys.stdout.reconfigure(encoding="utf-8")
from html.parser import HTMLParser
import re
import html
import json

class RobustContentExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.meta_description = ""
        self.og_title = ""
        self.og_description = ""
        self.json_ld_blocks = []
        
        # State tracking
        self.in_head = False
        self.in_title = False
        self.in_json_ld = False
        self.current_json_ld = ""
        
        # Blacklist tracking
        self.blacklist_tags = {
            "script", "style", "nav", "header", "footer", "aside",
            "noscript", "svg", "form", "select", "button", "iframe",
            "template", "dialog", "canvas", "audio", "video"
        }
        self.ignored_depth = 0
        self.tag_stack = []
        
        # Content accumulation
        self.lines = []
        self.current_line = []
        
        # Patterns for boilerplate classes/ids
        self.boilerplate_re = re.compile(
            r"(?i)\b(nav|menu|sidebar|breadcrumb|breadcrumbs|cookie|banner|advert|ad-|ad_|sponsor|"
            r"popup|modal|dialog|newsletter|promo|signup|social|share|catlinks|category-list|"
            r"categories|related|trending|popular|footer|widget|comments|pagination|disclaimer-banner|"
            r"author-bio|toolbar|header-nav)\b"
        )

    def handle_starttag(self, tag, attrs):
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        self.tag_stack.append((tag, attrs_dict))

        if tag == "head":
            self.in_head = True
            return

        if self.in_head:
            if tag == "title":
                self.in_title = True
            elif tag == "meta":
                name = attrs_dict.get("name", "").lower()
                prop = attrs_dict.get("property", "").lower()
                content = attrs_dict.get("content", "").strip()
                if name == "description":
                    self.meta_description = content
                elif prop == "og:title":
                    self.og_title = content
                elif prop == "og:description":
                    self.og_description = content
            elif tag == "script" and attrs_dict.get("type") == "application/ld+json":
                self.in_json_ld = True
                self.current_json_ld = ""
            return

        if tag == "script" and attrs_dict.get("type") == "application/ld+json":
            self.in_json_ld = True
            self.current_json_ld = ""
            return

        # Check blacklist tag
        if tag in self.blacklist_tags:
            self.ignored_depth += 1
            return

        # Check attributes for boilerplate
        cls_id = f"{attrs_dict.get('class', '')} {attrs_dict.get('id', '')} {attrs_dict.get('role', '')}"
        if self.boilerplate_re.search(cls_id):
            self.ignored_depth += 1
            return

        if self.ignored_depth > 0:
            self.ignored_depth += 1
            return

        # Block-level tags trigger paragraph break
        if tag in {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "article", "section", "blockquote"}:
            self._flush_line()

    def handle_endtag(self, tag):
        if tag == "head":
            self.in_head = False
            return

        if self.in_head:
            if tag == "title":
                self.in_title = False
            elif tag == "script" and self.in_json_ld:
                self.in_json_ld = False
                if self.current_json_ld.strip():
                    self.json_ld_blocks.append(self.current_json_ld.strip())
            return

        if tag == "script" and self.in_json_ld:
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

        if tag in {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "article", "section", "blockquote"}:
            self._flush_line()

    def handle_data(self, data):
        if self.in_title:
            self.title += data
            return
        if self.in_json_ld:
            self.current_json_ld += data
            return
        if self.in_head or self.ignored_depth > 0:
            return

        text = data.strip()
        if text:
            self.current_line.append(data)

    def _flush_line(self):
        if self.current_line:
            full = "".join(self.current_line).strip()
            if full:
                self.lines.append(full)
            self.current_line = []

    def get_extracted_text(self, max_chars=3500):
        self._flush_line()
        
        # Clean lines
        cleaned = []
        ui_phrases = [
            r"skip to (main )?content", r"report this post", r"sign in to (view|see)",
            r"join now to (see|view)", r"accept (all )?cookies", r"cookie policy",
            r"privacy policy", r"terms of (service|use)", r"all rights reserved",
            r"read our editorial process", r"was this page helpful\?", r"share this article",
            r"subscribe to our newsletter", r"advertisement", r"sponsored content"
        ]
        
        category_run = 0
        for line in self.lines:
            line_s = re.sub(r"\s+", " ", line).strip()
            if len(line_s) < 15:
                continue
            if any(re.search(bp, line_s, re.I) for bp in ui_phrases):
                continue
            
            # Filter repeated short category links without punctuation
            has_punct = any(p in line_s for p in [".", ",", ";", ":", "?", "!"])
            words = len(line_s.split())
            if not has_punct and words <= 4:
                category_run += 1
                if category_run > 2:
                    continue
            else:
                category_run = 0
                
            cleaned.append(line_s)
            
        body_text = "\n\n".join(cleaned)
        
        # If body text is very short or empty (e.g. client-hydrated sites like LinkedIn),
        # utilize meta description, og:description, and JSON-LD schema
        if len(body_text) < 100:
            meta_parts = []
            if self.og_title or self.title:
                meta_parts.append(self.og_title or self.title)
            if self.meta_description:
                meta_parts.append(self.meta_description)
            elif self.og_description:
                meta_parts.append(self.og_description)
                
            for b in self.json_ld_blocks:
                try:
                    jd = json.loads(b)
                    if isinstance(jd, dict):
                        desc = jd.get("description") or jd.get("headline")
                        if desc and desc not in meta_parts:
                            meta_parts.append(desc)
                except Exception:
                    pass
            if meta_parts:
                body_text = "\n\n".join(meta_parts)
                
        return body_text[:max_chars].strip()

# Test on the nested div bug sample
html_sample = """
<div id="content">
  <div class="breadcrumbs">
    <a href="/health">Health</a> &gt; <a href="/heart">Heart Disease</a> &gt; Categories
  </div>
  <div class="article-body">
    <h1>Understanding Chest Pain</h1>
    <p>Chest pain is a critical symptom that requires thorough medical evaluation.</p>
    <p>Common serious causes include acute coronary syndrome, angina, and pulmonary embolism.</p>
  </div>
</div>
"""
parser = RobustContentExtractor()
parser.feed(html_sample)
print("=== Output on Nested Div Sample ===")
print(parser.get_extracted_text())
