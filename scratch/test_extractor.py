import re
import html

def clean_mathematical_notation(text: str) -> str:
    """
    Converts raw LaTeX / math markup (e.g. {\\displaystyle p(y\\mid x)}) into readable text p(y | x).
    """
    if not text:
        return ""
    # Strip \displaystyle wrapper
    text = re.sub(r"\{\\displaystyle\s*([^\}]+)\}", r"\1", text)
    # Common math symbols
    replacements = [
        (r"\\mid", "|"),
        (r"\\times", "×"),
        (r"\\cdot", "·"),
        (r"\\leq", "<="),
        (r"\\geq", ">="),
        (r"\\neq", "!="),
        (r"\\approx", "≈"),
        (r"\\in", "in"),
        (r"\\sum", "sum"),
        (r"\\prod", "product"),
        (r"\\alpha", "alpha"),
        (r"\\beta", "beta"),
        (r"\\gamma", "gamma"),
        (r"\\theta", "theta"),
        (r"\\sigma", "sigma"),
        (r"\\mu", "mu"),
        (r"\\pi", "pi"),
        (r"\\int", "integral"),
        (r"\\sqrt\{([^}]+)\}", r"sqrt(\1)"),
        (r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1 / \2)"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    # Strip remaining single backslashes in math
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)
    text = re.sub(r"[{}\\]", "", text)
    return text

def extract_clean_article_text(html_str: str, max_chars: int = 3500) -> str:
    if not html_str:
        return ""

    # 1. Strip raw script, style, comments, head, svg, noscript, etc.
    clean = re.sub(r"<!--.*?-->", " ", html_str, flags=re.DOTALL)
    clean = re.sub(r"<(script|style|nav|footer|header|aside|noscript|svg|form|select|button|iframe)[^>]*>.*?</\1>", " ", clean, flags=re.DOTALL | re.IGNORECASE)

    # 2. Strip elements with navigation/sidebar/advertising/cookie class or id
    boilerplate_pattern = (
        r"<(div|section|aside|ul|ol|table)[^>]*"
        r"(class|id|role)=['\"][^'\"]*"
        r"(nav|menu|sidebar|breadcrumb|cookie|banner|advert|ad-|ad_|sponsor|popup|modal|dialog|newsletter|promo|signup|social|share|catlinks|category|categories|related|trending|popular|footer|widget)"
        r"[^'\"]*['\"][^>]*>.*?</\1>"
    )
    for _ in range(2):  # Two passes for nested boilerplate
        clean = re.sub(boilerplate_pattern, " ", clean, flags=re.DOTALL | re.IGNORECASE)

    # 3. Prefer main/article container if substantive
    main_match = re.search(r"<(?:article|main|div\s+[^>]*class=['\"][^'\"]*(?:article-body|entry-content|post-content|article-content|main-content)[^'\"]*['\"])[^>]*>(.*?)</(?:article|main|div)>", clean, flags=re.DOTALL | re.IGNORECASE)
    if main_match and len(main_match.group(1)) > 300:
        clean = main_match.group(1)

    # 4. Remove UI boilerplate lines (Skip to content, Report post, Cookie preferences, etc.)
    ui_boilerplate_phrases = [
        r"skip to (main )?content",
        r"report this post",
        r"sign in to (view|see)",
        r"join now to (see|view)",
        r"accept (all )?cookies",
        r"cookie policy",
        r"privacy policy",
        r"terms of (service|use)",
        r"all rights reserved",
        r"read our editorial process",
        r"was this page helpful\?",
        r"share this article"
    ]

    # 5. Block elements to newlines
    clean = re.sub(r"</?(?:p|div|h[1-6]|li|blockquote|tr|article|section)[^>]*>", "\n", clean, flags=re.IGNORECASE)

    # 6. Strip all remaining HTML tags
    clean = re.sub(r"<[^>]+>", " ", clean)

    # 7. Unescape HTML entities
    text = html.unescape(clean)

    # 8. Clean math notation
    text = clean_mathematical_notation(text)

    # 9. Clean and filter lines
    raw_lines = text.split("\n")
    cleaned_lines = []
    category_run_count = 0

    for line in raw_lines:
        line_clean = re.sub(r"\s+", " ", line).strip()
        if len(line_clean) < 15:
            continue

        # Check for UI boilerplate
        if any(re.search(r"^" + bp + r"$", line_clean, re.I) for bp in ui_boilerplate_phrases):
            continue

        # Check if line looks like an isolated navigation/category link
        # (e.g. short, title-cased, no sentence punctuation, or category list item)
        has_punctuation = any(p in line_clean for p in [".", ",", ";", ":", "?", "!"])
        word_count = len(line_clean.split())

        if not has_punctuation and word_count <= 4:
            category_run_count += 1
            if category_run_count > 2:
                # Consecutive short category items -> skip
                continue
        else:
            category_run_count = 0

        cleaned_lines.append(line_clean)

    result_text = "\n\n".join(cleaned_lines)
    return result_text[:max_chars].strip()

# Test sample with healthline-like nav and latex math
sample_html = """
<html>
<nav class="site-nav">
  <a href="/home">Home</a>
  <a href="/conditions">Health Conditions</a>
</nav>
<div class="sidebar category-menu">
  <div>Caregiving for Alzheimer's Disease</div>
  <div>Chronic Kidney Disease</div>
  <div>COPD</div>
  <div>Best Vegan and Plant-Based Meal Delivery for 2025</div>
</div>
<article class="article-body">
  <h1>What Are the Causes of Chest Pain and When Do I Need Help?</h1>
  <p>Chest pain is a common symptom that can range from a mild ache to a sharp, crushing sensation. While some causes of chest pain are benign, such as acid reflux or muscle strain, chest pain can also signal a medical emergency such as a myocardial infarction (heart attack).</p>
  <p>If you experience sudden, severe chest pain radiating to your left arm, neck, or jaw accompanied by shortness of breath, you should seek emergency medical attention immediately.</p>
  <p>In probabilistic modeling, conditional probability is defined as {\displaystyle p(y\mid x)}.</p>
</article>
<footer>
  <p>All rights reserved. Cookie Policy.</p>
</footer>
</html>
"""

print("EXTRACTED TEXT:")
print("-" * 50)
print(extract_clean_article_text(sample_html))
print("-" * 50)
