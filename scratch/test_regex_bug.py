import re

html_sample = """
<div id="content">
  <div class="breadcrumbs">
    <a href="/health">Health</a> &gt; <a href="/heart">Heart Disease</a> &gt; Categories
  </div>
  <div class="article-body">
    <h1>Understanding Chest Pain</h1>
    <p>Chest pain is a critical symptom that requires thorough evaluation...</p>
    <p>Common causes include myocardial infarction, angina, and costochondritis.</p>
  </div>
</div>
"""

pattern = r"<div\b[^>]*id=[\"'](?:mw-content-text|bodyContent|content)[\"'][^>]*>(.*?)</div>"
m = re.search(pattern, html_sample, re.DOTALL | re.IGNORECASE)
if m:
    print("MATCHED CONTENT:")
    print(repr(m.group(1)))
