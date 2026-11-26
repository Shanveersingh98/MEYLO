import urllib.request
import json

try:
    url = "https://huggingface.co/api/datasets/databricks/databricks-dolly-15k"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as response:
        data = json.loads(response.read().decode())
        print("Success fetching dataset metadata:", data.get("id"), "license:", data.get("cardData", {}).get("license"))
except Exception as e:
    print("Network test failed:", e)
