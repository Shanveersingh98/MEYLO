"""
scratch/inspect_qwen_hf.py
Verify official Qwen2.5-1.5B-Instruct-GGUF repo on Hugging Face.
"""
import sys
from huggingface_hub import HfApi

api = HfApi()
repo_id = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"

print(f"Inspecting repository: {repo_id}...")
try:
    info = api.model_info(repo_id)
    print("Model ID      :", info.id)
    print("Author        :", info.author)
    print("Tags          :", info.tags[:10])
    print("Downloads     :", getattr(info, "downloads", "N/A"))
    print("Likes         :", getattr(info, "likes", "N/A"))

    print("\nFiles in repository:")
    files = api.list_repo_files(repo_id)
    gguf_files = [f for f in files if f.endswith(".gguf")]
    for f in gguf_files:
        print("  -", f)
except Exception as e:
    print(f"Error inspecting {repo_id}: {e}")
