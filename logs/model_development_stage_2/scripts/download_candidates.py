"""
download_candidates.py
======================
Downloads candidate open datasets for Stage 2 into data/stage_2_candidates/raw/
Preserves raw files in their native JSON/JSONL format.
Includes metadata and license verification.
"""

import os
import sys
import json
import shutil
import urllib.request
from pathlib import Path

STAGE2_RAW_DIR = Path(r"D:\DEEP-LEARNING-PROJECT\data\stage_2_candidates\raw")
PROJECT_ROOT = Path(r"D:\DEEP-LEARNING-PROJECT")

def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"  [Already Exists] {dest_path.name} ({dest_path.stat().st_size:,} bytes)")
        return
    print(f"  Downloading {url} -> {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"  Done: {dest_path.name} ({dest_path.stat().st_size:,} bytes)")

def setup_dolly_15k():
    print("\n--- 1. Databricks Dolly 15K ---")
    dolly_dir = STAGE2_RAW_DIR / "dolly_15k"
    dolly_dir.mkdir(parents=True, exist_ok=True)
    # Direct raw JSONL from Hugging Face databricks/databricks-dolly-15k repository
    url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
    dest = dolly_dir / "databricks-dolly-15k.jsonl"
    download_file(url, dest)
    
    # Save metadata
    meta = {
        "dataset_name": "Databricks Dolly 15K",
        "huggingface_id": "databricks/databricks-dolly-15k",
        "official_source": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "license": "Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)",
        "intended_purpose": "Instruction following, open/closed Q&A, brainstorming, summarization",
        "raw_file": str(dest)
    }
    with open(dolly_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def setup_gsm8k():
    print("\n--- 2. OpenAI GSM8K (Grade School Math 8K) ---")
    gsm_dir = STAGE2_RAW_DIR / "gsm8k"
    gsm_dir.mkdir(parents=True, exist_ok=True)
    # Direct raw JSONL files for train and test splits from openai/gsm8k repository
    train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    
    download_file(train_url, gsm_dir / "train.jsonl")
    download_file(test_url, gsm_dir / "test.jsonl")
    
    meta = {
        "dataset_name": "OpenAI GSM8K",
        "huggingface_id": "openai/gsm8k",
        "official_source": "https://github.com/openai/grade-school-math",
        "license": "MIT License",
        "intended_purpose": "Step-by-step arithmetic and multi-step reasoning with natural language solution traces",
        "raw_files": [str(gsm_dir / "train.jsonl"), str(gsm_dir / "test.jsonl")]
    }
    with open(gsm_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def setup_sciq():
    print("\n--- 3. AllenAI SciQ (Science Exam QA with Explanations) ---")
    sciq_dir = STAGE2_RAW_DIR / "sciq"
    sciq_dir.mkdir(parents=True, exist_ok=True)
    # SciQ raw json splits from Allen Institute for AI
    train_url = "https://ai2-public-datasets.s3.amazonaws.com/sciq/sciq.tar.gz"
    # Or from HuggingFace SciQ parquet/json
    hf_train_url = "https://huggingface.co/datasets/allenai/sciq/resolve/main/data/train-00000-of-00001.parquet"
    hf_val_url = "https://huggingface.co/datasets/allenai/sciq/resolve/main/data/validation-00000-of-00001.parquet"
    hf_test_url = "https://huggingface.co/datasets/allenai/sciq/resolve/main/data/test-00000-of-00001.parquet"
    
    download_file(hf_train_url, sciq_dir / "train.parquet")
    download_file(hf_val_url, sciq_dir / "validation.parquet")
    download_file(hf_test_url, sciq_dir / "test.parquet")
    
    meta = {
        "dataset_name": "AllenAI SciQ",
        "huggingface_id": "allenai/sciq",
        "official_source": "https://allenai.org/data/sciq",
        "license": "Creative Commons Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0)",
        "intended_purpose": "Educational multiple-choice and conceptual science questions with detailed human-written support explanations",
        "raw_files": [str(sciq_dir / "train.parquet"), str(sciq_dir / "validation.parquet"), str(sciq_dir / "test.parquet")]
    }
    with open(sciq_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def setup_oasst1_reference():
    print("\n--- 4. OpenAssistant OASST1 (Preserved Existing Data) ---")
    oasst_dir = STAGE2_RAW_DIR / "oasst1"
    oasst_dir.mkdir(parents=True, exist_ok=True)
    
    # We already have train_paths.jsonl and validation_paths.jsonl in data/processed/conversation_paths/
    # And cached arrow files in .cache/datasets/OpenAssistant___oasst1/
    existing_train = PROJECT_ROOT / "data" / "processed" / "conversation_paths" / "train_paths.jsonl"
    existing_val = PROJECT_ROOT / "data" / "processed" / "conversation_paths" / "validation_paths.jsonl"
    
    meta = {
        "dataset_name": "OpenAssistant OASST1 (English Filtered)",
        "huggingface_id": "OpenAssistant/oasst1",
        "official_source": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "license": "Apache License 2.0",
        "intended_purpose": "Human-assistant conversational dialogues, role-following, polite multi-turn chit-chat",
        "existing_paths": {
            "train": str(existing_train),
            "validation": str(existing_val)
        },
        "note": "Preserved from existing project datasets without re-downloading per instruction #1"
    }
    with open(oasst_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Referenced existing files:\n    {existing_train}\n    {existing_val}")

def main():
    print("=" * 70)
    print("STAGE 2: CANDIDATE DATASET DOWNLOAD & PRESERVATION")
    print("=" * 70)
    setup_dolly_15k()
    setup_gsm8k()
    setup_sciq()
    setup_oasst1_reference()
    print("\nAll candidate datasets successfully set up under data/stage_2_candidates/raw/")

if __name__ == "__main__":
    main()
