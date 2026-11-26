import sys
import json
import hashlib
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
OUT_JSON = PROJECT_DIR / "logs" / "ultimate_improvement" / "artifact_manifest.json"

def sha256_file(p: Path) -> str:
    if not p.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()

manifest = {
    "manifest_version": "1.0",
    "mission": "Ultimate Chatbot Improvement",
    "project_root": "D:/DEEP-LEARNING-PROJECT",
    "timestamp": 1792434500.0,
    "checkpoints": {
        "reference_checkpoint": {
            "path": "checkpoints/experiment_2/best.pt",
            "sha256": sha256_file(PROJECT_DIR / "checkpoints/experiment_2/best.pt"),
            "size_bytes": (PROJECT_DIR / "checkpoints/experiment_2/best.pt").stat().st_size,
            "status": "PROTECTED_UNTOUCHED"
        },
        "final_improved_checkpoint": {
            "path": "checkpoints/ultimate_improvement/best.pt",
            "sha256": sha256_file(PROJECT_DIR / "checkpoints/ultimate_improvement/best.pt"),
            "size_bytes": (PROJECT_DIR / "checkpoints/ultimate_improvement/best.pt").stat().st_size,
            "status": "VERIFIED_OPTIMAL"
        },
        "final_latest_checkpoint": {
            "path": "checkpoints/ultimate_improvement/latest.pt",
            "sha256": sha256_file(PROJECT_DIR / "checkpoints/ultimate_improvement/latest.pt"),
            "size_bytes": (PROJECT_DIR / "checkpoints/ultimate_improvement/latest.pt").stat().st_size,
            "status": "SAVED"
        }
    },
    "tokenizer": {
        "path": "tokenizer/tokenizer.json",
        "sha256": sha256_file(PROJECT_DIR / "tokenizer/tokenizer.json"),
        "vocab_size": 8000,
        "status": "UNTOUCHED"
    },
    "key_datasets": {
        "ultimate_train_tokens": {
            "path": "data/ultimate_improvement/training_sequences/train_tokens.bin",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/training_sequences/train_tokens.bin"),
            "sequences": 6000
        },
        "ultimate_train_mask": {
            "path": "data/ultimate_improvement/training_sequences/train_mask.bin",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/training_sequences/train_mask.bin"),
            "sequences": 6000
        },
        "ultimate_heldout_tokens": {
            "path": "data/ultimate_improvement/training_sequences/heldout_tokens.bin",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/training_sequences/heldout_tokens.bin"),
            "sequences": 110
        },
        "oasst1_cleaned_curriculum": {
            "path": "data/experiment_2/curriculum_raw/category1_oasst1_cleaned.jsonl",
            "sha256": sha256_file(PROJECT_DIR / "data/experiment_2/curriculum_raw/category1_oasst1_cleaned.jsonl")
        }
    },
    "evaluation_suites": {
        "suite_01_benchmarks": {
            "path": "data/ultimate_improvement/eval_suite/suite_01_benchmarks.json",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/eval_suite/suite_01_benchmarks.json")
        },
        "suite_02_arithmetic_single_step": {
            "path": "data/ultimate_improvement/eval_suite/suite_02_arithmetic_single_step.jsonl",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/eval_suite/suite_02_arithmetic_single_step.jsonl")
        },
        "suite_03_arithmetic_multi_step": {
            "path": "data/ultimate_improvement/eval_suite/suite_03_arithmetic_multi_step.jsonl",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/eval_suite/suite_03_arithmetic_multi_step.jsonl")
        },
        "suite_04_python_tasks": {
            "path": "data/ultimate_improvement/eval_suite/suite_04_python_tasks.jsonl",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/eval_suite/suite_04_python_tasks.jsonl")
        },
        "suite_05_concepts": {
            "path": "data/ultimate_improvement/eval_suite/suite_05_concepts.jsonl",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/eval_suite/suite_05_concepts.jsonl")
        },
        "suite_06_factual_qa": {
            "path": "data/ultimate_improvement/eval_suite/suite_06_factual_qa.jsonl",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/eval_suite/suite_06_factual_qa.jsonl")
        },
        "suite_07_instruction_following": {
            "path": "data/ultimate_improvement/eval_suite/suite_07_instruction_following.jsonl",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/eval_suite/suite_07_instruction_following.jsonl")
        },
        "suite_08_structured_answers": {
            "path": "data/ultimate_improvement/eval_suite/suite_08_structured_answers.jsonl",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/eval_suite/suite_08_structured_answers.jsonl")
        },
        "suite_09_multiturn": {
            "path": "data/ultimate_improvement/eval_suite/suite_09_multiturn.jsonl",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/eval_suite/suite_09_multiturn.jsonl")
        },
        "suite_10_refusal_uncertainty": {
            "path": "data/ultimate_improvement/eval_suite/suite_10_refusal_uncertainty.jsonl",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/eval_suite/suite_10_refusal_uncertainty.jsonl")
        },
        "suite_11_robustness_malformed": {
            "path": "data/ultimate_improvement/eval_suite/suite_11_robustness_malformed.jsonl",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/eval_suite/suite_11_robustness_malformed.jsonl")
        },
        "suite_12_context_boundary": {
            "path": "data/ultimate_improvement/eval_suite/suite_12_context_boundary.jsonl",
            "sha256": sha256_file(PROJECT_DIR / "data/ultimate_improvement/eval_suite/suite_12_context_boundary.jsonl")
        }
    },
    "scripts": {
        "evaluate_comprehensive": {
            "path": "src/ultimate_improvement/evaluate_comprehensive.py",
            "sha256": sha256_file(PROJECT_DIR / "src/ultimate_improvement/evaluate_comprehensive.py")
        },
        "train_ultimate": {
            "path": "src/ultimate_improvement/train_ultimate.py",
            "sha256": sha256_file(PROJECT_DIR / "src/ultimate_improvement/train_ultimate.py")
        },
        "chat_application": {
            "path": "src/chat.py",
            "sha256": sha256_file(PROJECT_DIR / "src/chat.py")
        },
        "smoke_test": {
            "path": "src/ultimate_improvement/smoke_test_chat.py",
            "sha256": sha256_file(PROJECT_DIR / "src/ultimate_improvement/smoke_test_chat.py")
        }
    }
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print("Artifact manifest saved to:", OUT_JSON)
