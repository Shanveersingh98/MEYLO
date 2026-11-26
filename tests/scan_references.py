import os
from pathlib import Path

def scan():
    search_dirs = [Path("src"), Path("tests"), Path("configs"), Path("frontend/src")]
    files_to_scan = []
    for d in search_dirs:
        if d.exists():
            for ext in ["*.py", "*.ts", "*.tsx", "*.json", "*.bat"]:
                files_to_scan.extend(d.rglob(ext))
                
    # Root bat and py files
    for p in Path(".").glob("*.bat"):
        files_to_scan.append(p)
    for p in Path(".").glob("*.py"):
        files_to_scan.append(p)
        
    print(f"Total source/script files to scan: {len(files_to_scan)}")
    
    ckpt_names = [
        "step25_test.pt", "clean_dataset_test_latest.pt", "clean_dataset_test_best.pt",
        "diagnostic_overfit_latest.pt", "diagnostic_overfit_best.pt",
        "clean_50k_step_50000.pt", "clean_50k_step_40000.pt", "clean_50k_step_30000.pt",
        "clean_50k_step_20000.pt", "clean_50k_step_10000.pt", "clean_50k_step_5000.pt",
        "phase3_latest.pt", "clean_50k_latest.pt", "phase3_best.pt", "clean_50k_best.pt",
        "masked_latest.pt", "masked_best.pt", "eos_v2_pilot", "experiment_2",
        "ultimate_improvement", "best.pt", "best_10000.pt", "latest.pt"
    ]
    
    references = {c: [] for c in ckpt_names}
    
    for f in files_to_scan:
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            for c in ckpt_names:
                if c in content:
                    references[c].append(str(f).replace("\\", "/"))
        except Exception:
            pass
            
    for c, refs in sorted(references.items()):
        print(f"\n[{c}] -> {len(refs)} reference(s):")
        for r in refs:
            print(f"   • {r}")

if __name__ == "__main__":
    scan()
