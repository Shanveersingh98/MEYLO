import hashlib
import json
from pathlib import Path
from datetime import datetime
import torch

def hash_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 4):
            h.update(chunk)
    return h.hexdigest().upper()

def inspect_all():
    ckpt_dir = Path("checkpoints")
    files = sorted(list(ckpt_dir.rglob("*.pt")) + list(ckpt_dir.rglob("*.pth")))
    
    results = []
    
    for f in files:
        rel_path = str(f).replace("\\", "/")
        size_bytes = f.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        ctime = datetime.fromtimestamp(f.stat().st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        sha = hash_file(f)
        
        info = {
            "path": rel_path,
            "filename": f.name,
            "size_mb": round(size_mb, 2),
            "size_bytes": size_bytes,
            "modified": mtime,
            "created": ctime,
            "sha256": sha,
            "is_dict": False,
            "keys": [],
            "step": None,
            "epoch": None,
            "loss": None,
            "val_loss": None,
            "best_loss": None,
            "config": None,
            "model_kwargs": None,
            "has_optimizer": False,
            "has_scheduler": False,
            "has_scaler": False,
            "param_count_keys": 0,
            "other_metadata": {}
        }
        
        try:
            ckpt = torch.load(str(f), map_location="cpu")
            if isinstance(ckpt, dict):
                info["is_dict"] = True
                info["keys"] = list(ckpt.keys())
                
                # Check known scalar/dict keys
                for k in ["step", "global_step", "iteration", "iter"]:
                    if k in ckpt:
                        info["step"] = ckpt[k]
                        break
                        
                for k in ["epoch", "current_epoch"]:
                    if k in ckpt:
                        info["epoch"] = ckpt[k]
                        break
                        
                for k in ["loss", "train_loss", "current_loss"]:
                    if k in ckpt:
                        v = ckpt[k]
                        info["loss"] = float(v) if isinstance(v, (int, float, torch.Tensor)) else str(v)
                        break
                        
                for k in ["val_loss", "eval_loss", "validation_loss"]:
                    if k in ckpt:
                        v = ckpt[k]
                        info["val_loss"] = float(v) if isinstance(v, (int, float, torch.Tensor)) else str(v)
                        break
                        
                for k in ["best_val_loss", "best_loss", "min_val_loss"]:
                    if k in ckpt:
                        v = ckpt[k]
                        info["best_loss"] = float(v) if isinstance(v, (int, float, torch.Tensor)) else str(v)
                        break
                        
                if "config" in ckpt:
                    cfg = ckpt["config"]
                    if isinstance(cfg, dict):
                        info["config"] = cfg
                    else:
                        info["config"] = str(cfg)
                        
                if "model_kwargs" in ckpt:
                    kw = ckpt["model_kwargs"]
                    if isinstance(kw, dict):
                        info["model_kwargs"] = kw
                    else:
                        info["model_kwargs"] = str(kw)
                        
                info["has_optimizer"] = ("optimizer_state_dict" in ckpt or "optimizer" in ckpt)
                info["has_scheduler"] = ("scheduler_state_dict" in ckpt or "scheduler" in ckpt or "lr_scheduler" in ckpt)
                info["has_scaler"] = ("scaler_state_dict" in ckpt or "scaler" in ckpt)
                
                if "model_state_dict" in ckpt:
                    info["param_count_keys"] = len(ckpt["model_state_dict"])
                elif "state_dict" in ckpt:
                    info["param_count_keys"] = len(ckpt["state_dict"])
                else:
                    # Raw state dict
                    info["param_count_keys"] = len(ckpt)
                    
                # Collect remaining scalar metadata
                for k, v in ckpt.items():
                    if k not in ["model_state_dict", "state_dict", "optimizer_state_dict", "optimizer", "scheduler_state_dict", "scheduler", "scaler_state_dict", "scaler"]:
                        if isinstance(v, (int, float, str, bool)):
                            info["other_metadata"][k] = v
            else:
                info["is_dict"] = False
                info["other_metadata"]["type"] = str(type(ckpt))
        except Exception as e:
            info["error"] = str(e)
            
        results.append(info)
        print(f"Inspected: {rel_path} (step={info['step']}, loss={info['loss']}, val_loss={info['val_loss']})")
        
    out_file = Path("data/checkpoint_investigation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote full investigation results to {out_file}")

if __name__ == "__main__":
    inspect_all()
