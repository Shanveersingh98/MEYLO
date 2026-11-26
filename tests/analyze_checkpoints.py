import json
from pathlib import Path

def print_all_details():
    with open('data/checkpoint_investigation.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"=== FULL DETAILS FOR ALL {len(data)} CHECKPOINTS ===\n")
    for i, d in enumerate(data, 1):
        print(f"[{i:02d}] {d['path']}")
        print(f"     Filename:    {d['filename']}")
        print(f"     Size:        {d['size_mb']} MB ({d['size_bytes']:,} bytes)")
        print(f"     Timestamps:  Modified: {d['modified']} | Created: {d['created']}")
        print(f"     SHA-256:     {d['sha256']}")
        print(f"     Training:    Step: {d['step']} | Epoch: {d['epoch']} | Train Loss: {d['loss']} | Val Loss: {d['val_loss']} | Best Loss: {d['best_loss']}")
        print(f"     States:      Optimizer: {d['has_optimizer']} | Scheduler: {d['has_scheduler']} | GradScaler: {d['has_scaler']} | Param Keys: {d['param_count_keys']}")
        cfg = d.get('config') or d.get('model_kwargs') or {}
        print(f"     Config keys: {list(cfg.keys()) if isinstance(cfg, dict) else cfg}")
        if isinstance(cfg, dict):
            for ck in ['experiment', 'dataset', 'learning_rate', 'peak_lr', 'max_steps', 'context_length', 'vocab_size']:
                if ck in cfg:
                    print(f"       • {ck}: {cfg[ck]}")
        print(f"     Dict keys:   {d['keys']}")
        print()

if __name__ == '__main__':
    print_all_details()
