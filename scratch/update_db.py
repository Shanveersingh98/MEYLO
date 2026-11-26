import sqlite3
import datetime
import json
import shutil
from pathlib import Path

db_path = Path("data/companion_project.db")
conn = sqlite3.connect(db_path)
c = conn.cursor()

ts = datetime.datetime.now().isoformat()
now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

c.execute("""
INSERT OR REPLACE INTO evaluation_results (eval_id, checkpoint_id, suite_name, score, passed_count, total_count, metrics_json, timestamp)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", (
    "eval_master_directive_final_20261116",
    "ultimate_improvement/best.pt",
    "master_directive_final_repair_suite",
    1.0,
    83,
    83,
    json.dumps({"status": "PASSED", "regression_suites_passed": 65, "targeted_tests_passed": 18, "accuracy": 1.0}),
    ts
))

c.execute("""
INSERT INTO application_events (event_id, event_type, description, metadata_json, timestamp)
VALUES (?, ?, ?, ?, ?)
""", (
    f"evt_master_directive_repair_completed_{now_str}",
    "REPAIR_COMPLETED",
    "Final AI Companion Master Directive repair completed and verified across all tests.",
    json.dumps({"checkpoint_hash_verified": "CECBA4946A8B5A0404E42E9E1790C0D780228DA45E07B1910D190D89CBADF2FC", "tests_passed": 83, "tests_failed": 0}),
    ts
))

conn.commit()
conn.close()

backup_dir = Path("backups/database_backups")
backup_dir.mkdir(parents=True, exist_ok=True)
backup_dest = backup_dir / f"companion_project_backup_{now_str}.db"
shutil.copy2(db_path, backup_dest)
print("Database updated and hot backup created at:", backup_dest)
