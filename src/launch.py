"""
src/launch.py
=============
Robust, zero-friction launcher for Local AI Companion:
1. Detects if AI Companion backend is already running on port 8000.
2. If not running, launches src/api_server.py as a clean detached background process.
3. Redirects startup logs to logs/server_startup.log for instant debugging if needed.
4. Polls /api/health with live visual progress feedback.
5. Opens default browser to http://127.0.0.1:8000 automatically.
6. Exits launcher cleanly so CMD window closes automatically for the user.
"""

import os
import sys
import time
import json
import subprocess
import webbrowser
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SERVER_URL = "http://127.0.0.1:8000"
HEALTH_URL = f"{SERVER_URL}/api/health"
LOG_FILE = PROJECT_DIR / "logs" / "server_startup.log"


def is_server_ready() -> bool:
    """Checks if the local backend server is running and ready."""
    try:
        req = urllib.request.Request(HEALTH_URL)
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("status") == "ok"
    except Exception:
        return False
    return False


def launch():
    print("=" * 65)
    print("                      MEYLO (v2.0)")
    print("                 Your AI. Your Space.")
    print("=" * 65)
    print()

    # Step 1: Check if already running
    if is_server_ready():
        print("[INFO] MEYLO backend is already active!")
        print(f"Opening browser at: {SERVER_URL}")
        try:
            webbrowser.open(SERVER_URL)
        except Exception:
            os.system(f"start {SERVER_URL}")
        time.sleep(1.0)
        sys.exit(0)

    # Step 2: Spawn api_server.py in the background
    print("[1/3] Starting backend server & initializing AI models...")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    server_script = str(PROJECT_DIR / "src" / "api_server.py")
    python_exe = sys.executable

    # Windows detached flags
    CREATE_NO_WINDOW = 0x08000000
    DETACHED_PROCESS = 0x00000008

    with open(LOG_FILE, "a", encoding="utf-8") as log_out:
        log_out.write(f"\n--- Startup Attempt: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log_out.flush()
        try:
            proc = subprocess.Popen(
                [python_exe, server_script],
                stdout=log_out,
                stderr=log_out,
                cwd=str(PROJECT_DIR),
                creationflags=CREATE_NO_WINDOW | DETACHED_PROCESS
            )
        except Exception as e:
            print(f"[ERROR] Failed to spawn backend process: {e}")
            input("Press Enter to exit...")
            sys.exit(1)

    # Step 3: Poll health endpoint until ready (up to 30 seconds)
    print("[2/3] Loading PyTorch neural model checkpoint into memory...", end="", flush=True)
    ready = False
    start_time = time.time()
    for i in range(60):  # 60 * 0.5s = 30 seconds
        time.sleep(0.5)
        print(".", end="", flush=True)
        if is_server_ready():
            ready = True
            break

    elapsed = round(time.time() - start_time, 1)
    print()

    if not ready:
        print()
        print(f"[ERROR] Backend did not respond within {elapsed}s.")
        print(f"Check startup diagnostic log at: {LOG_FILE}")
        if LOG_FILE.exists():
            print("\nRecent Log Tail:")
            with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                print("".join(lines[-15:]))
        print()
        input("Press Enter to exit...")
        sys.exit(1)

    # Step 4: Open browser and exit
    print(f"[3/3] Ready in {elapsed}s! Launching browser...")
    print(f"URL: {SERVER_URL}")
    try:
        webbrowser.open(SERVER_URL)
    except Exception:
        os.system(f"start {SERVER_URL}")

    time.sleep(1.5)
    sys.exit(0)


if __name__ == "__main__":
    launch()
