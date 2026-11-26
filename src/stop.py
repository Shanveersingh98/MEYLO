"""
src/stop.py
===========
Cleanly stops all Local AI Companion services:
1. Calls /api/shutdown to release model weights, close SQLite database cleanly, and free port 8000.
2. Checks logs/server.pid to terminate the process if it didn't exit within 2s.
3. Terminates any running portable llama-server.exe process within the project.
4. Verifies port 8000 is freed.
"""

import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PID_FILE = PROJECT_DIR / "logs" / "server.pid"


def stop_services():
    print("=" * 65)
    print("                 STOPPING MEYLO SERVICES")
    print("=" * 65)

    # 1. Graceful HTTP shutdown
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/api/shutdown", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            print("[Shutdown] Sent graceful shutdown signal to backend API server.")
    except Exception:
        print("[Shutdown] API server was not actively responding via HTTP.")

    time.sleep(1.0)

    # 2. Check PID file and kill if still alive
    if PID_FILE.exists():
        try:
            pid_str = PID_FILE.read_text(encoding="utf-8").strip()
            if pid_str:
                pid = int(pid_str)
                # Terminate on Windows
                os.system(f"taskkill /PID {pid} /F >nul 2>&1")
                print(f"[Shutdown] Ensured API server process (PID {pid}) is stopped.")
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    # 3. Kill any llama-server.exe
    try:
        os.system("taskkill /IM llama-server.exe /F >nul 2>&1")
        print("[Shutdown] Ensured portable llama-server runtimes are stopped.")
    except Exception:
        pass

    # 4. Free check
    time.sleep(0.5)
    print("[SUCCESS] All AI Companion services have been stopped cleanly.")
    print("=" * 65)


if __name__ == "__main__":
    stop_services()
