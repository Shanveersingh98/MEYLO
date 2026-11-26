"""
scratch/test_chat_switching.py
Verify chat execution with model switching and web search.
"""
import sys
import time
import subprocess
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
PYTHON_EXE = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"

commands = [
    "/model status",
    "hello who are you",
    "/model external",
    "/model status",
    "hello who are you",
    "/web on",
    "who is india current prime minister",
    "/model custom",
    "what is machine learning",
    "/exit"
]

input_payload = "\n".join(commands) + "\n"

print("Starting chatbot subprocess...")
proc = subprocess.Popen(
    [str(PYTHON_EXE), "src/chat.py", "--model", "custom", "--web-search", "off"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    cwd=str(PROJECT_DIR)
)

out, _ = proc.communicate(input=input_payload, timeout=90)
print("=== CHATBOT SESSION OUTPUT ===")
print(out)
print("=== END SESSION OUTPUT ===")
print("Process return code:", proc.returncode)
