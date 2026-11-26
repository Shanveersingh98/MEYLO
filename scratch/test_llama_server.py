"""
scratch/test_llama_server.py
Test starting llama-server as a background process, making an inference request,
and terminating the process cleanly.
"""
import os
import sys
import time
import subprocess
import urllib.request
import json

RUNTIME_EXE = "D:/DEEP-LEARNING-PROJECT/runtime/llama.cpp/llama-server.exe"
MODEL_PATH = "D:/DEEP-LEARNING-PROJECT/models/external/qwen2.5-1.5b-instruct/model.gguf"
PORT = 28989

cmd = [
    RUNTIME_EXE,
    "-m", MODEL_PATH,
    "--port", str(PORT),
    "--host", "127.0.0.1",
    "-c", "2048",
    "-t", "4",
    "--log-disable"
]

print("Starting llama-server...")
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"llama-server PID: {proc.pid}")

# Wait for server to become ready
ready = False
for _ in range(30):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
        with urllib.request.urlopen(req, timeout=1) as resp:
            if resp.status == 200:
                ready = True
                break
    except Exception:
        time.sleep(0.5)

if not ready:
    print("Error: llama-server failed to become ready!")
    proc.terminate()
    sys.exit(1)

print("llama-server is READY! Sending test prompt...")

payload = {
    "messages": [
        {"role": "system", "content": "You are a helpful, friendly AI companion."},
        {"role": "user", "content": "What is machine learning in one simple sentence?"}
    ],
    "temperature": 0.7,
    "max_tokens": 128
}

req_data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    f"http://127.0.0.1:{PORT}/v1/chat/completions",
    data=req_data,
    headers={"Content-Type": "application/json"}
)

t0 = time.time()
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read().decode("utf-8"))
dur = time.time() - t0

reply = data["choices"][0]["message"]["content"]
tokens = data.get("usage", {}).get("completion_tokens", 0)
speed = tokens / dur if dur > 0 else 0

print(f"\nResponse from Qwen ({tokens} tokens in {dur:.2f}s, {speed:.1f} t/s):")
print("-" * 50)
print(reply)
print("-" * 50)

print("\nTerminating llama-server cleanly...")
proc.terminate()
proc.wait()
print("Process terminated successfully. Memory freed.")
