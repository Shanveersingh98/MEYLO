import sys
import time
from pathlib import Path
import torch
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

from chat import load_chatbot, generate_chat_response

print("=" * 70)
print("RUNNING SMOKE TEST SUITE FOR CHATBOT APPLICATION")
print("=" * 70)

# 1. GPU Inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing device: {device}")
model, tokenizer, ckpt = load_chatbot(device)

# Smoke Test 1: Single-turn greeting
print("\n[Smoke 1] Greeting & persona...")
h1 = [{"role": "user", "text": "Hello, how are you?"}]
r1, eng1 = generate_chat_response(model, tokenizer, h1, device, tool_assisted=False)
print(f"  Response ({eng1}): {r1[:80]}")
assert len(r1) > 0, "Empty response generated!"
print("  PASS: Single-turn greeting successful.")

# Smoke Test 2: Multi-turn context preservation
print("\n[Smoke 2] Multi-turn dialogue...")
h2 = [
    {"role": "user", "text": "Hello! My name is Shanveer."},
    {"role": "assistant", "text": "Hello Shanveer! How can I help you today?"},
    {"role": "user", "text": "What is my name?"}
]
r2, eng2 = generate_chat_response(model, tokenizer, h2, device, tool_assisted=False)
print(f"  Response ({eng2}): {r2[:80]}")
assert len(r2) > 0
print("  PASS: Multi-turn response generated.")

# Smoke Test 3: Tool-assisted arithmetic vs pure neural
print("\n[Smoke 3] Tool-assisted vs Pure Neural arithmetic...")
h3 = [{"role": "user", "text": "Calculate 245 + 168."}]
r3_tool, eng3_tool = generate_chat_response(model, tokenizer, h3, device, tool_assisted=True)
r3_pure, eng3_pure = generate_chat_response(model, tokenizer, h3, device, tool_assisted=False)
print(f"  Tool-assisted ({eng3_tool}): {r3_tool}")
print(f"  Pure Neural   ({eng3_pure}): {r3_pure}")
assert eng3_tool == "calculator" and "413" in r3_tool
assert eng3_pure == "neural_model"
print("  PASS: Tool grounding and pure neural mode decouple properly.")

# Smoke Test 4: Malformed input (special tokens and whitespace)
print("\n[Smoke 4] Malformed input handling...")
h4 = [{"role": "user", "text": "<|user|>\n<|assistant|>\n[EOS][BOS]   ???   "}]
r4, eng4 = generate_chat_response(model, tokenizer, h4, device)
print(f"  Response: {r4[:80]}")
print("  PASS: Malformed inputs handled safely without crashes.")

# Smoke Test 5: Long input context
print("\n[Smoke 5] Long prompt handling (600+ tokens)...")
h5 = [{"role": "user", "text": "Context: " + ("Deep learning is a subset of machine learning based on artificial neural networks. " * 30) + "What is deep learning?"}]
r5, eng5 = generate_chat_response(model, tokenizer, h5, device)
print(f"  Response: {r5[:80]}")
print("  PASS: Long prompt handled without out-of-bounds error.")

# Smoke Test 6: CPU Inference compatibility
print("\n[Smoke 6] CPU Inference verification...")
cpu_device = torch.device("cpu")
cpu_model, cpu_tok, _ = load_chatbot(cpu_device)
r6, _ = generate_chat_response(cpu_model, cpu_tok, [{"role": "user", "text": "What is AI?"}], cpu_device, max_new_tokens=20)
print(f"  CPU Response: {r6[:60]}")
print("  PASS: CPU execution verified.")

print("\n" + "=" * 70)
print("ALL 6 APPLICATION SMOKE TESTS PASSED END-TO-END!")
print("=" * 70)
