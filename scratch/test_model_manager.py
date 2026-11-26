"""
scratch/test_model_manager.py
Test ModelManager: loading, switching, independent generation, and memory.
"""
import sys
sys.path.insert(0, "src")
import torch
from model_manager import ModelManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing ModelManager with device: {device}...")
mgr = ModelManager(device=device, default_model="custom")

print("\n--- Initial Status ---")
print(mgr.get_status(web_search_enabled=False))

print("\n--- Test Generation with Custom Model ---")
prompt = "What is a neural network?"
reply, active, lat = mgr.generate(prompt)
print(f"Active Model: {active}")
print(f"Latency: {lat:.1f} ms")
print(f"Reply: {reply[:150]}...")
assert active == "custom", "Active model must be custom!"

print("\n--- Test Model List ---")
print(mgr.list_models())

print("\n--- Switching to External Model (/model external) ---")
ok, msg = mgr.set_model("external")
print(f"Switch result: {ok} -> {msg}")
assert ok, "Switch to external must succeed!"

print("\n--- Status After Switch ---")
print(mgr.get_status(web_search_enabled=False))

print("\n--- Test Generation with External Qwen Model ---")
reply_qwen, active_qwen, lat_qwen = mgr.generate(prompt)
print(f"Active Model: {active_qwen}")
print(f"Latency: {lat_qwen:.1f} ms")
print(f"Reply: {reply_qwen[:250]}...")
assert active_qwen == "external", "Active model must be external!"
assert len(reply_qwen) > 20, "Qwen must generate a coherent answer!"

print("\n--- Switching back to Custom Model (/model custom) ---")
ok2, msg2 = mgr.set_model("custom")
print(f"Switch back result: {ok2} -> {msg2}")
assert ok2, "Switch back to custom must succeed!"

print("\n--- Test Generation with Custom Model After Switch Back ---")
reply_custom2, active_custom2, lat_custom2 = mgr.generate(prompt)
print(f"Active Model: {active_custom2}")
print(f"Latency: {lat_custom2:.1f} ms")
print(f"Reply: {reply_custom2[:150]}...")
assert active_custom2 == "custom", "Active model must be custom!"

print("\nShutting down ModelManager...")
mgr.shutdown()
print("All ModelManager tests passed successfully!")
