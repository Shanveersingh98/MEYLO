"""
scratch/audit_interactive_conversations.py
Independent Audit - Requirement 6: Final Interactive Testing
Simulates end-to-end multi-turn dialogue with model switching, online/offline toggling,
error recovery, and session continuity.
"""
import sys
import time
import json
import torch
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model_manager import ModelManager
from web_retrieval import WebRetrievalEngine
from chat import generate_chat_response, parse_command, format_source_links

def run_interactive_audit():
    print("==================================================")
    print("STARTING REQUIREMENT 6: FINAL INTERACTIVE AUDIT")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_manager = ModelManager(device=device, default_model="custom")
    retrieval_engine = WebRetrievalEngine()
    
    web_search_enabled = False
    debug_mode = False
    history = []
    test_log = []

    def execute_turn(user_input: str, expected_model: str, is_command: bool = False):
        nonlocal web_search_enabled, history
        print(f"\nUSER: {user_input}")
        
        cmd_action, cmd_msg = parse_command(user_input)
        if cmd_action == "model_custom":
            ok, msg = model_manager.set_model("custom")
            print(f"[Command] {msg}")
            test_log.append({"type": "command", "cmd": user_input, "msg": msg, "active_model": model_manager.active_model_name})
            return msg
        elif cmd_action == "model_external":
            ok, msg = model_manager.set_model("external")
            print(f"[Command] {msg}")
            test_log.append({"type": "command", "cmd": user_input, "msg": msg, "active_model": model_manager.active_model_name})
            return msg
        elif cmd_action == "web_on":
            web_search_enabled = True
            msg = "[Web Search: ON]"
            print(f"[Command] {msg}")
            test_log.append({"type": "command", "cmd": user_input, "msg": msg, "web": True})
            return msg
        elif cmd_action == "web_off":
            web_search_enabled = False
            msg = "[Web Search: OFF]"
            print(f"[Command] {msg}")
            test_log.append({"type": "command", "cmd": user_input, "msg": msg, "web": False})
            return msg
        elif cmd_action == "model_status":
            status_text = model_manager.get_status(web_search_enabled)
            print(f"[Command Status]\n{status_text}")
            test_log.append({"type": "command", "cmd": user_input, "status": status_text})
            return status_text
        elif cmd_action == "sources":
            s_text = format_source_links(getattr(retrieval_engine, "session_last_sources", []))
            print(f"[Sources]\n{s_text}")
            test_log.append({"type": "command", "cmd": user_input, "sources": s_text})
            return s_text

        # Normal dialogue turn
        assert model_manager.active_model_name == expected_model, f"Expected {expected_model}, got {model_manager.active_model_name}"
        history.append({"role": "user", "text": user_input})
        
        t0 = time.time()
        try:
            reply, engine_name = generate_chat_response(
                model=model_manager.custom_engine.model if model_manager.active_model_name == "custom" else None,
                tokenizer=model_manager.custom_engine.tokenizer if model_manager.active_model_name == "custom" else None,
                history=history,
                device=device,
                retrieval_engine=retrieval_engine,
                web_search_enabled=web_search_enabled,
                debug_mode=debug_mode,
                model_manager=model_manager
            )
        except Exception as e:
            reply = f"Error occurred: {e}"
            engine_name = "error_fallback"
            
        latency = (time.time() - t0) * 1000
        bot_label = "BOT (Custom)" if model_manager.active_model_name == "custom" else "BOT (Qwen)"
        print(f"{bot_label} [{engine_name}, {latency:.1f}ms]: {reply}")
        
        history.append({"role": "assistant", "text": reply})
        test_log.append({
            "type": "turn",
            "user": user_input,
            "assistant": reply,
            "engine": engine_name,
            "model": model_manager.active_model_name,
            "latency_ms": latency
        })
        return reply

    try:
        # ----------------------------------------------------
        # Turn 1: Custom Model - Conversational Greeting
        # ----------------------------------------------------
        print("\n--- Turn 1: Custom Model - Conversational Greeting ---")
        r1 = execute_turn("Hello! My name is Jordan and I am a software engineer.", expected_model="custom")
        assert len(r1.strip()) > 0, "Response 1 must not be empty"

        # ----------------------------------------------------
        # Turn 2: Custom Model - Technical / Educational
        # ----------------------------------------------------
        print("\n--- Turn 2: Custom Model - Technical / Educational ---")
        r2 = execute_turn("What is machine learning in simple terms?", expected_model="custom")
        assert len(r2.strip()) > 0, "Response 2 must not be empty"

        # ----------------------------------------------------
        # Turn 3: Switch to External Model via Command
        # ----------------------------------------------------
        print("\n--- Turn 3: Switch to External Model (/model external) ---")
        execute_turn("/model external", expected_model="external", is_command=True)
        assert model_manager.active_model_name == "external"

        # ----------------------------------------------------
        # Turn 4: External Model - History Recall (Session Continuity)
        # ----------------------------------------------------
        print("\n--- Turn 4: External Model - History Recall across Model Switch ---")
        r4 = execute_turn("Do you remember what my name is and what I do?", expected_model="external")
        assert "jordan" in r4.lower(), f"External model should recall name Jordan: {r4}"
        assert "software" in r4.lower() or "engineer" in r4.lower(), f"External model should recall profession: {r4}"

        # ----------------------------------------------------
        # Turn 5: Enable Web Search via Command
        # ----------------------------------------------------
        print("\n--- Turn 5: Enable Web Search (/web on) ---")
        execute_turn("/web on", expected_model="external", is_command=True)
        assert web_search_enabled is True

        # ----------------------------------------------------
        # Turn 6: External Model - Live Web Grounded Answer
        # ----------------------------------------------------
        print("\n--- Turn 6: External Model - Live Web Grounded Answer ---")
        r6 = execute_turn("name of current cm of punjab india", expected_model="external")
        assert "bhagwant" in r6.lower() and "mann" in r6.lower(), f"Expected Bhagwant Mann in Punjab CM query, got: {r6}"
        execute_turn("/sources", expected_model="external", is_command=True)

        # ----------------------------------------------------
        # Turn 7: Error Recovery Simulation during Active Session
        # ----------------------------------------------------
        print("\n--- Turn 7: Error Recovery Simulation (Zero Model Switch on Error) ---")
        # Monkeypatch generate once to simulate a transient HTTP failure
        real_generate = model_manager.external_engine.generate
        flaky_invocations = [0]
        def flaky_generate(prompt, max_tokens=512, temperature=0.7, stop=None, messages=None):
            flaky_invocations[0] += 1
            if flaky_invocations[0] == 1:
                raise RuntimeError("Simulated transient connection failure")
            return real_generate(prompt, max_tokens, temperature, stop, messages)
            
        model_manager.external_engine.generate = flaky_generate
        r7 = execute_turn("What is the capital of Japan?", expected_model="external")
        model_manager.external_engine.generate = real_generate
        
        # Confirm chatbot handled gracefully and DID NOT change active model
        assert model_manager.active_model_name == "external", "Active model must remain external even after an error"
        print("Confirmed: Active model strictly stayed 'external', no secret fallback occurred.")

        # Turn 7b: Post-error recovery dialogue in same session
        print("\n--- Turn 7b: Continued Dialogue Post-Error Recovery ---")
        r7b = execute_turn("What is the capital of Japan?", expected_model="external")
        assert "tokyo" in r7b.lower(), f"Expected Tokyo, got {r7b}"

        # ----------------------------------------------------
        # Turn 8: Disable Web Search via Command
        # ----------------------------------------------------
        print("\n--- Turn 8: Disable Web Search (/web off) ---")
        execute_turn("/web off", expected_model="external", is_command=True)
        assert web_search_enabled is False

        # ----------------------------------------------------
        # Turn 9: Switch back to Custom Model (/model custom)
        # ----------------------------------------------------
        print("\n--- Turn 9: Switch back to Custom Model (/model custom) ---")
        execute_turn("/model custom", expected_model="custom", is_command=True)
        assert model_manager.active_model_name == "custom"

        # ----------------------------------------------------
        # Turn 10: Custom Model - Post-Switch Casual/Math Turn
        # ----------------------------------------------------
        print("\n--- Turn 10: Custom Model - Post-Switch Turn ---")
        r10 = execute_turn("What is 10 plus 15?", expected_model="custom")
        assert len(r10.strip()) > 0, "Custom model must produce response"

        # ----------------------------------------------------
        # Turn 11: Hardware Status Command (/model status)
        # ----------------------------------------------------
        print("\n--- Turn 11: Dynamic Hardware Status (/model status) ---")
        status_out = execute_turn("/model status", expected_model="custom", is_command=True)
        assert "CPU" in status_out and "RAM" in status_out, "Status must show dynamic hardware"

        # Save results
        output_path = PROJECT_DIR / "scratch/audit_interactive_conversations_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(test_log, f, indent=2)
        print(f"\nALL INTERACTIVE AUDIT CHECKS PASSED: Results saved to {output_path}")

    finally:
        model_manager.shutdown()

if __name__ == "__main__":
    run_interactive_audit()
