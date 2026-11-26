"""
src/model_manager.py
====================
Independent Dual-Model Orchestrator for Local AI Companion:
1. Model 1: Custom Trained FromScratchLM (checkpoints/ultimate_improvement/best.pt)
2. Model 2: External Pretrained Qwen2.5-1.5B-Instruct (models/external/qwen2.5-1.5b-instruct/model.gguf)

Strict Guarantees:
- Absolute Project Isolation: all binaries, models, caches remain inside D:/DEEP-LEARNING-PROJECT.
- True Independence: No hybrid mode, no automatic switching, no silent cross-model correction.
- Explicit lifecycle control: Unloads inactive model to conserve RAM and GPU VRAM.
"""

import os
import sys
import time
import json
import ctypes
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
from tokenizers import Tokenizer

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

from model import FromScratchLM

# Paths
CUSTOM_CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "ultimate_improvement" / "best.pt"
CUSTOM_TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"

EXTERNAL_MODEL_DIR = PROJECT_DIR / "models" / "external" / "qwen2.5-1.5b-instruct"
EXTERNAL_GGUF_PATH = EXTERNAL_MODEL_DIR / "model.gguf"
EXTERNAL_CONFIG_PATH = EXTERNAL_MODEL_DIR / "model-config.json"

LLAMA_SERVER_EXE = PROJECT_DIR / "runtime" / "llama.cpp" / "llama-server.exe"
LLAMA_CLI_EXE = PROJECT_DIR / "runtime" / "llama.cpp" / "llama-cli.exe"
LLAMA_SERVER_PORT = 28989


def get_hardware_info() -> Dict[str, Any]:
    """Returns dynamically detected CPU, RAM, GPU, and VRAM statistics."""
    info = {
        "cpu_name": "CPU",
        "ram_total_gb": 0.0,
        "ram_used_gb": 0.0,
        "ram_avail_gb": 0.0,
        "ram_percent": 0.0,
        "gpu_name": "None",
        "vram_total_gb": 0.0,
        "vram_used_mb": 0.0,
        "cuda_available": False,
        "cuda_version": "None"
    }
    # Dynamic CPU Detection
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        info["cpu_name"] = val.strip()
    except Exception:
        import platform
        info["cpu_name"] = platform.processor() or "CPU"

    # Dynamic RAM Detection
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        mem = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        total_ram = mem.ullTotalPhys / (1024**3)
        avail_ram = mem.ullAvailPhys / (1024**3)
        info["ram_total_gb"] = round(total_ram, 2)
        info["ram_avail_gb"] = round(avail_ram, 2)
        info["ram_used_gb"] = round(total_ram - avail_ram, 2)
        info["ram_percent"] = float(mem.dwMemoryLoad)
    except Exception:
        pass

    # Dynamic GPU / VRAM Detection
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_version"] = str(torch.version.cuda)
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            info["vram_used_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2), 2)
        except Exception:
            pass

    return info


def get_system_memory_info() -> Dict[str, float]:
    """Returns system RAM and GPU VRAM statistics."""
    hw = get_hardware_info()
    return {
        "ram_total_gb": hw["ram_total_gb"],
        "ram_used_gb": hw["ram_used_gb"],
        "ram_avail_gb": hw["ram_avail_gb"],
        "ram_percent": hw["ram_percent"],
        "vram_total_gb": hw["vram_total_gb"],
        "vram_used_mb": hw["vram_used_mb"]
    }

    return info


class CustomModelEngine:
    """
    Engine for Custom-Trained FromScratchLM checkpoint.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.model: Optional[FromScratchLM] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.is_loaded = False
        self.checkpoint_sha256 = ""

    def load(self) -> bool:
        if self.is_loaded and self.model is not None:
            return True

        if not CUSTOM_CHECKPOINT_PATH.exists():
            print(f"[ERROR] Custom checkpoint not found at: {CUSTOM_CHECKPOINT_PATH}")
            return False

        if not CUSTOM_TOKENIZER_PATH.exists():
            print(f"[ERROR] Custom tokenizer not found at: {CUSTOM_TOKENIZER_PATH}")
            return False

        # Load Tokenizer
        self.tokenizer = Tokenizer.from_file(str(CUSTOM_TOKENIZER_PATH))

        # Load Checkpoint
        ckpt = torch.load(str(CUSTOM_CHECKPOINT_PATH), map_location=self.device)
        model_kwargs = ckpt.get("model_kwargs", {})
        if not model_kwargs and "config" in ckpt:
            cfg = ckpt["config"]
            model_kwargs = {
                "vocab_size": cfg.get("vocab_size", 8000),
                "context_length": cfg.get("context_length", 1024),
                "d_model": cfg.get("d_model", 384),
                "num_heads": cfg.get("num_heads", 6),
                "num_layers": cfg.get("num_layers", 6),
                "d_ff": cfg.get("d_ff", 1536),
                "dropout": 0.0,
            }
        elif not model_kwargs and "vocab_size" in ckpt:
            model_kwargs = {
                "vocab_size": ckpt["vocab_size"],
                "d_model": ckpt.get("d_model", ckpt.get("embed_dim", 384)),
                "num_layers": ckpt.get("num_layers", 6),
                "num_heads": ckpt.get("num_heads", 6),
                "context_length": ckpt.get("context_length", 1024),
                "d_ff": ckpt.get("d_ff", 1536),
                "dropout": 0.0,
            }

        self.model = FromScratchLM(**model_kwargs)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # Strip DDP prefixes if present
        clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_sd)
        self.model.to(self.device)
        self.model.eval()

        self.is_loaded = True
        return True

    def unload(self):
        """Unloads model weights from memory and clears CUDA cache."""
        import gc
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def generate(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        context_evidence: Optional[str] = None,
        temperature: float = 0.35,
        top_p: float = 0.85,
        top_k: int = 20,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
        max_new_tokens: int = 160
    ) -> Tuple[str, float]:
        """
        Executes local neural generation on custom trained FromScratchLM.
        Returns (generated_text, latency_ms).
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Custom model is not loaded. Call load() first.")

        t0 = time.time()
        # Build prompt sequence respecting conversational context
        messages = []
        if history:
            for turn in history[-4:]:
                role = turn.get("role", "user")
                text = turn.get("text", "")
                if role == "user":
                    messages.append(f"<|user|>\n{text}")
                elif role == "assistant":
                    messages.append(f"<|assistant|>\n{text}")

        user_content = prompt
        if context_evidence:
            user_content = f"{prompt}\n\nRelevant Context:\n{context_evidence}"

        messages.append(f"<|user|>\n{user_content}")
        messages.append("<|assistant|>\n")
        full_prompt = "\n".join(messages)

        input_ids = self.tokenizer.encode(full_prompt).ids
        if len(input_ids) > 850:
            input_ids = input_ids[-850:]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated_ids = list(input_ids)

        stop_token_ids = {0, 2, 3, 4, 5}
        with torch.no_grad():
            for _ in range(max_new_tokens):
                curr_ctx = input_tensor[:, -1024:]
                logits = self.model(curr_ctx)[:, -1, :]

                # Repetition penalty
                if repetition_penalty != 1.0 and len(generated_ids) > len(input_ids):
                    for prev_token in set(generated_ids[len(input_ids):]):
                        logits[0, prev_token] /= repetition_penalty

                # No repeat n-gram penalty
                if no_repeat_ngram_size > 0 and len(generated_ids) >= no_repeat_ngram_size:
                    ngram = tuple(generated_ids[-(no_repeat_ngram_size - 1):])
                    for i in range(len(input_ids), len(generated_ids) - (no_repeat_ngram_size - 1)):
                        if tuple(generated_ids[i:i + no_repeat_ngram_size - 1]) == ngram:
                            forbidden_id = generated_ids[i + no_repeat_ngram_size - 1]
                            logits[0, forbidden_id] = -float("inf")

                # Temperature scaling & sampling
                if temperature <= 0.05:
                    next_token = torch.argmax(logits, dim=-1).item()
                else:
                    scaled_logits = logits / max(temperature, 1e-4)
                    if top_k > 0:
                        v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                        scaled_logits[scaled_logits < v[:, [-1]]] = -float("inf")
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = False
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        scaled_logits[:, indices_to_remove] = -float("inf")

                    probs = torch.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

                if next_token in stop_token_ids:
                    break

                generated_ids.append(next_token)
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=self.device)], dim=1)

        new_tokens = generated_ids[len(input_ids):]
        raw_text = self.tokenizer.decode(new_tokens)
        cleaned_text = raw_text.split("<|user|>")[0].split("<|assistant|>")[0].strip()
        latency_ms = (time.time() - t0) * 1000
        return cleaned_text, latency_ms


class ExternalModelEngine:
    """
    Engine for Pretrained Qwen2.5-1.5B-Instruct running via portable llama.cpp.
    """
    def __init__(self, port: int = LLAMA_SERVER_PORT):
        self.port = port
        self.server_process: Optional[subprocess.Popen] = None
        self.is_loaded = False
        self.model_path = EXTERNAL_GGUF_PATH
        self.runtime_exe = LLAMA_SERVER_EXE

    def load(self) -> bool:
        if self.is_loaded and self.is_server_alive():
            return True

        if not self.model_path.exists():
            print(f"[ERROR] External model GGUF not found at: {self.model_path}")
            return False

        if not self.runtime_exe.exists():
            print(f"[ERROR] llama-server executable not found at: {self.runtime_exe}")
            return False

        # Launch llama-server in background with 2048 context window (optimal memory footprint)
        cmd = [
            str(self.runtime_exe),
            "-m", str(self.model_path),
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "-c", "2048",
            "-t", "4",
            "--log-disable"
        ]

        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(self.runtime_exe.parent)
            )
        except Exception as e:
            print(f"[ERROR] Failed to spawn llama-server: {e}")
            return False

        # Wait for server readiness
        ready = False
        for _ in range(40):
            if self.is_server_alive():
                ready = True
                break
            time.sleep(0.25)

        if not ready:
            print("[ERROR] llama-server failed to respond to health check.")
            self.unload()
            return False

        self.is_loaded = True
        return True

    def is_server_alive(self) -> bool:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/health")
            with urllib.request.urlopen(req, timeout=1) as resp:
                return resp.status == 200
        except Exception:
            return False

    def unload(self):
        """Kills the llama-server process cleanly, releasing RAM."""
        if self.server_process is not None:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=3)
            except Exception:
                try:
                    self.server_process.kill()
                except Exception:
                    pass
            self.server_process = None
        self.is_loaded = False

    def generate(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        context_evidence: Optional[str] = None,
        system_instruction: str = "You are a helpful, friendly, and honest local AI Companion.",
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Tuple[str, float]:
        """
        Sends chat completion request to Qwen2.5-1.5B-Instruct via llama-server.
        Guarantees robust error handling and token budgeting:
        - Catches HTTPError (including 400 context exceeded), URLError, Timeout, JSON errors.
        - Automatically retries with compact context on context overflow.
        - NEVER raises uncaught exceptions that terminate the chatbot.
        Returns (generated_text, latency_ms).
        """
        t0 = time.time()
        if not self.is_loaded or not self.is_server_alive():
            ok = self.load()
            if not ok:
                return (
                    "Could not connect to the local Qwen model engine. Please check if the model is ready.",
                    0.0
                )

        messages = [{"role": "system", "content": system_instruction}]
        if history:
            # Budget conversational history (up to 6 turns); avoid duplicating the active prompt if present at end
            hist_turns = history[:-1] if (len(history) > 0 and history[-1].get("text", "").strip() == prompt.strip()) else history
            for turn in hist_turns[-6:]:
                role = turn.get("role", "user")
                text = turn.get("text", "")[:400]
                messages.append({"role": role, "content": text})

        user_content = prompt.strip()
        # Budget context evidence to max 3000 chars to fit comfortably within context
        if context_evidence:
            bounded_evidence = context_evidence.strip()
            if len(bounded_evidence) > 3000:
                bounded_evidence = bounded_evidence[:3000] + "\n...[Additional sources truncated for context limits]..."
            user_content = f"{prompt}\n\n[Verified Evidence Context]:\n{bounded_evidence}"

        messages.append({"role": "user", "content": user_content})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            latency_ms = (time.time() - t0) * 1000
            reply = data["choices"][0]["message"]["content"].strip()
            return reply, latency_ms

        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            print(f"[llama-server HTTPError {e.code}]: {e.reason} - {err_body}")

            # If context limit was exceeded (HTTP 400), automatically retry once with minimal compact evidence
            if e.code == 400 and ("exceed" in err_body.lower() or "context" in err_body.lower()):
                print("[llama-server] Context limit exceeded: retrying with compact evidence...")
                try:
                    compact_evidence = (context_evidence[:1200] + "...") if context_evidence else None
                    compact_user = prompt.strip()
                    if compact_evidence:
                        compact_user = f"{prompt}\n\n[Verified Evidence Context]:\n{compact_evidence}"
                    retry_payload = {
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": compact_user}
                        ],
                        "temperature": temperature,
                        "max_tokens": min(max_tokens, 256)
                    }
                    retry_req = urllib.request.Request(
                        f"http://127.0.0.1:{self.port}/v1/chat/completions",
                        data=json.dumps(retry_payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(retry_req, timeout=45) as retry_resp:
                        retry_data = json.loads(retry_resp.read().decode("utf-8"))
                    latency_ms = (time.time() - t0) * 1000
                    return retry_data["choices"][0]["message"]["content"].strip(), latency_ms
                except Exception as retry_err:
                    print(f"[llama-server retry error]: {retry_err}")

            latency_ms = (time.time() - t0) * 1000
            return (
                f"I encountered a server processing error (HTTP {e.code}) while generating the response. "
                "Please try rephrasing or asking a shorter, more specific question.",
                latency_ms
            )

        except (urllib.error.URLError, TimeoutError) as e:
            latency_ms = (time.time() - t0) * 1000
            print(f"[llama-server connection error]: {e}")
            return (
                "The local Qwen model server is currently unreachable or timed out. "
                "You can check its status with `/model status` or switch models with `/model custom`.",
                latency_ms
            )

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            latency_ms = (time.time() - t0) * 1000
            print(f"[llama-server malformed response]: {e}")
            return (
                "The local model engine returned an unexpected response format. Please try again.",
                latency_ms
            )

        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            print(f"[llama-server unexpected error]: {e}")
            return (
                f"An unexpected error occurred during generation: {e}",
                latency_ms
            )


class ModelManager:
    """
    Coordinates model selection, independent execution, and hardware memory management.
    """
    def __init__(self, device: torch.device, default_model: str = "custom"):
        self.device = device
        self.active_model_name = default_model  # "custom" or "external"
        self.custom_engine = CustomModelEngine(device)
        self.external_engine = ExternalModelEngine()

        # Load initial active model
        if default_model == "custom":
            self.custom_engine.load()
        elif default_model == "external":
            self.external_engine.load()

    def set_model(self, model_choice: str) -> Tuple[bool, str]:
        """
        Switches the active model. Unloads previous model weights to preserve memory.
        """
        choice = model_choice.strip().lower()
        if choice not in ["custom", "external"]:
            return False, f"Unknown model choice '{model_choice}'. Choose '/model custom' or '/model external'."

        if choice == self.active_model_name:
            status_desc = "Custom Trained (FromScratchLM)" if choice == "custom" else "External (Qwen2.5-1.5B-Instruct)"
            return True, f"Model '{status_desc}' is already active."

        if choice == "custom":
            # Unload external engine first to free RAM
            self.external_engine.unload()
            # Load custom model
            ok = self.custom_engine.load()
            if ok:
                self.active_model_name = "custom"
                return True, "[Switched to Custom Trained Model (checkpoints/ultimate_improvement/best.pt)]"
            else:
                return False, "[ERROR] Failed to load custom trained model checkpoint."

        elif choice == "external":
            # Unload custom engine from GPU/RAM
            self.custom_engine.unload()
            # Load external Qwen engine
            ok = self.external_engine.load()
            if ok:
                self.active_model_name = "external"
                return True, "[Switched to External Model: Qwen2.5-1.5B-Instruct (GGUF Q4_K_M)]"
            else:
                # Revert to custom if external failed to load
                self.custom_engine.load()
                return False, "[ERROR] Failed to initialize external Qwen model runtime. Reverted to custom model."

    def get_status(self, web_search_enabled: bool = False) -> str:
        """Returns comprehensive diagnostic status of active model and hardware."""
        hw = get_hardware_info()
        lines = [
            "==================================================",
            "          MEYLO AI COMPANION MODEL STATUS         ",
            "=================================================="
        ]
        if self.active_model_name == "custom":
            lines.extend([
                "Active Model      : Custom Trained Model (FromScratchLM)",
                f"Checkpoint Path   : {CUSTOM_CHECKPOINT_PATH.relative_to(PROJECT_DIR)}",
                f"Inference Engine  : PyTorch Autoregressive Decoder",
                f"Execution Device  : {hw['gpu_name']} (CUDA {hw['cuda_version']})" if hw["cuda_available"] else f"Execution Device  : {hw['cpu_name']} (CPU)",
                "Status            : Loaded & Ready"
            ])
        else:
            lines.extend([
                "Active Model      : External Model (Qwen2.5-1.5B-Instruct)",
                f"Model File Path   : {EXTERNAL_GGUF_PATH.relative_to(PROJECT_DIR)}",
                f"Quantization      : Q4_K_M (4-bit Medium Quantization, 1.04 GB)",
                f"Inference Engine  : llama.cpp (portable local server on port {LLAMA_SERVER_PORT})",
                f"Execution Device  : {hw['cpu_name']} (Multi-Threaded CPU)",
                "Status            : Loaded & Ready"
            ])

        lines.extend([
            "--------------------------------------------------",
            f"Hardware CPU      : {hw['cpu_name']}",
            f"Hardware GPU      : {hw['gpu_name']} ({hw['vram_total_gb']} GB VRAM, CUDA {hw['cuda_version']})" if hw["cuda_available"] else "Hardware GPU      : None",
            f"System RAM Load   : {hw['ram_used_gb']} GB / {hw['ram_total_gb']} GB ({hw['ram_percent']}%)",
            f"GPU VRAM Load     : {hw['vram_used_mb']} MB / {hw['vram_total_gb'] * 1024:.0f} MB" if hw["cuda_available"] else "GPU VRAM Load     : N/A",
            f"Web Search Mode   : {'ON (Live Search Enabled)' if web_search_enabled else 'OFF (Pure Offline Mode)'}",
            "=================================================="
        ])
        return "\n".join(lines)

    def list_models(self) -> str:
        """Lists available installed models and their active state."""
        custom_active = " [ACTIVE]" if self.active_model_name == "custom" else ""
        external_active = " [ACTIVE]" if self.active_model_name == "external" else ""

        custom_avail = "Available" if CUSTOM_CHECKPOINT_PATH.exists() else "Missing"
        external_avail = "Available" if EXTERNAL_GGUF_PATH.exists() else "Missing"

        lines = [
            "Available Models in D:/DEEP-LEARNING-PROJECT:",
            f"  1. custom   : FromScratchLM Custom Trained ({CUSTOM_CHECKPOINT_PATH.name}) - {custom_avail}{custom_active}",
            f"  2. external : Qwen2.5-1.5B-Instruct (GGUF Q4_K_M) - {external_avail}{external_active}",
            "",
            "To switch models, type: /model custom  or  /model external"
        ]
        return "\n".join(lines)

    def generate(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        context_evidence: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, str, float]:
        """
        Executes generation using the strictly active model.
        Guarantees that Custom and External models never generate for each other.
        Returns (reply_text, active_model_name, latency_ms).
        """
        if self.active_model_name == "custom":
            reply, lat = self.custom_engine.generate(
                prompt=prompt,
                history=history,
                context_evidence=context_evidence,
                temperature=kwargs.get("temperature", 0.35),
                max_new_tokens=kwargs.get("max_new_tokens", 160)
            )
            return reply, "custom", lat
        else:
            reply, lat = self.external_engine.generate(
                prompt=prompt,
                history=history,
                context_evidence=context_evidence,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_new_tokens", 512)
            )
            return reply, "external", lat

    def shutdown(self):
        """Cleanup upon application exit."""
        self.custom_engine.unload()
        self.external_engine.unload()
