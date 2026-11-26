import sys
from pathlib import Path
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import FromScratchLM

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "clean_50k_best.pt"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"
VAL_TOKENS_PATH = PROJECT_DIR / "data" / "processed" / "training_sequences_clean" / "validation_tokens.bin"
VAL_MASK_PATH = PROJECT_DIR / "data" / "processed" / "training_sequences_clean" / "validation_mask.bin"

OUTPUT_JSON = PROJECT_DIR / "logs" / "clean_50k_representation_geometry.json"
OUTPUT_JSONL = PROJECT_DIR / "logs" / "clean_50k_representation_geometry.jsonl"

TEST_PROMPTS = [
    "Hello",
    "What is artificial intelligence?",
    "What is machine learning?",
    "Explain Python in simple words.",
    "What is 2 + 2?",
    "Explain what a function is in Python.",
    "What is the difference between AI and machine learning?",
    "Explain monopsony in simple words."
]

def compute_effective_rank(matrix, center=True):
    """
    Computes effective rank of a matrix (N x d) using:
    1) Singular values entropy: erank_sigma = exp(- sum p_i ln p_i) where p_i = sigma_i / sum sigma
    2) Variance entropy: erank_var = exp(- sum p_i ln p_i) where p_i = lambda_i / sum lambda
    """
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)
    
    matrix = matrix.float()
    N, d = matrix.shape
    if N < 2:
        return {
            "erank_sigma": 1.0,
            "erank_var": 1.0,
            "erank_sigma_ratio": 1.0 / d,
            "erank_var_ratio": 1.0 / d,
            "singular_values": [1.0],
            "effective_dim": d
        }
    
    if center:
        mean_vec = matrix.mean(dim=0, keepdim=True)
        matrix = matrix - mean_vec

    # Compute covariance or SVD efficiently
    # Since d = 384 is much smaller than N, compute covariance eigenvalues:
    # Cov = (1 / N) * X^T * X  (shape d x d)
    cov = torch.mm(matrix.t(), matrix) / float(N)
    # Eigenvalues of cov are lambda_i >= 0
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals, min=0.0)
    eigvals = torch.flip(eigvals, dims=[0]) # sort descending

    sigmas = torch.sqrt(torch.clamp(eigvals * float(N), min=0.0))

    # 1. Singular value distribution
    sigma_sum = sigmas.sum().item()
    if sigma_sum > 0:
        p_sigma = sigmas / sigma_sum
        p_sigma_nonzero = p_sigma[p_sigma > 1e-12]
        h_sigma = -torch.sum(p_sigma_nonzero * torch.log(p_sigma_nonzero)).item()
        erank_sigma = math.exp(h_sigma)
    else:
        erank_sigma = 0.0

    # 2. Variance distribution (eigenvalues)
    eig_sum = eigvals.sum().item()
    if eig_sum > 0:
        p_var = eigvals / eig_sum
        p_var_nonzero = p_var[p_var > 1e-12]
        h_var = -torch.sum(p_var_nonzero * torch.log(p_var_nonzero)).item()
        erank_var = math.exp(h_var)
    else:
        erank_var = 0.0

    return {
        "erank_sigma": float(erank_sigma),
        "erank_var": float(erank_var),
        "erank_sigma_ratio": float(erank_sigma / d),
        "erank_var_ratio": float(erank_var / d),
        "top_5_singular_values": [float(s) for s in sigmas[:5].tolist()],
        "top_5_variance_ratio": [float(p) for p in (p_var[:5].tolist() if eig_sum > 0 else [])]
    }

def compute_pairwise_cosine_stats(vectors, num_pairs=100000, different_positions_only=False, positions=None):
    """
    Computes pairwise cosine similarity distribution via random pair sampling.
    vectors: Tensor of shape (N, d)
    positions: Optional 1D Tensor of length N with sequence positions
    """
    N = vectors.size(0)
    if N < 2:
        return {
            "mean": 1.0, "median": 1.0, "p05": 1.0, "p95": 1.0, "std": 0.0,
            "min": 1.0, "max": 1.0, "num_pairs_evaluated": 0
        }

    # Normalize vectors
    norms = torch.norm(vectors, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    unit_vecs = vectors / norms

    # Sample pairs
    if different_positions_only and positions is not None:
        idx1 = torch.randint(0, N, (num_pairs * 2,), device=vectors.device)
        idx2 = torch.randint(0, N, (num_pairs * 2,), device=vectors.device)
        mask = (positions[idx1] != positions[idx2]) & (idx1 != idx2)
        idx1 = idx1[mask][:num_pairs]
        idx2 = idx2[mask][:num_pairs]
    else:
        idx1 = torch.randint(0, N, (num_pairs * 2,), device=vectors.device)
        idx2 = torch.randint(0, N, (num_pairs * 2,), device=vectors.device)
        mask = (idx1 != idx2)
        idx1 = idx1[mask][:num_pairs]
        idx2 = idx2[mask][:num_pairs]

    actual_pairs = idx1.numel()
    if actual_pairs == 0:
        return {
            "mean": 1.0, "median": 1.0, "p05": 1.0, "p95": 1.0, "std": 0.0,
            "min": 1.0, "max": 1.0, "num_pairs_evaluated": 0
        }

    # Batch cosine computation
    sims = (unit_vecs[idx1] * unit_vecs[idx2]).sum(dim=-1).float().cpu().numpy()

    return {
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "p05": float(np.percentile(sims, 5)),
        "p95": float(np.percentile(sims, 95)),
        "std": float(np.std(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "num_pairs_evaluated": int(actual_pairs)
    }

def compute_norm_stats(vectors):
    norms = torch.norm(vectors, p=2, dim=-1).float().cpu().numpy()
    return {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "median": float(np.median(norms)),
        "min": float(np.min(norms)),
        "max": float(np.max(norms))
    }

def main():
    print("=" * 80)
    print("REPRESENTATION GEOMETRY DIAGNOSTIC SUITE")
    print("Model: FromScratchLM (14,120,768 parameters)")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 80)

    # ----------------------------------------------------
    # PART 1: CHECKPOINT AND ARCHITECTURE VERIFICATION
    # ----------------------------------------------------
    print("\n[PART 1] Verifying Checkpoint and Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    step = checkpoint.get("step")
    val_loss = checkpoint.get("validation_loss")
    best_val_loss = checkpoint.get("best_validation_loss")
    config = checkpoint.get("config", {})

    print(f"Checkpoint Step:        {step}")
    print(f"Validation Loss:        {val_loss}")
    print(f"Best Val Loss:          {best_val_loss}")
    print(f"Config:                 {config}")

    model = FromScratchLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        dropout=config.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    finite_check = all(torch.isfinite(p).all().item() for p in model.parameters())

    print(f"Total Model Parameters: {param_count:,}")
    print(f"All Parameters Finite:  {finite_check}")
    assert param_count == 14120768, f"Parameter count mismatch: got {param_count}, expected 14,120,768"
    assert finite_check, "Non-finite parameters detected in model checkpoint!"

    part1_info = {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "step": step,
        "validation_loss": val_loss,
        "best_validation_loss": best_val_loss,
        "config": config,
        "param_count": param_count,
        "all_parameters_finite": finite_check,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    }

    # ----------------------------------------------------
    # PART 4: TOKEN EMBEDDING GEOMETRY
    # ----------------------------------------------------
    print("\n[PART 4] Analyzing Token Embedding Geometry...")
    token_embed_weights = model.token_embedding.weight.detach() # (8000, 384)
    vocab_size, d_model = token_embed_weights.shape

    # All vocabulary tokens
    embed_sims_all = compute_pairwise_cosine_stats(token_embed_weights, num_pairs=100000)
    embed_norms_all = compute_norm_stats(token_embed_weights)
    embed_erank_centered_all = compute_effective_rank(token_embed_weights, center=True)
    embed_erank_uncentered_all = compute_effective_rank(token_embed_weights, center=False)

    # Exclude special tokens: [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3, <|user|>=4, <|assistant|>=5
    regular_embed_weights = token_embed_weights[6:] # (7994, 384)
    embed_sims_reg = compute_pairwise_cosine_stats(regular_embed_weights, num_pairs=100000)
    embed_norms_reg = compute_norm_stats(regular_embed_weights)
    embed_erank_centered_reg = compute_effective_rank(regular_embed_weights, center=True)
    embed_erank_uncentered_reg = compute_effective_rank(regular_embed_weights, center=False)

    part4_results = {
        "vocab_size": vocab_size,
        "embedding_dim": d_model,
        "all_tokens": {
            "token_count": vocab_size,
            "cosine_similarity": embed_sims_all,
            "vector_norms": embed_norms_all,
            "centered_effective_rank": embed_erank_centered_all,
            "uncentered_effective_rank": embed_erank_uncentered_all,
        },
        "regular_tokens_excluding_special": {
            "token_count": regular_embed_weights.shape[0],
            "excluded_special_tokens": ["[PAD](0)", "[UNK](1)", "[BOS](2)", "[EOS](3)", "<|user|>(4)", "<|assistant|>(5)"],
            "cosine_similarity": embed_sims_reg,
            "vector_norms": embed_norms_reg,
            "centered_effective_rank": embed_erank_centered_reg,
            "uncentered_effective_rank": embed_erank_uncentered_reg,
        }
    }
    print(f"Token Embeddings (All 8000 tokens):")
    print(f"  Mean Cosine Sim:     {embed_sims_all['mean']:.4f} (Median: {embed_sims_all['median']:.4f}, 5%: {embed_sims_all['p05']:.4f}, 95%: {embed_sims_all['p95']:.4f})")
    print(f"  Centered Erank:      {embed_erank_centered_all['erank_sigma']:.2f} / {d_model} (Ratio: {embed_erank_centered_all['erank_sigma_ratio']:.4f})")
    print(f"  Uncentered Erank:    {embed_erank_uncentered_all['erank_sigma']:.2f} / {d_model} (Ratio: {embed_erank_uncentered_all['erank_sigma_ratio']:.4f})")
    print(f"  Norm Mean:           {embed_norms_all['mean']:.4f} +/- {embed_norms_all['std']:.4f}")
    print(f"Token Embeddings (Regular tokens excluding specials):")
    print(f"  Mean Cosine Sim:     {embed_sims_reg['mean']:.4f} (Median: {embed_sims_reg['median']:.4f}, 5%: {embed_sims_reg['p05']:.4f}, 95%: {embed_sims_reg['p95']:.4f})")
    print(f"  Centered Erank:      {embed_erank_centered_reg['erank_sigma']:.2f} / {d_model} (Ratio: {embed_erank_centered_reg['erank_sigma_ratio']:.4f})")

    # ----------------------------------------------------
    # PART 2 & 3: COLLECT HIDDEN STATES & ANISOTROPY MEASUREMENTS
    # ----------------------------------------------------
    print("\n[PART 2 & 3] Collecting Hidden States on 100 Validation Sequences...")
    num_val_seqs = 100
    val_tokens_mm = np.memmap(VAL_TOKENS_PATH, dtype=np.uint16, mode="r")
    val_mask_mm = np.memmap(VAL_MASK_PATH, dtype=np.uint8, mode="r")

    layer_outputs_dict = {}
    hooks = []
    for i, block in enumerate(model.blocks):
        name = f"layer_{i+1}"
        hooks.append(block.register_forward_hook(lambda m, inp, out, n=name: layer_outputs_dict.update({n: out.detach()})))
    hooks.append(model.final_norm.register_forward_hook(lambda m, inp, out: layer_outputs_dict.update({"final_norm": out.detach()})))

    # Data collectors across the 100 sequences
    # We will gather hidden representations per layer for:
    # 1. All non-pad tokens
    # 2. Assistant tokens (mask == 1)
    # 3. User tokens (mask == 0 and token != 0)
    layer_names = [f"layer_{i+1}" for i in range(len(model.blocks))] + ["final_norm"]
    layer_tokens_all = {name: [] for name in layer_names}
    layer_positions_all = [] # 1D tensor of positions
    token_roles_all = [] # 'assistant' or 'user'

    total_real_tokens = 0
    total_asst_tokens = 0
    total_user_tokens = 0

    with torch.no_grad():
        for seq_idx in range(num_val_seqs):
            start = seq_idx * 1024
            end = start + 1024
            seq = np.asarray(val_tokens_mm[start:end], dtype=np.int64)
            msk = np.asarray(val_mask_mm[start:end], dtype=np.uint8)

            # Find non-pad length
            non_pad_indices = np.where(seq != 0)[0]
            if len(non_pad_indices) == 0:
                continue
            max_non_pad_idx = non_pad_indices[-1] + 1
            input_seq = seq[:max_non_pad_idx]
            mask_seq = msk[:max_non_pad_idx]

            input_tensor = torch.tensor(input_seq, dtype=torch.long, device=device).unsqueeze(0)
            _ = model(input_tensor)

            # Extract outputs
            # seq shape: (1, seq_len, 384) -> (seq_len, 384)
            # Find non-pad positions
            for pos in range(max_non_pad_idx):
                tok_id = input_seq[pos]
                if tok_id == 0:
                    continue # exclude padding
                
                is_asst = (mask_seq[pos] == 1)
                role = "assistant" if is_asst else "user"
                token_roles_all.append(role)
                layer_positions_all.append(pos)

                if is_asst:
                    total_asst_tokens += 1
                else:
                    total_user_tokens += 1
                total_real_tokens += 1

            for name in layer_names:
                # tensor on GPU, slice non-pad
                h = layer_outputs_dict[name][0, :max_non_pad_idx, :] # (seq_len, 384)
                # Keep only non-pad tokens
                non_pad_mask = torch.tensor(input_seq != 0, device=device)
                layer_tokens_all[name].append(h[non_pad_mask].cpu())

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"Collected hidden representations from {num_val_seqs} validation sequences:")
    print(f"  Total Real (Non-Pad) Tokens: {total_real_tokens}")
    print(f"  Assistant Tokens:            {total_asst_tokens}")
    print(f"  User/Prompt Tokens:          {total_user_tokens}")

    positions_tensor = torch.tensor(layer_positions_all, dtype=torch.long)
    token_roles_np = np.array(token_roles_all)
    asst_mask = (token_roles_np == "assistant")
    user_mask = (token_roles_np == "user")

    # Now compute metrics for each layer
    part3_results = {}
    print("\n--- Layer-by-Layer Representation Geometry (All Non-Pad Tokens) ---")
    print(f"{'Layer':<12} | {'Mean Sim':<9} | {'Median':<9} | {'5th-95th %':<15} | {'Diff-Pos Sim':<12} | {'Norm Mean(Std)':<16} | {'Erank(ratio)':<16}")
    print("-" * 105)

    for name in layer_names:
        H_all = torch.cat(layer_tokens_all[name], dim=0) # (N, 384) on CPU
        H_all_gpu = H_all.to(device)
        pos_gpu = positions_tensor.to(device)

        # 1. Pairwise cosine similarity among sampled tokens
        sims_all = compute_pairwise_cosine_stats(H_all_gpu, num_pairs=100000)

        # 2. Pairwise cosine similarity between DIFFERENT token positions
        sims_diff_pos = compute_pairwise_cosine_stats(H_all_gpu, num_pairs=100000, different_positions_only=True, positions=pos_gpu)

        # 3. Norm statistics
        norm_stats = compute_norm_stats(H_all_gpu)

        # 4. Effective rank (centered and uncentered)
        erank_centered = compute_effective_rank(H_all_gpu, center=True)
        erank_uncentered = compute_effective_rank(H_all_gpu, center=False)

        # Separate assistant and user subsets
        H_asst_gpu = H_all[asst_mask].to(device)
        sims_asst = compute_pairwise_cosine_stats(H_asst_gpu, num_pairs=50000)
        norm_asst = compute_norm_stats(H_asst_gpu)
        erank_asst_centered = compute_effective_rank(H_asst_gpu, center=True)

        H_user_gpu = H_all[user_mask].to(device)
        sims_user = compute_pairwise_cosine_stats(H_user_gpu, num_pairs=50000)
        norm_user = compute_norm_stats(H_user_gpu)
        erank_user_centered = compute_effective_rank(H_user_gpu, center=True)

        part3_results[name] = {
            "all_non_pad_tokens": {
                "token_count": total_real_tokens,
                "cosine_similarity": sims_all,
                "cosine_similarity_different_positions": sims_diff_pos,
                "vector_norms": norm_stats,
                "centered_effective_rank": erank_centered,
                "uncentered_effective_rank": erank_uncentered,
            },
            "assistant_tokens": {
                "token_count": int(asst_mask.sum()),
                "cosine_similarity": sims_asst,
                "vector_norms": norm_asst,
                "centered_effective_rank": erank_asst_centered,
            },
            "user_tokens": {
                "token_count": int(user_mask.sum()),
                "cosine_similarity": sims_user,
                "vector_norms": norm_user,
                "centered_effective_rank": erank_user_centered,
            }
        }

        p05_p95 = f"[{sims_all['p05']:.3f}, {sims_all['p95']:.3f}]"
        norm_str = f"{norm_stats['mean']:.2f} ({norm_stats['std']:.2f})"
        erank_str = f"{erank_centered['erank_sigma']:.1f} ({erank_centered['erank_sigma_ratio']:.3f})"
        print(f"{name:<12} | {sims_all['mean']:<9.4f} | {sims_all['median']:<9.4f} | {p05_p95:<15} | {sims_diff_pos['mean']:<12.4f} | {norm_str:<16} | {erank_str:<16}")

    # Free memory
    del layer_tokens_all
    torch.cuda.empty_cache()

    # ----------------------------------------------------
    # PART 6: GENERATION-STATE COMPARISON
    # ----------------------------------------------------
    print("\n[PART 6] Generation-State Comparison (4 Key States + Matched Length)...")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos_id = tokenizer.token_to_id("[EOS]")

    # Helper function to generate tokens and return token IDs
    def generate_sequence(prompt_text, do_sample=False, max_new_tokens=100, temperature=0.8, top_k=50, rep_penalty=1.15):
        formatted = f"[BOS]\n<|user|>\n{prompt_text}\n<|assistant|>\n"
        encoded = tokenizer.encode(formatted)
        prompt_ids = encoded.ids
        prompt_len = len(prompt_ids)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            input_context = input_ids[:, -1024:]
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(input_context)
            logits = logits[:, -1, :].float()

            if not do_sample:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                generated_tokens = set(input_ids[0, prompt_len:].tolist())
                for tid in generated_tokens:
                    if logits[0, tid] > 0:
                        logits[0, tid] /= rep_penalty
                    else:
                        logits[0, tid] *= rep_penalty
                logits = logits / temperature
                values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                probs = torch.softmax(filtered, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == eos_id:
                break

        full_ids = input_ids[0].tolist()
        gen_ids = full_ids[prompt_len:]
        return full_ids, prompt_ids, gen_ids

    # State A: Original Prompt 3 ("What is machine learning?") before generating anything
    prompt3_text = "What is machine learning?"
    formatted3 = f"[BOS]\n<|user|>\n{prompt3_text}\n<|assistant|>\n"
    state_a_ids = tokenizer.encode(formatted3).ids

    # State B: Short coherent prefix ("Hello" prompt + coherent response)
    hello_full_ids, hello_p_ids, hello_gen_ids = generate_sequence("Hello", do_sample=False, max_new_tokens=30)
    state_b_ids = hello_full_ids # coherent full sequence

    # State C: Repetitive greedy-generation prefix from Prompt 3
    greedy_full_ids, greedy_p_ids, greedy_gen_ids = generate_sequence(prompt3_text, do_sample=False, max_new_tokens=100)
    state_c_ids = greedy_full_ids # repetitive sequence

    # State D: Sampled response prefix from Prompt 3
    sampled_full_ids, sampled_p_ids, sampled_gen_ids = generate_sequence(prompt3_text, do_sample=True, max_new_tokens=100)
    state_d_ids = sampled_full_ids # sampled sequence

    # Validation reference: take a slice from validation sequence 0 of matched length
    # Let's inspect matched length:
    # Notice State C has length len(state_c_ids) = 115
    # Let's define a fixed matched length L_matched
    target_len = min(len(state_c_ids), len(state_d_ids))
    # Let's make matched sequences of exact length target_len
    state_c_matched_ids = state_c_ids[:target_len]
    state_d_matched_ids = state_d_ids[:target_len]
    
    # Real validation sequence matched to target_len
    val_seq0 = np.asarray(val_tokens_mm[:1024], dtype=np.int64)
    val_nonpad = val_seq0[val_seq0 != 0]
    val_matched_ids = val_nonpad[:target_len].tolist()

    states = {
        "State_A_Prompt_Only": {
            "description": "Original prompt 'What is machine learning?' before generation",
            "token_ids": state_a_ids,
            "prompt_length": len(state_a_ids),
            "generated_length": 0,
            "text": tokenizer.decode(state_a_ids, skip_special_tokens=False)
        },
        "State_B_Coherent_Hello": {
            "description": "Short coherent prefix from 'Hello' (greedy generation)",
            "token_ids": state_b_ids,
            "prompt_length": len(hello_p_ids),
            "generated_length": len(hello_gen_ids),
            "text": tokenizer.decode(state_b_ids, skip_special_tokens=False)
        },
        "State_C_Repetitive_Greedy": {
            "description": "Repetitive greedy generation prefix from 'What is machine learning?'",
            "token_ids": state_c_ids,
            "prompt_length": len(greedy_p_ids),
            "generated_length": len(greedy_gen_ids),
            "text": tokenizer.decode(state_c_ids, skip_special_tokens=False)
        },
        "State_D_Sampled_Response": {
            "description": "Sampled non-repetitive response prefix from 'What is machine learning?'",
            "token_ids": state_d_ids,
            "prompt_length": len(sampled_p_ids),
            "generated_length": len(sampled_gen_ids),
            "text": tokenizer.decode(state_d_ids, skip_special_tokens=False)
        },
        "State_C_Matched_Length": {
            "description": f"Repetitive greedy generation truncated to exactly {target_len} tokens",
            "token_ids": state_c_matched_ids,
            "prompt_length": len(greedy_p_ids),
            "generated_length": target_len - len(greedy_p_ids),
            "text": tokenizer.decode(state_c_matched_ids, skip_special_tokens=False)
        },
        "State_D_Matched_Length": {
            "description": f"Sampled response truncated to exactly {target_len} tokens",
            "token_ids": state_d_matched_ids,
            "prompt_length": len(sampled_p_ids),
            "generated_length": target_len - len(sampled_p_ids),
            "text": tokenizer.decode(state_d_matched_ids, skip_special_tokens=False)
        },
        "State_E_Validation_Matched": {
            "description": f"Real clean validation conversation slice of exactly {target_len} tokens",
            "token_ids": val_matched_ids,
            "prompt_length": 0,
            "generated_length": target_len,
            "text": tokenizer.decode(val_matched_ids, skip_special_tokens=False)
        }
    }

    # Evaluate geometry across all states
    part6_results = {}
    print("\n--- Generation States Representation Geometry Comparison ---")
    print(f"{'State Name':<28} | {'Len':<5} | {'FinalNorm Sim':<14} | {'Layer6 Sim':<12} | {'FinalNorm Erank':<16} | {'Layer6 Erank':<16}")
    print("-" * 105)

    for state_name, sinfo in states.items():
        t_ids = sinfo["token_ids"]
        seq_len = len(t_ids)
        inp = torch.tensor([t_ids], dtype=torch.long, device=device)

        # Hook to collect layers
        cur_outputs = {}
        cur_hooks = []
        for i, block in enumerate(model.blocks):
            cur_hooks.append(block.register_forward_hook(lambda m, in_, out, n=f"layer_{i+1}": cur_outputs.update({n: out.detach()})))
        cur_hooks.append(model.final_norm.register_forward_hook(lambda m, in_, out: cur_outputs.update({"final_norm": out.detach()})))

        with torch.no_grad():
            _ = model(inp)

        for h in cur_hooks:
            h.remove()

        state_layer_metrics = {}
        for lname in ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "layer_6", "final_norm"]:
            vecs = cur_outputs[lname][0] # (seq_len, 384)
            # Pairwise cosine similarity among all tokens in sequence
            # If seq_len is small (e.g. 15 to 115), compute all pairs or sample
            sims = compute_pairwise_cosine_stats(vecs, num_pairs=10000)
            norm_stats = compute_norm_stats(vecs)
            erank_cent = compute_effective_rank(vecs, center=True)
            erank_uncent = compute_effective_rank(vecs, center=False)

            # Consecutive token cosine similarity cos(h_t, h_{t+1})
            if seq_len > 1:
                v_norm = vecs / torch.norm(vecs, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                consec_sims = (v_norm[:-1] * v_norm[1:]).sum(dim=-1).cpu().numpy()
                consec_mean = float(np.mean(consec_sims))
                consec_std = float(np.std(consec_sims))
            else:
                consec_mean = 1.0
                consec_std = 0.0

            # If generated tokens exist, compute generated-only stats
            p_len = sinfo.get("prompt_length", 0)
            if sinfo.get("generated_length", 0) > 1 and seq_len > p_len:
                gen_vecs = vecs[p_len:]
                gen_sims = compute_pairwise_cosine_stats(gen_vecs, num_pairs=10000)
                gen_erank = compute_effective_rank(gen_vecs, center=True)
            else:
                gen_sims = None
                gen_erank = None

            state_layer_metrics[lname] = {
                "cosine_similarity": sims,
                "consecutive_token_similarity_mean": consec_mean,
                "consecutive_token_similarity_std": consec_std,
                "vector_norms": norm_stats,
                "centered_effective_rank": erank_cent,
                "uncentered_effective_rank": erank_uncent,
                "generated_tokens_only_similarity": gen_sims,
                "generated_tokens_only_erank": gen_erank,
                "rank_upper_bound": min(seq_len, 384)
            }

        part6_results[state_name] = {
            "description": sinfo["description"],
            "sequence_length": seq_len,
            "prompt_length": sinfo["prompt_length"],
            "generated_length": sinfo["generated_length"],
            "decoded_text_snippet": sinfo["text"][:120] + "..." if len(sinfo["text"]) > 120 else sinfo["text"],
            "layers": state_layer_metrics
        }

        fn_sim = state_layer_metrics["final_norm"]["cosine_similarity"]["mean"]
        l6_sim = state_layer_metrics["layer_6"]["cosine_similarity"]["mean"]
        fn_erank = state_layer_metrics["final_norm"]["centered_effective_rank"]["erank_sigma"]
        l6_erank = state_layer_metrics["layer_6"]["centered_effective_rank"]["erank_sigma"]
        print(f"{state_name:<28} | {seq_len:<5} | {fn_sim:<14.4f} | {l6_sim:<12.4f} | {fn_erank:<16.2f} | {l6_erank:<16.2f}")

    # ----------------------------------------------------
    # ALL 8 BENCHMARK PROMPTS: GREEDY VS SAMPLED COMPARISON
    # ----------------------------------------------------
    print("\n--- Benchmark Suite: Greedy vs Sampled Representation Geometry ---")
    benchmark_eval = {}
    for p_idx, prompt in enumerate(TEST_PROMPTS, 1):
        g_full, g_p, g_gen = generate_sequence(prompt, do_sample=False, max_new_tokens=60)
        s_full, s_p, s_gen = generate_sequence(prompt, do_sample=True, max_new_tokens=60)

        # Matched comparison length
        min_l = min(len(g_full), len(s_full))
        g_matched = torch.tensor([g_full[:min_l]], dtype=torch.long, device=device)
        s_matched = torch.tensor([s_full[:min_l]], dtype=torch.long, device=device)

        with torch.no_grad():
            cur_outputs = {}
            h_hook = model.final_norm.register_forward_hook(lambda m, in_, out: cur_outputs.update({"fn": out.detach()}))
            _ = model(g_matched)
            g_vecs = cur_outputs["fn"][0]
            _ = model(s_matched)
            s_vecs = cur_outputs["fn"][0]
            h_hook.remove()

        g_sim = compute_pairwise_cosine_stats(g_vecs, num_pairs=5000)["mean"]
        s_sim = compute_pairwise_cosine_stats(s_vecs, num_pairs=5000)["mean"]
        g_erank = compute_effective_rank(g_vecs, center=True)["erank_sigma"]
        s_erank = compute_effective_rank(s_vecs, center=True)["erank_sigma"]

        benchmark_eval[prompt] = {
            "matched_length": min_l,
            "greedy_mean_sim": g_sim,
            "sampled_mean_sim": s_sim,
            "greedy_erank": g_erank,
            "sampled_erank": s_erank,
            "greedy_text": tokenizer.decode(g_full[len(g_p):min_l], skip_special_tokens=True).strip()[:60],
            "sampled_text": tokenizer.decode(s_full[len(s_p):min_l], skip_special_tokens=True).strip()[:60]
        }
        print(f"[{p_idx}/8] '{prompt[:32]}' | L={min_l} | Greedy Sim: {g_sim:.4f}, Erank: {g_erank:.1f} | Sampled Sim: {s_sim:.4f}, Erank: {s_erank:.1f}")

    # ----------------------------------------------------
    # PART 7: SAVE OUTPUT FILES
    # ----------------------------------------------------
    print("\n[PART 7] Saving Results to Logs...")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    full_report_data = {
        "metadata": {
            "experiment": "Representation Geometry Diagnostic",
            "checkpoint": str(CHECKPOINT_PATH),
            "step": step,
            "date": "2026-10-13",
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "validation_sequences_analyzed": num_val_seqs,
            "total_tokens_analyzed": total_real_tokens,
            "hidden_dimension": 384,
            "num_layers": 6,
            "vocab_size": 8000
        },
        "checkpoint_verification": part1_info,
        "token_embedding_geometry": part4_results,
        "layer_by_layer_geometry": part3_results,
        "generation_states": part6_results,
        "benchmark_suite_comparison": benchmark_eval,
        "methodology_and_limitations": {
            "sampling_strategy": "100,000 uniformly sampled pairs without replacement per evaluation",
            "effective_rank_definition": "Roy & Vetterli (2007) Shannon entropy of normalized singular values: exp(-sum p_i ln p_i) where p_i = sigma_i / sum(sigma). Covariance eigenvalues used for exact computation.",
            "centering": "Mean-centered across tokens to distinguish directional cone (anisotropy) from dimensional variance collapse (low rank).",
            "sequence_length_limitation": "For sequence length L < d (where d=384), algebraic rank is strictly bounded by min(L, d) = L. Effective rank cannot exceed L. Cross-state comparisons must use matched length L."
        }
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_report_data, f, indent=2)
    print(f"Saved complete JSON report to: {OUTPUT_JSON}")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        # Line 1: Checkpoint info
        f.write(json.dumps({"record_type": "checkpoint_verification", "data": part1_info}) + "\n")
        # Line 2: Embedding geometry
        f.write(json.dumps({"record_type": "token_embedding_geometry", "data": part4_results}) + "\n")
        # Lines for each layer
        for lname, lmetrics in part3_results.items():
            f.write(json.dumps({"record_type": "layer_geometry", "layer": lname, "data": lmetrics}) + "\n")
        # Lines for each generation state
        for sname, smetrics in part6_results.items():
            f.write(json.dumps({"record_type": "generation_state", "state": sname, "data": smetrics}) + "\n")
        # Benchmark summary
        f.write(json.dumps({"record_type": "benchmark_summary", "data": benchmark_eval}) + "\n")
    print(f"Saved detailed JSONL log to: {OUTPUT_JSONL}")
    print("\nDiagnostic completed successfully.")

if __name__ == "__main__":
    main()
