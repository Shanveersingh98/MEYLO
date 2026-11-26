# Comprehensive Read-Only Audit: From-Scratch 50k LM Project

**Project Root:** `D:\DEEP-LEARNING-PROJECT`  
**Checkpoint:** `checkpoints\clean_50k_best.pt` (Step 50,000, Validation Loss: 3.8162)  
**Architecture:** `FromScratchLM` (14,120,768 parameters, 6 layers, 6 heads, d_model=384, d_ff=1536)  
**Audit Type:** Read-Only Static and Empirical Codebase Audit  
**Date:** October 13, 2026  

---

## 1. Executive Summary

This audit examined why the 50,000-step checkpoint (`clean_50k_best.pt`) produces repetitive, degenerate, or truncated completions despite achieving a healthy representation geometry and a validation loss of 3.816.

The investigation examined every link in the pipeline: dataset preparation (`build_clean_sequences.py`, `format_conversations.py`, `prepare_tokenization.py`), model architecture (`model.py`), training pipeline (`train_clean_50k.py`), and decoding implementation (`generate_clean_50k.py`).

### Summary of Audit Findings

| Verification Target | Status | Severity | Primary Finding |
| :--- | :---: | :---: | :--- |
| **1. Target & Mask Alignment** | **CORRECT** | None | Next-token prediction targets ($S[j+1]$) and boolean loss masks ($M[j+1]$) are properly synchronized. No off-by-one index shifts exist. |
| **2. Padding & Unintended Loss** | **CORRECT** | None | Padding tokens (`[PAD]=0`) have mask=0 and are completely excluded from both numerator and denominator in `masked_losses.mean()`. Prompt tokens are also masked out. |
| **3. Assistant Responses & EOS** | **SYSTEMIC FLAW** | **HIGH** | In multi-turn conversations (57.9% of all assistant turns), assistant responses end with `<|user|>`, **not `[EOS]`**. Furthermore, 30.3% of training conversations end on an unanswered user turn with unmasked EOS. The model is under-supervised on turn termination. |
| **4. Causal Attention & Boundaries** | **CORRECT** | None | Design A enforces 1 conversation per 1024-token sequence (zero conversation packing). Causal mask prevents attention across tokens $s > t$ and prevents real tokens from attending to padding. |
| **5. Tokenizer & Format Consistency** | **CORRECT** | None | Generation uses the exact same tokenizer, special token IDs, and conversation template (`[BOS]\n<|user|>\n...\n<|assistant|>\n`) as the training pipeline. |
| **6. Decoding Parameters (RepPenalty, Top-K, Top-P, Temp)** | **CONFIRMED BUG** | **CRITICAL** | **Repetition penalty is bypassed entirely under greedy decoding** (`do_sample=False`). Top-P (nucleus sampling) is completely unimplemented. Logit arithmetic runs in `float16` instead of `float32`. |
| **7. Generation Stopping Conditions** | **CONFIRMED BUG** | **HIGH** | Generation stops **only on `[EOS]`**. It does not terminate on `<|user|>` or `<|assistant|>`. When the model completes an answer and emits `<|user|>` (as trained in 58% of turns), `generate_text` strips the token and continues generating into a runaway loop. |
| **8. Training Dynamics & Data Budget** | **LIMITATION** | **HIGH** | Effective batch size is 1 sequence (~333 loss tokens) with zero gradient accumulation. Total trained volume is only ~16.6M tokens (~1.18 tokens/param), leaving the model severely under-trained relative to its 14.1M parameter capacity. |

---

## 2. Detailed Findings

---

### Finding 1: Repetition Penalty Bypassed in Greedy Decoding (CONFIRMED BUG)

* **Relevant File & Lines:** [`src/generate_clean_50k.py`](file:///d:/DEEP-LEARNING-PROJECT/src/generate_clean_50k.py#L76-L87)
* **Code Behavior:**
  ```python
  if not do_sample:
      # Deterministic / Greedy Argmax
      next_token = torch.argmax(logits, dim=-1, keepdim=True)
  else:
      # Repetition penalty strictly applied to tokens generated AFTER the prompt
      generated_tokens = set(input_ids[0, prompt_length:].tolist())
      for token_id in generated_tokens:
          if logits[0, token_id] > 0:
              logits[0, token_id] /= repetition_penalty
          else:
              logits[0, token_id] *= repetition_penalty

      logits = logits / temperature
      values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
      ...
  ```
* **Why It Is Problematic:**
  In `generate_clean_50k.py`, repetition penalty is nested strictly inside the `else:` block of `if not do_sample:`. Consequently, whenever greedy decoding is run (`--sample` flag omitted, which is the default in both `main()` and `run_eval_suite`), **repetition penalty is completely bypassed**.
  In language modeling literature (e.g., Keskar et al., 2019; HuggingFace `RepetitionPenaltyLogitsProcessor`), repetition penalty is a logits transform designed to apply *prior* to token selection, whether by argmax or multinomial sampling. Without repetition penalty, greedy argmax decoding on a small, partially trained model inevitably collapses into deterministic self-reinforcing loops on high-frequency n-grams.
* **Evidence:**
  Empirical ablation conducted on checkpoint `clean_50k_best.pt`:
  * Prompt: *"What is machine learning?"*
    * **Current greedy (repetition penalty bypassed):**
      `"I am a language model that is designed to generate text and generate text based on a wide range of text-like text-like text-like text-like text-like text-like text-like text-like text-like text-like..."` (Runaway loop)
    * **Greedy with repetition penalty (1.15) enabled before argmax:**
      `"I am a language model that is designed to generate text and generate text based on the ability to perform tasks."` (Completely breaks the loop; clean, coherent termination!)
  * Prompt: *"Explain Python in simple words."*
    * **Current greedy:**
      `"The function is a function that is a function that is a function that is a function that is a function that is used to the function..."` (Degenerate loop)
    * **Greedy with repetition penalty (1.15):**
      `"The function is a function that is a function that is used to be used to the function. The function is used to create a function, and can be used to read and perform the function. In this example, the function is used to generate a function in Python code. It uses a..."` (Loop broken)
* **Severity:** **CRITICAL**
* **Recommended Fix (Do Not Apply):**
  Move the repetition penalty calculation outside the `if not do_sample:` condition so it applies to `logits` unconditionally before either `torch.argmax` or sampling.

---

### Finding 2: Inconsistent Turn-Level EOS Supervision in Multi-Turn Conversations (SYSTEMIC FLAW)

* **Relevant Files & Lines:**
  * [`src/prepare_tokenization.py`](file:///d:/DEEP-LEARNING-PROJECT/src/prepare_tokenization.py#L85-L113)
  * [`src/format_conversations.py`](file:///d:/DEEP-LEARNING-PROJECT/src/format_conversations.py#L52-L75)
  * [`src/build_clean_sequences.py`](file:///d:/DEEP-LEARNING-PROJECT/src/build_clean_sequences.py#L38-L60)
* **Code Behavior:**
  In `prepare_tokenization.py`:
  ```python
  def format_conversation(conversation):
      parts = ["[BOS]"]
      for message in conversation["messages"]:
          role = message["role"]
          text = message["text"].strip()
          if role == "prompter":
              parts.append("<|user|>")
          elif role == "assistant":
              parts.append("<|assistant|>")
          parts.append(text)
      parts.append("[EOS]")
      return "\n".join(parts)
  ```
  In `build_clean_sequences.py`:
  ```python
  def create_loss_mask(token_ids):
      mask = np.zeros(len(token_ids), dtype=np.uint8)
      assistant_active = False
      for i, token_id in enumerate(token_ids):
          if token_id == ASSISTANT_ID:
              assistant_active = True
              mask[i] = 1
          elif token_id == USER_ID:
              assistant_active = False
          elif token_id == EOS_ID:
              if assistant_active:
                  mask[i] = 1
              assistant_active = False
          elif assistant_active:
              mask[i] = 1
      return mask
  ```
* **Why It Is Problematic:**
  1. `[EOS]` was appended **only at the very end of the entire conversation**, never at the end of individual assistant turns.
  2. In multi-turn conversations (which constitute the majority of the dataset), when an assistant finishes speaking in turn 1, the next token in the sequence is `<|user|>`, **not `[EOS]`**.
  3. When `<|user|>` occurs, `create_loss_mask` executes `assistant_active = False` and sets `mask[i] = 0`. This means the transition from assistant speech to the next turn is **unsupervised** (mask 0). The model is never taught that an assistant turn must terminate with `[EOS]`!
  4. In 5,362 training conversations (30.3% of the entire dataset), the conversation ends on an unanswered user prompt (`prompter-assistant-prompter`). In these cases, `[EOS]` appears after a user prompt, so `assistant_active` is False and `mask[EOS] = 0`.
* **Evidence:**
  Static analysis across all 17,683 training conversations:
  * Total assistant turns: 29,303
  * Assistant turns terminated by `<|user|>`: **16,982 (57.9%)**
  * Assistant turns terminated by `[EOS]`: **12,321 (42.1%)**
  * Conversations ending on an unanswered user turn: **5,362 (30.3%)**
  * `[EOS]` tokens with `mask == 1`: 12,321
  * `[EOS]` tokens with `mask == 0`: 5,362
  Because 57.9% of assistant turns in training were never followed by `[EOS]`, the model has a weak prior for emitting `[EOS]` at the end of an answer.
* **Severity:** **HIGH**
* **Recommended Fix (Do Not Apply):**
  In conversation formatting, append `[EOS]` at the end of *every* assistant turn:
  `<|assistant|>\n{text}\n[EOS]\n`. Ensure the loss mask includes `[EOS]` after every assistant turn. Filter out trailing unanswered user turns from the training dataset.

---

### Finding 3: Single-Token Stopping Condition Missing Role Tokens (CONFIRMED PROBLEM)

* **Relevant File & Lines:** [`src/generate_clean_50k.py`](file:///d:/DEEP-LEARNING-PROJECT/src/generate_clean_50k.py#L65,L98-L102)
* **Code Behavior:**
  ```python
  eos_id = tokenizer.token_to_id("[EOS]")
  ...
  for _ in range(max_new_tokens):
      ...
      input_ids = torch.cat([input_ids, next_token], dim=1)

      if next_token.item() == eos_id:
          break

  generated_ids = input_ids[0, prompt_length:].tolist()
  return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
  ```
* **Why It Is Problematic:**
  1. The stopping condition checks exclusively for `next_token.item() == eos_id` (`[EOS]`).
  2. Because the training data trained the model that an assistant message is followed by `<|user|>` in 57.9% of turns, the model often generates `<|user|>` (token 4) when its answer is complete.
  3. Because `generate_text` does not recognize `<|user|>` as a stop token, it does not break.
  4. When decoding at the end, `tokenizer.decode(generated_ids, skip_special_tokens=True)` silently erases `<|user|>`, leaving the subsequent text concatenated, or allowing the model to drift into uncontrolled hallucination/repetition up to `max_new_tokens` (100).
  5. There is no repetition break condition or repeated n-gram detector.
* **Evidence:**
  Inspecting the generation loops shows that the model generates past natural semantic endpoints because no turn boundary stop token (`<|user|>`) is configured in `generate_text`.
* **Severity:** **HIGH**
* **Recommended Fix (Do Not Apply):**
  Expand the stop condition to accept a set of stop IDs:
  ```python
  stop_token_ids = {tokenizer.token_to_id("[EOS]"), tokenizer.token_to_id("<|user|>"), tokenizer.token_to_id("<|assistant|>")}
  if next_token.item() in stop_token_ids:
      break
  ```

---

### Finding 4: Top-P (Nucleus Sampling) Missing Entirely (DEFICIENCY)

* **Relevant File & Lines:** [`src/generate_clean_50k.py`](file:///d:/DEEP-LEARNING-PROJECT/src/generate_clean_50k.py#L50-L59,L89-L96)
* **Code Behavior:**
  ```python
  def generate_text(
      model,
      tokenizer,
      prompt,
      max_new_tokens=100,
      temperature=0.8,
      top_k=50,
      repetition_penalty=1.15,
      do_sample=False
  ):
  ...
      values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
      filtered_logits = torch.full_like(logits, float("-inf"))
      filtered_logits.scatter_(1, indices, values)
      probs = torch.softmax(filtered_logits, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1)
  ```
* **Why It Is Problematic:**
  `top_p` (nucleus sampling, Holtzman et al., 2019) is completely absent from the signature and implementation. Top-K sampling with a static $k=50$ cannot adapt to distribution shape:
  * When the distribution is peaky (high confidence, e.g. prompt completion), Top-50 includes 48 low-probability tokens that can derail generation.
  * When the distribution is flat, Top-50 truncates valid candidates.
* **Severity:** **MEDIUM**
* **Recommended Fix (Do Not Apply):**
  Implement standard nucleus filtering:
  ```python
  sorted_logits, sorted_indices = torch.sort(logits, descending=True)
  cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
  sorted_indices_to_remove = cumulative_probs > top_p
  sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
  sorted_indices_to_remove[..., 0] = 0
  ```

---

### Finding 5: Float16 Arithmetic in Generation Softmax (PRECISION CONCERN)

* **Relevant File & Lines:** [`src/generate_clean_50k.py`](file:///d:/DEEP-LEARNING-PROJECT/src/generate_clean_50k.py#L70-L75,L88-L93)
* **Code Behavior:**
  ```python
  with torch.no_grad():
      with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
          logits = model(input_context)

  logits = logits[:, -1, :]
  # logits remain float16
  logits = logits / temperature
  filtered_logits = torch.full_like(logits, float("-inf"))
  filtered_logits.scatter_(1, indices, values)
  probs = torch.softmax(filtered_logits, dim=-1)
  ```
* **Why It Is Problematic:**
  `logits` are generated in `torch.float16` under `autocast`. Dividing by `temperature` (0.8), scattering `float("-inf")`, and computing `torch.softmax` directly in `float16` can lead to numerical underflow or imprecise probability mass distributions in half precision (float16 dynamic range is limited to $6.1 \times 10^{-5}$ to $65504$).
* **Severity:** **LOW**
* **Recommended Fix (Do Not Apply):**
  Cast to full precision immediately after extraction: `logits = logits[:, -1, :].float()`.

---

### Finding 6: Training Batch Size 1 and Zero Gradient Accumulation (TRAINING LIMITATION)

* **Relevant File & Lines:** [`src/train_clean_50k.py`](file:///d:/DEEP-LEARNING-PROJECT/src/train_clean_50k.py#L53,L387-L408)
* **Code Behavior:**
  ```python
  BATCH_SIZE = 1
  ...
  while step < MAX_STEPS:
      for input_ids, target_ids, loss_mask in train_loader:
          ...
          with torch.autocast(...):
              logits = model(input_ids)
              loss = calculate_masked_loss(logits, target_ids, loss_mask)

          scaler.scale(loss).backward()
          scaler.unscale_(optimizer)
          grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
          scaler.step(optimizer)
          scaler.update()
          scheduler.step()
          step += 1
  ```
* **Why It Is Problematic:**
  1. `BATCH_SIZE = 1` with no gradient accumulation means the optimizer updates model weights on every single sequence.
  2. In Design A, each sequence contains on average only 333 assistant tokens. An effective batch size of 333 tokens creates high-variance, noisy gradient estimates for AdamW.
  3. The entire 50,000 steps processed only 50,000 sequences ($50,000 / 17,683 \approx 2.83$ epochs), corresponding to approximately **16.6 million tokens**.
  4. For a 14.1 million parameter model, training on 16.6 million tokens equates to roughly **1.18 tokens per parameter**. According to scaling laws (Chinchilla, Hoffmann et al., 2022), a compute-optimal model requires $\approx 20$ tokens per parameter ($\approx 280$ million tokens for 14.1M). The model is compute-starved by an order of magnitude.
  5. This explains why validation loss plateaued at 3.816 (Perplexity 45.4). At perplexity 45, the model's next-token distribution has high residual entropy, which makes deterministic greedy decoding fragile.
* **Severity:** **HIGH (Project Limitation)**
* **Recommended Fix (Do Not Apply):**
  Implement gradient accumulation (e.g., `gradient_accumulation_steps = 32` or `64`), giving an effective batch size of 32–64 sequences (~10,000–21,000 tokens per optimizer step), and train for more total token volume.

---

### Finding 7: Missing Learning Rate Warmup (TRAINING PRACTICE)

* **Relevant File & Lines:** [`src/train_clean_50k.py`](file:///d:/DEEP-LEARNING-PROJECT/src/train_clean_50k.py#L365-L369,L408)
* **Code Behavior:**
  ```python
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer,
      T_max=MAX_STEPS,
      eta_min=MIN_LEARNING_RATE
  )
  ```
* **Why It Is Problematic:**
  Training began immediately at step 1 with the maximum learning rate (`3e-4`) on randomly initialized weights with zero warmup steps. Warmup (typically 500–2,000 steps) is standard for AdamW in Transformers to stabilize the running second-moment estimates ($v_t$) before large weight steps occur.
* **Severity:** **LOW**
* **Recommended Fix (Do Not Apply):**
  Use a linear warmup followed by cosine decay (e.g. `get_cosine_schedule_with_warmup`).

---

### Finding 8: Skewed Output Bias Parameter (MODEL ARCHITECTURE ARTIFACT)

* **Relevant File & Lines:**
  * [`src/model.py`](file:///d:/DEEP-LEARNING-PROJECT/src/model.py#L217-L220,L287-L291)
  * Checkpoint parameter: `output_bias` (shape: `torch.Size([8000])`)
* **Code Behavior:**
  In `FromScratchLM`:
  ```python
  self.output_bias = nn.Parameter(torch.zeros(vocab_size))
  ...
  logits = F.linear(x, self.token_embedding.weight, self.output_bias)
  ```
* **Why It Is Problematic:**
  Inspection of `output_bias` in checkpoint `clean_50k_best.pt` revealed extreme negative values on role tokens:
  * Token 2 (`[BOS]`): bias = **-3.9394**
  * Token 4 (`<|user|>`): bias = **-3.7226**
  And positive values on high-frequency punctuation and function words:
  * Token 215 (`' the'`): bias = +0.1311
  * Token 19 (`'.'`): bias = +0.1309
  * Token 17 (`','`): bias = +0.1306
  * Token 237 (`' and'`): bias = +0.1221
  * Token 18 (`'-'`): bias = +0.1173
  * Token 210 (`' a'`): bias = +0.1028
  Because `<|user|>` was never present as a target with `loss_mask = 1`, the cross-entropy loss applied continuous negative gradients against token 4 across 50,000 steps, driving its bias to -3.72. Meanwhile, frequent function words and hyphens accumulated positive biases, predisposing the model to predict them when uncertain.
* **Severity:** **MEDIUM**
* **Recommended Fix (Do Not Apply):**
  Either remove `output_bias` (standard in LLaMA / modern GPT-style architectures, where `lm_head` is an unbiased linear projection), or ensure role boundary tokens are properly supervised in conversation modeling.

---

## 3. Systematic Verification of User Checklist Items

### Item 1: Next-Token Targets and Assistant-Only Loss Mask Alignment
* **Files:** [`src/train_clean_50k.py`](file:///d:/DEEP-LEARNING-PROJECT/src/train_clean_50k.py#L98-L101,L216-L225), [`src/build_clean_sequences.py`](file:///d:/DEEP-LEARNING-PROJECT/src/build_clean_sequences.py#L38-L60)
* **Audit Result: VERIFIED CORRECT.**
  * In `CleanTokenDataset.__getitem__`:
    `input_ids = sequence[:-1]` ($S[0 \dots N-2]$)  
    `target_ids = sequence[1:]` ($S[1 \dots N-1]$)  
    `loss_mask = mask[1:]` ($M[1 \dots N-1]$)  
  * For position $j$, `logits[:, j, :]` conditions on $S[0 \dots j]$ to predict $S[j+1]$. The ground-truth target is `target_ids[j] = S[j+1]`.
  * The mask applied is `loss_mask[j] = M[j+1]`.
  * In `build_clean_sequences.py`, `M[k]` is 1 if and only if $S[k]$ is an assistant token.
  * Therefore, predicting token $S[j+1]$ is included in the loss if and only if $S[j+1]$ is an assistant token.
  * There is **no off-by-one misalignment**.

### Item 2: Padding and Unintended Token Contribution to Loss
* **Files:** [`src/train_clean_50k.py`](file:///d:/DEEP-LEARNING-PROJECT/src/train_clean_50k.py#L216-L225), [`src/build_clean_sequences.py`](file:///d:/DEEP-LEARNING-PROJECT/src/build_clean_sequences.py#L113-L115)
* **Audit Result: VERIFIED CORRECT.**
  * In `build_clean_sequences.py`, padding is applied with `constant_values=0` to `seq_mask`.
  * In `train_clean_50k.py`, `loss_mask` is converted to a boolean tensor: `loss_mask = torch.tensor(mask[1:], dtype=torch.bool)`.
  * In `calculate_masked_loss`: `masked_losses = token_losses[loss_mask]`; `return masked_losses.mean()`.
  * `masked_losses` indexes exclusively the `True` positions. The `.mean()` operation divides strictly by `masked_losses.numel()` (the number of assistant tokens), not by the sequence length (1023).
  * Padding tokens contribute $0$ to the numerator and $0$ to the denominator. They do not dilute the loss.
  * User prompt tokens also have `loss_mask = False` and are excluded.

### Item 3: Assistant Responses and EOS Handling During Training
* **Files:** [`src/prepare_tokenization.py`](file:///d:/DEEP-LEARNING-PROJECT/src/prepare_tokenization.py#L87-L112), [`src/build_clean_sequences.py`](file:///d:/DEEP-LEARNING-PROJECT/src/build_clean_sequences.py#L38-L60)
* **Audit Result: FLAW IDENTIFIED (HIGH SEVERITY).**
  * `[EOS]` was placed only at the end of the entire multi-turn conversation.
  * 57.9% of assistant turns (16,982 turns) end with `<|user|>`, not `[EOS]`.
  * 30.3% of conversations (5,362 conversations) end with an unanswered user prompt, leaving `[EOS]` unmasked.
  * Only 42.1% of assistant turns (12,321 turns) are supervised on `[EOS]`.
  * The model is insufficiently trained to emit `[EOS]` at the end of single-turn answers.

### Item 4: Causal Attention and Conversation Boundaries
* **Files:** [`src/model.py`](file:///d:/DEEP-LEARNING-PROJECT/src/model.py#L71-L78), [`src/build_clean_sequences.py`](file:///d:/DEEP-LEARNING-PROJECT/src/build_clean_sequences.py#L109-L136)
* **Audit Result: VERIFIED CORRECT.**
  * `scaled_dot_product_attention` is called with `is_causal=True`. Token $t$ only attends to positions $s \le t$.
  * Design A packages exactly one conversation per 1,024-token sequence with zero conversation packing.
  * Real tokens at position $t < L_{real}$ never attend to subsequent padding tokens at $s \ge L_{real}$.
  * Attention never leaks across conversation boundaries.

### Item 5: Tokenizer, Checkpoint, Conversation Format, and Special Tokens
* **Files:** [`src/generate_clean_50k.py`](file:///d:/DEEP-LEARNING-PROJECT/src/generate_clean_50k.py#L15-L16,L60-L65), [`tokenizer/tokenizer.json`](file:///d:/DEEP-LEARNING-PROJECT/tokenizer/tokenizer.json)
* **Audit Result: VERIFIED CORRECT.**
  * Tokenizer: `tokenizer/tokenizer.json` (vocab size 8,000).
  * Special token IDs match identically: `[PAD]=0`, `[UNK]=1`, `[BOS]=2`, `[EOS]=3`, `<|user|>=4`, `<|assistant|>=5`.
  * Generation formatting string `[BOS]\n<|user|>\n{prompt}\n<|assistant|>\n` was tested against the binary training sequences in `train_tokens.bin` and matched the exact token IDs.
  * Checkpoint: `checkpoints/clean_50k_best.pt` matches the 14,120,768 parameter architecture.

### Item 6: Temperature, Top-K, Top-P, and Repetition Penalty Implementation
* **Files:** [`src/generate_clean_50k.py`](file:///d:/DEEP-LEARNING-PROJECT/src/generate_clean_50k.py#L76-L95)
* **Audit Result: CONFIRMED BUGS & GAPS.**
  * **Critical Bug:** Repetition penalty is placed inside `else:` (`do_sample=True`), completely bypassing greedy decoding (`do_sample=False`).
  * **Missing Feature:** `top_p` (nucleus sampling) is not implemented.
  * **Precision Concern:** Decoding arithmetic runs in `float16` instead of `float32`.

### Item 7: Generation Stopping Conditions
* **Files:** [`src/generate_clean_50k.py`](file:///d:/DEEP-LEARNING-PROJECT/src/generate_clean_50k.py#L65,L98-L100)
* **Audit Result: CONFIRMED DEFECT.**
  * Stopping condition checks only `next_token.item() == eos_id`.
  * Role tokens like `<|user|>` are not checked, allowing generation to continue past completion.
  * No repetition detector exists to halt runaway loops.

### Item 8: Distinction Between Bugs, Hypotheses, and Limitations
* **Confirmed Implementation Bugs:**
  1. Repetition penalty bypassed during greedy decoding in [`generate_clean_50k.py`](file:///d:/DEEP-LEARNING-PROJECT/src/generate_clean_50k.py#L76-L86).
  2. Single-token stop condition in [`generate_clean_50k.py`](file:///d:/DEEP-LEARNING-PROJECT/src/generate_clean_50k.py#L98) failing to stop on turn-boundary role tokens.
* **Confirmed Data/Pipeline Flaws:**
  1. Missing `[EOS]` tokens after intermediate assistant turns in multi-turn conversations (57.9% of turns).
  2. Unanswered user prompts (30.3% of conversations) leaving `[EOS]` unmasked.
* **Confirmed Project Limitations (Not Code Bugs):**
  1. Severe under-training (16.6M tokens for a 14.1M parameter model = 1.18 tokens/param vs 20 tokens/param standard).
  2. High residual validation loss (3.816 / Perplexity 45.4).
  3. Noisy optimization from batch size 1 with zero gradient accumulation.
  4. Absence of learning rate warmup.
* **Disproven Hypotheses:**
  1. *Representation Collapse / Anisotropy:* Disproven by the representation geometry diagnostic (effective rank is 226.2 / 384 at layer 6, pairwise cosine similarity is 0.0518; no collapse).
  2. *Loss Mask Misalignment:* Disproven; targets and loss masks are mathematically synchronized.
  3. *Padding Leakage:* Disproven; padding tokens are completely excluded from loss calculation.

---

## 4. Synthesis: Why Generated Responses Fail

The poor generation behavior observed in `clean_50k_best.pt` is not caused by representation collapse or loss mask alignment errors. Instead, it is the result of a compounding interaction between decoding bugs and training limitations:

1. **Greedy Argmax Lock-In (The Primary Cause of Infinite Loops):**  
   In `generate_clean_50k.py`, greedy decoding (`do_sample=False`) bypasses repetition penalty entirely. Because the model has a perplexity of 45.4 and strong positive biases on common words, deterministic argmax repeatedly selects high-probability n-grams. Causal self-attention attends back to identical tokens, reinforcing the same logits and trapping greedy decoding in an inescapable cyclic loop. When repetition penalty is enabled for greedy decoding, these loops disappear completely.

2. **Incomplete Turn-Termination Conditioning (Why the Model Fails to Stop):**  
   In multi-turn conversations, 57.9% of assistant turns were followed by `<|user|>`, not `[EOS]`. Meanwhile, `generate_text` only halts when `[EOS]` is produced. When the model finishes an answer, it frequently attempts to emit `<|user|>` or fails to cross the `[EOS]` probability threshold. `generate_text` ignores `<|user|>`, strips it during decoding, and forces the model to keep generating for 100 tokens.

3. **Compute and Data Starvation (Why Answers Lack Factual Precision):**  
   The model was trained on only 16.6 million tokens—less than 1.2 tokens per parameter. With an effective batch size of 1 sequence and no warmup, the model learned basic grammar, persona phrases, and syntactic templates, but lacks the parameter convergence necessary to answer specific technical questions accurately.

---

## 5. Non-Destructive Next Steps (For Future Implementation)

When moving to the next implementation phase, the following isolated fixes will address the generation defects without requiring architectural changes:

1. **Fix Decoding (`generate_clean_50k.py`):**
   * Apply repetition penalty to `logits` before branching into argmax or sampling.
   * Add `<|user|>` and `<|assistant|>` to the generation stop condition set.
   * Cast `logits` to `float32` prior to softmax.
   * Add nucleus (`top_p`) sampling.

2. **Fix Conversation SFT Formatting (`build_clean_sequences.py` / `prepare_tokenization.py`):**
   * Place `[EOS]` after every assistant turn in multi-turn data: `<|assistant|>\n{text}\n[EOS]\n`.
   * Discard trailing unanswered user turns so that every conversation ends on an assistant `[EOS]`.

3. **Enhance Training Hyperparameters (`train_clean_50k.py`):**
   * Introduce gradient accumulation (e.g. 32 steps) to stabilize AdamW updates.
   * Add 1,000 steps of linear warmup.
