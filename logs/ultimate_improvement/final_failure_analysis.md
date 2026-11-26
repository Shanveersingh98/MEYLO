# Final Failure Analysis & Capacity Boundaries: Ultimate Improved Model

**Evaluated Checkpoint:** `checkpoints/ultimate_improvement/best.pt`  
**Verified SHA-256:** `cecba4946a8b5a0404e42e9e1790c0d780228da45e07b1910d190d89cbadf2fc`  
**Evaluation Suite:** 248 items across 12 suites  
**Date:** 2026-10-27  

---

## 1. Executive Summary

While the Ultimate Improvement curriculum and decoding enhancements resolved several critical defects (eliminating 100% of repetition loops, boosting instruction following from 6.67% to 33.33%, improving multi-turn consistency from 30.0% to 50.0%, and enabling 68.0% arithmetic accuracy via tool assistance), specific capability boundaries remain.

This document details exactly what remains weak, categorizes the persistent failure modes, and explains why certain capabilities cannot be acquired by a 14.1M parameter model trained from scratch without massive internet-scale pretraining.

---

## 2. Detailed Breakdown of Persistent Failure Modes

### Category 1: Zero-Shot Algorithmic Code Synthesis (Executable Python: 0/15 functional)
* **What Works:**
  - The model has mastered Markdown code fencing (` ```python ... ``` `).
  - It generates valid function headers (`def func_name(params):`).
  - It generates realistic PEP-257 docstrings explaining parameters and returns.
  - Subword bigram repetition loops (`celsius_to_to_to...`) are completely eliminated by 3-gram repetition blocking.
* **What Fails:**
  - Generating novel algorithms that pass multi-condition unit tests zero-shot (e.g. `celsius_to_kelvin`, `is_power_of_two`, `remove_none_values`).
  - Example generation on `celsius_to_kelvin(celsius)`:
    ```python
    def celsius_to_to(celsius):
        """Return the temperature from Celsius * 9`.
        # Return the maximum temperature from Felsiu
    ```
* **Root Cause & Capacity Ceiling:**
  - Writing functional code for arbitrary unseen tasks requires state tracking, variable binding, and symbolic program reasoning.
  - A 14.1M parameter Transformer has only 56 MB of parameter memory and 6 transformer layers.
  - In frontier AI research, even models with 1 billion parameters struggle with zero-shot code synthesis unless pretrained on hundreds of billions of tokens of source code (e.g. CodeLlama, StarCoder). Attempting to teach novel algorithmic synthesis to a 14M model via 1,200 fine-tuning sequences is fundamentally bounded by parameter capacity.

### Category 2: Zero-Shot Mental Multi-Digit Arithmetic (4.0% Neural Alone vs 68.0% Tool-Assisted)
* **What Works:**
  - Single-digit subtraction and addition with small numbers (e.g., `8 - 5 = 3`, `7 - 0 = 7`).
  - When tool assistance is active, accuracy immediately jumps to **68.0%** (68/100) on single-step arithmetic and solves complex numbers deterministically.
* **What Fails (Model Alone):**
  - Multi-digit mental arithmetic (3-digit and 4-digit numbers) without external tools remains near 0%.
  - Under pure neural execution, the model attempts intermediate reasoning steps (e.g. *"First add tens: 330 + 123 = 569..."*), but hallucinates the intermediate arithmetic.
* **Root Cause:**
  - In a fixed-depth Transformer, each token is generated in a constant number of layer passes ($L=6$). Computing multi-column addition with cascading carry bits requires sequential state updates that cannot be packed into 6 attention layers for arbitrary multi-digit numbers.
  - This validates why Part 9 (Tool Grounding) is the only mathematically sound solution for deterministic calculation on small models.

### Category 3: Open-Domain Encyclopedic Knowledge (Factual QA: 5.0% - 15.0%)
* **What Works:**
  - Specialized concepts that were explicitly included in the curriculum (`monopsony`, `artificial intelligence`, `machine learning`, `photosynthesis`, `gravity`) are recalled with high fidelity.
* **What Fails:**
  - General world knowledge not reinforced in the curriculum (e.g., "What is the capital of France?", "What is the boiling point of water?", "What does CPU stand for?").
  - The model outputs generic conversational responses or tangential definitions rather than precise facts.
* **Root Cause:**
  - Factual recall in language models is a direct function of pretraining dataset size. Frontier models store facts after ingesting trillions of tokens from Wikipedia, books, and web crawls.
  - Our educational model was trained from scratch on only ~16 million tokens of OASST1 and curated curriculum. It is physically impossible for 14.1 million parameters to store the entirety of human encyclopedic knowledge.

### Category 4: Context Boundary Degradation (Suite 12: 0/5)
* **What Fails:**
  - When prompts reach 750 to 920 tokens (near the 1024 context limit), the model fails to extract specific needles in the haystack.
* **Root Cause:**
  - In `FromScratchLM`, learned absolute positional embeddings were trained mostly on sequences averaging 170–315 tokens.
  - Positions $750$ through $1023$ received sparse gradient updates compared to positions $0$ through $300$, resulting in degraded attention focus near the upper context boundary.

---

## 3. Verified Best Practices & Recommendations

1. **Always Use 3-Gram Repetition Blocking:**
   - Setting `no_repeat_ngram_size = 3` eliminates 100% of degenerate loops and should be the non-negotiable default in all inference code.
2. **Keep Tool Grounding Enabled for Math:**
   - The hybrid assistant mode (`/tool on` in `src/chat.py`) provides 68% accuracy on arithmetic that the neural network alone cannot compute.
3. **Keep Model Scoped to Conversational & Educational Tutoring:**
   - Deploy the model for concept explanations, tutoring dialogues, and friendly interaction where it excels, while routing complex code synthesis and calculation to specialized external tools.
