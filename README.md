# Recovery-Aware Structured Pruning (RASP)

Code and data for the paper **"How Pruning Criteria Shape Post-RLVR Recovery: A Controlled Study of Structured MLP Pruning in a Reasoning LLM"**.

## Abstract

Reinforcement learning with verifiable rewards (RLVR) is widely regarded as essential for recovering pruned reasoning models, substantially outperforming supervised fine-tuning (SFT). Under fine-grained structured MLP pruning of a Qwen2.5-family 7B model at moderate sparsity, we find that this advantage does not hold: across four criteria, SFT and RL recover to near-identical accuracy for each criterion tested. We term this *reward-accuracy decoupling*.

What the pruning criterion does control is RL training dynamics. To study this, we introduce Recovery-Aware Structured Pruning (RASP), a scoring function that balances gradient importance with representational diversity via Centered Kernel Alignment (CKA). The resulting *recovery capacity spectrum* shows that diversity-preserving criteria sustain RL optimization while importance-dominated criteria exhibit reward collapse, yet none of these differences translate into accuracy gains over SFT.

These findings reframe the role of RL in pruning recovery: SFT warmup captures most recoverable reasoning capacity, while RL serves as a diagnostic that reveals structural deficiencies invisible to accuracy. Directional trends hold across three benchmarks, with preliminary cross-scale and cross-model evidence.

## Setup

```bash
pip install -r requirements.txt
```

## Repository Structure

- `src/` — Core implementation
  - `rasp_scoring.py` — RASP scoring function (Eq. 3-5)
  - `cka_overlap.py` — CKA functional overlap computation
  - `importance_score.py` — Gradient-based importance scoring
  - `prune.py` — Structured pruning module
  - `model_utils.py` — Model loading and manipulation utilities
  - `train_grpo.py` — GRPO recovery training
  - `eval_gsm8k.py` — GSM8K evaluation
  - `eval_mbpp.py` — MBPP evaluation
  - `grasp_scoring.py` — GraSP baseline scoring
  - `gradient_flow_analysis.py` — Gradient flow analysis
  - `cka_stability_analysis.py` — CKA stability analysis
  - `correlation_analysis.py` — Score correlation analysis
- `tables/` — LaTeX table sources
- `figures/` — Paper figures

## Quick Start

### 1. Compute importance and CKA scores

```bash
python src/importance_score.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --output_path scores/importance.pt
python src/cka_overlap.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --output_path scores/cka.pt
```

### 2. Pruning

```bash
python src/prune.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --importance_path scores/importance.pt \
    --cka_path scores/cka.pt \
    --mode rasp --alpha 2.0 --sparsity 0.2 \
    --output_path models/pruned_rasp_a2
```

### 3. SFT Recovery

See `src/run_mvp.sh` for the complete two-stage pipeline (SFT warmup + GRPO).

### 4. GRPO Training

```bash
python src/train_grpo.py \
    --model_path models/pruned_rasp_a2 \
    --output_path models/recovered_rasp_a2 \
    --learning_rate 1e-6 --temperature 0.4 \
    --max_new_tokens 1024 --num_generations 4
```

### 5. Evaluation

```bash
python src/eval_gsm8k.py --model_path models/recovered_rasp_a2 --max_new_tokens 2048
python src/eval_mbpp.py --model_path models/recovered_rasp_a2
```

## Data

- **GSM8K**: automatically downloaded via HuggingFace `datasets`
- **MATH-500**: subset of MATH benchmark (Hendrycks et al., 2021)
- **MBPP**: automatically downloaded via HuggingFace `datasets`
