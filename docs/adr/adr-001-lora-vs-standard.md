# ADR-001: LoRA vs Standard Fine-Tuning

**Date**: 2026-02-20
**Status**: Accepted
**Project**: P3 — Fine-Tuning & Guardrails

## Context

I needed to fine-tune `all-MiniLM-L6-v2` embeddings for dating compatibility. The baseline model produces inverted embeddings (Spearman -0.219), meaning incompatible pairs score higher than compatible ones. The two options were standard fine-tuning (update all 22.7M parameters) and LoRA fine-tuning (add low-rank adapters, 73K trainable parameters, 0.32% of total).

P2 benchmarks (ADR-003) established MiniLM as the quality baseline for local models, making it the natural starting point for fine-tuning.

## Decision

I implemented both and compared empirically using identical hyperparameters: 4 epochs, batch size 16, learning rate 2e-5 (standard) and 2e-4 (LoRA, higher LR is common for adapter tuning), 100 warmup steps, CosineSimilarityLoss, and 8 evaluation metrics.

LoRA configuration: rank 8, alpha 16 (scaling factor = 2r), dropout 0.1, target modules `["query", "value"]` (attention Q/V matrices only, K excluded per best practices).

Both approaches fixed the inverted embeddings. Standard is the better performer, LoRA gets close with far fewer parameters:

| Metric | Baseline | Standard | LoRA | LoRA % of Standard |
|--------|----------|----------|------|---------------------|
| Spearman Correlation | -0.219 | 0.852 | 0.820 | 96.2% |
| Compatibility Margin | -0.083 | +0.941 | +0.748 | 79.5% |
| Cohen's d (effect size) | -0.419 | 7.451 | 3.510 | 47.1% |
| AUC-ROC | 0.373 | 0.993 | 0.974 | 98.1% |
| Best F1 | 0.698 | 0.988 | 0.946 | 95.8% |
| Cluster Purity | 0.839 | 0.986 | 0.912 | 92.5% |
| Training Time (M2 Mac) | n/a | 45 min | 38 min | 84% |
| Trainable Parameters | 0 | 22.7M (100%) | 73K (0.32%) | 0.32% |

The biggest gap is Cohen's d: standard achieves 7.451 vs LoRA's 3.510 (47.1%). Standard creates much wider separation between compatible and incompatible pairs in the embedding space. For classification metrics (AUC-ROC, F1), LoRA is within 2-4% of standard.

## Alternatives Considered

**LoRA (r=8)** - 0.32% of trainable parameters, 16% faster training, and adapters can be swapped at inference for multi-task scenarios. But there's a slight performance drop (~3% Spearman) and more hyperparameters to tune (r, alpha, dropout, target modules). Compelling for multi-task or memory-constrained environments.

**LoRA (r=16 or higher)** - Would likely close the gap with standard performance while staying under 5% of total parameters. But for a 22M parameter model that already fits in 8GB memory, the efficiency gain doesn't justify the added complexity.

**QLoRA (4-bit quantization)** - Even lower memory via quantization, enables training on consumer GPUs. But it requires CUDA and bitsandbytes, and is unnecessary for a 22M model that fits comfortably in M2 Mac unified memory. Deferred to larger model work (ADR-002).

## Quantified Validation

Both approaches flip Spearman from -0.219 to positive (0.852 standard, 0.820 LoRA). LoRA achieves 96.2% of standard's Spearman correlation with 0.32% of the trainable parameters. Training time dropped from 45 to 38 minutes on an M2 Mac due to fewer backward passes through frozen layers. AUC-ROC (0.993 vs 0.974) and F1 (0.988 vs 0.946) show LoRA is within striking distance on classification metrics. The Cohen's d gap (7.451 vs 3.510) is the most significant difference: standard creates much stronger separation in the embedding space.

An adapter merge bug in `generate_finetuned_embeddings` initially produced incorrect LoRA evaluation metrics. After fixing the merge logic, the full 8-metric post-training evaluation completed for both models.

## Consequences

For this dating compatibility task, standard fine-tuning is sufficient. There's no multi-task requirement, and memory is not constrained on M2 Mac. Standard gets the best absolute performance across all 8 metrics.

LoRA becomes relevant when you need multi-task learning (separate adapters per domain), A/B testing (swap adapters at inference without reloading the base model), or when memory is constrained (edge deployment, low-RAM servers). The 96% Spearman match at 0.32% of parameters is a strong efficiency result, just not needed for this single-task project.

Adapter management adds complexity: tracking base model + adapter checkpoint pairs, adding adapter loading logic to inference code, and tuning additional hyperparameters (r, alpha, target modules). (This is analogous to the Decorator pattern in Java: LoRA wraps the frozen base model with lightweight adapters rather than modifying the base class directly, enabling adapter stacking and swapping at runtime.)
