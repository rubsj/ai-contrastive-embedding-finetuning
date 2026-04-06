# ADR-002: QLoRA Skipped for Small Models

**Date**: 2026-02-20
**Status**: Accepted
**Project**: P3 — Fine-Tuning & Guardrails

## Context

QLoRA (Quantized Low-Rank Adaptation) extends LoRA by quantizing the base model to 4-bit precision, which cuts memory by roughly 75% and enables fine-tuning 7B+ parameter models on consumer hardware. For this project I'm fine-tuning `all-MiniLM-L6-v2` (22.7M parameters) on an M2 Mac with 8GB unified memory. Standard LoRA already fits comfortably in that budget (ADR-001), so the question is whether QLoRA adds anything here.

## Decision

I skipped QLoRA for this project and used standard LoRA (fp32 weights + fp32 adapters) instead.

The primary blocker is hardware: QLoRA requires `bitsandbytes`, which depends on CUDA. The M2 Mac uses Metal, not CUDA, so QLoRA is not runnable on this hardware without switching to a cloud GPU. Beyond that, the model is too small to benefit. 22.7M parameters fit easily in 8GB RAM, so QLoRA's 75% memory savings solve a problem I don't have. Adding quantization logic, calibration data requirements, and potential numerical instability for a 22M model is complexity with no payoff. There's also a performance risk: 4-bit quantization could degrade a 22M model more than a 7B+ model, since fewer parameters means less redundancy to absorb quantization errors.

## Alternatives Considered

**QLoRA (4-bit + LoRA)** - Extreme memory efficiency, proven on 7B to 65B models. But it requires CUDA (incompatible with M2 Mac), is overkill for a 22M model, and adds quantization complexity for no benefit at this scale.

**8-bit quantization** - Less extreme than 4-bit with better precision. But it still requires bitsandbytes and CUDA, so the same hardware blocker applies. Unnecessary for a 22M model.

**Standard fine-tuning (fp32)** - Maximum performance with no adapters. Used as the baseline comparison in ADR-001. 77x more trainable parameters than LoRA and slower training, but no adapter overhead.

## Quantified Validation

The memory math makes the case clearly. A 7B model in fp16 needs roughly 14GB for the base model plus 2GB for LoRA adapters, totaling 16GB, which exceeds M2 Mac's 8GB. The same 7B model with 4-bit QLoRA drops to roughly 3.5GB base plus 2GB adapters, totaling 5.5GB, which fits. For the 22.7M model in this project, standard LoRA uses a fraction of the available 8GB. QLoRA's compression is solving for a constraint that doesn't exist here.

## Consequences

No CUDA dependency means the project runs natively on M2 Mac via PyTorch MPS backend. No quantization numerical issues to debug, and no calibration data to prepare. Iteration stays fast.

If a future project fine-tunes 7B+ models locally, M2 Mac's 8GB will be insufficient and QLoRA (or a cloud GPU) becomes necessary. That tradeoff is worth revisiting when the model size actually demands it. (This is similar to choosing PNG over JPEG for small images: lossless is the right call when the file already fits in memory, and lossy compression only matters when it doesn't.)
