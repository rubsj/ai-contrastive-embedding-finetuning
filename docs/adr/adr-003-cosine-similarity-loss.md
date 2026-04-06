# ADR-003: CosineSimilarityLoss Selection

**Date**: 2026-02-20
**Status**: Accepted
**Project**: P3 — Fine-Tuning & Guardrails

## Context

I needed a loss function for fine-tuning `all-MiniLM-L6-v2` embeddings on dating compatibility pairs. The baseline model produces inverted embeddings (Spearman -0.219), so the loss function must push compatible pairs closer and incompatible pairs farther apart in embedding space. Sentence-Transformers supports several options: CosineSimilarityLoss, ContrastiveLoss, TripletLoss, and MultipleNegativesRankingLoss.

## Decision

I used **CosineSimilarityLoss** for training.

```python
from sentence_transformers import losses, InputExample

train_examples = [
    InputExample(texts=[profile1, profile2], label=1.0),  # compatible
    InputExample(texts=[profile3, profile4], label=0.0),  # incompatible
]

loss = losses.CosineSimilarityLoss(model)
```

The main argument is training-evaluation alignment: I evaluate with cosine similarity (Spearman correlation, margin), so the loss should optimize the same metric. CosineSimilarityLoss also accepts continuous floats in [0, 1], which leaves room for graded compatibility scores in future work. And it has no margin hyperparameter to tune, unlike ContrastiveLoss or TripletLoss.

## Alternatives Considered

**ContrastiveLoss** - Explicit margin enforcement, well-studied for metric learning. But it optimizes Euclidean distance, not cosine similarity, creating a metric mismatch with evaluation. Sentence-BERT normalizes embeddings to the unit sphere, where Euclidean and cosine are monotonically related, but optimizing Euclidean explicitly is less direct. It also requires grid-searching a margin value (0.5, 1.0, 2.0 are common) and only accepts binary labels.

**TripletLoss** - Strong performance on retrieval via relative ranking. But it requires constructing (anchor, positive, negative) triplets from pairs, which means 3x data prep complexity and a triplet mining strategy (random, semi-hard, or hard). Same margin tuning problem as ContrastiveLoss.

**MultipleNegativesRankingLoss** - No explicit negatives needed, scales well with in-batch negatives. But it requires large batch sizes (64+) to work effectively. My batch size is 16 due to memory constraints on M2 Mac, so this was a non-starter.

## Quantified Validation

Fine-tuning with CosineSimilarityLoss (4 epochs, batch size 16, LR 2e-5):

| Metric | Baseline | Fine-Tuned | Improvement |
|--------|----------|------------|-------------|
| Spearman Correlation | -0.219 | 0.853 | +1.072 |
| Compatibility Margin | -0.083 | +0.940 | +1.023 |
| Cohen's d | -0.419 | 7.727 | +8.146 |
| AUC-ROC | 0.373 | 0.994 | +0.621 |

CosineSimilarityLoss reversed the inverted embeddings and achieved near-perfect separation (AUC 0.994). These are the standard fine-tuning results from ADR-001; LoRA results are slightly lower but follow the same pattern.

## Consequences

The loss function directly optimizes the metric I evaluate on, so there's no translation gap between training and evaluation. One fewer hyperparameter to tune (no margin), and the data pipeline stays simple with pair-label format instead of triplet construction.

CosineSimilarityLoss doesn't explicitly push hard negatives farther apart the way ContrastiveLoss or TripletLoss would. Empirically it performed well on this dataset, but if a future task requires retrieval ranking rather than similarity scoring, TripletLoss might outperform. (This is the same principle as choosing a distance metric for KNN: if you evaluate with cosine distance, train with cosine loss.)
