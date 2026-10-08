# Leaf & Gill: accuracy and trust meter

Fresh iNaturalist research-grade photos observed since 2025-06-01 (after BioCLIP 2's training data), openly licensed. 3000 photos, 2334 of species in the pack, 666 not in it.

| | Plants | Fungi | All |
|---|---|---|---|
| Top-1 species (in pack) | 85.3% | 85.4% | 85.3% |
| Top-5 species (in pack) | 97.4% | 96.6% | 97.1% |
| Top-1 genus (all photos) | 80.2% | 82.5% | 80.9% |

## Trust meter (held-out 1219 photos)

| | Raw model score | TabPFN trust meter |
|---|---|---|
| Separates right from wrong (AUROC, higher is better) | 0.817 | 0.844 |
| Brier score (lower is better) | 0.167 | 0.148 |
| Right answers among the 40% most trusted | 89.5% | 90.8% |
| Answers shown as Confident | 1.1% | 23.0% |
| Confident answers that were right | 92.3% | 91.4% |
| Wrong answers shown as Confident | 1 | 24 |

Overall, 66.4% of all photos (including species outside the pack) got the right species as the top match. Features: top-1 probability, its margin over the runner-up, top-1 cosine similarity, kingdom.

Both cut-offs were chosen on the training split to keep Confident answers at least 95% right (raw score >= 0.999, TabPFN >= 0.930).
