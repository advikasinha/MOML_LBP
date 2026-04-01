# 4-Objective MOC-FS (Partition Entropy)

This README summarizes the 4-objective runs (Compactness, Connectedness, Simplicity, Partition Entropy) for both NSGA-II and LHFiD. Data source: CWRU (balanced to 7,335 segments; 12 severity-aware classes). Fixes applied: per-load normalization, class balancing, severity labels.

## Quick View
- LHFiD summary: [moc_results_entropy/lfhid/summary_v2.json](moc_results_entropy/lfhid/summary_v2.json)
- NSGA-II summary: [moc_results_entropy/nsga/summary_v2.json](moc_results_entropy/nsga/summary_v2.json)
- Plots (per optimizer): 00_convergence_v2.png, 01_pareto_3d_v2.png, 02_feature_importance_v2.png, 03_cluster_simple_v2.png, 04_cluster_compact_v2.png

## Headline Metrics

| Optimizer | Pop | Gen | ARI-4 max | ARI-16 max | Sil max | Pareto size | f3 range | Wall time (s) |
|-----------|-----|-----|-----------|------------|---------|-------------|----------|---------------|
| NSGA-II   | 200 | 200 | 0.3785    | 0.5173     | 0.8450  | 200         | 1–2      | 273.3         |
| LHFiD     | 200 | 200 | 0.3680    | 0.5166     | 0.8435  | 107         | 1–2      | 693.9         |

## Representative Solutions

| Optimizer | Role    | K   | #feat | Active feats        | ARI-4 |
|-----------|---------|-----|-------|---------------------|-------|
| NSGA-II   | Simple  | 10  | 1     | Std Dev             | 0.3785|
| NSGA-II   | Compact | 9   | 1     | Std Dev             | 0.3466|
| LHFiD     | Simple  | 9   | 1     | Std Dev             | 0.3680|
| LHFiD     | Compact | 10  | 1     | RMS                 | 0.2855|

## What the 4th Objective Did
- Added f4 (Partition Entropy/Gini) but behaved largely redundant with compactness/connectedness in this low-D (7-feature) setting; both solvers converged to 1-feature masks.
- LHFiD produced a smaller frontier (107 vs 200) but at ~2.5× runtime. NSGA-II kept more diverse trade-offs and slightly higher ARI/Sil.

## Notes on Method
- Ref dirs (LHFiD): Das-Dennis, 4 objs, n_partitions=8 (165 vectors).
- Masks encoded with MIN_K=2, MAX_K=10; HybridSampling mixes KMeans seeds and datapoint seeds.
- All objectives minimized; Partition Entropy uses inverse-distance soft assignments and Gini impurity.

## Next Experiments (ideas)
1) Make f4 more orthogonal (e.g., Gaussian-kernel soft assignments) so entropy is less coupled to compactness.
2) Force richer masks: raise min_active in HybridSampling to 3–4 and/or penalize masks <3 to see if ARI16 improves.
3) Increase ref-dir granularity for LHFiD (partitions 10–12) only after f4 is decorrelated; otherwise it just adds cost.

## Files
- LHFiD results: [moc_results_entropy/lfhid](moc_results_entropy/lfhid)
- NSGA-II results: [moc_results_entropy/nsga](moc_results_entropy/nsga)
