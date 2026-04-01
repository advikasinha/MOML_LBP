# Workflow

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/f6493b0e-bb25-40b0-a969-dc3faf4c4c31" />













------------------------------------------------------------------------------------------------------------

# Optimizer Comparison: NSGA-II vs. LHFiD

multi-objective clustering experiment using **NSGA-II** (baseline) and **LHFiD** (Localized High-Fidelity Dominance). Both algorithms were run with a population size of 200 over 200 generations.

## 1. Performance Metics

| Metric                   | NSGA-II    | LHFiD  | Notes                                                                                       |
| :----------------------- | :--------- | :----- | :------------------------------------------------------------------------------------------ |
| **Max ARI (4-class)**    | **0.3595** | 0.3049 | NSGA-II achieved a slightly better alignment with the main fault type grouping.             |
| **Max ARI (16-class)**   | **0.5000** | 0.3500 | NSGA-II was significantly better at isolating severity sub-clusters within the fault types. |
| **Max Silhouette Score** | **0.8152** | 0.8135 | Both algorithms converged on solutions with extremely compact and well-separated clusters.  |

## 2. Pareto Front Characteristics

| Metric                                                 | NSGA-II | LHFiD               | Notes                                                                                                                                                                                                                          |
| :----------------------------------------------------- | :------ | :------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pareto Front Size**                                  | 200     | **116**             | LHFiD produced a sparser, more concise Pareto front by heavily pruning weakly dominated or geometrically redundant solutions. NSGA-II retained the maximum allowed size (pop. size), likely carrying near-duplicate solutions. |
| **Representative "Simple" Solution (Active Features)** | `RMS`   | `Spectral Centroid` | The algorithms fell into different local minima for their most aggressive feature selection trade-offs.                                                                                                                        |
| **Representative "Simple" Solution (Clusters $K$)**    | K=10    | K=3                 | LHFiD's simple solution favoured coarse grouping ($K=3$), while NSGA-II favoured tighter, more granular subgrouping ($K=10$).                                                                                                  |

## 3. Computational Efficiency

| Metric                | NSGA-II     | LHFiD    | Notes                                                                                                                                                                                                             |
| :-------------------- | :---------- | :------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Optimization Time** | **~34.4 s** | ~427.9 s | NSGA-II is roughly 12$\times$ faster. LHFiD involves computationally heavy high-fidelity dominance comparisons and reference line distance associations in the environmental selection phase at every generation. |
| **Total Run Time**    | ~124.9 s    | ~536.2 s | Includes data loading, K-NN graph building, and evaluation loops.                                                                                                                                                 |

## Key Takeaways

1. **Granularity vs Coarseness:** NSGA-II leaned towards finding a higher number of clusters ($K=10$), doing an exceptional job (ARI=0.50) separating the inner structure of the faults by their severity. Conversely, LHFiD gravitated towards coarse representation ($K=3$), settling on broader categories.
2. **Feature Preference:** NSGA-II exploited the **RMS** feature to segregate vibration severity, while LHFiD preferred **Spectral Centroid**, which identifies frequency shifts but is less sensitive to pure amplitude/severity variations.
3. **Pacing and Front Density:** LHFiD successfully yielded a much more structured and filtered Pareto front (116 solutions), validating its design goal of providing a manageable, high-fidelity set of trade-offs. However, this comes at a substantial computational penalty compared to the highly optimized, fast non-dominated sorting used by NSGA-II.

**NSGA-II worked better if your goal is:**

- **Accuracy/Clustering Performance:** NSGA-II achieved higher ARI scores for both the 4-class (0.3595) and 16-class (0.5000) labels. It was much better at discovering the underlying severity sub-clusters.
- **Speed / Computational Efficiency:** NSGA-II was about 12$\times$ faster (~34s vs ~428s).

**LHFiD worked better if your goal is:**

- **Decision-Making & Interpretability:** LHFiD successfully did what it was designed to do—it filtered out weakly dominated/redundant solutions to give you a much more concise and manageable Pareto front (116 solutions instead of 200).
- **Simplicity:** LHFiD's "simple" solution found a much more natural grouping ($K=3$, representing broad fault types) compared to NSGA-II, which forced high-granularity ($K=10$) even in its most compact solution.

**Summary**
_"While NSGA-II achieved higher clustering accuracy (ARI) and converged much faster, LHFiD provided a superior set of trade-offs for decision-making. LHFiD successfully mitigated Pareto front crowding, yielding a concise set of 116 high-fidelity solutions and discovering models with lower, more interpretable cluster counts (e.g., $K=3$) that NSGA-II missed."_
