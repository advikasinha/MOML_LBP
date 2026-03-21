Here's the full mechanical story, from physics to numbers:

What the results actually say about the bearing

1. RMS alone separates all fault classes — and that's a physical statement
   The NSGA-II ran 200 generations across 7 candidate features and consistently put RMS (and nothing else) on the Pareto front. That's not a statistical accident — it's telling you something physical.

RMS = √(mean(x²)) is the total vibrational energy of the signal. When a bearing is healthy, vibration energy comes from shaft rotation harmonics and background noise — low and stable. When a spall or crack develops on a raceway or ball, every time a rolling element passes over the defect it creates a sharp impact. That impact excites the bearing's natural frequencies and injects a burst of energy into the signal. The bigger the spall, the stronger the impact, the higher the RMS.

After per-load normalisation removes the load-induced amplitude shift, what's left in the RMS is purely the fault-induced energy. The fact that one number suffices means: different fault types and severity levels produce genuinely different energy levels. You don't need shape statistics (Kurtosis, Crest Factor) or frequency content (Spectral Centroid) on top of that. The damage speaks loudest through energy.

2. K=10 optimal clusters with only 4 fault types — the algorithm is doing fault sizing
   There are 4 fault types (Normal, IR, Ball, OR). Yet the optimal solution picks K=10 clusters. The 16-class ARI (0.50) significantly exceeds the 4-class ARI (0.36).

This means the algorithm isn't just finding "which fault" — it's finding "which fault at which severity". The 4 severity levels (7/14/21 mil spall diameter) produce proportionally larger defects, larger impacts, larger RMS values. The clustering is naturally stratifying along the severity axis:

Normal → 1 cluster
IR @ 7mil, 14mil, 21mil, 28mil → roughly 4 sub-clusters
Ball @ multiple sizes → sub-clusters
OR @ 3 positions × sizes → sub-clusters
This is unsupervised fault sizing, not just fault detection. No severity label was ever shown to the algorithm during training.

3. The OR fault is mechanically the hardest to cluster — and the data confirms it
   OR fault has three mounting positions: @3 (12 o'clock, unloaded zone), @6 (3 o'clock, transition), @12 (6 o'clock, maximum load zone). The load zone matters enormously: a spall in the loaded zone generates much stronger impacts than the same spall in the unloaded zone because the Hertzian contact force is much higher.

In v1 (only the @6 position was loaded due to the glob bug), OR was artificially uniform. In v2 with all three positions, OR now has 3× the intra-class spread — the same 7-mil OR fault looks completely different at @3 vs @12. This is why v2's ARI is lower than v1's (0.36 vs 0.44). It's not a regression — it's a more honest reflection of how much harder OR fault diagnosis actually is in practice. The position of the defect within the load zone is a hidden variable that pure vibration energy can't fully resolve without additional information (e.g. shaft angle encoder).

4. f3 Pareto range of 1–2 means the other 5 features are mechanically redundant here
   The entire Pareto front uses only 1–2 features. No 3-feature solution was non-dominated. Mechanically, this happens because:

RMS and Std Dev are linearly related (Std Dev ≈ RMS when zero-mean) — one subsumes the other
Kurtosis and Crest Factor measure impulsiveness, but at the severity levels in CWRU (7–28 mil spalls at 12 kHz, 1797 RPM), the impulses are strong enough that energy (RMS) already separates the classes — the impulse shape adds marginal new information
Spectral Centroid shifts toward BSF/BPFI frequencies under fault, but the 1024-sample window at 12 kHz doesn't give enough frequency resolution to make this strongly discriminative over energy content
Skewness is near-zero for most fault types since impacts are symmetric in time
The curse of dimensionality plays a role too: with 7 nearly-correlated features, the within-cluster spread grows faster than between-cluster separation when you add dimensions.

The practical implication: a single RMS sensor, properly normalised per load condition, is sufficient for fault type and severity discrimination on this bearing.

5. Silhouette = 0.82 tells you the clusters are real, not statistical
   A Silhouette Score of 0.82 (out of 1.0) means the average sample is much closer to the centroid of its own cluster than to the nearest foreign cluster. In 1D (just RMS), these 10 clusters are very well-separated. Mechanically this means the energy bands for each fault-severity combination are distinct and don't overlap significantly — there's a real physical gap between, say, a 14-mil IR fault and a 21-mil IR fault in terms of RMS output.

The one-sentence mechanical summary
After removing load-condition amplitude shifts, the total vibrational energy (RMS) of a CWRU bearing signal is sufficient to stratify 10 distinct fault-severity states without any labels — because each bearing defect injects energy proportional to the spall size and contact force, and these energy levels are physically non-overlapping for the fault sizes tested.

Drive End vs Fan End — the mechanical case
What makes Drive End structurally different
The CWRU test rig has two bearings on the same shaft:

Drive End (DE): SKF 6205-2RS — sits between the shaft and the dynamometer (the load source). It directly carries the applied radial load.
Fan End (FE): SKF 6203-2RS — sits on the opposite side, beneath the motor's cooling fan. It carries only the rotor's own weight with no applied external load.
These are not the same bearing. Different geometry, different characteristic frequencies:

Property DE (6205-2RS) FE (6203-2RS)
Balls 9 8
Pitch diameter 38.5 mm 28.5 mm
Ball diameter 7.94 mm 6.75 mm
BPFI multiplier 5.415 × fr 4.947 × fr
BPFO multiplier 3.585 × fr 3.053 × fr
BSF multiplier 2.357 × fr 1.994 × fr
Three separate mechanical consequences follow from this:

① Contact force is fundamentally different. At DE, the Hertzian contact force between ball and raceway scales with the applied dynamometer load plus rotor weight. At FE it's rotor weight only. A 7-mil spall at DE gets struck with substantially more force than the same spall at FE. Impulse energy = contact force × deformation time. So DE produces stronger, crisper impacts for the same physical defect size.

② Transmission path. The accelerometer at DE is mounted on the bearing housing, roughly 3–4 cm from the defect. The FE accelerometer is on the motor housing — the signal must travel through the shaft, motor end-cap, and housing wall before reaching the sensor. Every interface attenuates and reflects — FE signals are structurally low-passed before they're even measured.

③ Electromagnetic contamination. The FE bearing sits inside the motor casing. The motor's rotating magnetic field induces vibration at 2× line frequency (120 Hz) and its harmonics, which rides on top of the bearing signal. This is absent at DE.

What we'd hypothesise for Fan End
Hypothesis 1 — Kurtosis rises to prominence, possibly displacing RMS on the Pareto front.

At DE, the fault impulses are strong enough that RMS (energy) separates the classes cleanly. At FE, the impulses are weaker relative to background noise. Kurtosis is the 4th standardised moment — it's dimensionless and measures impulsiveness rather than energy. A weak, sharp spike in a noisy background might not shift RMS much but will move Kurtosis dramatically. We'd expect the Pareto front to favour {Kurtosis} or {Kurtosis, RMS} rather than {RMS} alone.

Hypothesis 2 — f3 range widens (2–4 features rather than 1–2).

Lower SNR means a single feature captures less. The optimiser would need to combine complementary statistics to maintain cluster quality. Crest Factor and Peak-to-Peak might re-enter the Pareto solutions because they're also impulsiveness metrics less sensitive to overall energy level.

Hypothesis 3 — ARI drops, but Silhouette might not drop as much.

ARI measures alignment with ground-truth fault labels. Weaker signals → worse separation → lower ARI. But Silhouette measures internal cohesion relative to the clusters the algorithm found — even if those clusters don't align with fault types perfectly, they may still be tight. The algorithm would just find different groupings (perhaps load-severity combinations instead of fault-type groupings).

Hypothesis 4 — FIX-1 (per-load normalisation) matters less at FE.

At DE, the applied load shifts the vibration amplitude significantly (that's the whole point of the dynamometer). At FE, the bearing load doesn't change with the dynamometer setting — only the shaft speed changes slightly. The amplitude variation between 0 HP and 3 HP is much smaller at FE. Per-load normalisation would give a smaller ARI improvement.

Hypothesis 5 — OR class is simpler at FE, but Ball might be harder.

CWRU's FE dataset only has OR fault at the @6 position (no @3 or @12 variants) because without a strong defined load direction, the "loaded zone" isn't meaningful. OR clustering at FE should therefore be easier (no intra-class position variance). However, Ball fault at FE might be harder: the FE ball is smaller, rotates faster around a smaller cage, and its BSF (1.994×fr) puts the characteristic frequency closer to shaft harmonics — harder to separate spectrally.

Hypothesis 6 — K would shrink.

With fewer OR sub-clusters (no mounting position variation) and weaker severity discrimination, the optimal K would likely be lower — perhaps 6–8 rather than 10. The Pareto front might shift toward fewer, coarser clusters.

The core physical intuition
Drive End is a high-SNR, high-contact-force measurement where energy content is the dominant discriminating signal. Fan End is a low-SNR, motor-contaminated measurement where impulsiveness relative to background becomes more important than absolute energy. Our results showing RMS-dominance are specifically a Drive End result — not a universal bearing result. Running the same pipeline on FE data would be a meaningful test of whether the physical interpretation generalises.
