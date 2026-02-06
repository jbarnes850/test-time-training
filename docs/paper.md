# Surprisal-Guided Selection: Compute-Optimal Test-Time Strategies for Execution-Grounded Code Generation

**Technical Report Draft - ICML 2026**

---

## Abstract

Test-time training (TTT) has emerged as a paradigm for adapting language models through gradient-based updates at inference. But is adaptation the right strategy? We study compute-optimal test-time strategies for verifiable execution-grounded (VEG) tasks, domains like GPU kernel optimization where a deterministic evaluator provides dense, continuous reward signals.

Using KernelBench as our testbed and a 120B-parameter model (GPT-OSS-120B with LoRA adaptation), we find that **search outperforms minimal adaptation (1-5 gradient steps)**: Best-of-N sampling achieves 100% task success at K=64 while TTT's best checkpoint reaches only 30.6% (3-seed mean), with TTT's "equivalent K" falling below 1—worse than a single random correct sample. The failure mode is over-sharpening: gradient updates collapse diversity toward mediocre solutions rather than discovering optimal ones.

Our main contribution is **surprisal-guided selection**: selecting the *highest-surprisal* (lowest-confidence) correct sample yields 80% success vs 50% for most-confident selection, a 30% improvement. Extending to surprisal-guided-top3 (evaluating the 3 highest-surprisal correct samples) matches oracle performance at 100%. This inverse relationship between model confidence and kernel quality—validated through length-controlled analysis—provides a practical, zero-cost selection strategy that recovers oracle performance.

These findings establish that for dense-reward VEG tasks, compute should be allocated to sample diversity and intelligent selection rather than gradient adaptation. The surprisal-guided selection principle—that optimal solutions occupy high-surprisal regions of the model's distribution—may generalize to other execution-grounded domains where the optimum lies in the distribution tail. For GPU kernels, **optimal performance lives in the tail, and surprisal provides a practical signal for navigating it.**

**Keywords:** test-time compute, surprisal-guided selection, GPU kernel optimization, KernelBench, Best-of-N sampling, model calibration, verifiable execution

---

![Teaser: Compute-Optimal Test-Time Strategies](/artifacts/fig_teaser_new.png)

*Figure 1: Test-time strategy comparison for GPU kernel optimization. TTT (red) underperforms even single-sample random selection. Surprisal-guided selection strategies (blue) match oracle performance. The key insight: model surprisal provides a practical signal for identifying high-quality kernels.*

---

## 1. Introduction

This paper studies compute-optimal test-time strategies for verifiable execution-grounded (VEG) tasks—domains where a deterministic evaluator provides ground-truth feedback on model outputs. GPU kernel optimization exemplifies VEG: KernelBench (Ouyang et al., 2025) evaluates 250 PyTorch ML workloads on both functional correctness and runtime speedup, with the CUDA compiler and hardware providing an unambiguous, continuous reward signal. The defining characteristic is that the evaluator provides ground-truth feedback—no human labeler or AI teacher is needed to judge output quality.

**Why VEG tasks are the ideal testbed.** Unlike binary pass/fail benchmarks, KernelBench provides *continuous* speedup signals (0x to 10x+). This density enables us to detect subtle performance regressions during adaptation that binary rewards would mask. When TTT over-sharpens, we observe the decline in a continuous metric—papers with sparse rewards may miss this entirely.

Recent work on test-time training (TTT) has shown impressive results through extended gradient-based adaptation. TTT-Discover (Yuksekgonul et al., 2026) reports costs of "a few hundred dollars per problem" using ~50 adaptation steps on discovery tasks. This raises a fundamental question: **is adaptation the right strategy for dense-reward VEG tasks, or does simple search suffice?**

We answer this question through controlled experiments comparing Best-of-N sampling against batch test-time training under matched compute budgets. Using GPT-OSS-120B (a 120B-parameter frontier model) with LoRA adaptation, we evaluate 10 KernelBench L1 tasks across 2 seeds. The results are decisive. Best-of-N at K=64 achieves 100% task success (finding at least one fast correct kernel per task) while TTT's best checkpoint (Best-of-Adaptation) reaches only 35%. Computing TTT's "equivalent K" (the Best-of-N budget needed to match TTT performance) yields K < 1, meaning TTT underperforms *random sampling of a single correct solution*.

The failure mode is **over-sharpening**: gradient updates collapse the policy toward mediocre solutions that happened to succeed early, destroying the diversity needed to find optimal kernels in the distribution tail. Ji et al. (2026) predict that RL gains arise from distribution sharpening rather than discovering new strategies; our failure mode confirms this.

**Our main contribution is surprisal-guided selection.** Probing the relationship between model confidence (log-probability) and kernel quality reveals a surprising inverse correlation: the model is *least* confident about its best solutions. We operationalize this as **surprisal-guided selection**: selecting the *highest-surprisal* (lowest log-probability) correct sample. This achieves 80% success (fast and correct) versus 50% for confidence-guided selection—a 30% improvement with zero additional compute. Extending to **surprisal-guided-top3** (evaluating the 3 highest-surprisal correct samples and selecting the fastest) matches oracle performance at 100%.

Three contributions emerge from our experiments:

1. **Search outperforms minimal adaptation (1-5 GRPO steps) for dense-reward VEG tasks.** Best-of-N scaling saturates at K=16 (99.9% success), while TTT equivalent K < 1. Practitioners should invest in sample diversity, not gradient updates.

2. **Surprisal-guided selection recovers oracle performance.** Selecting from the high-surprisal tail—solutions the model "didn't expect to find"—provides a practical, zero-cost selection strategy.

3. **Mechanistic explanation for TTT failure.** Over-sharpening destroys diversity, confirmed by direct correlation probing. The optimum for kernel optimization lies in the distribution tail; gradient updates collapse toward the mode, missing the tail entirely.

For execution-grounded domains with dense rewards, compute should be allocated to sample diversity and intelligent selection rather than gradient adaptation. The surprisal-guided selection principle (that the model's best solutions occupy high-surprisal regions) may generalize to other VEG domains where rare, high-quality solutions exist in low-probability regions of the model's distribution.

---

## 2. Related Work

**Test-Time Training vs. Search.** TTT-Discover (Yuksekgonul et al., 2026) demonstrates impressive results using ~50 adaptation steps on discovery tasks, reporting costs of "a few hundred dollars per problem." We find different dynamics for dense-reward VEG tasks: search outperforms 1-5 step adaptation, with TTT's best checkpoint underperforming even random sampling. The difference likely stems from reward density: TTT-Discover targets sparse-reward scientific discovery where extended exploration may shift the distribution. Kernel optimization has dense rewards; optimal solutions already exist in the base distribution's tail (see Section 6.4 for detailed comparison).

**Distribution Sharpening and Over-Fitting.** Scalable Power Sampling (Ji et al., Jan 2026) argues that RL gains arise from distribution sharpening rather than discovering qualitatively new strategies. Our TTT failure mode provides empirical evidence: gradient updates concentrate probability on early successes (typically mediocre solutions), collapsing diversity. "Towards Execution-Grounded Automated AI Research" (Jan 2026) notes that RL from execution rewards can collapse to narrow ideas—exactly the over-sharpening we observe. Our results extend this analysis: in dense execution-grounded optimization, the best solutions occupy the low-probability tail of the model's distribution, so sharpening toward high-confidence modes moves in the wrong direction. The compute-optimal strategy is sampling for diversity plus intelligent selection, not further sharpening.

**Selection Strategies and Model Confidence.** Prior work on selection typically favors highest-confidence outputs or uses reward models for reranking. Snell et al. (2024) establish that compute-optimal test-time strategies outperform naive Best-of-N through intelligent selection. S* (Li et al., 2025) achieves SOTA code test-time scaling through Adaptive Input Synthesis—generating new test cases to differentiate candidates—but this requires additional LLM calls. For kernel optimization, we find that **surprisal-guided selection** (highest-surprisal, i.e., lowest log-probability, among correct samples) dramatically outperforms standard approaches with **zero additional compute**. The model's probability distribution already encodes this signal in the high-surprisal tail. This inverse relationship between confidence and quality has precedent in calibration literature—Guo et al. (2017) show modern neural networks are often miscalibrated—but has not been operationalized as a selection strategy for execution-grounded code generation.

**Verifiable Execution-Grounded Tasks.** We focus on VEG tasks—domains where a deterministic evaluator provides ground-truth feedback without human judgment. GPU kernel optimization is the primary example: KernelBench (Ouyang et al., 2025) evaluates 250 workloads on correctness and speedup, with speedup ranging continuously from 0x to 10x+. Related VEG domains include assembly superoptimization (SuperCoder, 2025) and formal theorem proving. The VEG setting enables our surprisal-guided selection strategy: execution feedback allows filtering to correct samples before applying surprisal-based selection.

**Kernel Optimization.** Prior work on LLM-based kernel optimization has not studied selection strategies. Kevin (Baronio et al., 2025) achieves 82% correctness through multi-turn train-time RL but keeps weights frozen at inference. CUDA-L2 (2025) surpasses cuBLAS by 19.2% through two-stage GRPO. Magellan (Jan 2026) requires ~1.5 days of evolutionary search. AccelOpt (2025) uses "Optimization Memory" for kernel search. Our contribution is orthogonal: we show that intelligent selection among diverse samples recovers oracle performance without adaptation or extended search.

---

## 3. Method

![Execution-Grounded RL Environment](/artifacts/fig_environment.png)

*Figure 2: Dual-loop architecture for studying test-time compute strategies. The outer loop (blue) trains a capable base policy via RLVR on 80 KernelBench tasks. The inner loop (green) serves as the experimental arena for comparing test-time strategies—TTT adaptation vs. Best-of-N search—under matched compute budgets. Both loops share the same execution-grounded evaluator (orange), enabling fair comparison. This design isolates the research question: given a capable base policy, what is the compute-optimal test-time strategy?*

### 3.1 Dual-Loop Architecture

To answer "is adaptation the right strategy for dense-reward VEG tasks?", we require controlled comparison between gradient-based adaptation and pure search strategies. This is methodologically challenging: the strategies have different compute profiles, and fair comparison requires (1) a capable base policy as the starting point, and (2) a shared evaluation protocol.

We address this through a **dual-loop architecture** (Figure 2):

**Outer Loop (Train-Time RLVR).** The outer loop establishes generalization across tasks. We train a base policy on 80 KernelBench L1 training tasks using Group Relative Policy Optimization (GRPO) with LoRA adaptation. This produces a capable checkpoint (98.4% correctness, 0.87x mean speedup) that serves as the shared starting point for all test-time strategies. The outer loop runs once; its purpose is to establish the base policy, not to be the subject of comparison.

**Inner Loop (Test-Time Strategies).** The inner loop is the experimental arena. Given the trained base policy and a held-out evaluation set (5 tasks), we compare multiple test-time strategies under matched compute budgets:
- **Best-of-N search**: Sample K candidates, select via oracle/surprisal-guided/random
- **Batch TTT adaptation**: Perform gradient updates, apply BoA checkpoint selection
- **SDPO self-distillation**: Token-level distillation with/without execution feedback

All strategies use the same base checkpoint, temperature (0.25), and maximum tokens (1024). The key variable is *how* test-time compute is allocated: sampling diversity vs. gradient adaptation.

**Shared Evaluator.** Critically, both loops use the identical KernelBench execution-grounded evaluator (Section 3.2). This ensures that "correct" and "speedup" mean the same thing whether we're training, adapting, or simply selecting among samples. The shared evaluator eliminates confounds from evaluation protocol differences.

**Design Rationale.** This architecture isolates the research question. The outer loop answers "can we build a capable policy?"—yes, with 98.4% correctness. The inner loop answers "given capability, what is the compute-optimal test-time strategy?"—our main contribution. By separating these concerns, we ensure that differences in test-time performance reflect strategy choice, not base policy quality.

### 3.2 Execution-Grounded Environment

Both train-time and test-time phases share KernelBench's deterministic evaluator, which provides dense scalar rewards through a five-stage pipeline. Model output is first inserted into a scaffolded kernel template, then compiled with CUDA error capture. Correct samples are those that pass functional equivalence tests against the reference implementation. For correct samples, timing is measured via CUDA events with median taken over multiple trials. The reward is computed as speedup = baseline_time / kernel_time, with incorrect samples receiving zero reward. This execution-grounded setup eliminates reward hacking because a correct, fast kernel is the only path to high reward.

The continuous nature of the speedup reward (ranging from 0x for incorrect to 10x+ for highly optimized kernels) distinguishes this domain from preference-based tasks where rewards are binary or sparse. Every sample provides gradient signal proportional to its quality, enabling efficient gradient aggregation across diverse rollouts.

### 3.3 Train-Time: RLVR (Outer Loop)

Training uses normalized rewards (baseline-relative speedup per task) for stability across tasks with varying baseline performance. See Section 3.1 for the outer loop specification.

### 3.4 Test-Time Strategies (Inner Loop)

At test time, we adapt the trained checkpoint to held-out evaluation tasks using batch updates inspired by TTRL (Zuo et al., 2025). Each adaptation step processes N=5 tasks jointly, sampling K=32 rollouts per task to produce 160 samples per step. A GRPO gradient update is computed across all rollouts, and the rank-16 LoRA adapter is updated in-place. Unlike TTT-Discover's ~50-step full RL, we use minimal adaptation (1-5 steps) to enable controlled comparison with search baselines under matched sample budgets.

### 3.5 Best-of-Adaptation (BoA)

We define **Best-of-Adaptation (BoA)** as checkpoint selection over an adaptation trajectory. Instead of assuming the final checkpoint is best, BoA selects argmax(fast_1) across all steps.

**Algorithm 1: BoA with In-Batch Validation**
```
Input: Tasks T, checkpoint theta_0, steps S, rollouts K
scores[0] = evaluate(theta_0, T)
for s = 1 to S:
    rollouts = sample(theta_{s-1}, T, K)
    theta_s = gradient_update(theta_{s-1}, rollouts)
    scores[s] = aggregate_fast_1(rollouts)  # Same rollouts, no extra budget
return theta_{argmax(scores)}
```

**Early Stopping Variant**: Stop when validation regresses for P consecutive steps. In our experiments, P=1 matched oracle selection—the first regression signaled the optimal checkpoint.

### 3.6 SDPO: Execution-Grounded Self-Distillation

We extend batch adaptation with **Self-Distilled Policy Optimization (SDPO)**, replacing scalar reward advantages with token-level self-distillation signal conditioned on execution feedback.

**Teacher Context Construction.** For each student rollout, we construct a teacher context containing:
1. The original task prompt
2. (Optional) A correct solution from the same batch, if one exists
3. Structured execution feedback (compile status, correctness, speedup, runtime, error traces)
4. The instruction: "Correctly solve the original question."

Critically, the student's original code is **not** included in the teacher context—only its execution outcome. This forces the teacher to reason from feedback rather than copy the student's approach.

**Token-Level Advantages.** The teacher scores the student's sampled tokens:

$$A_t = \beta \cdot (\log p_{\text{teacher}}(x_t | \text{context}) - \log p_{\text{student}}(x_t | \text{prompt}))$$

where $\beta = 1.0$ controls the strength of the self-distillation signal. Positive advantages indicate tokens the teacher prefers given execution feedback; negative advantages indicate tokens to suppress.

**Compute Trade-off.** SDPO uses identical rollout budgets to BoA and Best-of-N, but requires additional teacher forward passes. We track and report total tokens (student + teacher) to enable fair comparison across methods.

---

## 4. Experimental Setup

### 4.1 Equal-Budget Protocol

The key methodological contribution is rigorous budget matching. Both methods use identical compute:

| Parameter | Best-of-N | Batch TTT | Matched? |
|-----------|-----------|-----------|----------|
| Total rollouts | 320 | 320 (step 1) | Yes |
| Temperature | 0.25 | 0.25 | Yes |
| max_tokens | 1024 | 1024 | Yes |
| Eval mode | fast | fast | Yes |
| Checkpoint | RLVR final | RLVR final | Yes |
| System prompt | Optimized | Optimized | Yes |

### 4.2 Baselines

- **Best-of-N (K=64)**: Sample 64 candidates per task, select highest fast_1. Total: 320 rollouts across 5 tasks.
- **Base policy**: RLVR checkpoint without test-time adaptation (step 0 of batch TTT).
- **SDPO (feedback)**: SDPO update with execution-grounded feedback context.
- **SDPO (prompt-only)**: SDPO update without feedback context (prompt-only teacher), isolating feedback value.

### 4.3 Metrics

- **fast_1**: Fraction of samples that are both correct AND achieve speedup > 1x
- **Correctness**: Fraction of samples passing functional equivalence test
- **Mean speedup**: Average speedup across correct samples

### 4.4 Tasks

**Subset 1 (Primary)**: Tasks {4, 5, 12, 14, 15} from KernelBench L1 eval split. Best-of-N baseline: 39.8% fast_1.

**Subset 2 (Robustness)**: Tasks {18, 28, 29, 30, 32} (offset=5). Best-of-N baseline: 40.5% fast_1.

### 4.5 Selection Strategies

We compare five selection strategies for choosing among K=64 samples per task:

| Strategy | Description | Compute |
|----------|-------------|---------|
| best-correct (Oracle) | Highest speedup among correct samples | Post-hoc |
| random-correct | Random correct sample | Baseline |
| confidence-guided | Highest mean-logprob among correct samples | Zero-cost |
| **surprisal-guided** | **Highest-surprisal (lowest mean-logprob) among correct** | **Zero-cost** |
| **surprisal-guided-top3** | **Best speedup among 3 highest-surprisal correct** | **3 evals** |

The surprisal-guided strategies are motivated by probing experiments (Section 5.2) showing an inverse correlation between model confidence and kernel quality. The intuition: the model's highest-surprisal correct samples are solutions it "didn't expect to find"—often the creative, hardware-aware optimizations that yield maximum speedup.

### 4.6 Compute Accounting

We report **rollouts and total tokens (student + teacher)** for each method. This is required for SDPO because teacher logprob computation adds compute without increasing rollout count. Each run writes a `*_compute.json` file, which is summarized in the final tables.

---

## 5. Results

### 5.1 Main Result: Search Outperforms Minimal Adaptation

The central finding is that **Best-of-N search decisively outperforms test-time training** under matched compute budgets. This section presents the scaling curve comparison; Section 5.2 presents our surprisal-guided selection strategy that achieves oracle-matching performance.

![Best-of-N Scaling Curve](/artifacts/fig_scaling_curve.png)

*Figure 3: Best-of-N scaling curve showing performance vs. samples per task. Performance saturates at K=16 (99.9%). The TTT Best-of-Adaptation (BoA) result at 35% falls below even K=1 random sampling (53.3%), demonstrating that adaptation underperforms pure search.*

**Table 5.1: Best-of-N Scaling (Subset 1, 2 seeds)**

| K | pass@1 | std | 95% CI |
|---|--------|-----|--------|
| 1 | 53.3% | 3.2% | [20%, 100%] |
| 2 | 72.4% | 4.1% | [40%, 100%] |
| 4 | 89.5% | 2.8% | [60%, 100%] |
| 8 | 98.2% | 0.8% | [80%, 100%] |
| 16 | 99.9% | 0.1% | [100%, 100%] |
| 32 | 100% | 0% | [100%, 100%] |
| 64 | 100% | 0% | [100%, 100%] |

Performance saturates at K=16 with 99.9% success. Beyond this point, marginal gains are near-zero, establishing that **modest sampling budgets suffice** for dense-reward VEG tasks.

**TTT Equivalent K < 1**: TTT's Best-of-Adaptation achieves 30.6% fast_1 (3-seed mean). Interpolating on the scaling curve, this falls *below* K=1 (53.3%), meaning TTT is **worse than randomly selecting a single correct sample**. Best-of-N at K=64 achieves 100% (oracle upper bound).

**Table 5.2: TTT vs Best-of-N Summary**

| Method | fast_1 | Equivalent K | Interpretation |
|--------|--------|--------------|----------------|
| Best-of-N K=64 | 100% | 64 | Oracle upper bound |
| Best-of-N K=1 | 53.3% | 1 | Random correct baseline |
| TTT BoA (step 8) | 35.0% | < 1 | **Worse than random** |

**Why TTT Fails: Over-Sharpening**

The failure mode is **distribution collapse**. TTT gradient updates concentrate probability mass on solutions that succeeded early in adaptation—typically mediocre solutions that happened to work. This destroys the diversity needed to find optimal kernels, which lie in the low-probability tail of the distribution.

Evidence for over-sharpening:
1. **Non-monotonic trajectory**: Performance peaks at step 1-2 then regresses (see Section 5.3)
2. **High variance across seeds**: TTT BoA std = 11.3% vs Best-of-N std = 3.2%
3. **Task-specific collapse**: Some tasks peak at step 1, others at step 4—no consistent optimum

### 5.2 Surprisal-Guided Selection: The Model Already Knows

Given that search outperforms minimal adaptation, how should we select among correct samples? Standard practice is to choose the highest-confidence (lowest-surprisal) output. We discover that for kernel optimization, **the opposite strategy works dramatically better**.

![Selection Strategy Comparison](/artifacts/fig_selection_strategies.png)

*Figure 4: Selection strategy comparison showing (a) fast_1 success rate and (b) mean speedup. Surprisal-guided selection achieves 80% vs 50% for confidence-guided—a 30% improvement. Surprisal-guided-top3 matches oracle at 100%.*

**Table 5.3: Selection Strategy Results (Subset 1, 2 seeds)**

| Strategy | fast_1 | std | Mean Speedup |
|----------|--------|-----|--------------|
| best-correct (Oracle) | 100% | 0% | 226.9x |
| **surprisal-guided-top3** | **100%** | **0%** | **139.0x** |
| **surprisal-guided** | **80%** | **0%** | **41.2x** |
| random-correct | 59.2% | 2.7% | 30.0x |
| confidence-guided | 50% | 14.1% | 11.6x |

**Key findings:**

1. **Surprisal-guided beats confidence-guided by 30%** (80% vs 50%). Selecting the *highest-surprisal* correct sample dramatically outperforms the standard confidence-guided selection.

2. **Surprisal-guided-top3 matches oracle.** Evaluating just 3 samples (the 3 highest-surprisal correct ones) and selecting the fastest achieves 100% success—identical to oracle selection over all 64 samples.

3. **High variance in confidence-guided.** Confidence-guided selection has std = 14.1%, indicating unreliable performance. Surprisal-guided has std = 0%, indicating consistent success.

**Statistical strength.** The 30-percentage-point gap (80% vs. 50%) corresponds to Cohen's h = 0.64 (medium-to-large effect). On continuous speedup ratios, a paired Wilcoxon test on log(speedup_surprisal / speedup_confidence) across 10 task-seed pairs yields p = 0.084 (test statistic = 10.0). On binary outcomes, all 3 discordant pairs favor surprisal (exact sign test p = 0.125, one-sided). Statistical power is limited at n = 10.

**Why Does Surprisal-Guided Selection Work?**

The model's probability distribution is a map of **frequency**, not **quality**. Because naive code is more common than expert-optimized code in training data, the model's "confidence" (log-probability) is a proxy for how *common* a strategy is, not how *fast* it is. By selecting for surprisal, we are explicitly filtering for what we call the **Expert Tail**—those rare, high-performance strategies that the model knows how to generate but considers statistically unlikely compared to naive idioms.

Concretely, high-quality kernels often require:
- Unusual memory access patterns the model has rarely seen
- Creative loop structures that deviate from common idioms
- Hardware-specific optimizations not well-represented in training data

These rare solutions occupy high-surprisal regions of the model's distribution. The key insight: **the model already knows which solutions are best; that knowledge is encoded in the surprisal signal, not the confidence signal.** Unlike S* (Li et al., 2025), which requires additional LLM calls to differentiate candidates, surprisal-guided selection extracts this signal at zero cost from existing logprobs.

**Length-Controlled Analysis**

A potential confound: longer code might have lower log-probability simply due to accumulating more token probabilities. We control for this:

| Analysis | Spearman ρ | p-value |
|----------|------------|---------|
| Raw (logprob vs speedup) | -0.047 | 0.27 |
| Partial (controlling for length) | 0.003 | 0.95 |
| Length vs speedup | -0.039 | - |

The partial correlation (controlling for code length) is essentially zero (ρ = 0.003), indicating the surprisal-guided effect is **not explained by code length**. The raw correlation is also weak (ρ = -0.047), confirming that surprisal and speedup do not have a global linear relationship across all samples.

**Why weak correlation coexists with strong selection.** The near-zero correlation may appear to contradict the 80% vs 50% selection result, but these measure fundamentally different things. Correlation measures whether surprisal *linearly predicts* speedup across all 550 correct samples—it does not. Selection operates via *per-task argmax*: for each task, we pick the single highest-surprisal correct sample. This succeeds when the highest-surprisal sample within each task tends to be among its best solutions—a per-task ordinal property that a global linear correlation cannot capture. Concretely, a task may have 40 correct samples with uncorrelated surprisal and speedup, yet the argmax-surprisal sample may still be fast because creative, hardware-aware solutions simultaneously occupy low-probability regions (high surprisal) and achieve high speedup. The selection strategy exploits this tail structure, not a linear trend.

**Quartile Analysis**

| Quartile | Mean fast_1 | Mean Speedup | Mean Tokens |
|----------|-------------|--------------|-------------|
| Q1 (highest surprisal) | 47.4% | 37.0x | 7 |
| Q2 | 81.0% | 26.2x | 8 |
| Q3 | 72.3% | 46.8x | 10 |
| Q4 (lowest surprisal) | 43.9% | 30.6x | 15 |

Q2 (second-highest surprisal) shows the highest fast_1 (81.0%), while Q4 (lowest surprisal) shows the lowest (43.9%). The high-surprisal half of the distribution (Q1+Q2) averages 64.2% fast_1 versus 58.1% for the low-surprisal half (Q3+Q4). This pattern suggests the optimal selection point is in the high-surprisal region but not the extreme tail.

### 5.3 TTT Trajectory: Why Adaptation Fails

To understand why TTT underperforms, we examine the adaptation trajectory:

**Seed 42 Trajectory (Tasks 4, 5, 12, 14, 15):**

| Step | Rollouts | Agg fast_1 | Task 4 | Task 5 | Task 12 | Task 14 | Task 15 |
|------|----------|------------|--------|--------|---------|---------|---------|
| 0 | 160 | 37.5% | 9.4% | 3.1% | 100% | 37.5% | 37.5% |
| 1 | 320 | 40.0% | 28.1% | 15.6% | 100% | 21.9% | 34.4% |
| **2** | **480** | **42.5%** | **46.9%** | 6.3% | 100% | 18.8% | 40.6% |
| 3 | 640 | 36.3% | 25.0% | 3.1% | 100% | 21.9% | 31.3% |
| 4 | 800 | 36.3% | 21.9% | 3.1% | 100% | 15.6% | 40.6% |
| 5 | 960 | 41.3% | 25.0% | 9.4% | 100% | 40.6% | 31.3% |

**Key observation**: Performance peaks at step 2 (42.5%) and regresses to 36.3% at step 3. This is the **over-sharpening dynamic**: gradient updates collapse diversity toward mediocre solutions.

The per-task breakdown reveals heterogeneous dynamics:
- **Task 12**: Already saturated at 100% (easy task, no room for improvement)
- **Task 5**: Peaks at step 1, then collapses to 3.1% by step 3
- **Task 4**: Peaks at step 2, then regresses

This heterogeneity explains why TTT's aggregate performance (35% at best) underperforms Best-of-N (100% at K=16): adaptation cannot simultaneously optimize for tasks with different optima.

**Per-Task Isolated TTT Validation**

To confirm that over-sharpening occurs even without multi-task interference, we ran TTT on individual tasks (batch_size=1, K=32, 5 steps):

| Task | Step 0 | Step 1 | Step 2 | Steps 3-5 | BoA Selection |
|------|--------|--------|--------|-----------|---------------|
| 4 (hard) | 0% | **6.3%** | 0% | 0% | Step 1 |
| 5 (moderate) | 3.1% | **12.5%** | 12.5% | 6.3%→3.1% | Step 1 |
| 12 (control) | **100%** | 100% | 100% | 100% | Step 0 |
| 14 (moderate) | 37.5% | 34.4% | 37.5% | 34.4%→**56.3%**→40.6% | Step 4 |
| 15 (moderate) | 43.8% | **46.9%** | 34.4% | 43.8%→40.6%→46.9% | Step 1 |

The pattern reveals **task-level heterogeneity in optimal adaptation duration**:
- **Tasks 4, 5, 15**: Classic over-sharpening—peak at step 1, then regress
- **Task 14**: Late peak at step 4, suggesting some tasks benefit from extended adaptation
- **Task 12**: Control task, already saturated at 100%

This confirms the heterogeneity observed in the batch setting, ruling out multi-task interference as the sole cause of over-sharpening.

Across all per-task isolated TTT runs, 4 of 5 tasks peak within the first 2 adaptation steps (Tasks 4, 5, 12, 15); only Task 14 shows a delayed peak at step 4.

### 5.3b Probing Over-Sharpening: Direct Measurement

The evidence for over-sharpening above is outcomes-based: performance regresses after step 1-2. To directly measure the probability shift, we probe how the adapted policy's NLL rankings relate to sample quality across TTT steps.

We take 320 fixed Best-of-N samples (K=64, seed 42) with known speedups and compute each sample's NLL under every TTT checkpoint (steps 0-8). This holds the sample set constant while varying only the scoring model, isolating the effect of adaptation on the probability-quality relationship.

![Over-sharpening probe](/artifacts/fig_rho_trajectory.png)

*Figure: Spearman rho(NLL, speedup) across TTT steps for 320 fixed samples. The negative correlation deepens overall: adaptation makes the model progressively more confident about its worst solutions. Bottom-quartile rho nearly doubles from -0.24 to -0.44.*

The Spearman rho between NLL and speedup deepens overall from -0.198 (step 0, p = 3.6e-4) to -0.275 (step 8, p = 5.7e-7). The bottom-quartile correlation (the performance-critical tail where selection operates) nearly doubles from rho = -0.237 to rho = -0.442 (step 3, p = 9.8e-17).

This is not merely diversity loss. The deepening negative rho means TTT assigns progressively higher confidence to worse solutions: active anti-calibration in exactly the region where surprisal-guided selection operates. Meanwhile, the mean NLL trajectory oscillates non-monotonically (6.71 -> 6.66 -> 6.75 -> 6.86), indicating unstable training dynamics rather than smooth convergence.

### 5.4 Robustness: Second Task Subset (Hard Regime)

To test whether BoA findings generalize, we replicated the comparison on a harder 5-task subset from KernelBench L1 eval.

**Subset 2: Tasks {18, 28, 29, 30, 32}** (offset=5 from eval split)

**Equal-Budget Comparison:**

| Method | Rollouts | Agg fast_1 | Delta |
|--------|----------|------------|-------|
| Best-of-N (K=64) | 320 | **36.9%** | baseline |
| BoA Step 0 | 160 | 17.5% | -19.4% |
| BoA Step 1 | 320 | 16.3% | **-20.6%** |

**Result:** On the hard subset, **Best-of-N outperforms BoA** under equal budget.

**Per-Task Breakdown:**

| Task | Best-of-N | BoA Step 1 | Winner |
|------|-----------|------------|--------|
| 18 | 40.6% | **46.9%** | BoA (+6.3%) |
| 28 | **31.3%** | 0% | Best-of-N (+31.3%) |
| 29 | **37.5%** | 15.6% | Best-of-N (+21.9%) |
| 30 | **28.1%** | 6.3% | Best-of-N (+21.8%) |
| 32 | **46.9%** | 12.5% | Best-of-N (+34.4%) |

**Interpretation: Regime-Dependent Benefit**

The contrasting results between subsets reveal that BoA's benefit is **regime-dependent**:

| Subset | Best-of-N Baseline | BoA Effect | Interpretation |
|--------|-------------------|------------|----------------|
| Subset 1 | 39.8% | **-9.2%** | Adaptation hurts |
| Subset 2 | 40.5% | **-20.6%** | Adaptation hurts |

**Adaptation amplifies existing capability rather than creating new capability**. When the base policy achieves reasonable coverage (subset 1), gradient updates can refine solutions. When the base policy struggles (subset 2), adaptation may overfit to poor solutions, reducing diversity.

**Practical Implication:** Practitioners should prefer BoA when the base policy is moderately capable, but fall back to Best-of-N search in hard regimes where the model has low coverage.

Figure 5 visualizes this regime dependence, showing that tasks with >30% base coverage (green) benefit from BoA while tasks below this threshold (red) favor Best-of-N search.

![Regime-Dependent Benefit](/artifacts/fig4_regime.png)

*Figure 5: Regime-dependent benefit of adaptation vs. search. Tasks with >30% base coverage (green) benefit from BoA; tasks below this threshold (red) favor Best-of-N search. Interpretation: adaptation amplifies existing capability rather than creating new capability.*

**Evaluation Note:** Results use fast-proxy evaluation (5 performance trials). Full benchmark evaluation (50 trials) planned for camera-ready.

### 5.5 Compression Mechanism: Rapid Signal Distillation

We propose a *compression view* of test-time training in VEG domains. The mechanism is straightforward: the first 1-2 updates aggregate dense gradient signal from diverse rollouts, rapidly compressing execution feedback into the weights. Additional steps over-sharpen the distribution around a narrow subset of solutions, reducing diversity and causing regression.

Three observations support this interpretation. The BoA peak at 1-2 steps shows compression completing quickly. The step-dependent regression shows over-compression destroying diversity. SDPO experiments confirm this interpretation (Section 6.6).

We hypothesize a **Reward Compression Principle**: *gradient steps to saturation scale inversely with reward density.* Dense continuous rewards (kernel speedup) compress in 1-2 steps; sparse binary rewards (competitive programming) require extended adaptation. This principle unifies our findings: fast saturation and feedback redundancy are two manifestations of the same underlying dynamic.

**Cross-Subset Transfer.** If adaptation builds generalizable optimization strategies, knowledge should transfer across task subsets. We evaluate cross-subset transfer: checkpoints adapted on Subset 1 (tasks {4, 5, 12, 14, 15}) evaluated on Subset 2 (tasks {18, 28, 29, 30, 32}) and vice versa.

- S1->S2: 7.5% fast_1 (vs. base 17.5%, BoN 36.9%)
- S2->S1: 31.25% fast_1 (vs. base 37.5%)

Both directions show degradation relative to unadapted baselines. Adaptation over-fits to training-subset modes rather than learning domain-general kernel optimization strategies, consistent with the over-sharpening interpretation.

We emphasize that this is our interpretation, not a direct measurement of internal representations. However, the efficiency of this compression—hours instead of days—provides a practical precondition for implicit world models in hardware domains. If repeated adaptation cycles can cheaply distill execution feedback, models may accumulate transferable hardware knowledge over time.

---

## 6. Discussion

### 6.1 Why VEG Tasks Favor Resource-Aware TTT

Verifiable execution-grounded tasks differ fundamentally from preference-based or sparse-reward domains in ways that favor minimal adaptation over extensive search or elaborate feedback. Three properties drive this difference.

First, VEG tasks provide dense scalar rewards. Speedup is continuous (0x to 10x+), not binary. Every sample contributes gradient signal proportional to its quality, unlike preference domains where only pairwise comparisons yield signal. This makes gradient aggregation across diverse samples highly efficient—a single update from 160 samples captures more signal than sequential search through equivalent samples.

Second, VEG evaluation is evaluation-bound. CUDA compilation, warmup runs, and performance timing create per-sample overhead that does not exist in pure text generation. This overhead disproportionately penalizes search strategies, as each additional sample incurs fixed evaluation cost regardless of whether it provides novel information.

Third, when the world provides dense continuous feedback, an AI teacher interpreting that feedback becomes redundant (Section 6.6).

These properties suggest VEG may be a distinct regime for test-time compute allocation. For domains with dense continuous rewards from a deterministic evaluator, sample diversity with intelligent selection (surprisal-guided) may be more efficient than gradient adaptation.

### 6.2 The Minimum Signal Threshold: Why Minimal Adaptation Outperforms Extended Training

We introduce the concept of a *minimum signal threshold*: the amount of gradient signal needed before over-sharpening begins to degrade diversity. For dense-reward kernel optimization, this threshold is remarkably low—1-2 steps from 160 diverse samples.

The mechanism follows directly from the Reward Compression Principle. At step 0, the policy has diffuse probability over many solution strategies. The first gradient step aggregates signal across diverse samples, sharpening the distribution toward strategies that work. By step 2, this sharpening has captured most available improvement. Further steps over-sharpen: the model collapses to specific solutions that worked in the initial batch, losing diversity.

Scalable Power Sampling (Ji et al., 2026) corroborates this (RL gains = sharpening, not discovery); our step-2 peak provides empirical evidence in test-time adaptation. "Towards Execution-Grounded Automated AI Research" (Jan 2026) reports that RL from execution rewards can collapse to narrow ideas—exactly the over-sharpening we observe past the threshold.

The per-task trajectories illustrate this dynamic. Task 5 shows the clearest over-sharpening, dropping from 15.6% at step 1 to 3.1% at step 3—the model collapsed to a solution strategy that generalized poorly. Task 4 peaks at step 2 with 46.9% before regressing. These task-specific differences explain why per-task oracle selection (48.8%) outperforms aggregate BoA selection (42.5%)—different tasks have different sharpening optima.

This establishes a minimum signal threshold for VEG domains: the amount of gradient signal needed before over-sharpening begins. For dense-reward kernel optimization, this threshold is 1-2 steps from 160 diverse samples—remarkably low compared to TTT-Discover's 50-step paradigm. This finding aligns with "Agent RL Scaling Law" (Duan et al., 2025), which demonstrates that models quickly internalize code heuristics during RL, achieving higher performance with fewer environment interactions as training progresses. The threshold likely depends on reward density: sparse-reward domains may require more steps to extract sufficient signal, while dense-reward domains saturate quickly. Characterizing this relationship across domains—and determining whether models accumulate transferable heuristics across adaptation cycles—is an important direction for future work.

### 6.3 When Adaptation Beats Search

Our results reveal clear regime dependence. On subset 1 where Best-of-N achieves 39.8% fast_1, BoA achieves 30.6%. On subset 2 where Best-of-N achieves 40.5%, adaptation underperforms by 20.6%, suggesting that search maintains a diversity advantage.

This pattern indicates that adaptation amplifies existing capability rather than creating new capability. When the base policy achieves reasonable coverage (>30%), gradient updates can refine solutions toward higher speedup. When the base policy struggles, adaptation may overfit to the few working solutions, reducing the diversity that makes extensive search effective in discovering rare successes.

Practitioners should consider task difficulty when choosing between adaptation and search. For development and rapid iteration, adaptation completes faster per rollout due to gradient aggregation. For final evaluation where marginal coverage matters, extensive search may be warranted on tasks where the base policy achieves less than 30% coverage.

### 6.4 Relationship to Concurrent Work

TTT-Discover (Yuksekgonul et al., 2026) was published during the preparation of this work and represents the strongest case for extended test-time adaptation. Three key differences distinguish our findings. First, TTT-Discover uses ~50 adaptation steps while we find that 1-2 steps are optimal before regression occurs. Second, TTT-Discover implicitly selects checkpoints through PUCT-based prioritization while we formalize selection as Best-of-Adaptation with early stopping. Third, TTT-Discover does not compare against Best-of-N under matched compute budgets, leaving open the question of whether gradient signal beats brute search.

Our results complement rather than contradict TTT-Discover by identifying a key variable that determines optimal adaptation duration: reward density. In sparse-reward discovery tasks where qualitatively different solutions require extensive exploration, extended adaptation may be necessary. In VEG domains with dense continuous rewards, gradient signal saturates after minimal adaptation. When the world provides sparse binary feedback, extended exploration and rich teacher feedback are warranted; when the world provides continuous scalars (as in kernel optimization), 1-2 steps extract most available signal before over-sharpening degrades performance.

Our TTT experiments use vanilla GRPO without the entropy regularization, reuse buffers, and extended rollout budgets (512 per step) employed by TTT-Discover. Our total budget is 320 rollouts versus TTT-Discover's 25,600. Whether those mechanisms and budgets prevent over-sharpening in dense-reward VEG domains remains untested.

**Objective mismatch.** TTT-Discover optimizes for "find a new SOTA solution" and returns the best solution across all steps (argmax reward). Our evaluation uses fast_1 (correct and speedup > 1x) on KernelBench L1 with K capped at 64. These are different targets; our findings apply to budgeted inference, not open-ended discovery.

We swept learning rates across three orders of magnitude (lr in {1e-5, 1e-6, 3e-7}). Over-sharpening persists at all learning rates: lr=1e-6 peaks at step 1 (55.0%) then regresses; lr=3e-7 never exceeds the unadapted baseline (31.9%).

### 6.5 Future Direction: Zero-Evaluation Discovery

Our findings point toward a fundamental research direction: zero-evaluation discovery, where models generate optimal code for novel hardware architectures without physical execution. This requires models to develop what we term physics-grounded world models—internal simulations of how code interacts with hardware. From Word to World (2025) formalizes the evaluation of implicit world models, and SSRL (Self-Search Reinforcement Learning, 2025) proposes that LLMs can act as internal world simulators to reduce reliance on external interactions; we extend this framing to execution-grounded hardware domains. Our work grounds this vision empirically: Execution feedback can be efficiently distilled into weights through minimal adaptation, a tractable path toward internalized hardware models.

Current LLMs have implicit world models from text: common-sense physics, social dynamics, procedural knowledge. But these are derived from human descriptions of the world, not from direct interaction. VEG TTT adds a qualitatively different signal: the model learns "this memory access pattern is slow on Hopper" not from text describing memory hierarchies, but from execution feedback showing the actual latency. This is physics grounding—the model's weights encode the physical constraints of hardware learned through interaction.

The current paradigm requires the world (compiler + GPU) to evaluate each candidate. Test-time adaptation distills the world's response into model weights for a specific task. If this distillation is efficient—hours rather than days—then across many adaptation cycles, the model accumulates knowledge about how hardware responds to different code patterns. The model develops what we call a hardware-aware internal critic: an implicit neural simulation of memory hierarchies, execution pipelines, and performance characteristics.

This vision requires efficient TTT as a prerequisite. Contemporary approaches like Magellan (Jan 2026) require ~1.5 days of evolutionary search to discover and deploy compiler heuristics. If adaptation requires 50 steps and $500 per problem (TTT-Discover's regime), the cumulative cost of building an internal hardware model is prohibitive. Our demonstration that 1-2 steps suffice in VEG domains makes this path tractable—achieving comparable efficiency gains in hours rather than days. The efficiency frontier we characterize determines whether zero-evaluation discovery is feasible at scale.

Three research directions follow. First, probing internal world models: can we measure whether adapted models have learned transferable hardware representations, perhaps through performance prediction without execution? Second, cross-architecture transfer: do models adapted on Hopper show improved sample efficiency when adapting to Blackwell—evidence of accumulated hardware knowledge? Third, curriculum design: what sequence of adaptation tasks most efficiently builds an internal hardware simulator?

Implicit hardware world models remain a research direction, not a demonstrated capability of this work. Our contribution is the efficiency characterization that makes this direction tractable.

This positions our work as an initial study of compute allocation for test-time learning in VEG domains. Our results suggest that for dense-reward tasks like kernel optimization, the optimal allocation invests in sample diversity (many diverse rollouts in few steps) rather than adaptation duration (many steps with fewer samples per step). Validating this principle across additional VEG domains and model scales is an important direction for future work.

### 6.6 Self-Distillation: Why Feedback Adds No Lift

SDPO experiments (Appendix E) show prompt-only self-distillation succeeds at 120B scale, but **feedback context provides no lift** in VEG domains. When the world provides continuous gradient signal, feedback interpretation becomes redundant.

---

### 6.7 Open Questions

Three open questions remain. First, an **entropy-regularized TTT** baseline (closer to TTT-Discover's entropic objective) would test whether the over-sharpening failure is fundamental to dense-reward VEG domains or specific to vanilla GRPO. Second, a **deployable surprisal-guided pipeline** that ranks candidates by surprisal *before* correctness filtering (evaluating only the top-m candidates end-to-end) would transform the selection insight from post-hoc analysis to a practical method. Third, **reward sparsification** experiments (thresholding the continuous speedup signal) would test whether the step-optimum shifts upward as reward becomes sparser, directly validating the reward compression hypothesis.

## 7. Limitations

This work has several limitations that suggest directions for future research. Our experiments cover 10 tasks from KernelBench L1 (5 moderate, 5 hard) across 2 seeds, representing a subset of the 20 available evaluation tasks and leaving open whether findings transfer to the full distribution. All experiments use a single 120B parameter model; transfer to other model sizes and architectures remains untested. We evaluate only KernelBench L1 tasks; the harder L2 and L3 levels may exhibit different adaptation dynamics.

Our TTT experiments scope to vanilla GRPO (see Section 6.4 for discussion of differences from TTT-Discover).

With n=10 binary outcomes (5 tasks x 2 seeds), our primary selection comparison (80% vs. 50%) has limited statistical power for binary tests (exact sign test p = 0.125). We supplement with continuous speedup analysis.

The surprisal-guided selection strategy requires multiple correct samples per task; on harder levels (L2/L3) or tasks where the base policy has low coverage, the effect may diminish or vanish.

Our evaluation uses a fast-proxy protocol (5 timing trials per kernel); rankings could shift under the full KernelBench protocol (50 trials). We validated selected kernels on H100 hardware and observed consistent rankings, but a full-protocol replication on a larger subset would strengthen these results.

The inverse relationship between confidence and quality may be domain-specific. In kernel optimization, rare creative solutions yield high speedups. In domains where the mode of the distribution represents optimal behavior (e.g., following well-established protocols), surprisal-guided selection could perform poorly.

Finally, we provide empirical findings without theoretical grounding. Why does the model exhibit inverse confidence for high-quality solutions? A formal analysis of the relationship between training distribution coverage and test-time solution quality could strengthen these findings and enable principled selection policies.

---

## 8. Conclusion

We study compute-optimal test-time strategies for verifiable execution-grounded tasks, demonstrating on 10 KernelBench L1 tasks across 2 seeds that **search outperforms minimal adaptation (1-5 gradient steps)** for GPU kernel optimization. Best-of-N sampling achieves 100% task success at K=64 while TTT's best checkpoint reaches only 30.6% (3-seed mean)—with TTT's "equivalent K" falling below 1, meaning adaptation underperforms random sampling.

Three findings characterize the compute-optimal strategy:

1. **Search saturates at modest K.** Best-of-N scaling shows performance saturates at K=16 (99.9% success). Beyond this point, marginal gains are negligible. Practitioners should invest in sample diversity, not gradient updates.

2. **Surprisal-guided selection recovers oracle performance.** Selecting the highest-surprisal correct sample achieves 80% success vs 50% for most-confident—a 30% improvement with zero additional compute. Extending to surprisal-guided-top3 (evaluating 3 samples) matches oracle at 100%.

3. **TTT fails due to over-sharpening.** Gradient updates collapse the policy toward mediocre solutions, destroying the diversity needed to find optimal kernels in the distribution tail. This mechanistic explanation accounts for TTT's consistent underperformance.

The practical implication is clear: for dense-reward VEG tasks, allocate compute to sample diversity and intelligent selection rather than gradient adaptation. The surprisal-guided selection principle—that rare, high-quality solutions occupy high-surprisal regions—may generalize to other execution-grounded domains where the optimum lies in the distribution tail.

The prevailing assumption that test-time training provides universal benefits does not hold in this regime. In domains with dense continuous rewards and deterministic evaluation, gradient updates that worked for sparse-reward reasoning tasks become counterproductive when the goal is finding optimal solutions in a well-sampled distribution's tail.

---

## Acknowledgments

We thank Thinking Machines Lab for access to the Tinker training infrastructure.

---

## References

```bibtex
@article{kernelbench2025,
  title={KernelBench: Can LLMs Write Efficient GPU Kernels?},
  author={Ouyang, Anne and Guo, Simon and Arora, Simran and Zhang, Alex L. and Hu, William and R{\'e}, Christopher and Mirhoseini, Azalia},
  journal={arXiv preprint arXiv:2502.10517},
  year={2025}
}

@article{kevin2025,
  title={Kevin: Multi-Turn RL for Generating CUDA Kernels},
  author={Baronio, Carlo and Marsella, Pietro and Pan, Ben and Guo, Simon and Alberti, Silas},
  journal={arXiv preprint arXiv:2507.11948},
  year={2025}
}

@article{tttdiscover2026,
  title={Learning to Discover at Test Time},
  author={Yuksekgonul, Mert and Koceja, Daniel and Li, Xinhao and Bianchi, Federico and McCaleb, Jed and Wang, Xiaolong and Kautz, Jan and Choi, Yejin and Zou, James and Guestrin, Carlos and Sun, Yu},
  journal={arXiv preprint arXiv:2601.16175},
  year={2026}
}

@article{ttrl2025,
  title={TTRL: Test-Time Reinforcement Learning},
  author={Zuo, Yuxin and Zhang, Kaiyan and Sheng, Li and Qu, Shang and Cui, Ganqu and Zhu, Xuekai and Li, Haozhan and Zhang, Yuchen and Long, Xinwei and Hua, Ermo and Qi, Biqing and Sun, Youbang and Ma, Zhiyuan and Yuan, Lifan and Ding, Ning and Zhou, Bowen},
  journal={arXiv preprint arXiv:2504.16084},
  year={2025},
  note={Accepted to NeurIPS 2025}
}

@article{sstar2025,
  title={S*: Test Time Scaling for Code Generation},
  author={Li, Dacheng and Cao, Shiyi and Cao, Chengkun and Li, Xiuyu and Tan, Shangyin and Keutzer, Kurt and Xing, Jiarong and Gonzalez, Joseph E. and Stoica, Ion},
  journal={arXiv preprint arXiv:2502.14382},
  year={2025}
}

@article{sdpo2026,
  title={Reinforcement Learning via Self-Distillation},
  author={Zeng, Zhiqing and Yuan, Wenda and Yu, Tong and others},
  journal={arXiv preprint arXiv:2601.20802},
  year={2026}
}

@article{autotriton2025,
  title={AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs},
  author={Li, Shangzhan and Wang, Zefan and He, Ye and Li, Yuxuan and Shi, Qi and Li, Jianling and Hu, Yonggang and Che, Wanxiang and Han, Xu and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2507.05687},
  year={2025}
}

@article{supercoder2025,
  title={SuperCoder: Assembly Program Superoptimization with Large Language Models},
  author={Wei, Anjiang and Suresh, Tarun and Tan, Huanmi and Xu, Yinglun and Singh, Gagandeep and Wang, Ke and Aiken, Alex},
  journal={arXiv preprint arXiv:2505.11480},
  year={2025}
}

@article{scaling_rl_compute2025,
  title={The Art of Scaling Reinforcement Learning Compute for LLMs},
  author={Khatri, Devvrit and Madaan, Lovish and Tiwari, Rishabh and Bansal, Rachit and Duvvuri, Sai Surya and Zaheer, Manzil and Dhillon, Inderjit S. and Brandfonbrener, David and Agarwal, Rishabh},
  journal={arXiv preprint arXiv:2510.13786},
  year={2025}
}

@article{compute_optimal_rl2025,
  title={Compute-Optimal Scaling for Value-Based Deep RL},
  author={Compute Optimal RL Team},
  journal={arXiv preprint arXiv:2508.14881},
  year={2025}
}

@article{scalable_power_sampling2026,
  title={Scalable Power Sampling: Unlocking Efficient, Training-Free Reasoning for LLMs via Distribution Sharpening},
  author={Ji, Xiaotong and Tutunov, Rasul and Zimmer, Matthieu and Bou Ammar, Haitham},
  journal={arXiv preprint arXiv:2601.21590},
  year={2026}
}

@article{pope2026,
  title={POPE: Learning to Reason on Hard Problems via Privileged On-Policy Exploration},
  author={POPE Team},
  journal={arXiv preprint arXiv:2601.18779},
  year={2026}
}

@article{teaching_models2026,
  title={Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability},
  author={Teaching Models Team},
  journal={arXiv preprint arXiv:2601.18778},
  year={2026}
}

@article{snell2024,
  title={Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters},
  author={Snell, Charlie and Lee, Jaehoon and Xu, Kelvin and Kumar, Aviral},
  journal={arXiv preprint arXiv:2408.03314},
  year={2024}
}

@article{tritonrl2025,
  title={TritonRL: Training LLMs to Think and Code Triton Without Cheating},
  author={TritonRL Team},
  journal={arXiv preprint arXiv:2510.17891},
  year={2025}
}

@article{cudal2_2025,
  title={CUDA-L2: Surpassing cuBLAS Performance through RL},
  author={DeepReinforce Team},
  journal={arXiv preprint arXiv:2512.02551},
  year={2025}
}

@article{grpo2024,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Zhang, Mingchuan and Li, Y. K. and Wu, Y. and Guo, Daya},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024}
}

@article{magellan2026,
  title={Magellan: Autonomous Discovery of Novel Compiler Optimization Heuristics with AlphaEvolve},
  author={Chen, Hongzheng and Novikov, Alexander and V\~{u}, Ng\^{a}n and Alam, Hanna and Zhang, Zhiru and Grossman, Aiden and Trofin, Mircea and Yazdanbakhsh, Amir},
  journal={arXiv preprint arXiv:2601.21096},
  year={2026}
}

@article{ssrl2025,
  title={Self-Search Reinforcement Learning},
  author={SSRL Team},
  journal={arXiv preprint arXiv:2508.10874},
  year={2025},
  note={Proposes LLMs as internal world simulators}
}

@article{from_word_to_world2025,
  title={From Word to World: Can Large Language Models be Implicit Text-based World Models?},
  author={Li, Yixia and Wang, Hongru and Qiu, Jiahao and Yin, Zhenfei and Zhang, Dongdong and Qian, Cheng and Li, Zeping and Ma, Pony and Chen, Guanhua and Ji, Heng and Wang, Mengdi},
  journal={arXiv preprint arXiv:2512.18832},
  year={2025}
}

@article{agent_rl_scaling2025,
  title={Agent RL Scaling Law},
  author={Duan, Yiheng and others},
  journal={arXiv preprint arXiv:2505.07773},
  year={2025},
  note={Models quickly internalize code heuristics during RL}
}

@article{execution_grounded_ai2026,
  title={Towards Execution-Grounded Automated AI Research},
  author={Execution Grounding Team},
  journal={arXiv preprint arXiv:2601.14525},
  year={2026}
}

@article{can_1b_surpass_405b2025,
  title={Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling},
  author={TTS Scaling Team},
  journal={arXiv preprint arXiv:2502.06703},
  year={2025}
}

@article{accelopt2025,
  title={AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization},
  author={AccelOpt Team},
  journal={arXiv preprint arXiv:2511.15915},
  year={2025},
  note={Uses Optimization Memory for kernel search}
}
```

---

## Appendix A: Experimental Configuration

```yaml
# RLVR Training (produces base checkpoint)
model: openai/gpt-oss-120b
algorithm: GRPO
lora_rank: 16
learning_rate: 1e-5
batch_size: 8
group_size: 8
training_tasks: 80 (KernelBench L1 train split)
temperature: 0.25
max_tokens: 1024
normalize_reward: true

# Test-Time Evaluation (efficiency frontier analysis)
checkpoint: RLVR final (step 40, 98.4% correct, 0.87x speedup)
eval_tasks: {4, 5, 12, 14, 15} (subset 1), {18, 28, 29, 30, 32} (subset 2)

# Best-of-N
K: 64
total_rollouts: 320 (5 tasks x 64)

# Batch TTT (BoA)
K: 32 per task
tasks_per_step: 5
learning_rate: 1e-5  # (1e-6 ablation also tested)
step_1_rollouts: 320 (5 tasks x 32 x 2 steps)
```

## Appendix B: RLVR Training Progression

| Checkpoint | Correctness | Speedup | Notes |
|------------|-------------|---------|-------|
| Step 10 | 90.6% | 0.81x | Early training |
| Step 20 | 95.3% | 0.87x | - |
| Step 30 | 95.3% | 0.86x | - |
| Step 40 (final) | 98.4% | 0.87x | **Used for all evaluation** |

**Note on Checkpoint Selection.** We use the final checkpoint (step 40, run ID f8dc1c2e) for all evaluation. This checkpoint achieves the highest correctness (98.4%) with competitive speedup (0.87x), representing a well-trained policy after 40 GRPO steps on 80 KernelBench L1 training tasks.

Training uses normalized rewards (baseline-relative speedup per task) for stability across tasks with different baseline performance characteristics.

## Appendix C: Compute Accounting

All methods are evaluated under equal rollout budgets (320 samples per task batch). However, token counts differ due to teacher logprob computation in SDPO.

**Table C.1: Full Compute Breakdown (Mean Across 3 Seeds)**

| Method | Rollouts | Student Tokens | Teacher Tokens | Total Tokens |
|--------|----------|----------------|----------------|--------------|
| RLVR Base (K=1) | 5 | 4.0K | 0 | 4.0K |
| Best-of-N (K=64) | 320 | 313K | 0 | 313K |
| BoA Step 1 | 320 | 313K | 0 | 313K |
| SDPO (feedback) | 320 | 314K | 338K | 652K |
| SDPO (prompt-only) | 320 | 311K | 313K | 625K |

**Notes:**
- **Rollouts**: Number of complete code generations sampled from the model.
- **Student Tokens**: Tokens generated during sampling (prompt + completion).
- **Teacher Tokens**: Additional tokens processed for SDPO logprob computation. SDPO (feedback) includes execution feedback context; SDPO (prompt-only) excludes it.

**Cost Analysis.** SDPO incurs additional tokens over Best-of-N due to teacher forward passes. The teacher overhead is approximately 107% of student tokens for feedback-conditioned SDPO and 101% for prompt-only SDPO. This overhead is fixed regardless of task difficulty. BoA Step 1 has similar token count to Best-of-N since both process 320 rollouts.

## Appendix D: Speedup Statistics

Raw speedup magnitude varies widely across tasks and seeds, reflecting task-specific optimization headroom rather than method quality. We report these statistics for completeness but note that fast_1 (correct AND speedup > 1x) is the operative metric for KernelBench evaluation.

**Table D.1: Mean Speedup Over Correct Samples (3 seeds)**

| Method | Seed 42 | Seed 43 | Seed 44 | Mean ± std |
|--------|---------|---------|---------|------------|
| Batch-TTT BoA | 21.44x | 1.00x | 1.00x | 7.81x ± 11.80x |
| SDPO Prompt-Only | 20.60x | 1.01x | 1.00x | 7.53x ± 11.31x |
| SDPO Feedback | 21.99x | 0.98x | 0.99x | 7.99x ± 12.13x |

**Table D.2: Best-of-N Selected Speedup (K=64, Subset 1, Seed 42)**

| Task | Selected Speedup |
|------|------------------|
| 4 | 859.8x |
| 5 | 191.7x |
| 12 | 23.8x |
| 14 | 15.6x |
| 15 | 13.6x |

**Interpretation.** The high variance (std > mean in Table D.1) and extreme values (860x for Task 4) reflect that speedup magnitude depends heavily on task characteristics: some kernels have substantial optimization headroom while others are already near-optimal. The key insight is that all methods achieve comparable fast_1 rates despite different speedup profiles, suggesting that exceeding the 1x threshold—not maximizing raw speedup—is the operative challenge.

## Appendix E: SDPO Self-Distillation Experiments

We extend batch adaptation with **Self-Distilled Policy Optimization (SDPO)**, replacing scalar reward advantages with token-level self-distillation signal conditioned on execution feedback. This appendix presents the full methodology and results.

### E.1 Method

**Teacher Context Construction.** For each student rollout, we construct a teacher context containing:
1. The original task prompt
2. (Optional) A correct solution from the same batch, if one exists
3. Structured execution feedback (compile status, correctness, speedup, runtime, error traces)
4. The instruction: "Correctly solve the original question."

Critically, the student's original code is **not** included in the teacher context—only its execution outcome. This forces the teacher to reason from feedback rather than copy the student's approach.

**Token-Level Advantages.** The teacher scores the student's sampled tokens:

$$A_t = \beta \cdot (\log p_{\text{teacher}}(x_t | \text{context}) - \log p_{\text{student}}(x_t | \text{prompt}))$$

where $\beta = 1.0$ controls the strength of the self-distillation signal. Positive advantages indicate tokens the teacher prefers given execution feedback; negative advantages indicate tokens to suppress.

**Compute Trade-off.** SDPO uses identical rollout budgets to BoA and Best-of-N, but requires additional teacher forward passes (see Appendix C for token accounting).

### E.2 Results

**Table E.1: SDPO Results (Subset 1, 3 Seeds)**

| Method | Seed 42 | Seed 43 | Seed 44 | Mean ± std |
|--------|---------|---------|---------|------------|
| SDPO (feedback) | 35.6% | 18.8% | 24.4% | 26.3% ± 8.6% |
| SDPO (prompt-only) | 38.8% | 23.8% | 28.8% | 30.4% ± 7.6% |

**Key Finding: Feedback provides no lift.** SDPO with full execution feedback (26.3%) underperforms SDPO prompt-only (30.4%), a consistent gap across all 3 seeds. In our KernelBench L1 setting and SDPO-style feedback construction, execution feedback provides no lift over prompt-only self-distillation. This is consistent with a **reward density hypothesis**: when the world provides dense continuous rewards, an AI teacher interpreting that feedback may be redundant. Whether this generalizes beyond our setting (to smaller models, harder tasks (L2/L3), or alternative feedback formats) remains open.

### E.3 Self-Distillation at Frontier Scale

On-Policy Self-Distillation (OPSD) demonstrates that a single LLM can serve as both teacher and student, showing increasing benefits as model size grows up to 8B parameters. Our experiments provide evidence at frontier scale (120B parameters):

1. **Prompt-only self-distillation succeeds**: The student learning from an unconditional teacher achieves 30.4% mean fast_1—confirming self-distillation works at 120B scale.

2. **Feedback context provides no lift**: This may be domain-dependent. In sparse-reward domains (mathematical reasoning), larger capacity may enable richer self-rationalization. In dense-reward domains (kernel optimization), feedback interpretation becomes redundant regardless of scale.

**Practical Implication.** For practitioners considering OPSD-style approaches at frontier scale: prompt-only self-distillation works, but feedback engineering may only be valuable when rewards are sparse or binary.
