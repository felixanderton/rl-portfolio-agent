# ML Development Process Retrospective

Lessons from building rl-portfolio-agent across 14 hypotheses. The focus is on process failures that caused wasted experiment cycles, and the practices that would have prevented them.

---

## The core problem: jumping to experiments before understanding the failure mode

The most expensive mistake in this project was treating hypothesis testing as the primary tool for *diagnosis* rather than *validation*. When a problem appeared (the train/val Sharpe gap), the response was to run experiments that might fix it — rather than first establishing what was actually causing it.

The train/val Sharpe gap (training Sharpe ~7, val Sharpe ~0.77) was visible from H4. The next five hypotheses (H5, H7, H12, H13, H14) were attempts to fix overfitting by throwing different regularisers at it: weight decay, block bootstrap, observation noise, concentration penalty, new features. All failed. The gap was only properly understood after the experiments: reward-space regularisers (H6's TC curriculum) work; weight-space and input-space regularisers disrupt the gradient dynamics that produce the late-training val Sharpe surge.

That understanding should have preceded the experiments, not followed them. The right sequence is:

1. Observe the failure mode
2. Form a mechanistic hypothesis about *why* it's happening
3. Check whether existing literature has already solved or explained it
4. Then design an experiment to test the proposed fix

Skipping steps 2 and 3 is how you run five experiments to answer a question that has a known answer.

---

## Lesson 1: Do a literature review before writing a line of code

This project started with implementation. The differential Sharpe reward, the PPO setup, the feature engineering — all designed from first principles, without first checking what the RL-for-portfolio literature had already found.

A day of reading before the first experiment would have surfaced:

- **Known failure modes of RL on financial data**: the train/val gap and policy collapse are well-documented. The field has settled on a short list of mitigations (TC penalties, entropy regularisation, data augmentation) with known tradeoffs. Knowing the failure modes in advance means you arrive at your first experiment with a much shorter list of things to try.

- **Feature engineering baselines**: cross-sectional momentum at 3–12 month horizons is the most replicated factor in sector ETF allocation. Starting the feature set without it (and only adding it as H14, the final hypothesis) meant every prior experiment was trained on a potentially incomplete signal. This should have been in the baseline, not an afterthought.

- **Reward function choices**: the differential Sharpe formulation has known sensitivity to EMA initialisation (the accumulator reset bug fixed in H4 was a known issue in the literature). Reading Moody & Saffell before implementation would likely have caught it.

**Practice**: before starting a new ML project, spend time on: (a) a domain survey (what has been tried on this specific problem type), (b) a methods survey (what techniques are known to work/fail for this reward/architecture combination), and (c) a failure modes survey (what are the most common ways this class of model breaks). Budget this as real project time, not optional background reading.

---

## Lesson 2: Define your diagnostic metrics before your first training run

The metrics that mattered most for understanding this project were: training Sharpe, val Sharpe, entropy loss, episode turnover, and the train/val Sharpe gap. Several of these were not tracked from the start.

Turnover (`episode_turnover`) was particularly important — it was the clearest behavioural signal of overfitting (a policy that rebalances aggressively every day is almost certainly memorising rather than generalising). But it wasn't a first-class metric from the beginning. When it was eventually tracked, it immediately clarified the problem.

Similarly, the train/val Sharpe gap as an explicit metric (rather than two numbers you compare manually) would have made it impossible to ignore from H1 onward. Instead, the gap was noted but not treated as the primary diagnostic signal until much later.

**Practice**: before the first training run, write down:

- What does a healthy run look like? (e.g. train and val Sharpe should track each other; entropy should decay gradually)
- What does each failure mode look like in the metrics? (e.g. a widening train/val gap = overfitting; entropy collapse before val Sharpe plateaus = premature exploitation)
- Which metrics will you check at a fixed interval during training to decide whether to stop early?

These should be in a `METRICS_GUIDE.md` before you run anything — not reverse-engineered from failed experiments.

---

## Lesson 3: Establish a clean baseline before testing improvements

The H4 bug fixes (action space bounds, EMA warm-up) should have been in the baseline. They were not enhancements — they were corrections to the environment that made it behave as designed. Running H1–H3 on a broken environment produced results that were not comparable to anything run after H4, and effectively wasted three experiments.

More broadly: before testing a hypothesis about *improving* performance, you need to be confident that your baseline implementation is correct. This means:

- Code review of the environment and reward function before the first training run
- A sanity check episode: does the agent achieve near-equal-weight allocations with random actions? Does the reward have the right sign and magnitude?
- A unit test that verifies the observation vector has the expected shape and value range

The cost of a broken baseline is that every experiment run on it produces results you cannot interpret or reuse.

---

## Lesson 4: Understand *why* a hypothesis will work before testing it

Most of the failed hypotheses (H5, H7, H12, H13) were reasonable ideas without a clear mechanistic story for why they would work given the specific training dynamics of this model.

For example, H7 (block bootstrap) had a plausible intuition — synthetic episodes prevent trajectory memorisation — but no account of how bootstrap resampling would interact with the differential Sharpe EMA, which accumulates a running signal over the episode. Resampling data on reset means the EMA accumulators are perpetually recalibrated to a different sequence, destroying the stable gradient signal. This interaction was predictable from first principles, but the hypothesis was tested without considering it.

A useful pre-experiment question: **what is the specific mechanism by which this change produces the expected effect, and is there any reason it would be disrupted by the existing training dynamics?** If you cannot answer this, the hypothesis is not ready to test.

This is not about demanding certainty before experimentation — it's about ensuring that each experiment is testing a specific causal claim rather than a vague hope.

---

## Lesson 5: Control for warm-starting in your experimental design

A significant confound in this project was the use of warm-starting (loading a prior checkpoint as the starting point for a new run). H2, H10, and some intermediate runs warm-started from previous checkpoints. This made it impossible to cleanly attribute performance differences to the hypothesis being tested.

H10 reached a peak val Sharpe of 0.813 — but this was built on 3M effective training steps across two warm-start chains (H1 → H6 → H10). Whether the val Sharpe improvement was due to additional training budget, the warm-start initialisation, or the H10-specific changes was never cleanly isolated.

**Practice**: for any hypothesis that changes the training protocol (rather than the architecture or features), run both a warm-start version and a from-scratch version. The from-scratch version is the honest baseline; the warm-start version tells you whether the improvement compounds. If you can only afford one run, start from scratch — the result is interpretable.

---

## Lesson 6: Set falsification criteria before running

Several experiments ran to completion despite showing clear negative signals early. H12 (observation noise) showed val Sharpe declining monotonically from step 300k — the run was terminated at 1.1M steps, 800k steps after the outcome was clear.

The better hypotheses in this project had explicit falsification criteria written in advance: "if val Sharpe is below X at step Y, the hypothesis is falsified — stop the run." This should be standard practice for every experiment. It:

- Saves compute on runs that are clearly failing
- Forces you to commit to a specific prediction before seeing the data
- Makes the experimental record honest (the falsification criterion was set before, not chosen after seeing the results)

The criterion should be based on where the prior best result stood at the same training step, not the final val Sharpe — because failure often appears mid-training before any recovery is possible.

---

## Summary: a better process

| Phase | What we did | What we should have done |
|---|---|---|
| Project start | Implemented from first principles | Literature review: known failure modes, baseline features, reward function choices |
| Baseline | First training run on unreviewed code | Sanity checks, unit tests, code review before any training |
| Metrics | Added metrics reactively as problems appeared | Define diagnostic metrics and health checks before first run |
| Hypothesis design | "Let's try X and see" | Written mechanistic hypothesis, falsification criterion, and interaction analysis before running |
| Failure analysis | Ran more experiments to diagnose a failure | Pause, diagnose with metrics and literature, then experiment |
| Experimental control | Mixed warm-start and from-scratch runs | Clean from-scratch baseline for every new protocol change |

The pattern across all of these is the same: **invest more time in understanding before investing compute in experimenting**. The experiments that worked (H4, H6, H10) were all grounded in a specific diagnosis. The experiments that failed were mostly attempts to fix a symptom without understanding the cause.
