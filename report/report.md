# Does verification make a data-analysis agent more accurate?

**A controlled measurement on a small open model — and the two harness artifacts the measurement caught.**

*Project: `da-verify`. Model under test: MiniMax-M2.7 (a reasoning model), with MiniMax-M3 as a stronger verifier. Benchmark: InfiAgent-DABench (DAEval).*

---

## TL;DR

I built a data-analysis agent and a deliberately-trustworthy evaluation harness, then asked a single question: **how much does adding a verification step raise the agent's task accuracy?** I compared three conditions on 40 stratified tasks at temperature 0:

| Condition | Accuracy | Δ vs C0 | tasks fixed / broke | McNemar (exact) |
|---|---|---|---|---|
| **C0** — no verification | 82.5% | — | — | — |
| **C1** — self-verification (introspection) | 82.5% | +0.0% | 0 / 0 | p = 1.00 |
| **C2** — independent verifier, *same* model | 85.0% | +2.5% | 1 / 0 | p = 1.00 |
| **C2** — independent verifier, *stronger* model (M3) | 82.5% | +0.0% | 0 / 0 | p = 1.00 |

**Finding:** at this scale (n = 40, k = 1, temp 0), **no verification variant — self, same-model, or a stronger model — produced a statistically detectable accuracy gain.** A stronger verifier did not help.

The more useful output is *how* I got there. Two intermediate runs produced striking numbers — a temp-0.7 collapse to 17.5%, and a "stronger verifier makes it 10% **worse**" — and **both were harness artifacts that error analysis removed**, not facts about the model. Catching those is the actual point: the project is about *trustworthy measurement of agents*, and a measurement you can't trust to reject its own false positives isn't measurement.

---

## 1. Why measure verification at all?

"Generate-then-verify" is a popular recipe: let a model produce an answer, then have it (or another model) check the work. The intuition is that checking is easier than producing. But the intuition is rarely *measured* — most reports show a cherry-picked win, not a controlled estimate with uncertainty.

This matters most for **data analysis**, where a wrong answer is not obviously wrong: a model can compute *a* number through a plausible-looking but incorrect method (wrong column, sign error, leaked data, wrong outlier rule) and report it with full confidence. The error is silent. Verification is supposed to catch exactly this. Does it?

To answer that honestly you need two things this project is built around:
1. a **grader you can trust** (otherwise every downstream number is fiction), and
2. **paired statistics** (otherwise "85% > 82.5%" is noise dressed as a result).

---

## 2. Setup

**Benchmark.** [InfiAgent-DABench](https://arxiv.org/abs/2401.05507) (ICML 2024) turns open-ended data-analysis questions into *closed-form* answers (`@name[value]`) so they can be graded automatically. The public validation split used here is **257 questions over 68 CSV files**, each with a real dataset and a programmatically-checkable gold answer. (Data is CC BY-NC 4.0 and is *not* vendored — a fetch script pulls it.)

**The grader (W1).** Before trusting any score, I validated the grader against itself: render each gold answer in the model's output format, run it back through extraction + comparison, and demand it scores as correct. Result: **255/257 round-trip correct**; the 2 failures are the *benchmark's own* malformed gold (a stray bracket; a `nan` value that can never equal itself in IEEE-754), logged as disputes. On the cleaned set and the evaluation subset, the grader is **100%**. The grader is byte-compatible with the official scorer by default, with documented, tested divergences (balanced-bracket extraction, numeric tolerance derived from the gold's stated precision, ordered-tuple list handling, multi-part all-or-nothing).

> A concrete reminder of why this step exists: a later fix to the extractor moved the measured C0 accuracy from 77.5% to **82.5%** on the *same* model outputs — a buggy ruler had mis-stated the baseline by 5 points.

**Subset.** A stratified, verifiability-first sample of **40 tasks** (13 easy / 14 medium / 13 hard), biased toward robustly-checkable numeric/list answers and away from free-text categorical gold whose vocabulary is inconsistent across the benchmark (`'no'` / `'False'` / `'not normal'` all appear for the same kind of question — an auto-grading hazard documented separately).

**Agent (C0).** A hand-written ReAct loop (Thought → Action → Observation) over a stateful, sandboxed Jupyter kernel (per-cell timeout, read-only source data) with function-calling tools (`run_python`, schema introspection). No self-check — whatever it lands on first is the answer.

**Conditions.**
- **C1 (self-verification):** C0, then one round where the *same* agent re-checks its own code and answer and may revise.
- **C2 (independent verification):** C0, then a *fresh, skeptical* verifier agent — new context, its own clean sandbox — that recomputes the answer from scratch ("don't trust the candidate; re-derive it yourself, ideally a different way") and reconciles. The verifier can be the same model or a different (stronger) one.

---

## 3. Metrics & statistics (and why each)

- **pass@1 / pass@k / pass^k.** Accuracy (pass@1), capability (pass@k = ≥1 of k correct, via the unbiased Chen-2021 estimator), and *reliability* (pass^k = all k correct). Verification is expected to move reliability most, so the axis must be measured, not assumed.
- **Wilson score interval** for each accuracy — better than the normal approximation at small n / extreme p, which is our regime.
- **Exact McNemar test** for k = 1 paired binary outcomes. Conditions run on the same tasks, so the outcomes are *paired*; McNemar looks only at the discordant tasks (one condition right, the other wrong). A t-test would be wrong here (paired binary on the same items, not independent continuous samples).
- **Paired bootstrap** on the mean per-task rate difference for k > 1 (where the unit is a rate, not a single binary), resampling tasks to avoid pseudo-replication across correlated within-task samples.

Everything is pure-Python (no SciPy dependency) and unit-tested.

---

## 4. Results (temp 0, n = 40, k = 1)

The four conditions are in the TL;DR table. In words:

- **C1 (introspection) changed nothing** — zero tasks moved. A deterministic model told to "check again" restates its answer; there is no new signal. This is consistent with the literature that LLMs struggle to self-correct reasoning without external feedback.
- **C2 with the same model is directionally positive but not significant** — it fixed 1 task (a regression-accuracy question C0 got slightly wrong) and broke none, for +2.5%. With a single discordant pair, McNemar p = 1.00. And the one fix sits on a task whose gold is non-reproducible (the question fixes no random seed for its train/test split), so even that edge is shaky.
- **A stronger verifier (M3) did not help** — Δ = 0.

The defensible claim is narrow and true: **on a weak model, at temp 0, n = 40, verification does not produce a measurable accuracy gain — and verifier strength did not change that.**

---

## 5. The part that matters: two artifacts the analysis caught

A measurement is only as good as its ability to reject its own false positives. Two intermediate results looked like findings and were not.

### 5.1 The temp-0.7 "collapse" (a rate-limit artifact)

To create headroom for verification, I moved to temperature 0.7 with k = 5 samples and `--workers 3` for speed. The run reported **pass@1 = 17.5%**, with **82% of samples producing no answer at all** — a catastrophic collapse versus the temp-0 numbers.

It would have been easy to write "the model falls apart at temp 0.7." Instead, I ran *one* of the empty tasks **serially, in isolation** — and it produced a correct, well-formatted answer. The difference was concurrency: at 3 parallel requests the provider was **rate-limiting**, and the errored calls surfaced as empty answers. The 17.5% was an artifact of the harness, not the model. **Discarded.** Lesson recorded: do not parallelize temp>0 against a rate-limited API without backoff.

### 5.2 The "stronger verifier makes it 10% worse" (a reconciliation artifact)

Using MiniMax-M3 (stronger) as the verifier first reported **72.5% — a 10-point drop, with 4 tasks broken and none fixed.** "A stronger verifier hurts" is a surprising, tweetable claim.

Error analysis killed it. **All four broken tasks were multi-part questions, and in every one M3's computed values were correct — it had simply omitted one of the required `@name` fields** (e.g. it reported `mean` but dropped `std_dev`). The reconciliation logic then replaced M2.7's *complete, correct* answer with M3's *partial* one.

So the "−10%" was not about verifier strength at all; it was a reconciliation policy that trusted the verifier wholesale. The fix is principled: **adopt the verifier's answer only if it covers every required field**; otherwise keep the candidate. (This subsumes an earlier fix — never let a verifier that produced *no* parseable answer override a good one.) Re-running gave **82.5% (Δ 0)**.

The real conclusion the M3 experiment supports is the opposite of the tweet: **the bottleneck here is reconciliation policy and multi-part field completeness, not verifier strength.** Two reconciliation bugs were found and fixed by error analysis — which is exactly what this project exists to do.

---

## 6. Limitations & threats to validity

I would rather state these than have a reader find them.

- **Underpowered.** n = 40, k = 1, with 0–1 discordant pairs. The study cannot detect a small (a-few-percent) effect; "not significant" here means "no detectable effect," not "no effect."
- **temp 0 limits self-verification structurally** — a deterministic re-attempt reproduces the original answer. An early 8-task probe at temp 0.7 showed the model is *capable-but-unreliable* (pass@1 ≈ 45% vs pass@5 ≈ 75%), which is the regime where verification *should* matter most; measuring it there cleanly (serially) is unfinished.
- **Same provider.** Solver and verifier share a vendor and likely correlated blind spots; "stronger" (M3) is by reputation, not established on these tasks.
- **Benchmark ambiguities.** Some gold answers are non-reproducible (unfixed random seeds) or under-specified ("outliers" without a definition); these are logged, and the subset is biased toward robustly-checkable tasks.

---

## 7. What a powered, fair study would need

- **More tasks** (the full DAEval, not 40) and **k > 1 samples**, run **serially** (or with backoff) to avoid §5.1.
- **temp > 0**, to give verification correctable variance to work on.
- **Field-aware reconciliation** (merge per-field, or require completeness) so multi-part answers aren't an artifact source.
- **Verification grounded in a programmatic check** (re-derive via an independent method and require agreement, or assert invariants), rather than trusting another LLM's re-derivation — the §4/§5 results suggest same-family LLM verification has limited headroom.

---

## 8. Reproducibility

```bash
bash scripts/fetch_data.sh                 # pull DAEval (CC BY-NC, not vendored)
python3 -m pytest tests/ -q                # 82 tests
python3 scripts/make_subset.py             # the stratified 40-task subset
python3 scripts/gold_self_check.py         # grader self-check gate
python3 scripts/run_eval.py --condition c0 --n 40 --k 1
python3 scripts/run_eval.py --condition c2 --verifier-model MiniMax-M3 --n 40 --k 1
python3 scripts/compare_conditions.py --a <c0>.jsonl --b <c2>.jsonl --a-name C0 --b-name C2
```

LLM responses are content-addressed and cached, so re-runs are deterministic and free; a reviewer reproduces the headline numbers without spending on the API.

---

## 9. What this project actually demonstrates

Not "verification works" (it didn't, here) — but the thing the study set out to show regardless of the sign of the effect: **the ability to measure an agent's quality in a way you can trust, including trusting it to reject its own false positives.** A verified grader, the right paired statistics, an honest null, and two self-caught artifacts are, for an evaluation problem, the result.

---

*Benchmark: Hu et al., "InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks," ICML 2024 ([arXiv:2401.05507](https://arxiv.org/abs/2401.05507)), CC BY-NC 4.0. pass@k estimator: Chen et al. 2021.*
