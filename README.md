# da-verify

**A verification-in-the-loop data-analysis agent + a rigorous evaluation harness.**

> **Thesis.** On data-analysis tasks where answers can be checked by a program,
> *how much does adding verification raise an agent's success rate — and at what
> cost?* This repo measures that with a faithful, reproducible verifier and a
> controlled study, not a vibes demo.
>
> The product is the **measurement**, not "I built an agent." Lineage:
> upgrades the evaluation harness and the *harness > fine-tuning on 8B* finding
> from the earlier ArkNarrator work.

This README is **evidence-first**: every technique appears only where it earns
its place, with a one-line engineering reason. No keyword stuffing.

> 📄 **Full technical write-up: [`report/report.md`](report/report.md)** — the
> C0/C1/C2 result, the honest null, and the two harness artifacts error analysis
> caught.

---

## Status — W1 done: the verifier (the foundation)

Before any agent or any accuracy claim, the grader itself has to be trustworthy.
W1 delivers exactly that, and proves it.

| Piece | File |
|---|---|
| Load + join DAEval (questions ⨝ gold labels) | `src/da_verify/tasks/loader.py` |
| Type-aware verifier (numeric / categorical / list / multi-part) | `src/da_verify/tasks/verifier.py` |
| Stratified, verifiability-first 40-question subset | `src/da_verify/tasks/sampler.py` |
| Gold self-check gate + dispute log | `scripts/gold_self_check.py` |
| Tests (one per behaviour, the spec) | `tests/test_verifier.py` |

### Run it

```bash
bash scripts/fetch_data.sh           # pull DAEval into data/daeval/ (CC BY-NC, not vendored)
python3 -m pytest tests/ -q          # 23 passed
python3 scripts/make_subset.py       # -> data/subsets/headline_40.json
python3 scripts/gold_self_check.py   # -> PASS + data/disputes.md
```

> The benchmark data is **not** committed (it's CC BY-NC 4.0). `fetch_data.sh`
> clones it from the official InfiAgent repo. Our own artifacts — the subset id
> list and the dispute log — *are* committed.

### W1 acceptance — met

- **Gold self-check:** 255/257 round-trip; the 2 failures are *the benchmark's
  own malformed/degenerate gold* (machine-diagnosed in `data/disputes.md`).
  **Cleaned set = 100%, headline-40 = 100%.**
- **Subset:** 40 tasks balanced over level (13 easy / 14 medium / 13 hard) ×
  complexity (21 single / 19 multi); 0 noisy-categorical (verifiability-first).

---

## Status — W2 done: C0 baseline runs end-to-end

A hand-written ReAct agent (Thought→Action→Observation) drives a sandboxed Python
kernel via function-calling; answers are scored by the W1 verifier.

| Piece | File |
|---|---|
| Sandbox kernel (stateful, timeout-interrupt, read-only data) | `src/da_verify/sandbox/kernel.py` |
| Tools (`run_python` + data introspection) | `src/da_verify/agent/tools.py` |
| C0 ReAct loop (no self-check, by design) | `src/da_verify/agent/react.py` |
| LLM client + record-replay cache | `src/da_verify/llm/client.py` |
| Baseline runner | `scripts/run_baseline.py` |

### C0 result — model `MiniMax-M2.7`, first 10 headline tasks

- accuracy (all_correct): **60%** (6/10)
- format-miss rate: **10%** (1/10) — id=132 computed the right value (`20`) but
  emitted the wrong field name → a formatting failure masking a correct answer.
  This is why format-miss is tracked separately from accuracy.
- candidate rate: **100%** — no floor effect; the model is weak-but-functional,
  the right regime to study verification.
- API errors: 0/10.

The 4 misses are single-step slips, not floor failures: a sign-flip on
daily-return mean (id=75), an outlier-definition difference (id=62), a
non-deterministic train/test split with no fixed seed (id=7), and the wrong
field name (id=132). These are exactly the errors a verification step (C2) could
plausibly catch — i.e. real headroom for the study.

> **Comparability caveat.** Do not read this 60% against published
> leaderboards (e.g. InfiAgent reports GPT-4 ≈ 60% ABQ). Those numbers use a
> *different* harness, add a GPT-3.5 reformat step we deliberately omit (so we
> eat the format penalty they don't), a different model, and the full task set.
> External numbers are only a loose "not catastrophically broken" sanity check.
> **The only authoritative comparison is internal: our C0 vs C2** — same model,
> harness, tasks, and grader. The study is controlled precisely so it doesn't
> depend on cross-setup comparability.

### Sandbox isolation boundary

**Enforced:** per-cell wall-clock timeout (interrupt; kernel survives), separate
process, cwd-scoped workdir, source CSV copied in read-only (`0o444`),
best-effort memory cap (`RLIMIT_AS`; macOS may ignore). **Not enforced here**
(deferred to a container with `--network none` + cgroups): network isolation,
filesystem access beyond cwd. The module docstring states this — no false
sense of security.

### Benchmark ambiguities found (logged, not hidden)

- id=7: accuracy depends on an unfixed train/test split → gold not reproducible.
- id=62: "outliers" undefined → the without-outliers mean depends on method.

These are task-quality caveats alongside `data/disputes.md`.

### Run it

```bash
cp .env.example .env   # add a valid OpenAI-compatible key (MiniMax/DeepSeek/…)
python3 scripts/run_baseline.py --n 10   # cached after first run
```

---

## Status — W3: stable full-set eval + pass@k harness

W2's baseline was 10 tasks (noisy). W3 runs the full 40-task subset and adds a
capability-vs-reliability split via repeated sampling.

### C0 over the full 40-task subset (`MiniMax-M2.7`, temp 0)

- **pass@1: 82.5%** — by level: easy **92%** · medium **79%** · hard **77%**;
  format-ok **95%** · candidate **97.5%** (no floor).
- (The 10-task W2 number was 60% — small-sample noise. An earlier full-set run
  read **77.5%**; the post-review verifier fixes — nested-tag extraction,
  short-comma lists — re-scored ~2 previously-misjudged tasks, moving it to
  82.5% on the *same* cached agent outputs. A buggy ruler had mis-stated the
  baseline by 5 points; this is exactly why the verifier ships with a self-check.)

### Capability vs reliability (pass@k)

`src/da_verify/eval/metrics.py` reports three numbers from k repeated samples:
pass@1 (accuracy), **pass@k** (≥1 of k correct — capability; unbiased Chen-2021
estimator), **pass^k** (all k correct — reliability).

Slice (8 tasks × 5 samples, temp 0.7):

| metric | value | reading |
|---|---|---|
| pass@1 | 45% | one shot |
| pass@5 | 75% | capability (≥1 of 5) |
| pass^5 | 12.5% | reliability (all 5) — 1/8 tasks |

The **45%→75% gap** is the study's target: the model is *capable but
unreliable*. Verification's job is to convert "sometimes right" into "reliably
right," and `pass^5 = 12.5%` shows that axis is wide open.

Two findings logged: (1) **temperature hurts format** — format-ok dropped 90%→50%
and candidate 92.5%→50% from temp 0→0.7, so the headline C0/C2 study likely runs
at temp 0 (decision for W5). (2) **resampling ≠ verification** — id=62
(outlier-definition) and id=75 (sign-flip) were wrong all 5 times; these are
*systematic* errors resampling can't fix but a re-derive/re-read verification
step might. That distinguishes flaky headroom from systematic headroom.

---

## Status — W4: C1 self-verification (a clean null at temp 0)

C1 = C0 + one self-verification round (the same agent re-checks its own
code+answer and may revise once). Conditions share one loop (`_react_loop`);
only the verification policy differs, so the comparison is controlled. **MCP and
RAG are deliberately NOT added** — neither is earned with a single backend and
directly-provided data; they wait for the multi-backend layer.

### C0 vs C1 (`MiniMax-M2.7`, 40 tasks, temp 0)

| | accuracy | 95% CI (Wilson) |
|---|---|---|
| C0 (no verification) | 82.5% | 68.0–91.3% |
| C1 (self-verification) | 82.5% | 68.0–91.3% |

Δ = **0.0%**. C1 fixed 0, broke 0. **McNemar exact p = 1.00** (no discordant
pairs). C1 marginally improved *format* (format-ok 95%→97.5%, candidate
97.5%→100%) but changed **no** final-answer correctness.

**Reading (honest).** Pure self-verification gives this weak model **no accuracy
gain at temp 0** — expected: a deterministic model asked to "check again" mostly
restates its answer (no new signal). Consistent with the literature that LLMs
struggle to self-correct reasoning without external feedback. It's a clean null,
and it **motivates C2**: external/multi-step verification (re-derive by a
different method, run sanity checks) rather than introspection — the pass@k
slice showed the failures are *systematic* (sign-flip, outlier-definition),
which introspection can't see but a re-derive might. *Caveat:* temp 0
structurally limits self-verification (deterministic re-attempt); whether C1
helps at temp>0 is open for W5.

Stats: `eval/stats.py` (Wilson + exact McNemar, pure-math); compare two runs with
`scripts/compare_conditions.py`.

---

## Status — W5: C2 external verification, and the C0/C1/C2 picture

C2 = C0, then an INDEPENDENT verifier (a fresh skeptical agent in its own sandbox)
recomputes the answer from scratch and reconciles. Unlike C1's introspection, the
external re-execution is a genuine new signal.

### C0 / C1 / C2 (`MiniMax-M2.7`, 40 tasks, temp 0)

| condition | accuracy | 95% CI (Wilson) | vs C0 | fixed / broke | McNemar p |
|---|---|---|---|---|---|
| C0 no verification | 82.5% | 68.0–91.3% | — | — | — |
| C1 self-verification | 82.5% | 68.0–91.3% | +0.0% | 0 / 0 | 1.00 |
| C2 external verification | **85.0%** | 70.9–92.9% | **+2.5%** | 1 / 0 | 1.00 |

**Reading (honest — this is the result, not a disappointment).**
- **C1 (introspection) moved nothing.** A deterministic model asked to "check
  again" restates its answer — no new signal (consistent with the literature on
  LLM self-correction).
- **C2 (external re-derivation) is directionally better** and, unlike C1,
  actually moves answers: it fixed id=7, broke none.
- **But it is not statistically significant.** +2.5% = 1 task in 40, a single
  discordant pair, McNemar p=1.00. At n=40 / k=1 the study is **underpowered** to
  detect a small effect — and the one fix (id=7) sits on a task whose gold is
  non-reproducible (unfixed train/test split), so even that edge is shaky.

The defensible claim is narrow and true: *on a weak model, temp 0, n=40 —
self-verification gives no gain; external re-derivation is directionally positive
but underpowered.* This extends the ArkNarrator thesis: not all harness helps —
**same-model verification alone is insufficient; you need a stronger or
externally-grounded checker.**

**Bug found + fixed mid-experiment:** the first C2 run *broke* id=587 — the
verifier produced no parseable answer and the naive reconciliation (`vfinal or
final`) overrode a correct candidate with a non-answer. Fixed: adopt the
verifier's answer only if it parses; otherwise keep the candidate
(`tests/test_agent.py` pins it).

**Path to a powered result (W6+):** more tasks (full DAEval) + k>1 samples +
temp>0 (the pass@k slice showed capable-but-unreliable behavior there) + a
verifier stronger than or grounded beyond the solver (multi-model / programmatic
checks). The harness and stats are ready — this is now a sample-size + design lever.

---

## Status — W6: multi-model verification + two artifacts honest analysis caught

W6 tested the most promising lever — a STRONGER verifier (MiniMax-M3 checking a
MiniMax-M2.7 solution) — and pushed for power.

### C0 / C1 / C2 (40 tasks, temp 0, k=1)

| condition | accuracy | Δ vs C0 | fixed/broke | McNemar p |
|---|---|---|---|---|
| C0 no verification | 82.5% | — | — | — |
| C1 self-verification | 82.5% | +0.0% | 0/0 | 1.00 |
| C2 verifier = M2.7 (same) | 85.0% | +2.5% | 1/0 | 1.00 |
| C2 verifier = M3 (stronger) | 82.5% | +0.0% | 0/0 | 1.00 |

**Honest conclusion:** at n=40 / k=1 / temp 0, no verification variant — self,
same-model, or a *stronger* model — produced a statistically detectable gain.

**Two artifacts that error analysis caught (the real W6 story):**

1. **temp>0 parallelism collapse.** A temp-0.7, k=5 run with `--workers 3`
   reported pass@1 17.5% with 82% of samples empty. A serial standalone run of
   the same task produced a correct answer → the provider was **rate-limiting
   concurrent requests**. Discarded. Lesson: don't parallelize temp>0 against a
   rate-limited API without backoff.
2. **The M3 "−10%" that wasn't.** Naive reconciliation reported M3 verification
   at 72.5% (−10%, 4 broke). All 4 breaks were multi-part questions where M3 gave
   *correct values but omitted a required field*, and the override replaced
   M2.7's complete answer with M3's partial one. Fix: adopt the verifier's answer
   only if it covers **all** required fields → re-run gave 82.5% (Δ0). "Stronger
   verifier hurts" was a reconciliation artifact, not a fact.

**What this says:** the bottleneck is **reconciliation policy + multi-part field
completeness, not verifier strength.** Two reconciliation bugs (override with a
non-answer; override with a partial answer) were found and fixed by error
analysis — which is what this project is built to do. A real verification signal,
if one exists, needs larger n, k>1 diversity (run serially), and verification
grounded in a programmatic check rather than another LLM's re-derivation.

---

## Verifier design — what's earned, and why (defend each in interview)

- **Balanced-bracket extraction** over the official non-greedy `@(\w+)\[(.*?)\]`,
  because real list answers like `[1,2,3]` contain brackets the lazy regex
  truncates. (`test_balanced_scanner_handles_bracketed_list_value`)
- **Tolerance, not `==`, for numbers** — IEEE-754 + rounding make exact decimal
  equality unreliable. Default `abs_tol=1e-6, rel_tol=0` is **byte-compatible
  with the official benchmark** (leaderboard comparability); a `rel_tol` knob
  enables a separate robustness study.
- **casefold + whitespace-collapse for categorical** — fixes case/spacing only.
  It does **not** fake synonym matching (`'not normal'` ≠ `'False'`); that
  semantic-variant hazard is documented, not hidden (`data/disputes.md` §B).
- **Ordered element-wise for lists by default** — DAEval "lists" are usually
  *ordered tuples* (e.g. `month, year, price`), so blanket set-equality would be
  wrong. `set` mode is available and explicit.
- **Multi-part = all sub-answers must pass**; a missing field is recorded for
  error analysis. Matches the official "Accuracy by Question".
- **`official_*` reference kept verbatim** so tests pin down exactly where we
  agree with and diverge from the benchmark.

## Known benchmark issues we surface (not paper over)

See `data/disputes.md`: 2 malformed/degenerate gold answers (excluded), 81
tasks with semantically-inconsistent categorical gold (de-prioritised in the
headline subset), 1 duplicate-field task, 3 empty-gold tasks.

## Tests & quality

`python3 -m pytest tests/ -q` → **59 tests**. Covers the load-bearing pieces
(verifier extraction/comparison/multi-part, pass@k estimator) and the modules
with subtle logic (sandbox timeout-drain + read-only, LLM cache-key semantics,
ReAct loop termination + tool-error handling, loader join contracts, sampler
determinism). A two-reviewer pass (Python + architecture) drove fixes to the
verifier (nested-tag extraction, short-comma lists, dict-literal gold), the
pass@k estimator (reject c>n), tool-error propagation, and atomic cache writes.
`requirements.txt` pins deps so a fresh clone runs.

## Data / attribution

DAEval is InfiAgent-DABench (ICML 2024), CC BY-NC 4.0 — see `NOTICE`.
