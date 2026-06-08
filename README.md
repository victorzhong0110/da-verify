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

## Data / attribution

DAEval is InfiAgent-DABench (ICML 2024), CC BY-NC 4.0 — see `NOTICE`.
