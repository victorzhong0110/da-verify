"""Regenerate assets/c0-c3.svg — the headline figure for README.md.

Numbers are the reported study results (see report/report.md):
  temp 0, k=1: C0 82.5 / C1 82.5 / C2 85.0 (95% Wilson CI)
  temp 0.7, k=5, serial: pass@1 and pass^5 for C0 / C2 / C3;
  C3 vs C0 paired bootstrap 95% CI [+10.5, +30.0].

Run from the repo root:  python3 scripts/make_headline_figure.py
Requires only matplotlib + numpy.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["svg.fonttype"] = "path"

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "c0-c3.svg"

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1, 1.35]}
)

GREY = "#9ca3af"
MID = "#6b9bd1"
BLUE = "#0078D7"
DARK = "#111827"
ERR = "#374151"

# ---- Left panel: temp 0, k=1 (underpowered) ----
conds = ["C0\nno verif.", "C1\nself-check", "C2\nLLM re-derive"]
vals = [82.5, 82.5, 85.0]
lo = [68.0, 68.0, 70.9]
hi = [91.3, 91.3, 92.9]
err = [
    [v - l for v, l in zip(vals, lo)],
    [h - v for v, h in zip(vals, hi)],
]
ax1.bar(
    conds,
    vals,
    color=[GREY, GREY, MID],
    width=0.58,
    yerr=err,
    capsize=4,
    error_kw=dict(ecolor=ERR, lw=1.2),
)
for i, v in enumerate(vals):
    ax1.text(i, hi[i] + 2.5, f"{v}%", ha="center", fontsize=9, color=DARK)
ax1.set_title("temp 0 · k=1 · n=40\nunderpowered: nothing moves", fontsize=11)
ax1.set_ylabel("pass@1 (%)  ·  95% Wilson CI", fontsize=9)
ax1.set_ylim(0, 108)
ax1.tick_params(labelsize=8.5)

# ---- Right panel: temp 0.7, k=5 (powered) ----
groups = ["C0\nno verif.", "C2\nLLM verifier", "C3\nprogrammatic"]
p1 = [64.5, 75.5, 84.5]
pk = [40.0, 57.5, 75.0]
x = np.arange(3)
w = 0.36
ax2.bar(x - w / 2, p1, w, label="pass@1", color=[GREY, MID, BLUE])
ax2.bar(
    x + w / 2,
    pk,
    w,
    label="pass^5 (reliability)",
    color=["#d1d5db", "#a9c3e4", "#58a6e8"],
    hatch="//",
    edgecolor="white",
)
ax2.axhline(82.5, ls="--", lw=1, color="#b45309")
ax2.text(
    -0.45,
    84.5,
    "temp-0 baseline 82.5%",
    fontsize=8,
    color="#b45309",
    ha="left",
)
for xi, v in zip(x - w / 2, p1):
    ax2.text(xi, v + 2, f"{v}%", ha="center", fontsize=8, color=DARK)
for xi, v in zip(x + w / 2, pk):
    ax2.text(xi, v + 2, f"{v}%", ha="center", fontsize=8, color=DARK)
ax2.annotate(
    "C3 vs C0: +20.0pt\n95% CI [+10.5, +30.0]",
    xy=(2 - w / 2, 86.5),
    xytext=(0.75, 99),
    fontsize=8.5,
    color=BLUE,
    ha="center",
    arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8),
)
ax2.set_xticks(x)
ax2.set_xticklabels(groups, fontsize=8.5)
ax2.set_title(
    "temp 0.7 · k=5 · serial\nverification works where instability lives",
    fontsize=11,
)
ax2.set_ylim(0, 108)
ax2.legend(fontsize=8, frameon=False, loc="upper left")
ax2.tick_params(labelsize=8.5)

fig.suptitle(
    "da-verify: verification repairs variance, not capability",
    fontsize=12.5,
    fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, bbox_inches="tight")
print(f"saved {OUT}")
