import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from extract import norm_acts, extract_or_load, RUN_EVALS
from probe import BASE_PAIRS, direction, theta_90
from implant import base_sessions, LADDER, acts_eval, acts_neut, raw_proj
from followup_data import EVAL_CASUAL, NATURAL_FORMAL


def auroc(pos, neg):
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def unit(v):
    return v / np.linalg.norm(v)


def paired_p(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float("nan") if np.allclose(d, 0) else wilcoxon(a, b).pvalue


# ---------------------------------------------------------------------------
print("\n" + "=" * 70); print("FOLLOW-UP 1: FORMAT-SWAPPED CONTRAST SET"); print("=" * 70)
print("  If format is the mechanism: original probe scores natural_formal > eval_casual,")
print("  and the format direction is close to the original probe direction.")

cells = {
    "eval_formal":    [e for e, _ in BASE_PAIRS],
    "eval_casual":    EVAL_CASUAL,
    "natural_formal": NATURAL_FORMAL,
    "natural_casual": [p for _, p in BASE_PAIRS],
}
for k, v in cells.items():
    assert len(v) == 24 and len(set(v)) == 24, f"{k} n collapsed"
    print(f"  {k:<15} n={len(v)}  mean chars={np.mean([len(t) for t in v]):.0f}")

X = {k: norm_acts(extract_or_load(v, f"activations/acts_fmt_{k}.npy")) for k, v in cells.items()}
S = {k: X[k] @ direction for k in cells}

print(f"\n  Original probe (theta_90={theta_90:+.4f}):")
print(f"  {'cell':<15} | {'mean':>8} | {'std':>7} | {'fires':>6}")
for k in cells:
    print(f"  {k:<15} | {S[k].mean():+.4f} | {S[k].std():.4f} | {(S[k] >= theta_90).mean()*100:5.1f}%")

conflict = auroc(S["eval_casual"], S["natural_formal"])
print(f"\n  AUROC eval_casual vs natural_formal : {conflict:.3f}   (<0.5 means format beats framing)")
print(f"  AUROC formal vs casual, eval only    : {auroc(S['eval_formal'], S['eval_casual']):.3f}")
print(f"  AUROC formal vs casual, natural only : {auroc(S['natural_formal'], S['natural_casual']):.3f}")
print(f"  AUROC eval vs natural, formal only   : {auroc(S['eval_formal'], S['natural_formal']):.3f}")
print(f"  AUROC eval vs natural, casual only   : {auroc(S['eval_casual'], S['natural_casual']):.3f}")

# Balanced probe: framing label, format balanced, grouped CV so one task never
# appears in both train and test.
order  = list(cells)
X_all  = np.concatenate([X[k] for k in order])
y_frm  = np.array([1] * 48 + [0] * 48)                      # eval vs natural
y_fmt  = np.array([1] * 24 + [0] * 24 + [1] * 24 + [0] * 24)  # formal vs casual
groups = np.tile(np.arange(24), 4)

oof = np.zeros(len(X_all))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X_all, y_frm, groups):
    d = unit(X_all[tr][y_frm[tr] == 1].mean(0) - X_all[tr][y_frm[tr] == 0].mean(0))
    oof[te] = X_all[te] @ d

print("\n  Balanced probe (trained on all 4 cells, grouped 5-fold CV):")
print(f"  framing AUROC, all       : {roc_auc_score(y_frm, oof):.3f}")
print(f"  framing AUROC, formal only: {roc_auc_score(y_frm[y_fmt == 1], oof[y_fmt == 1]):.3f}")
print(f"  framing AUROC, casual only: {roc_auc_score(y_frm[y_fmt == 0], oof[y_fmt == 0]):.3f}")

d_bal = unit(X_all[y_frm == 1].mean(0) - X_all[y_frm == 0].mean(0))
d_fmt = unit(X_all[y_fmt == 1].mean(0) - X_all[y_fmt == 0].mean(0))
theta_bal = np.percentile(X_all[y_frm == 1] @ d_bal, 10)
print(f"\n  cos(original, format)   = {direction @ d_fmt:+.3f}")
print(f"  cos(original, balanced) = {direction @ d_bal:+.3f}")
print(f"  cos(balanced, format)   = {d_bal @ d_fmt:+.3f}")

print(f"\n  Balanced probe on the bracketed ladder (theta_bal={theta_bal:+.4f}):")
print(f"  {'Rung':<5} | {'Eval TPR':>9} | {'Neut FPR':>9} | {'AUROC':>7}")
bal_ladder = {}
for r in RUN_EVALS:
    ep, npj = norm_acts(acts_eval[r]) @ d_bal, norm_acts(acts_neut[r]) @ d_bal
    bal_ladder[r] = (ep, npj)
    print(f"  {r:<5} | {(ep >= theta_bal).mean()*100:8.1f}% | {(npj >= theta_bal).mean()*100:8.1f}% | {auroc(ep, npj):7.3f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70); print("FOLLOW-UP 2: SAME CUES, NO BRACKETS"); print("=" * 70)
print("  Cue text is identical to the ladder; only the wrapper changes.")
print("  end : cue appended to the last turn as a sentence (no [Context: ...])")
print("  mid : cue appended to the first turn; last turn and last token unchanged")
print("  If format is the mechanism: firing drops toward 0% on both eval and neutral.")


def inline_end(turns, cue):
    m = list(turns); m[-1] += f" {cue}"; return "\n".join(m)


def inline_mid(turns, cue):
    m = list(turns); m[0] += f" {cue}"; return "\n".join(m)


placements = {"end": inline_end, "mid": inline_mid}
proj = {}
for place, fn in placements.items():
    for r in RUN_EVALS:
        t = LADDER[r]
        ev = [fn(s, t[i % len(t)][0]) for i, s in enumerate(base_sessions)]
        nv = [fn(s, t[i % len(t)][1]) for i, s in enumerate(base_sessions)]
        assert len(set(ev)) == len(base_sessions), f"{place} {r} eval n collapsed"
        assert len(set(nv)) == len(base_sessions), f"{place} {r} neut n collapsed"
        if place == "mid":
            assert all(e.endswith(s[-1]) for e, s in zip(ev, base_sessions)), "mid changed last turn"
        ea = norm_acts(extract_or_load(ev, f"activations/acts_{place}_eval_{r}.npy")) @ direction
        na = norm_acts(extract_or_load(nv, f"activations/acts_{place}_neut_{r}.npy")) @ direction
        proj[(place, r)] = (ea, na)

for r in RUN_EVALS:
    proj[("bracket", r)] = (norm_acts(acts_eval[r]) @ direction, norm_acts(acts_neut[r]) @ direction)

print(f"\n  Original probe, theta_90={theta_90:+.4f}. Uninjected sessions fire "
      f"{(raw_proj >= theta_90).mean()*100:.0f}%.")
print(f"  {'Wrapper':<8} | {'Rung':<4} | {'Eval TPR':>9} | {'Neut FPR':>9} | {'AUROC':>6} | {'paired p':>8}")
print("-" * 70)
for place in ["bracket", "end", "mid"]:
    for r in RUN_EVALS:
        ea, na = proj[(place, r)]
        print(f"  {place:<8} | {r:<4} | {(ea >= theta_90).mean()*100:8.1f}% | "
              f"{(na >= theta_90).mean()*100:8.1f}% | {auroc(ea, na):6.3f} | {paired_p(ea, na):8.3f}")

print("\n  RAW EXAMPLES (read these):")
rng = np.random.default_rng(1)
for place, fn in placements.items():
    i = int(rng.integers(len(base_sessions)))
    print(f"  --- {place}, L2, session {i} ---")
    print("  " + fn(base_sessions[i], LADDER["L2"][i % 8][0]).replace("\n", "\n  "))

# ---------------------------------------------------------------------------
np.savez("data/followup_results.npz",
         **{f"fmt_{k}": S[k] for k in cells},
         oof=oof, y_frm=y_frm, y_fmt=y_fmt,
         d_bal=d_bal, d_fmt=d_fmt, theta_bal=theta_bal,
         **{f"{p}_{r}_eval": proj[(p, r)][0] for p in ["bracket", "end", "mid"] for r in RUN_EVALS},
         **{f"{p}_{r}_neut": proj[(p, r)][1] for p in ["bracket", "end", "mid"] for r in RUN_EVALS})
print("\nsaved -> data/followup_results.npz")

plt.rcParams.update({"figure.dpi": 140, "font.size": 9})

fig, ax = plt.subplots(figsize=(5.2, 2.8))
names = ["eval_formal", "eval_casual", "natural_formal", "natural_casual"]
cols = ["#c0392b", "#e6a09a", "#2980b9", "#9cc3e0"]
for j, (k, c) in enumerate(zip(names, cols)):
    ax.scatter(np.full(24, j) + np.random.default_rng(j).uniform(-.15, .15, 24), S[k], s=10, c=c)
    ax.hlines(S[k].mean(), j - .3, j + .3, colors="k", lw=1.2)
ax.axhline(theta_90, ls="--", c="k", lw=.8)
ax.set_xticks(range(4)); ax.set_xticklabels([n.replace("_", "\n") for n in names])
ax.set_ylabel("original probe score")
ax.set_title("Original probe on format-swapped prompts")
fig.tight_layout(); fig.savefig("results/figures/fig5_format_swap.png"); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)
w = .38
for ax, place in zip(axes, ["bracket", "end", "mid"]):
    tpr = [(proj[(place, r)][0] >= theta_90).mean() * 100 for r in RUN_EVALS]
    fpr = [(proj[(place, r)][1] >= theta_90).mean() * 100 for r in RUN_EVALS]
    x = np.arange(len(RUN_EVALS))
    ax.bar(x - w/2, tpr, w, color="#c0392b", label="eval cue")
    ax.bar(x + w/2, fpr, w, color="#7f8c8d", label="neutral twin")
    ax.set_xticks(x); ax.set_xticklabels(RUN_EVALS); ax.set_title(place); ax.set_ylim(0, 110)
axes[0].set_ylabel("firing rate (%)"); axes[0].legend(fontsize=7)
fig.suptitle("Same cues, different wrapper")
fig.tight_layout(); fig.savefig("results/figures/fig6_wrapper.png"); plt.close(fig)
print("saved -> results/figures/fig5_format_swap.png, fig6_wrapper.png")
print("=" * 70); print("DONE"); print("=" * 70)
