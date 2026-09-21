import json, random
import numpy as np
import matplotlib.pyplot as plt

d = np.load("data/results_all.npz")
theta = float(d["theta_90"])

plt.rcParams.update({"figure.dpi": 140, "font.size": 9})

fig, ax = plt.subplots(figsize=(5.2, 2.8))
cp, yc = d["clean_proj"], d["y_clean"]
ax.hist(cp[yc == 1], bins=18, alpha=.65, label="eval-framed", color="#c0392b")
ax.hist(cp[yc == 0], bins=18, alpha=.65, label="natural", color="#2980b9")
ax.axvline(theta, ls="--", c="k", lw=1, label=f"θ₉₀ = {theta:+.4f}")
ax.set_xlabel("projection onto probe direction")
ax.set_ylabel("count")
ax.set_title("Clean contrast set: 5-fold CV AUROC 0.971 ± 0.048")
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("results/figures/fig1_clean_separation.png"); plt.close(fig)

rungs = ["L1\nexplicit", "L2\nmoderate", "L3\nsubtle"]
tpr = [100, 90, 65]
fpr = [100, 90, 70]
x = np.arange(3); w = .36

fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.bar(x - w/2, tpr, w, label="eval cue (TPR)", color="#c0392b")
ax.bar(x + w/2, fpr, w, label="matched neutral cue (FPR)", color="#7f8c8d")
for i, (a, b) in enumerate(zip(tpr, fpr)):
    ax.text(i - w/2, a + 2, f"{a}%", ha="center", fontsize=8)
    ax.text(i + w/2, b + 2, f"{b}%", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(rungs)
ax.set_ylabel("firing rate (%)"); ax.set_ylim(0, 118)
ax.set_title("Probe fires equally on eval and matched neutral cues")
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig("results/figures/fig2_ladder_tpr_fpr.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 2.8))
au = [0.635, 0.5175, 0.395]
ax.axhspan(0.5 - 0.117, 0.5 + 0.117, color="#bdc3c7", alpha=.4,
           label="null band (100 random dirs)")
ax.axhline(.5, c="k", lw=.8, ls=":")
ax.bar(rungs, au, .5, color="#8e44ad")
ax.axhline(.971, c="#27ae60", ls="--", lw=1.2, label="clean-set CV AUROC 0.971")
for i, a in enumerate(au):
    ax.text(i, a + .02, f"{a:.3f}", ha="center", fontsize=8)
ax.set_ylabel("AUROC (eval vs matched neutral)"); ax.set_ylim(0, 1.05)
ax.set_title("Clean-set AUROC does not transfer")
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig("results/figures/fig3_auroc_collapse.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 3.0))
x = np.arange(3); w = .2
ax.bar(x - 1.5*w, [100, 90, 65],  w, label="probe: eval",    color="#c0392b")
ax.bar(x - 0.5*w, [100, 90, 70],  w, label="probe: neutral", color="#e59866")
ax.bar(x + 0.5*w, [100, 100, 95], w, label="prompt: eval",   color="#2471a3")
ax.bar(x + 1.5*w, [70, 100, 70],  w, label="prompt: neutral",color="#85c1e9")
ax.set_xticks(x); ax.set_xticklabels(rungs); ax.set_ylabel("firing rate (%)")
ax.set_ylim(0, 118)
ax.set_title("Neither method discriminates against matched controls")
ax.legend(fontsize=7, ncol=2, loc="lower left")
fig.tight_layout(); fig.savefig("results/figures/fig4_probe_vs_prompting.png"); plt.close(fig)

print("wrote fig1..fig4 PNGs")


print("\n\n" + "=" * 72)
print("RESULTS BLOCK — copy these into the write-up, do not retype from memory")
print("=" * 72)

d = np.load("data/results_all.npz")
theta = float(d["theta_90"])
cp, yc = d["clean_proj"], d["y_clean"]
cv = d["cv_aurocs"]
raw = d["raw_proj"]
su, sa, delta = d["su"], d["sa"], d["d"]

print(f"\nSTAGE 1")
print(f"  5-fold CV AUROC   : {cv.mean():.4f} ± {cv.std():.4f}")
print(f"  per-fold          : {np.round(cv, 4).tolist()}")
print(f"  theta_90          : {theta:+.4f}")
print(f"  clean TPR         : {(cp[yc==1] >= theta).mean()*100:.1f}%")
print(f"  clean FPR         : {(cp[yc==0] >= theta).mean()*100:.1f}%")
print(f"  n prompts         : {len(cp)}")

print(f"\nSTAGE 4 (unmodified sessions, n={len(raw)})")
print(f"  firing rate       : {(raw >= theta).mean()*100:.1f}%")
print(f"  min/mean/max      : {raw.min():+.4f} / {raw.mean():+.4f} / {raw.max():+.4f}")
print(f"  std               : {raw.std():.4f}")

print(f"\nSTAGE 6 speaker-role ablation (n={len(delta)} paired)")
print(f"  user-end          : {su.mean():+.4f} ± {su.std():.4f}, fires {(su>theta).mean()*100:.0f}%")
print(f"  asst-end          : {sa.mean():+.4f} ± {sa.std():.4f}, fires {(sa>theta).mean()*100:.0f}%")
print(f"  paired delta      : {delta.mean():+.4f} ± {delta.std():.4f}")
print(f"  delta>0           : {(delta>0).sum()}/{len(delta)}")
print(f"  |delta| vs theta  : {abs(delta.mean()):.4f} vs {theta:.4f} "
      f"(ratio {theta/abs(delta.mean()):.1f}x)")

print("\n" + "=" * 72)
print("Stage 3 and Stage 5 numbers are in your run log — copy them from there.")
print("=" * 72)
