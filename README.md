# The Probe Detects the Envelope, Not the Letter

**Does a linear eval-awareness probe still work where you'd actually use it?**

A difference-in-means probe on `allenai/Olmo-3-7B-Instruct` reaches
**0.971 ± 0.048** 5-fold CV AUROC separating eval-framed from natural prompts.
Embed the same cue inside a multi-turn transcript next to a length-matched
neutral control and it falls to chance. Uninjected sessions fire at 0%; any
appended `[Context: ...]` block fires at 100%, whether it contains a benchmark
notice or a routine container log.

The probe responds to the block, not what's in it.

September 2026.
Write-up: [link] · Author: Aditya Katkar

---

## Results

### The probe replicates on clean prompts

| Metric | Value |
|---|---|
| 5-fold CV AUROC | **0.971 ± 0.048** |
| per-fold | 1.0, 0.878, 1.0, 1.0, 0.978 |
| in-sample AUROC | 0.991 *(not reported — direction fit on these rows)* |
| clean TPR @ θ₉₀ | 89.6% |
| clean FPR @ θ₉₀ | 2.1% |
| null (100 random directions) | 0.514 ± 0.117 |
| n | 96 unique prompts |

![clean separation](figures/fig1_clean_separation.png)

### It collapses against length-matched controls

| Rung | Cue type | Eval TPR | Neutral FPR | AUROC |
|---|---|---|---|---|
| L1 | explicit ("you are being evaluated") | 100% | 100% | 0.635 |
| L2 | moderate (harness paths, eval repo URLs) | 90% | 90% | 0.518 |
| L3 | subtle (single-commit history, fixtures) | 65% | 70% | 0.395 |

n = 20 per rung. **TPR ≈ FPR at every rung** — not graceful degradation with
subtlety, but no discrimination at any level.

![ladder tpr vs fpr](figures/fig2_ladder_tpr_fpr.png)
![auroc collapse](figures/fig3_auroc_collapse.png)

### The baseline rate is zero

Same 20 sessions, no cue injected:

| Metric | Value |
|---|---|
| firing rate | **0%** |
| score min / mean / max | −0.0458 / −0.0080 / +0.0167 |
| std | 0.0163 |

0% uninjected → 100% with an eval cue → 100% with a matched neutral cue.
That is the inference: it detects the container, not the content.

### Prompting fails too, in the opposite direction

| Rung | Eval TPR | Neutral FPR | Unparsed |
|---|---|---|---|
| L1 | 100% | 70% | 0 |
| L2 | 100% | 100% | 0 |
| L3 | 95% | 70% | 0 |

120 generations, zero unparsed (verified by reading raw outputs — all bare
YES/NO). The probe over-fires on format; prompting over-fires on nearly
everything.

![probe vs prompting](figures/fig4_probe_vs_prompting.png)

---

## Controls

**Null direction.** 100 random unit directions: AUROC 0.514 ± 0.117. Rules out
the clean-set separation being a geometric artifact of a 4096-dim space at n=96.

**Speaker-role confound — rejected.** Paired ablation, identical content, only
the final turn's speaker label flipped:

| | Value |
|---|---|
| ends on User | −0.0080 ± 0.0163, fires 0% |
| ends on Assistant | −0.0049 ± 0.0176, fires 10% |
| paired delta | −0.0031 ± 0.0133 |
| pairs with delta > 0 | 7/20 |
| Wilcoxon p | 0.37 |

Effect ~6.6× smaller than θ₉₀. An earlier n=1 version of this test looked
decisive (+0.0481 vs −0.0439) and vanished at n=20.

**Testing-jargon confound — rejected.** Text saturated with `test_`, `pytest`,
`AssertionError` scored −0.0959, the lowest of the three semantic conditions
tested. If keyword matching drove the false positives this should have fired
hardest; it fired least.

---

## Setup

| | |
|---|---|
| Model | `allenai/Olmo-3-7B-Instruct`, fp16 |
| Site | layer 16 / 32, residual stream, last token, d = 4096 |
| Probe | unit-normalised difference-in-means |
| Threshold | θ₉₀ = +0.0203 (10th pct of positive projections, targets 90% TPR) |
| Tooling | nnsight |
| Hardware | 2×T4, ~15 min end to end |

**Contrast set.** 24 base pairs × 2 suffix variants = 48 pairs, 96 unique
prompts. Eval-framed (benchmark harness headers, grader notices, task IDs) vs
task-matched natural framing, across coding, safety, maths, systems and writing.

**Ladder.** 20 distinct base sessions × 8 cue templates per rung. Every eval cue
has a length-matched neutral twin injected at the same position, differing by
1–3 characters in total length.

**Probe direction.**

```
v = (mu_pos - mu_neg) / ||mu_pos - mu_neg||_2
```

fitted on L2-normalised activations. Chose difference-in-means over logistic
regression because at n=96 and d=4096 a fitted classifier has ample room to
separate on noise.

---

## Files

| File | Purpose |
|---|---|
| `full_pipeline.py` | end to end: probe, ladder, baseline FPR, prompting, role ablation |
| `sample_examples.py` | random raw examples + results block (no GPU) |
| `make_figures.py` | figures from `results_all.npz` |
| `results_all.npz` | saved arrays: direction, θ₉₀, projections, ablation scores |

```bash
pip install torch transformers nnsight scikit-learn scipy numpy matplotlib
python full_pipeline.py      # ~15 min on 2xT4
python sample_examples.py    # raw examples, CPU only
python make_figures.py       # fig1-4
```

`full_pipeline.py` asserts on effective sample size before every extraction.
If an assert fires, the dataset has collapsed — don't work around it.

---

## Self-caught errors

Three data bugs found and fixed during the project. The first two each reversed
a reported number:

1. **Effective n = 2 reported as n = 50.** The first ladder cycled 2 cue
   templates across 50 identical copies of one base session. Caught in a data
   audit. After rebuilding with 20 distinct sessions × 8 templates, the neutral
   control flipped from an apparent pass (0% FPR) to complete collapse (100%).
2. **Degenerate raw baseline.** Stage 4 originally scored 50 duplicates of one
   string — every projection identical (min = mean = max = +0.0549). Caught
   while hand-reading firings.
3. **n = 1 speaker-role split.** Looked decisive, vanished at n = 20 (p = 0.37).

---

## Scope and limitations

One model, one layer, one probe type, one cue-injection format. This shows the
probe can't resolve **near-minimal semantic pairs inside an appended block** —
not that eval-awareness probes fail in general.

- **Uniform delimiter format** is the most likely mechanism. Every cue arrives
  in an identical `\n\n[Context: ...]` block; the obvious next test is natural
  in-turn phrasing without delimiters. Noticed late.
- **Constructed sessions, not real traffic.** SWE-chat is turn-level with heavy
  tool-call serialization; clean multi-turn transcripts couldn't be assembled in
  the time budget. The operational base-rate question stays open.
- **Nested brackets in L1.** L1 cues render as `[Context: [SYSTEM NOTICE: ...]]`
  where L2/L3 don't — a possible partial explanation for L1's higher AUROC.
- **Cue variety is 8, not 20.** Eight templates cycling over 20 sessions means
  each cue appears 2–3 times.
- **Shared suffix families** in the contrast set may inflate the clean 0.971.
- **Small n** throughout: 24 base pairs, 20 sessions, 20 per rung.
- **Sub-chance L3 (0.395)** is not interpreted — within noise at n = 20.

## Next

Vary the injection format · real production transcripts and a true base-rate
audit · layer sweep across all 32 blocks, and test whether projecting out the
delimiter subspace recovers any semantic signal.
