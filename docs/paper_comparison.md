# Relationship to the Deep Hedging paper

Reference: Hans Buehler, Lukas Gonon, Josef Teichmann and Ben Wood, [*Deep Hedging*, arXiv:1802.03042v1](https://arxiv.org/abs/1802.03042v1). Locations below refer to that version's [PDF](https://arxiv.org/pdf/1802.03042v1), especially Sections 2, 4.3 and 5.1–5.4.

This repository implements a paper-inspired design and investigates training-data efficiency. It does not numerically reproduce every published experiment.

## Design correspondence

| Component | Paper location / design | Current implementation |
|---|---|---|
| Decision rule | §5.1: daily feedforward policies linked by previous holdings | Separate PyTorch daily networks; previous holdings remain in the gradient graph. |
| Hidden layers | §5.1: two ReLU layers, width $d+15$ | Default paper preset uses `D+15` with normalization. Compact preset has two 16-unit Tanh layers. |
| One-stock inputs | §5.2, Figure 1: log price, variance, previous positions | Active `forward()` scales log-moneyness and relative variance for **both** presets. `forward1()` is inactive. |
| Normalization / optimizer | §5.1: normalization before activation; Adam, batch 256, rate 0.005 | Paper preset keeps this ordering and CVaR defaults. Recalibration, stage budgets and entropic rate are explicit local choices. |
| Market grid | §5.1: 30 daily intervals, $T=30/365$ | 30 daily intervals, 31 stored observations, 64 substeps per interval by default. |
| Simulator | §5.2: CIR transitions, simplified Broadie–Kaya scheme | Exact CIR endpoints, weighted endpoint integration and corrected stock update; identical joint simulation laws or paths are not asserted. |
| Model hedge | §5.1: QuantLib reference | Custom Fourier/interpolation sensitivities; one synthetic call with stock and optional variance claim. |
| Larger networks | §5.4: widths $12n_H$, five independent models | Three independent stocks, explicit 36/36 layers and six instruments; matched networks with CVaR/entropic objectives. |
| Terminal cost | §2: numerical examples have zero maturity fees | Opening/rebalancing fees, no extra maturity fee: consistent with that convention. |

`D` is the number of instrument channels, including a forced-zero channel. The one-stock paper preset is 4 → 17 → 17 → 2. Its stock-only control retains two outputs but zeros the variance holding. Our stock-only preset is 3 → 16 → 16 → 1; supplying two instruments gives 4 → 16 → 16 → 2. Every configuration observes variance.

Shared architecture does not mean matching feature scaling, initialization, simulator details, training budgets, stopping rules, software or results. [Methods](methods.md) describes the active formulas, including certainty-equivalent entropic training and the directly reported entropic score.

## What the comparisons isolate

The single-call comparison changes preset architecture and data mode together, while separately varying trading instruments. Its differences cannot be attributed solely to streaming or solely to architecture.

Small-portfolio and memory comparisons explicitly match networks, features and instruments across method labels, isolating data mode within the local protocol. The path notebook also fixes the method preset across all four conditions and varies only data mode/generator.

The stored-pool baseline is a local implementation choice. A paper-style neural architecture can also train on streaming data. A finite-sample formulation alone does not require retaining the full training pool in RAM. Our measured improvement is against this repository's in-memory baseline, not a theoretical memory lower bound for every implementation of the source methodology.

Both built-in generators preserve float64 and the local numerical scheme. Fast adds concurrent blocks and bounded prefetch and changes training-stream partitioning. Evaluation uses common scenarios. Parallel elapsed-time gains use CPU capacity and do not establish lower asymptotic simulation complexity.

## Premiums, figures and evidence

Current single-call P&L figures use a shared `call_price` premium; the code does not perform a general utility-indifference-pricing study. `show_published_figure` displays source artwork from `assets/paper/`, separately from locally generated results.

The committed memory result has one training seed and concerns retained training scenarios. Repeated path tests use quick budgets, not established full-budget speed/quality equivalence. Portfolio-information compression remains a research objective.

A precise description is: **we implement daily neural hedging policies and compare matched fixed-data and fresh-batch training, with isolated memory measurements and optional parallel simulation.** See [Experiments](experiments.md) for artifact provenance and limitations and [the code guide](code_guide.md) for usage.
