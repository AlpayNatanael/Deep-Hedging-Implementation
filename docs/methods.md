# Model, networks and training

This guide describes [general_nn.py](../scripts/general_nn.py) and [settings.py](../scripts/settings.py). See [the code guide](code_guide.md) for runnable examples and [Experiments](experiments.md) for configurations and measurements.

## Market and liability

The built-in market contains $M$ independent Heston stocks. Within stock $i$, price and variance noises have correlation $\rho_i$:

$$
dS_t^i=S_t^i\sqrt{v_t^i}\,dW_t^i,\qquad
dv_t^i=\kappa_i(\theta_i-v_t^i)dt+\xi_i\sqrt{v_t^i}\,dB_t^i,
\qquad d\langle W^i,B^i\rangle_t=\rho_i\,dt.
$$

Interest and stock drift are zero in the synthetic examples. The default grid has $N=30$ intervals of length $1/365$, hence $T=30/365$ years and 31 stored observations. `time_step` specifies the calendar convention; dates supplied later do not redefine it.

The synthetic terminal liability is

$$
Z=\sum_{i=1}^{M}q_i(S_T^i-K_i)^+.
$$

`call_quantity` supplies $q_i$, defaulting to one. One scenario contains complete paths for all stocks, their variance states, instrument quotes and the liability payment. Scenario count and stock count are separate quantities.

Let $A_t^i=\int_0^t v_s^i\,ds$. An optional variance claim pays $A_T^i$ and has raw quote

$$
V_t^i=A_t^i+\theta_i(T-t)
+\frac{v_t^i-\theta_i}{\kappa_i}(1-e^{-\kappa_i(T-t)}).
$$

This conditional continuous-Heston pricing formula is applied to the simulated states. The variance claim is a tradable instrument; $v_t^i$ is a feature. Instrument multipliers, including the default variance notional of 1,000, enter accounting once.

## Numerical simulation and generators

`simulate_paths` uses CPU float64 arrays and 64 internal substeps per interval by default. Each substep samples an exact CIR variance endpoint from a scaled noncentral chi-square distribution, approximates integrated variance by positive weighted endpoint values, and applies a stock update with a conditional martingale correction. Only daily observations are retained.

Exact variance endpoints do not make the joint stock/integral simulation exact. Internal substeps improve simulation resolution; they are not additional trades. `transition_parameters`, `_simulate_arrays` and `variance_claim_price` define the numerical conventions.

Regular processes stocks sequentially and uses Numba when available, otherwise a Python/NumPy fallback. Fast reuses the compiled kernel in independent blocks: two blocks per stock and at most `min(fast_workers, 2*M)` worker threads. Fresh/Fast can prefetch into a queue of size `fast_prefetch`.

Fast requires Numba and partitions random streams differently, so equal seed labels do not imply identical training arrays. Built-in validation and test generation always uses Regular. The queue bounds pending batches; calibration/evaluation datasets, models and diagnostic records also consume memory.

## Daily hedging networks

There is a separate feedforward network for each trading date, linked by previous holdings:

$$
\delta_k=F_k(x_k,\delta_{k-1}),\qquad\delta_{-1}=0.
$$

The **active `forward()` uses the same market features for both method names**:

$$
x_{k,i}^{S}=\frac{\log(S_{k,i}/K_i)}{\sqrt{v_{0,i}T}},\qquad
x_{k,i}^{v}=v_{k,i}/v_{0,i}-1.
$$

These $2M$ features are followed by the previous holdings in all $D$ configured instruments. Input size is $2M+D$ and output size $D$. Outputs are new target holdings, not trade increments. The alternative `forward1()` is not called by `model(...)`.

| Configuration | Input → hidden → output widths | Activation | Batch normalization |
|---|---|---|---|
| Default `method="ours"`, one stock instrument | 3 → 16 → 16 → 1 | Tanh | No |
| Default `method="paper"`, stock and variance claim | 4 → 17 → 17 → 2 | ReLU | Yes |
| Matched three-stock experiments | 12 → 36 → 36 → 6 | Tanh | Yes |

Explicit network/instrument settings override presets. `method` does not choose data mode. Matched three-stock policies have 64,980 trainable network parameters over 30 dates, excluding the auxiliary threshold and normalization buffers. In the paper-style stock-only control the variance output remains but its holding is forced to zero. Every neural configuration observes variance. Future market states are not inputs.

## Gains, costs and losses

With raw quote $P_{k,j}$, multiplier $m_j$, per-unit interval cashflow $a_{k,j}$ and fee rate $c_j$, accounting is

$$
G=\sum_{k=0}^{N-1}\sum_{j=1}^{D}\delta_{k,j}m_j(P_{k+1,j}-P_{k,j}+a_{k,j}),
$$

$$
C=\sum_{k=0}^{N-1}\sum_{j=1}^{D}c_jm_jP_{k,j}|\delta_{k,j}-\delta_{k-1,j}|,
\qquad L=Z-G+C.
$$

Instrument-specific rates override the general rate. The default is 0.001, or 0.1% of traded notional. Opening/rebalancing fees are charged; no extra maturity liquidation fee is charged. Built-in synthetic instruments have zero intermediate cashflows.

Reported risks concern loss **before premium** $L$. P&L is $p-L$, with a shared premium. `call_price` supplies the zero-interest single-call model price; it is not a general portfolio indifference-pricing routine.

## Training objectives and reported risks

CVaR training jointly optimizes the policy and scalar threshold $\tau$:

$$
J_C=\tau+\frac{\operatorname{mean}(L-\tau)^+}{1-\alpha}.
$$

Default $\alpha=0.5$ targets the worst half of losses. Test CVaR is recomputed from the empirical distribution by `cvar`; it does not reuse the learned threshold as the test quantile.

Entropic training uses the exponential certainty-equivalent objective

$$
J_E=\tau+\frac{\operatorname{mean}[e^{\lambda(L-\tau)}-1]}{\lambda}.
$$

Minimizing over $\tau$ yields entropic risk. Validation and test directly report

$$
\rho_E(L)=\frac{1}{\lambda}\log\operatorname{mean}(e^{\lambda L}),
$$

using log-sum-exp. `risk_aversion` is $\lambda$, default one. Thus training batch objectives and reported entropic scores use different computational expressions.

`train()` performs a CVaR warm-up per replicate, then starts each requested continuation from its selected warm-up weights. Each stage gets a new Adam optimizer and threshold. Defaults include batch size 256, CVaR learning rate 0.005, entropic rate 0.0005 and gradient-norm limit 10. Notebooks override the longer class-default update budgets.

Each stage retains one training-source calibration sample of size `normalization_size`. Batch-normalization means/variances are calibrated from that sample before validation checkpoints, rather than from evaluation data. The lowest-validation-risk checkpoint is restored; test evaluation is separate. This fits hedging policies and normalization statistics, not Heston parameters to observed market prices.

## Data modes and classical reference

Fixed mode samples with replacement from a stored `static_training_size` pool. Fresh mode generates new update batches and stage calibration samples. Raw batches are discarded by default; `retain_training_batches=True` retains them. Evaluation/calibration data and diagnostic records remain in memory in both modes.

The Heston hedge supports one synthetic call, one tradable stock and optionally a variance claim. It uses numerical Fourier/interpolation sensitivities and the same loss accounting as neural policies. It is not a multi-stock or arbitrary historical-payoff hedge. The stock-only reference leaves variance risk unhedged; a model hedge is not automatically optimal under every objective/cost setting.

The current memory result concerns retained training scenarios. Statistical equivalence, full-budget runtime parity and reduction in required portfolio information remain separate research questions. See [the paper comparison](paper_comparison.md).
