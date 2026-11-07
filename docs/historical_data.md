# Evaluating supplied historical observations

`model.evaluate_historical(data)` evaluates current policies using supplied prices, variance features and liability payments. Use a trained or loaded model. It does not fit weights, recalibrate normalization on evaluation data, generate missing quotes or substitute a simulated call payoff.

No historical dataset or historical performance result is supplied. Current notebook results use simulated data. See [the code guide](code_guide.md) for training and [Methods](methods.md) for accounting.

## Required observations

Let $B$ be episodes, $N$ configured intervals, $D$ instrument channels and $M$ stocks. Default policies use $N=30$, hence 31 price observations per episode.

| Field | Shape / contents | Meaning |
|---|---|---|
| `prices` | `(B, N+1, D)` | Nonnegative observed quotes, raw currency per unit |
| `stock_prices` | `(B, N+1, M)` | Positive underlying feature prices |
| `variances` | `(B, N+1, M)` | Nonnegative features in annual variance units |
| `payoffs` | `(B,)` | Required terminal liability payments |
| `dates` | `(B, N+1)` | Strictly increasing timestamps per episode |
| `feature_times` | `(B, N)` | Latest input availability time for each decision |
| `instrument_names`, `stock_names` | Ordered lists | Exact configured channel/stock names |
| `cashflows` | optional `(B, N, D)` | Per-unit asset payments during each interval |
| `metadata` | Dictionary below | Provenance and conventions |

All numbers must be finite and dimensions match the policy. Even nontradable configured channels require observed quotes: the paper-stock control retains a forced-zero variance channel. Historical variance-claim quotes are not synthesized.

Normalize timestamps to UTC before supplying nanosecond-compatible arrays/strings. No `NaT` values are allowed. Each `feature_times[b,k]` must be no later than `dates[b,k]`. Dates describe observations; they do not resample the policy or redefine its configured time step, maturity or scaling.

Variance can be a declared trailing estimate using only information available at the decision. Supply variance, not volatility or percentage volatility. A historical proxy differs from the simulated latent Heston state; transfer performance needs evaluation.

## Units and payments

Loss is $L=Z-G+C$ before premium, with zero interest, opening/rebalancing fees and no additional maturity trading fee. Multipliers apply once inside `calculate_losses`; do not pre-multiply quotes or cashflows. Instrument-specific rates override the general fee rate.

An interval payment belongs to the holding carried from its starting to ending timestamp. Its gain is $\delta m(P_{k+1}-P_k+a_k)$ for raw quote $P$, per-unit payment $a$ and multiplier $m$. Keep price adjustments, distributions and corporate actions consistent so payments are counted once. Intermediate liability payments and other interest conventions are unsupported.

Use `asset_cashflows="none"` for no payments; omitted cashflows become zeros. Otherwise supply the array and declare `"included"`. Nonzero payments with `"none"`, or `"included"` without an array, are rejected.

## Template for your own data

The names below are arrays/descriptions you must prepare. This is a schema template, not a supplied market dataset.

```python
historical_data = {
    "prices": observed_instrument_prices,
    "stock_prices": observed_stock_prices,
    "variances": supplied_variance_features,
    "payoffs": observed_terminal_payments,
    "dates": observation_dates_utc,
    "feature_times": feature_availability_times_utc,
    "instrument_names": instrument_names,
    "stock_names": stock_names,
    "metadata": {
        "kind": "historical",
        "source": price_source_description,
        "payoff_source": payment_source_description,
        "variance_source": variance_source_description,
        "price_basis": "raw_per_unit",
        "interest_rate": 0.0,
        "liability_timing": "terminal",
        "asset_cashflows": "none",
    },
}

historical = model.evaluate_historical(
    historical_data, name="Historical evaluation"
)
```

The three source fields must be nonempty descriptions. For included payments add `"cashflows": observed_asset_payments` and change `asset_cashflows` to `"included"`.

The result is stored in `model.historical_results` and supports ordinary loss/risk/tail/P&L plots. Its metrics use supplied payments and preserve the parent's original synthetic scores. `historical.evaluate()` re-evaluates those observations; synthetic analytical benchmarks on that view are rejected. `plot_learning(historical)` shows original training history, not historical retraining.

## Separation and caller responsibilities

Historical checking rejects reserved training/validation seeds, identical retained training/calibration/validation datasets and overlapping date ranges with those dated datasets. It deliberately skips an existing `test` split. The general loader applies role-specific checks for supplied training/validation/test data.

Fingerprints recognize identical whole datasets in the same row order, not every partial overlap, reordered subset or dependence. Streamed batches are normally discarded, so observations cannot be checked against every past training batch. Undated training sources do not establish calendar separation.

Callers must establish provenance, causal features and out-of-sample periods. Overlapping episodes are dependent observations. For a separately prepared validation set, call `model.load_data(..., split="validation")`, then `model.validate()`; it measures frozen policies rather than fitting them.
