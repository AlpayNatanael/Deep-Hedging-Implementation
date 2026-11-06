# Evaluate supplied historical data

`model.evaluate_historical(data)` applies the saved policies to observations you supply. It uses the supplied liability payments and asset prices, with the saved feature scaling and frozen network parameters. It does not generate missing market data, replace the liability with a simulated call, or retrain the model.

No historical dataset is included, and no historical performance result is claimed by this project.

## Required observations

Let $B$ be the number of recorded episodes, $T$ the number of trading decisions, $D$ the configured instrument count, and $M$ the stock count. The saved default models require $T=30$, with $31$ price observations per episode.

| Dictionary field | Shape or content | Meaning |
|---|---|---|
| `prices` | $(B,T+1,D)$ | Nonnegative observed instrument prices, in raw currency per unit. |
| `stock_prices` | $(B,T+1,M)$ | Positive stock prices used as network inputs. |
| `variances` | $(B,T+1,M)$ | Nonnegative variance inputs, in the saved model's annual variance units. |
| `payoffs` | $(B,)$ | Recorded terminal liability payments, in currency; always required. |
| `dates` | $(B,T+1)$ | Strictly increasing observation and trading timestamps. |
| `feature_times` | $(B,T)$ | Latest availability time of the inputs used at each decision. |
| `instrument_names`, `stock_names` | Ordered names | Must match the saved settings exactly. |
| `cashflows` | Optional $(B,T,D)$ | Observed per-unit asset payments received during each holding interval. |

Normalize both timestamp arrays to UTC before calling the method. Every feature must have been available by its corresponding decision time. Use the saved model's horizon and time convention; supplying dates does not change its daily networks or normalization.

Historical variance may be a trailing estimate, but its source and construction must be declared. Supply variance, not volatility or percentage volatility, and use only information available at the decision. Such an estimate is an input proxy; it is not an observed latent Heston state. A model trained on simulated states is not automatically calibrated to that proxy.

## Units, cashflows, and loss

Each `InstrumentSettings` record specifies its name, kind, stock association, multiplier, trading flag, and optional cost rate. Supply observed prices for every configured channel; the method never constructs a historical variance claim from a Heston formula. The fixed data schema also requires observed quotes for a nontradable paper-stock output, although its forced-zero holdings contribute no gains or fees; fabricated placeholder quotes are not supported.

For raw price $P_{k,j}$, interval cash payment $A_{k,j}$, multiplier $m_j$, and holding $h_{k,j}$, accounting uses

$$
G=\sum_{k,j}h_{k,j}m_j(P_{k+1,j}-P_{k,j}+A_{k,j}),
\qquad
C=\sum_{k,j}c_jm_jP_{k,j}|h_{k,j}-h_{k-1,j}|,
\qquad L=Z-G+C.
$$

Here $Z$ is the supplied liability payment and $h_{-1,j}=0$. Multipliers apply once: do not pre-multiply prices or asset cashflows. An interval cashflow belongs to the holding carried from its starting timestamp to its ending timestamp. Keep prices, distributions, and corporate actions consistent so each payment is counted once.

Results are losses **before premium**, with zero interest, opening and rebalancing fees, terminal marking to market, and no maturity trading fee. Unsupported interest conventions and intermediate liability payments are rejected. An instrument-specific cost rate overrides the general configured rate.

## Example with your own arrays

The variables below represent data you must obtain and prepare; they are not a supplied dataset.

```python
historical_data = {
    "prices": observed_instrument_prices,
    "stock_prices": observed_stock_prices,
    "variances": supplied_variance_states,
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

If asset payments occur, supply `cashflows` and set `asset_cashflows` to `"included"`. The returned object works with the same plotting helpers and is stored in `model.historical_results`; the parent's original experiment arrays remain unchanged.

For a separately prepared validation dataset, use `model.load_data(data, split="validation")`, then `model.validate()`. Validation measures frozen policies. Known reuse and dated overlap across training and evaluation are rejected; callers must still establish provenance and independence. Overlapping historical episodes are dependent observations.
