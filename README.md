# Deep Hedging in the Heston Model

**Can we preserve hedging performance while reducing the memory needed for training?** We study neural hedging with transaction costs, comparing training from a stored dataset with training on fresh simulated batches.

Our longer-term goal is portfolio compression: representing an option portfolio with less information while preserving hedging performance. This implementation builds on [*Deep Hedging* by Buehler et al.](https://arxiv.org/abs/1802.03042).

## Our contributions

- **Training without a large stored dataset:** generate Heston paths as needed for each training batch.
- **Matched comparisons:** compare fixed-data and fresh-batch training using identical network architectures and common test scenarios in a three-stock experiment.
- **Performance and memory evaluation:** measure CVaR, entropic risk, P&L distributions, and peak process RAM.

## Preliminary results

In our initial single-call experiment, our networks achieve lower test CVaR than our paper-style reconstruction, both with stock-only trading and with a variance claim. Architecture and training-data mode change together in this experiment. The full Heston benchmark remains stronger in the saved stock-plus-variance comparison.

In the three-stock experiment, both configurations use the same architecture and achieve closely matched observed CVaR. A separate CPU benchmark found approximately **77.5% lower peak process RAM** with fresh batches compared with our in-memory fixed pool. Network size is unchanged, and fresh-batch training currently takes longer.

These findings support **memory-efficient training with comparable observed hedging performance**. They are preliminary results from one training replicate per configuration; repeated runs are needed to assess reliability.

## Experiments

| Notebook | Description |
|---|---|
| [Single_Call_Comparison.ipynb](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/Single_Call_Comparison.ipynb) | Hedge one call using stock alone or stock plus a variance claim. Compare neural strategies with classical Heston hedges. |
| [Large_Portfoloio_Comparison.ipynb](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/Large_Portfoloio_Comparison.ipynb) | Hedge three calls on three simulated stocks. Compare fixed-data and fresh-batch training with the same network architecture. |
| [Large_Portfolio_Memory_Comparison.ipynb](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/Large_Portfolio_Memory_Comparison.ipynb) | Run the same portfolio experiment in separate processes and compare peak RAM, runtime, and test risk. |

For a guided walkthrough, see [Single_Call_Comparison_Explained.ipynb](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/Single_Call_Comparison_Explained.ipynb).

## Getting started

From the repository folder:

```bash
python -m pip install -r environment/requirements.txt
python -m notebook
```

Open a comparison notebook and run its cells in order. Training cells fit new models; saved outputs can be viewed directly on GitHub.

## Code

- `scripts/general_nn.py`: simulation, networks, training, and evaluation.
- `scripts/settings.py`: market, network, and experiment settings.
- `scripts/plots.py`: learning curves, risk comparisons, and P&L plots.
- `scripts/memory_tracker.py`: RAM tracking and isolated memory comparisons.
