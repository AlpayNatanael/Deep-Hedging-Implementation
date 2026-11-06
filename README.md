# Deep Hedging in the Heston Model

**How much information does a neural network need to hedge an option portfolio effectively?** This project studies neural hedging with transaction costs, working toward portfolio compression: representing a portfolio with less information while preserving hedging performance.

Our current experiments compare neural-network designs trained on fixed datasets or fresh simulated batches in Heston markets. Strategies trade stock alone or stock together with variance claims and are evaluated using CVaR, entropic risk, and profit-and-loss distributions. The implementation builds on [*Deep Hedging* by Buehler et al.](https://arxiv.org/abs/1802.03042).

**Preliminary results.** In our initial single-call experiment, our network achieves lower test CVaR than our paper-style implementation, both with stock-only trading and when both networks can trade a variance claim. These results are preliminary, with network design and training-data mode varied together.

## Experiments

| Notebook | Description |
|---|---|
| [Single_Call_Comparison.ipynb](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/Single_Call_Comparison.ipynb) | Hedge one call using stock alone or stock plus a variance claim. Compare neural strategies with classical Heston hedges. |
| [Large_Portfoloio_Comparison.ipynb](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/Large_Portfoloio_Comparison.ipynb) | Hedge a portfolio of calls on three simulated stocks and compare fixed-data and fresh-batch training. |

## Getting started

From the repository folder:

```bash
python -m pip install -r environment/requirements.txt
python -m notebook
```

Open a comparison notebook and run its cells in order. Running the training cells fits new models; saved outputs can be viewed directly on GitHub.

## Code

- `scripts/general_nn.py`: simulation, neural networks, training, and evaluation.
- `scripts/settings.py`: market, network, and experiment settings.
- `scripts/plots.py`: learning curves, risk comparisons, and P&L plots.
