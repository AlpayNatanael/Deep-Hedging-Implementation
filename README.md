# Deep Hedging in the Heston Model

**Can we preserve hedging performance while reducing training memory and simulation time?** We study neural hedging with transaction costs, comparing training from a stored dataset with training on fresh simulated batches.

Our longer-term goal is portfolio compression: representing an option portfolio with less information while preserving hedging performance. This implementation builds on [*Deep Hedging* by Buehler et al.](https://arxiv.org/abs/1802.03042).

## Our contributions

- **Training without a large stored dataset:** generate Heston paths as needed for each training batch.
- **Matched comparisons:** compare fixed-data and fresh-batch training using identical network architectures and common test scenarios in a three-stock experiment.
- **Isolated performance and memory evaluation:** measure CVaR, entropic risk, and sampled CPU RAM in fresh training processes, with saved reports and comparison plots.
- **Accelerated path generation — prototype:** generate independent path blocks concurrently and prepare upcoming batches through a bounded queue, preserving the existing precision and simulation scheme.

## Preliminary results

**Hedging performance.** In our initial single-call experiment, our networks achieve lower test CVaR than our paper-style reconstruction, both with stock-only trading and with a variance claim. Architecture and training-data mode change together in this experiment. The full Heston benchmark remains stronger in the saved stock-plus-variance comparison.

**Training memory.** The committed three-stock memory comparison measured **78.1% lower sampled peak CPU RAM** with fresh batches compared with our in-memory fixed pool. Both configurations use identical networks and achieve closely matched observed CVaR. Fresh-batch training took **70.5% longer** in this run. These measurements cover data preparation, training, and test evaluation after simulator warm-up.

**Path-generation speed.** For equal path counts, a standalone six-worker CPU prototype generated fresh batches approximately **4.6× faster than the original sequential pool generator**. Fresh-batch creation was approximately **12% slower than accelerated bulk creation**, while using a small, bounded data buffer. These elapsed-time improvements use additional CPU capacity.

The memory and path-generation results were measured separately. **Combining the accelerated producer with neural training remains ongoing work; full-training runtime parity has not yet been established.** Training results currently use one replicate per configuration, and repeated runs are needed to assess reliability.

## Experiments

| Notebook | Description |
|---|---|
| [Single_Call_Comparison.ipynb](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/Single_Call_Comparison.ipynb) | Hedge one call using stock alone or stock plus a variance claim. Compare neural strategies with classical Heston hedges. |
| [Large_Portfoloio_Comparison.ipynb](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/Large_Portfoloio_Comparison.ipynb) | Hedge three calls on three simulated stocks. Compare fixed-data and fresh-batch training with the same network architecture. |
| [Large_Portfolio_Memory_Comparison.ipynb](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/Large_Portfolio_Memory_Comparison.ipynb) | Measure fixed-pool and fresh-batch training in separate CPU processes, then compare RAM, runtime, and test risk. |

For a guided walkthrough, see [Single_Call_Comparison_Explained.ipynb](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/blob/main/Single_Call_Comparison_Explained.ipynb).

## Getting started

From the repository folder:

```bash
python -m pip install -r environment/requirements.txt
python -m notebook
```

Open a comparison notebook and run its cells in order. Training cells fit new models; saved outputs can be viewed directly on GitHub.

**Run All in the memory-comparison notebook trains both configurations sequentially in separate processes.** Its [saved reports and plot](https://github.com/AlpayNatanael/Deep-Hedging-Implementation/tree/main/results/portfolio_memory_comparison) can be inspected without retraining.

## Code

- `scripts/general_nn.py`: simulation, networks, training, and evaluation.
- `scripts/settings.py`: market, network, and experiment settings.
- `scripts/plots.py`: learning curves, risk comparisons, and P&L plots.
- `scripts/memory_tracker.py`: CPU RAM tracking with `track_memory` and isolated training benchmarks with `run_memory_experiment`.
