# Deep Hedging with the Heston Model and Data-Driven Strategies

## Project Goal

This project implements and extends the deep hedging methodology outlined in [*Deep Hedging*](https://arxiv.org/abs/1802.03042) by Buehler et al. (2018), specifically incorporating the Heston stochastic volatility model. The project aims to:

- **Reproduce** the original deep hedging approach using pre-generated training data
- **Develop** a modified, data-driven approach in which stock price paths are simulated dynamically at each training iteration using the Heston model, rather than being fixed in advance
- **Compare** the performance of the traditional shared-weights neural network with the step-specific, parameterised networks described in the paper, where each time step $k$ has its own independent parameter set $\theta_k$:

$$\delta_k^\theta = F^{\theta_k}(I_k,\, \delta_{k-1}^\theta)$$

- **Analyse** differences in risk metrics (CVaR, entropic risk) and overall hedging effectiveness between the two strategies

---

## Directory Structure

```
.
├── figures/
├── scripts/
│   ├── deep_hedging_original.py
│   └── deep_hedging_step_specific.py
└── deep_hedging_experiment.ipynb
```

### `figures/`
Contains images and plots generated during experiments, illustrating the overall performance comparisons between models and strategies.

### `scripts/`
A folder housing two Python class files:

- **`deep_hedging_original.py`**: Implements the original deep hedging approach with pre-generated training data and shared neural network weights across all time steps.
- **`deep_hedging_step_specific.py`**: Implements the modified approach where stock price paths are generated dynamically at each training iteration using the Heston model, and separate neural networks with independent weights are used for each trading step, as described in the paper.

### `deep_hedging_experiment.ipynb`
A Jupyter notebook serving as a sandbox environment to:
- Generate figures and performance metrics comparing the two strategies
- Test and compare the two class implementations side by side
- Visualise results including CVaR, entropic risk, and P&L distributions

---

## References

- Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2018). [Deep Hedging](https://arxiv.org/abs/1802.03042). *arXiv:1802.03042*
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility. *The Review of Financial Studies*, 6(2), 327–343.
- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21–41.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *Journal of Computational Finance*, 11(3).
