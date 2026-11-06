"""Teaching figures for one HedgingExperiment or a matched list of experiments."""

import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECTIVES = ("CVaR", "Entropic")
COLORS = {"paper_full": "#286DA8", "paper_stock": "#C76B2C", "ours": "#29846B"}
MATCHED_SETTINGS = (
    "initial_price", "strike", "initial_variance", "long_run_variance",
    "mean_reversion", "vol_of_vol", "correlation", "number_of_days",
    "time_step", "simulation_substeps", "cost_rate", "variance_notional",
    "alpha", "risk_aversion",
)


def _experiments(value):
    experiments = list(value) if isinstance(value, (list, tuple)) else [value]
    if not experiments:
        raise ValueError("Pass an experiment or a nonempty list of experiments.")
    if len(experiments) > 1:
        reference = experiments[0]
        identity = reference.metadata.get("test_data_id")
        if identity is None:
            raise ValueError("Matched comparisons need evaluated experiments with a test_data_id.")
        for experiment in experiments[1:]:
            if experiment.metadata.get("test_data_id") != identity:
                raise ValueError("These experiments use different test paths; compare them on common test data first.")
            differences = [
                key for key in MATCHED_SETTINGS
                if getattr(experiment.settings, key) != getattr(reference.settings, key)
            ]
            if differences:
                raise ValueError(f"Matched comparisons need the same market, costs, and risk settings: {differences}")
            first_quotes = reference.metadata.get("instrument_data_ids", {})
            next_quotes = experiment.metadata.get("instrument_data_ids", {})
            for name in first_quotes.keys() & next_quotes.keys():
                if first_quotes[name] != next_quotes[name]:
                    raise ValueError(f"The common instrument {name} has different observations.")
            first_units = {item.name: (item.multiplier, item.cost_rate) for item in reference.settings.instruments}
            next_units = {item.name: (item.multiplier, item.cost_rate) for item in experiment.settings.instruments}
            if any(first_units[name] != next_units[name] for name in first_units.keys() & next_units.keys()):
                raise ValueError("Common instruments must have matching units and trading costs.")
    return experiments


def _runs(experiment, objective):
    runs = [run for run in experiment.runs if run["objective"].lower() == objective.lower()]
    if not runs:
        raise ValueError(f"No {objective} runs are available. Load saved results or fit the experiment first.")
    return sorted(runs, key=lambda run: run["replicate_id"])


def _replicate(experiment):
    return int(experiment.settings.display_replicate)


def _color(experiment, experiments):
    if sum(item.family == experiment.family for item in experiments) > 1:
        return plt.get_cmap("tab10")(experiments.index(experiment) % 10)
    return COLORS[experiment.family]


def _label(experiment, experiments):
    label = experiment.name
    same_name = [item for item in experiments if item.name == label]
    if len(same_name) > 1:
        mode = "fresh batches" if experiment.dynamic_data else "fixed pool"
        label += f" ({mode})"
        if sum(item.dynamic_data == experiment.dynamic_data for item in same_name) > 1:
            label += f" #{experiments.index(experiment) + 1}"
    return label


def _panels(experiments, height):
    objectives = [
        objective for objective in OBJECTIVES
        if all(any(run["objective"] == objective for run in experiment.runs)
               for experiment in experiments)
    ]
    if not objectives:
        raise ValueError("The experiments need at least one common fitted risk objective.")
    figure, axes = plt.subplots(1, len(objectives), figsize=(11.5, height), squeeze=False)
    return figure, axes.ravel(), objectives


def _objective_title(objective, settings):
    if objective == "CVaR":
        return f"CVaR: worst {100 * (1-settings.alpha):g}% of losses"
    return f"Entropic risk: $\\lambda={settings.risk_aversion:g}$"


def _finish(figure, axes, title, paragraph, subtitle=None, image=False):
    if not image:
        for axis in np.asarray(axes, dtype=object).reshape(-1):
            axis.grid(alpha=0.17)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(labelsize=9)

    caption = textwrap.fill(
        paragraph, width=140, break_long_words=False, break_on_hyphens=False,
    )
    bottom = ((len(caption.splitlines()) * 13 + 18) / 72) / figure.get_figheight() + 0.025
    top = 0.87 if subtitle else 0.91
    figure.suptitle(title, fontsize=14, fontweight="semibold", y=0.98)
    if subtitle:
        figure.text(0.5, 0.925, subtitle, ha="center", va="center", fontsize=9.5)
    figure.tight_layout(rect=(0.01, bottom, 0.99, top), w_pad=2.5)
    figure.text(0.025, 0.025, caption, ha="left", va="bottom", fontsize=9.5, linespacing=1.35)
    plt.show()
    plt.close(figure)


def show_published_figure(number):
    """Display original author artwork, with its original setting identified."""
    metadata = json.loads((PROJECT_ROOT / "assets" / "paper" / "metadata.json").read_text())
    entry = metadata["figures"][str(number)]
    artwork = plt.imread(PROJECT_ROOT / entry["image"])
    figure, axis = plt.subplots(figsize=(10.5, 7.2))
    axis.imshow(artwork)
    axis.axis("off")
    if int(number) == 2:
        paragraph = (
            "Blue is the authors' learned CVaR hedge; orange is their model hedge. The horizontal axis is profit and loss "
            r"after receiving the common premium $q\approx1.69$; values left of zero mean losses. "
            "Taller bars count more simulated outcomes. "
            "The similar histograms show similar outcomes in this example. Both hedges trade stock and a variance claim, "
            "with zero transaction costs. Our later graphs include costs and report loss before the premium, so their numerical levels have a different meaning."
        )
    elif int(number) == 11:
        paragraph = (
            "Red points show the authors' price premiums at five trading-cost levels; the blue line fits their relationship. "
            "Both axes are logarithmic. Higher trading costs require a larger price premium. The displayed slope 0.71 describes "
            "how the premium scales with costs, rather than how quickly a network learns. "
            r"The printed $p_0$ uses $q$ as the numerical reference. Our fixed-cost comparison does not estimate this slope."
        )
    else:
        paragraph = entry["interpretation_markdown"]
    _finish(
        figure, axis, entry["title"], paragraph,
        subtitle="Original artwork from Buehler et al., Deep Hedging (2018)", image=True,
    )


def plot_paths(experiments):
    """Explain the common stock scenarios and the call payment at maturity."""
    experiments = _experiments(experiments)
    experiment = experiments[0]
    settings = experiment.settings
    if experiment.metadata.get("evaluation_kind") == "historical" or not experiment.metadata.get("default_call_payoff", True):
        _plot_supplied_payments(experiment)
        return
    prices = np.asarray(experiment.arrays["example_prices"], dtype=float)
    stock = prices[:, :, 0] if prices.ndim == 3 else prices
    stock = stock[:6]
    days = np.arange(stock.shape[1])
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(stock)))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for path, color in zip(stock, colors):
        axes[0].plot(days, path, color=color, linewidth=1.6)
    axes[0].axhline(settings.strike, color="#68727D", linestyle="--", linewidth=1, label="Strike")
    axes[0].set(xlabel="Trading date", ylabel="Stock price", title="Six possible stock paths")
    axes[0].legend(fontsize=9)

    terminal = stock[:, -1]
    lower = min(float(stock.min()), settings.strike * 0.95)
    upper = max(float(stock.max()), settings.strike * 1.05)
    price_grid = np.linspace(lower, upper, 200)
    axes[1].plot(price_grid, np.maximum(price_grid-settings.strike, 0), color="#42515E", linewidth=1.7)
    axes[1].scatter(terminal, np.maximum(terminal-settings.strike, 0), c=colors, s=35, zorder=3)
    axes[1].set(xlabel="Stock price at maturity", ylabel="Option payment", title=r"The bill we must pay: $(S_T-K)^+$")
    paragraph = (
        "Each line is a possible future for the stock, not a forecast. Matching colored dots show the payment owed at the end "
        f"of those paths: zero below the strike $K={settings.strike:g}$, and the amount above the strike otherwise. "
        "A hedge buys and sells assets before these future prices are known, so its trading gains can help pay this bill. "
        "The comparison uses common market paths for all methods."
    )
    _finish(figure, axes, "What are we trying to hedge?", paragraph)


def _plot_supplied_payments(experiment):
    historical = experiment.metadata.get("evaluation_kind") == "historical"
    prices = np.asarray(experiment.arrays["example_prices"])
    payments = np.asarray(experiment.arrays["example_payoffs"])
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(prices)))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for index, color in enumerate(colors):
        axes[0].plot(np.arange(prices.shape[1]), prices[index, :, 0], color=color)
    axes[0].set(xlabel="Trading step within each episode", ylabel="First stock price",
                title="Observed historical episodes" if historical else "Example stock paths")
    axes[1].bar(np.arange(1, len(payments) + 1), payments, color=colors)
    axes[1].set(xlabel="Episode", ylabel="Terminal liability payment", title="Supplied payments for these episodes")
    paragraph = (
        "Colors match each first-stock path to that episode's supplied liability payment. "
        "The right panel uses the payments in the dataset; it does not reconstruct a call from the last stock price. "
        "A portfolio payment can depend on more than the first stock shown here. "
        + ("These are observed historical windows, evaluated with the already-fitted policy."
           if historical else "The same episode payments enter the loss calculation.")
    )
    _finish(figure, axes, "The market observations and the liability", paragraph)


def plot_learning(experiments):
    """Show continuation validation histories for one method or a matched list."""
    experiments = _experiments(experiments)
    settings = experiments[0].settings
    figure, axes, objectives = _panels(experiments, 4.9)
    logarithmic = []
    for axis, objective in zip(axes, objectives):
        values = []
        for experiment in experiments:
            histories = []
            for run in _runs(experiment, objective):
                history = {int(row["update"]): float(row["risk"]) for row in run["history"]}
                histories.append(history)
                updates = np.asarray(sorted(history))
                risks = np.asarray([history[update] for update in updates])
                values.extend(risks.tolist())
                axis.plot(updates, risks, color=_color(experiment, experiments), alpha=0.3, linewidth=0.8)
                chosen = int(run["best_update"])
                axis.scatter(chosen, history[chosen], s=28, facecolor="white", edgecolor=_color(experiment, experiments), zorder=4)
            common = np.asarray(sorted(set.intersection(*(set(history) for history in histories))))
            risks = np.asarray([[history[update] for update in common] for history in histories])
            axis.plot(common, np.median(risks, axis=0), color=_color(experiment, experiments), linewidth=2, label=_label(experiment, experiments))
        axis.set(xlabel="Updates after warm-up", ylabel="Validation risk", title=_objective_title(objective, settings))
        if min(values) > 0 and max(values) / min(values) > 6:
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            axis.yaxis.set_major_formatter(ScalarFormatter())
            axis.yaxis.set_minor_formatter(NullFormatter())
            axis.set_ylabel("Validation risk (logarithmic scale)")
            logarithmic.append(objective.lower())
        axis.legend(fontsize=8.5)

    paragraph = (
        "Lower values mean less risk on the validation paths. Thin curves are separate training runs; the thick curve is their median. "
        "Circles mark the checkpoints chosen for testing. The horizontal axis starts after the common CVaR warm-up, so it does not show learning "
        "from a random starting network. A flat curve means little further improvement here; test results must still check performance on new paths."
    )
    if logarithmic:
        paragraph += " The " + " and ".join(logarithmic) + " panel uses a logarithmic vertical scale to include large spikes."
    modes = {experiment.dynamic_data for experiment in experiments}
    data_note = "fresh paths per update" if modes == {True} else "resampling a fixed training pool" if modes == {False} else "different training-data modes"
    subtitle = f"{settings.warmup_updates:,} warm-up updates precede this plot; training data: {data_note}."
    _finish(figure, axes, "How does the network learn?", paragraph, subtitle)


def plot_losses(experiments):
    """Compare full-sample loss histograms with saved no-hedge and stock-delta baselines."""
    experiments = _experiments(experiments)
    reference = experiments[0]
    figure, axes, objectives = _panels(experiments, 4.9)
    no_hedge = np.asarray(reference.arrays["benchmark_loss_no_hedge"], dtype=float)
    delta = reference.arrays.get("benchmark_loss_stock_delta")
    if delta is not None:
        delta = np.asarray(delta, dtype=float)
    for axis, objective in zip(axes, objectives):
        losses = [np.asarray(experiment.losses(objective, _replicate(experiment)), dtype=float) for experiment in experiments]
        benchmarks = [no_hedge] if delta is None else [no_hedge, delta]
        all_values = np.concatenate(benchmarks + losses)
        bins = np.linspace(float(all_values.min()), float(all_values.max()), 81)
        axis.hist(no_hedge, bins=bins, weights=np.full(len(no_hedge), 100/len(no_hedge)), color="#D4D9DE", label="No hedge")
        if delta is not None:
            axis.hist(delta, bins=bins, weights=np.full(len(delta), 100/len(delta)), histtype="step", color="#626A73", linestyle="--", linewidth=1.4, label="Stock delta")
        for experiment, values in zip(experiments, losses):
            axis.hist(values, bins=bins, weights=np.full(len(values), 100/len(values)), histtype="step", color=_color(experiment, experiments), linewidth=1.8, label=_label(experiment, experiments))
        axis.set(xlabel="Loss before premium", ylabel="Test paths in each bin (%)", title=_objective_title(objective, reference.settings))
        axis.legend(fontsize=8.2)
    delta_note = "The dashed line is a stock-only delta hedge. " if delta is not None else ""
    paragraph = (
        "A taller bin means more paths have losses in that range. The gray no-hedge distribution is the option bill with no trading; "
        "its mass at zero comes from options expiring without a payment. "
        f"{delta_note}"
        "The learned policies try to reduce large losses, while paying trading costs. A narrow main distribution can still hide rare large losses; "
        "the later tail graph examines them directly. All losses here are measured before receiving the option premium."
    )
    if reference.metadata.get("evaluation_kind") == "historical" or not reference.metadata.get("default_call_payoff", True):
        paragraph = (
            "Each bin shows the percentage of episodes with losses in that range. The gray reference uses the supplied "
            "liability payments with no hedge. Learned-policy losses include trading gains, supplied asset cashflows, "
            "and the configured transaction costs, before any premium. Rare large losses may be hard to see in a histogram."
        )
    _finish(figure, axes, "What changes when we hedge?", paragraph, f"Preselected training replicate {_replicate(reference)}; all observed losses are included.")


def plot_risk(experiments):
    """Compare objective-specific test risks, including the common no-hedge baseline."""
    experiments = _experiments(experiments)
    reference = experiments[0]
    figure, axes, objectives = _panels(experiments, 5.2)
    for axis, objective in zip(axes, objectives):
        metric = objective.lower()
        baseline = float(reference.benchmarks["no_hedge"][metric])
        axis.scatter(0, baseline, marker="s", s=60, color="#87929D", zorder=3)
        axis.text(0, 0.97, f"{baseline:.3f}", transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=9, color="#626A73")
        all_values = [baseline]
        for index, experiment in enumerate(experiments, start=1):
            scores = np.asarray([run["metrics"][metric] for run in _runs(experiment, objective)], dtype=float)
            offsets = np.linspace(-0.07, 0.07, len(scores)) if len(scores) > 1 else np.zeros(1)
            axis.scatter(index+offsets, scores, s=27, color=_color(experiment, experiments), zorder=3)
            axis.scatter(index, scores.mean(), marker="D", s=70, color=_color(experiment, experiments), edgecolor="white", zorder=4)
            axis.text(index, 0.97, f"Mean: {scores.mean():.3f}", transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color=_color(experiment, experiments))
            all_values.extend(scores.tolist())
        span = max(float(np.ptp(all_values)), 0.2)
        axis.set_ylim(min(all_values)-0.10*span, max(all_values)+0.23*span)
        axis.set_xlim(-0.4, len(experiments)+0.4)
        labels = [textwrap.fill(_label(experiment, experiments), width=19) for experiment in experiments]
        axis.set_xticks(np.arange(len(experiments)+1), labels=["No hedge"] + labels)
        axis.set(ylabel="Test risk before premium", title=_objective_title(objective, reference.settings))

    paragraph = (
        "Lower is better within each panel. The gray square is no hedging. Small colored dots are separate training runs; diamonds and labels show means. "
        "Each column uses policies trained for that panel's risk objective. We evaluate them on the same untouched test paths. "
    )
    by_family = {experiment.family: experiment for experiment in experiments}
    if {"paper_full", "paper_stock", "ours"} <= set(by_family) and len(objectives) == 2:
        full = by_family["paper_full"]
        ours = by_family["ours"]
        full_mean = np.mean([run["metrics"]["cvar"] for run in _runs(full, "CVaR")])
        ours_mean = np.mean([run["metrics"]["cvar"] for run in _runs(ours, "CVaR")])
        er = [run["metrics"]["entropic"] for run in _runs(full, "Entropic")]
        paragraph += (
            f"Here the two-asset method has mean CVaR {full_mean:.3f}, versus {ours_mean:.3f} for ours, but its entropic scores range "
            f"from {min(er):.3f} to {max(er):.3f}. The variance-trading control and the stock-only implementation comparison answer different questions."
        )
    else:
        paragraph += "This measures the performance of these fitted policies; a small average does not establish reliable control of every rare loss."
    subtitle = "Matched market, costs, risk definitions, and test scenarios."
    if reference.metadata.get("evaluation_kind") == "historical":
        paragraph = (
            "Lower is better within each risk panel. Dots are the previously trained policies and diamonds are their mean scores. "
            "All are evaluated on the same supplied historical episodes, using their recorded liability payments. "
            "Weights and normalization remain frozen. These finite-sample scores depend on the selected historical windows and accounting assumptions."
        )
        subtitle = "Historical scenario evaluation; supplied observations and configured trading costs."
        for axis in axes:
            axis.set_ylabel("Historical risk before premium")
    _finish(figure, axes, "Which saved hedge has less risk?", paragraph, subtitle)


def plot_loss_tails(experiments, threshold=5.0):
    """Show the probability of exceeding a loss, with no-hedge reference curves."""
    experiments = _experiments(experiments)
    reference = experiments[0]
    figure, axes, objectives = _panels(experiments, 5.0)
    sample_size = len(np.asarray(reference.arrays["benchmark_loss_no_hedge"]))
    no_hedge = np.asarray(reference.arrays["benchmark_loss_no_hedge"], dtype=float)
    for axis, objective in zip(axes, objectives):
        values, indices = np.unique(np.sort(no_hedge), return_index=True)
        axis.step(values, 100*(sample_size-indices)/sample_size, where="pre", color="#9CA5AE", linewidth=1.2, linestyle="--", label="No hedge")
        for experiment in experiments:
            losses = np.asarray(experiment.losses(objective, _replicate(experiment)), dtype=float)
            values, indices = np.unique(np.sort(losses), return_index=True)
            axis.step(values, 100*(len(losses)-indices)/len(losses), where="pre", color=_color(experiment, experiments), linewidth=1.7, label=_label(experiment, experiments))
        axis.axvline(threshold, color="#8E98A2", linestyle=":", linewidth=1)
        axis.set_yscale("log")
        ticks = [value for value in [0.001, 0.01, 0.1, 1, 10, 100] if value >= 100/sample_size]
        axis.set_yticks(ticks, labels=[f"{value:g}" for value in ticks])
        axis.set_ylim(100/(1.3*sample_size), 125)
        axis.set(xlabel="Loss threshold before premium", ylabel="Paths with at least this loss (%)", title=_objective_title(objective, reference.settings))
        axis.legend(fontsize=8.2)

    paragraph = (
        "Choose a loss on the horizontal axis, then read the percentage of paths that lose at least that much. "
        "The vertical scale is logarithmic, so very rare outcomes stay visible. A lower curve at one threshold need not mean lower risk at every threshold. "
    )
    full = next((experiment for experiment in experiments if experiment.family == "paper_full"), None)
    if full is not None and "Entropic" in objectives and len(full.settings.instruments) == 2:
        run = full.get_run("Entropic", _replicate(full))
        largest = run["metrics"]["max_loss"]
        weight = 100 * run["metrics"]["entropic_max_weight"]
        counts = [int(np.count_nonzero(np.asarray(experiment.losses("Entropic", _replicate(experiment))) >= threshold)) for experiment in experiments]
        paragraph += (
            f"At {threshold:g}, entropic-panel counts are " + " / ".join(f"{count:,}" for count in counts)
            + " in colored-legend order. "
            f"The two-asset policy's largest hedging loss is {largest:.3f}; one such path contributes {weight:.1f}% of its exponential sum. "
        )
    paragraph += "Each curve ends at its sample maximum, which is not a bound on future losses. "
    paragraph += "The displayed training replicate was chosen before examining test performance."
    subtitle = f"Preselected replicate {_replicate(reference)}; {sample_size:,} test paths; losses before premium."
    if reference.metadata.get("evaluation_kind") == "historical":
        paragraph = (
            "At a chosen horizontal threshold, read the percentage of supplied historical episodes with at least that loss. "
            "The logarithmic vertical scale reveals rare losses. Each curve ends at its observed sample maximum, "
            "which is not a bound on future loss. Scores use the configured frozen training replicate and no premium."
        )
        subtitle = f"Historical evaluation; configured replicate {_replicate(reference)}; {sample_size:,} episodes."
    _finish(figure, axes, "Why can a few bad hedges matter so much?", paragraph, subtitle)
