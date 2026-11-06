"""Readable, high-resolution plots for generalNN experiments.

Existing calls still work. Each risk objective gets its own full-size figure.
Use objective="CVaR" or objective="Entropic" to display just one objective.
Use explain=True for a separate interpretation below the figure, and
save_dir="figures" to export 300-dpi PNGs and vector SVGs.
No data, scores, model weights, or training settings are changed.
"""

import io
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECTIVES = ("CVaR", "Entropic")
MATCHED_SETTINGS = (
    "initial_price", "strike", "initial_variance", "long_run_variance",
    "mean_reversion", "vol_of_vol", "correlation", "number_of_days",
    "time_step", "simulation_substeps", "cost_rate", "variance_notional",
    "alpha", "risk_aversion",
)
DEFAULT_SIZE = (9, 6)
DEFAULT_DPI = 200


def _experiments(value, match_test=True):
    experiments = list(value) if isinstance(value, (list, tuple)) else [value]
    if not experiments:
        raise ValueError("Pass an experiment or a nonempty list of experiments.")
    if len(experiments) == 1:
        return experiments

    reference = experiments[0]
    for experiment in experiments[1:]:
        differences = [
            key for key in MATCHED_SETTINGS
            if getattr(experiment.settings, key) != getattr(reference.settings, key)
        ]
        if differences:
            raise ValueError("Different comparison settings: " + ", ".join(differences))
        if not match_test:
            continue
        identity = reference.metadata.get("test_data_id")
        if identity is None or experiment.metadata.get("test_data_id") != identity:
            raise ValueError("Evaluate the experiments on common test paths first.")
        first_quotes = reference.metadata.get("instrument_data_ids", {})
        next_quotes = experiment.metadata.get("instrument_data_ids", {})
        for name in first_quotes.keys() & next_quotes.keys():
            if first_quotes[name] != next_quotes[name]:
                raise ValueError("Different observed quotes for " + name)
        first_units = {
            item.name: (item.multiplier, item.cost_rate)
            for item in reference.settings.instruments
        }
        next_units = {
            item.name: (item.multiplier, item.cost_rate)
            for item in experiment.settings.instruments
        }
        for name in first_units.keys() & next_units.keys():
            if first_units[name] != next_units[name]:
                raise ValueError("Different units or trading costs for " + name)
    return experiments


def _objectives(experiments, objective):
    common = [
        name for name in OBJECTIVES
        if all(any(run["objective"] == name for run in item.runs) for item in experiments)
    ]
    if objective is not None:
        names = {name.lower(): name for name in OBJECTIVES}
        selected = names.get(str(objective).lower())
        if selected not in common:
            raise ValueError("Choose an available objective: " + ", ".join(common))
        return [selected]
    if not common:
        raise ValueError("No common fitted objective is available.")
    return common


def _runs(experiment, objective):
    runs = [run for run in experiment.runs if run["objective"] == objective]
    return sorted(runs, key=lambda run: run["replicate_id"])


def _replicate(experiment, replicate):
    return int(experiment.settings.display_replicate if replicate is None else replicate)


def _label(experiment, experiments):
    label = experiment.name
    matches = [item for item in experiments if item.name == label]
    if len(matches) > 1:
        mode = "fresh batches" if experiment.dynamic_data else "fixed pool"
        label += " (" + mode + ")"
        if sum(item.dynamic_data == experiment.dynamic_data for item in matches) > 1:
            label += " #" + str(experiments.index(experiment) + 1)
    return label


def _criterion(objective, settings):
    if objective == "CVaR":
        return f"CVaR: worst {100 * (1 - settings.alpha):g}% of losses"
    return rf"Entropic risk: $\lambda={settings.risk_aversion:g}$"


def _array(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty, finite, one-dimensional loss array.")
    return values


def _figure(title, xlabel, ylabel, dpi, figsize):
    figure = plt.figure(figsize=figsize, dpi=dpi)
    axis = figure.add_subplot(111)
    axis.set_title(title, fontsize=15, pad=18)
    axis.set_xlabel(xlabel, fontsize=13, labelpad=9)
    axis.set_ylabel(ylabel, fontsize=13, labelpad=9)
    axis.tick_params(labelsize=11)
    axis.grid(axis="y", alpha=0.18)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    return figure, axis


def _legend(axis):
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.20),
            ncol=2 if len(handles) > 1 else 1, frameon=False, fontsize=10.5,
            handlelength=2.8, columnspacing=1.6,
        )


def _finish(figure, axis, filename, explanation, show, explain, save_dir):
    """Use vector display in Jupyter; keep interpretation out of the image."""
    _legend(axis)
    figure.tight_layout(pad=1.3)
    if save_dir is not None:
        folder = Path(save_dir)
        folder.mkdir(parents=True, exist_ok=True)
        figure.savefig(folder / (filename + ".png"), dpi=300, bbox_inches="tight")
        figure.savefig(folder / (filename + ".svg"), format="svg", bbox_inches="tight")
    if show:
        try:
            from IPython import get_ipython
            from IPython.display import Markdown, SVG, display
            in_notebook = get_ipython() is not None and hasattr(get_ipython(), "kernel")
        except ImportError:
            in_notebook = False
        if in_notebook:
            buffer = io.StringIO()
            figure.savefig(buffer, format="svg", bbox_inches="tight")
            display(SVG(buffer.getvalue()))
            if explain:
                display(Markdown("**How to read this graph.** " + explanation))
        else:
            plt.show()
            if explain:
                print(explanation)
    plt.close(figure)
    return figure, axis


def _filename(kind, experiments, objective=""):
    families = "_".join(str(item.family) for item in experiments)
    return re.sub(r"[^a-z0-9_-]+", "_", f"{kind}_{families}_{objective}".lower()).strip("_")


def plot_learning(experiments, objective=None, replicate=None, *, dpi=DEFAULT_DPI,
                  figsize=DEFAULT_SIZE, show=True, explain=False, save_dir=None):
    """Plot validation histories, one full-size figure per objective.

    By default, show every run and its pointwise median. Set replicate=0
    to display only run zero. The recorded checkpoints are not smoothed.
    With show=False, return a dictionary of (figure, axis) pairs.
    """
    experiments = _experiments(experiments, match_test=False)
    output = {}
    for name in _objectives(experiments, objective):
        figure, axis = _figure(
            "How does the network learn?\n" + _criterion(name, experiments[0].settings),
            "Updates after warm-up", "Validation risk (lower is better)", dpi, figsize,
        )
        values = []
        for experiment in experiments:
            runs = _runs(experiment, name)
            if replicate is not None:
                runs = [experiment.get_run(name, int(replicate))]
            histories = [
                {int(row["update"]): float(row["risk"]) for row in run["history"]}
                for run in runs
            ]
            if any(not history for history in histories):
                raise ValueError("A run has an empty validation history.")
            common = sorted(set.intersection(*(set(history) for history in histories)))
            if not common:
                raise ValueError("The runs have no common recorded update.")
            matrix = np.array([[history[step] for step in common] for history in histories])
            median, = axis.plot(
                common, np.median(matrix, axis=0), linewidth=2.4,
                marker="o" if len(common) <= 15 else None, markersize=4,
                label=_label(experiment, experiments),
            )
            for run, history in zip(runs, histories):
                updates = sorted(history)
                risks = [history[step] for step in updates]
                values.extend(risks)
                if len(runs) > 1:
                    axis.plot(updates, risks, color=median.get_color(), alpha=0.28, linewidth=1)
                chosen = int(run["best_update"])
                axis.scatter(
                    chosen, history[chosen], s=80, facecolors="none",
                    edgecolors=median.get_color(), linewidths=1.7, zorder=5,
                )
        if min(values) > 0 and max(values) / min(values) > 6:
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            axis.yaxis.set_major_formatter(ScalarFormatter())
            axis.yaxis.set_minor_formatter(NullFormatter())
            axis.set_ylabel("Validation risk (log scale; lower is better)", fontsize=13)
        explanation = (
            "The horizontal axis counts optimizer updates after the CVaR warm-up, not market days. "
            "Lower validation risk is better. With multiple runs, faint lines show individual runs "
            "and the bold line is their median at common recorded checkpoints. With one run, "
            "the bold line is that run. Open circles mark validation-selected checkpoints. "
            "Straight segments join recorded observations; there is no smoothing. "
            "Warm-up updates are not displayed. CVaR and entropic risk use different criteria."
        )
        if axis.get_yscale() == "log":
            explanation += " The vertical axis is logarithmic to retain large validation spikes."
        output[name] = _finish(
            figure, axis, _filename("learning", experiments, name),
            explanation, show, explain, save_dir,
        )
    return output if not show else None


def plot_learning_simple(experiment, objective="CVaR", replicate=0, dpi=DEFAULT_DPI):
    """Backward-compatible single-run version of the earlier helper."""
    result = plot_learning(
        experiment, objective=objective, replicate=replicate, dpi=dpi, show=False,
    )
    figure, axis = next(iter(result.values()))
    _finish(figure, axis, "learning_simple", "", True, False, None)
    return figure, axis


def plot_losses(experiments, objective=None, replicate=None, *, bins=80,
                dpi=DEFAULT_DPI, figsize=DEFAULT_SIZE, show=True, explain=False, save_dir=None):
    """Plot full-range loss histograms; no tail observations are removed."""
    experiments = _experiments(experiments)
    reference = experiments[0]
    no_hedge = _array(reference.arrays["benchmark_loss_no_hedge"])
    output = {}
    for name in _objectives(experiments, objective):
        series = [
            (_label(item, experiments), _array(item.losses(name, _replicate(item, replicate))))
            for item in experiments
        ]
        series.append(("No hedge", no_hedge))
        if "benchmark_loss_stock_delta" in reference.arrays:
            series.append(("Stock delta", _array(reference.arrays["benchmark_loss_stock_delta"])))
        edges = np.histogram_bin_edges(np.concatenate([v for _, v in series]), bins=bins)
        figure, axis = _figure(
            "What changes when we hedge?\n" + _criterion(name, reference.settings),
            "Terminal loss before premium", "Scenarios in each bin (%)", dpi, figsize,
        )
        for index, (label, values) in enumerate(series):
            axis.hist(
                values, bins=edges, weights=np.full(len(values), 100 / len(values)),
                histtype="step", linewidth=2.0,
                linestyle="-" if index < len(experiments) else "--", label=label,
            )
        explanation = (
            "The horizontal axis is the terminal loss before an option premium. "
            "Each bin shows a percentage of scenarios, not a density. Curves share the same bins, "
            "and all losses, including the extremes, are included. Dashed curves are benchmarks. "
            "A narrow central distribution can still have rare large losses; use plot_loss_tails "
            "to see their frequencies. This shows the configured display run (or the requested "
            "replicate), not an average of separately trained policies."
        )
        output[name] = _finish(
            figure, axis, _filename("losses", experiments, name), explanation, show, explain, save_dir,
        )
    return output if not show else None


def plot_risk(experiments, objective=None, *, dpi=DEFAULT_DPI, figsize=DEFAULT_SIZE,
              show=True, explain=False, save_dir=None):
    """Compare objective-specific test risks and variation across training runs."""
    experiments = _experiments(experiments)
    reference = experiments[0]
    output = {}
    for name in _objectives(experiments, objective):
        figure, axis = _figure(
            "Which hedge has less risk?\n" + _criterion(name, reference.settings),
            "", "Measured risk before premium (lower is better)", dpi, figsize,
        )
        metric = name.lower()
        all_values = []
        for index, experiment in enumerate(experiments):
            scores = np.array([run["metrics"][metric] for run in _runs(experiment, name)])
            offsets = np.linspace(-0.08, 0.08, len(scores)) if len(scores) > 1 else [0]
            points = axis.scatter(
                index + np.asarray(offsets), scores, s=45, zorder=4,
                label="Individual training run" if index == 0 else None,
            )
            axis.scatter(
                index, scores.mean(), s=115, marker="D", facecolors="none",
                edgecolors=points.get_facecolor(), linewidths=2, zorder=5,
                label="Mean across runs" if index == 0 else None,
            )
            axis.annotate(
                f"Mean {scores.mean():.3f}", (index, scores.mean()),
                xytext=(0, 13), textcoords="offset points", ha="center", fontsize=11,
            )
            all_values.extend(scores.tolist())
        baseline = float(reference.benchmarks["no_hedge"][metric])
        position = len(experiments)
        axis.scatter(position, baseline, s=65, marker="s", zorder=4)
        axis.annotate(
            f"{baseline:.3f}", (position, baseline), xytext=(0, 13),
            textcoords="offset points", ha="center", fontsize=11,
        )
        all_values.append(baseline)
        span = max(float(np.ptp(all_values)), 0.2)
        axis.set_ylim(min(all_values) - 0.13 * span, max(all_values) + 0.25 * span)
        axis.set_xlim(-0.5, position + 0.5)
        labels = [_label(item, experiments).replace(": ", ":\n") for item in experiments]
        axis.set_xticks(range(position + 1))
        axis.set_xticklabels(labels + ["No hedge"], fontsize=10.5)
        explanation = (
            "Lower is better. Small dots are individual training runs, open diamonds are their "
            "arithmetic mean, and the square is no hedging. All policies shown here were trained "
            "for the named objective and evaluated on common scenarios. Variation across training "
            "runs is not a confidence interval. The mean of their risk scores is not the risk of "
            "an averaged or ensemble trading policy. Different risk objectives have separate plots."
        )
        output[name] = _finish(
            figure, axis, _filename("risk", experiments, name), explanation, show, explain, save_dir,
        )
    return output if not show else None


def plot_loss_tails(experiments, threshold=5.0, *, objective=None, replicate=None,
                    dpi=DEFAULT_DPI, figsize=DEFAULT_SIZE, show=True, explain=False, save_dir=None):
    """Plot empirical P(loss >= threshold) on a logarithmic percentage axis.

    Each objective has a separate full-size figure. All sample extremes are
    retained. Open endpoint circles mark the observed maxima; no smoothing,
    binning, truncation, or extrapolation is used.
    """
    experiments = _experiments(experiments)
    reference = experiments[0]
    no_hedge = _array(reference.arrays["benchmark_loss_no_hedge"])
    output = {}
    for name in _objectives(experiments, objective):
        series = [
            (_label(item, experiments), _array(item.losses(name, _replicate(item, replicate))))
            for item in experiments
        ]
        series.append(("No hedge", no_hedge))
        if any(len(values) != len(no_hedge) for _, values in series):
            raise ValueError("Tail comparisons require the same number of evaluation scenarios.")
        figure, axis = _figure(
            "Why can a few bad hedges matter so much?\n" + _criterion(name, reference.settings),
            "Loss threshold before premium", "Scenarios with at least this loss (%)", dpi, figsize,
        )
        for index, (label, losses) in enumerate(series):
            values, first_indices = np.unique(np.sort(losses), return_index=True)
            percentages = 100 * (len(losses) - first_indices) / len(losses)
            line, = axis.step(
                values, percentages, where="pre", linewidth=2.2,
                linestyle="--" if index == len(experiments) else "-", label=label,
            )
            axis.scatter(
                values[-1], percentages[-1], s=50, facecolors="none",
                edgecolors=line.get_color(), linewidths=1.5, zorder=4,
            )
        one_path = 100 / len(no_hedge)
        axis.set_yscale("log")
        ticks = [10.0 ** power for power in range(-10, 3) if one_path <= 10.0 ** power <= 100]
        axis.set_yticks(ticks)
        axis.set_yticklabels([f"{tick:g}%" for tick in ticks])
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_ylim(one_path / 1.8, 125)
        axis.axhline(one_path, linestyle=":", linewidth=1, alpha=0.45)
        axis.text(
            0.99, one_path / 1.12, f"1 of {len(no_hedge):,} scenarios = {one_path:.4g}%",
            transform=axis.get_yaxis_transform(), ha="right", va="top", fontsize=9,
        )
        if threshold is not None:
            if not np.isfinite(threshold):
                raise ValueError("threshold must be finite or None.")
            axis.axvline(threshold, linestyle=":", linewidth=1, alpha=0.6)
            axis.text(
                threshold, 0.98, f"  Threshold = {threshold:g}",
                transform=axis.get_xaxis_transform(), ha="left", va="top", fontsize=9,
            )
        explanation = (
            "Choose a loss threshold on the horizontal axis, then read the percentage of "
            "evaluation scenarios with loss at least that large. A lower curve at that threshold "
            "means fewer exceedances. The vertical axis is logarithmic: adjacent decade ticks "
            "represent a factor of ten. Each open endpoint is the largest loss in that sample, "
            "not a bound on future loss. The dotted horizontal line marks one scenario. "
            "A horizontal tail is an interval with no intervening observed losses, not missing data. "
            "All sample extremes are shown, without smoothing or interpolation of probabilities. "
            "These are scenario frequencies, not the exponential weights used by entropic risk. "
            "The figure uses the configured display replicate (or the requested replicate), "
            "not a pooled sample of training runs."
        )
        if name == "Entropic":
            details = []
            lam = reference.settings.risk_aversion
            for label, losses in series[:-1]:
                weights = np.exp(lam * (losses - losses.max()))
                largest_weight = 100 / weights.sum()
                details.append(
                    f"{label}: maximum loss {losses.max():.3f}; one maximum-loss scenario "
                    f"has {largest_weight:.1f}% of the exponential weight"
                )
            explanation += " For the displayed runs, " + "; ".join(details) + "."
        output[name] = _finish(
            figure, axis, _filename("tails", experiments, name), explanation, show, explain, save_dir,
        )
    return output if not show else None


def plot_paths(experiments, *, dpi=DEFAULT_DPI, figsize=DEFAULT_SIZE,
               show=True, explain=False, save_dir=None):
    """Display example stock paths, then the corresponding liability payments."""
    experiments = _experiments(experiments)
    experiment = experiments[0]
    prices = np.asarray(experiment.arrays["example_prices"], dtype=float)
    stock = prices[:, :, 0] if prices.ndim == 3 else prices
    stock = stock[:6]
    supplied = (
        experiment.metadata.get("evaluation_kind") == "historical"
        or not experiment.metadata.get("default_call_payoff", True)
    )
    figure, axis = _figure(
        "Example stock paths", "Trading step", "First stock price", dpi, figsize,
    )
    line_colors = []
    for index, path in enumerate(stock, start=1):
        line, = axis.plot(np.arange(len(path)), path, linewidth=1.8, label=f"Path {index}")
        line_colors.append(line.get_color())
    if not supplied:
        axis.axhline(experiment.settings.strike, linestyle="--", linewidth=1.3, label="Strike")
    explanation = (
        "Each line follows the first stock in one stored evaluation scenario. "
        "The plotted scenarios are examples, not forecasts or independent training runs. "
        "For a multi-stock portfolio this is only its first stock."
    )
    output = {"paths": _finish(
        figure, axis, _filename("paths", experiments), explanation, show, explain, save_dir,
    )}
    if supplied:
        payments = np.asarray(experiment.arrays["example_payoffs"], dtype=float)[:len(stock)]
        figure, axis = _figure(
            "Liability payments for the displayed paths", "Path", "Supplied liability payment",
            dpi, figsize,
        )
        axis.bar(np.arange(1, len(payments) + 1), payments, color=line_colors)
        axis.set_xticks(np.arange(1, len(payments) + 1))
        explanation = (
            "Each bar is the supplied payment for the correspondingly numbered path. "
            "Colors match the stock-path graph. Payments are read from the evaluation data; "
            "they are not reconstructed from the first stock."
        )
    else:
        strike = experiment.settings.strike
        terminal = stock[:, -1]
        grid = np.linspace(min(stock.min(), strike * 0.95), max(stock.max(), strike * 1.05), 200)
        figure, axis = _figure(
            r"The call payment we owe: $(S_T-K)^+$", "Stock price at maturity", "Call payment",
            dpi, figsize,
        )
        axis.plot(grid, np.maximum(grid - strike, 0), linewidth=1.8, linestyle="--")
        for value, color in zip(terminal, line_colors):
            axis.scatter(value, max(value - strike, 0), s=70, color=color, zorder=3)
        explanation = (
            f"The call pays zero below the strike {strike:g}, and the excess above it otherwise. "
            "The dots use the same path colors as the preceding stock-path graph. "
            "This is a liability payment, before trading gains, costs, or an option premium."
        )
    output["payoffs"] = _finish(
        figure, axis, _filename("payoffs", experiments), explanation, show, explain, save_dir,
    )
    return output if not show else None


def show_published_figure(number, *, dpi=DEFAULT_DPI, figsize=(10, 6),
                          show=True, explain=False, save_dir=None):
    """Show original paper artwork, using its original SVG in Jupyter."""
    metadata = json.loads((PROJECT_ROOT / "assets" / "paper" / "metadata.json").read_text())
    entry = metadata["figures"][str(number)]
    image_path = PROJECT_ROOT / entry["image"]
    svg_path = PROJECT_ROOT / entry.get("original_svg", "missing.svg")
    if save_dir is not None:
        import shutil
        folder = Path(save_dir)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path, folder / f"paper_figure_{int(number):02d}.png")
        if svg_path.is_file():
            shutil.copyfile(svg_path, folder / f"paper_figure_{int(number):02d}.svg")
    if show and svg_path.is_file():
        try:
            from IPython import get_ipython
            from IPython.display import Markdown, SVG, display
            if get_ipython() is not None and hasattr(get_ipython(), "kernel"):
                display(Markdown("### " + entry["title"]))
                display(SVG(filename=str(svg_path)))
                if explain:
                    note = entry["interpretation_markdown"] + " " + entry["comparison_note_markdown"]
                    display(Markdown(note))
                return None
        except ImportError:
            pass
    figure = plt.figure(figsize=figsize, dpi=dpi)
    axis = figure.add_subplot(111)
    axis.imshow(plt.imread(image_path))
    axis.axis("off")
    axis.set_title(entry["title"], fontsize=14, pad=16)
    pair = _finish(
        figure, axis, f"paper_display_{number}", entry["interpretation_markdown"],
        show, explain, None,
    )
    return {"published": pair} if not show else None
