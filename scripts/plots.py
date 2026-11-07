"""Readable, high-resolution plots for generalNN experiments.

Existing calls still work. Each risk objective gets its own full-size figure.
Use objective="CVaR" or objective="Entropic" to display just one objective.
Use explain=True for a separate interpretation below the figure, and
save_dir="figures" to export 300-dpi PNGs and vector SVGs.
Use plot_results_grid to arrange learning, risk and loss tails in one figure.
No data, scores, model weights, or training settings are changed.

All eleven public plotting functions are available:
    plot_learning, plot_learning_simple, plot_losses, plot_pnl_histogram,
    plot_pnl_comparison, plot_risk, plot_loss_tails, plot_paths,
    show_published_figure, plot_memory_comparison, and plot_results_grid.

The first seven and plot_results_grid accept objective="CVaR" or objective="Entropic".
Omitting objective displays all common fitted objectives, except that
plot_learning_simple and plot_pnl_comparison default to CVaR.
plot_pnl_comparison requires one common premium and shows full and central views. Objective names are
case-insensitive. Market paths and published figures do not depend on an
objective, so plot_paths and show_published_figure do not take that argument.

Example notebook calls after installing this file as scripts/plots.py:
    from scripts import plots
    plots.plot_risk([ours, ours_variance], objective="CVaR")
    plots.plot_loss_tails([ours, ours_variance], objective="Entropic")
    plots.plot_results_grid([paper, ours])

If plots was already imported, run importlib.reload(plots) after replacing
the file. Use print(plots.__file__) to check which file the notebook imports.
"""

# Functions beginning with an underscore are helpers used by the plotting functions.
# The public functions near the end are the ones you normally call in a notebook.
# An experiment supplies stored data and training records; this file draws them.
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


# Put one experiment, or a list/tuple of experiments, into a common list format.
# For multiple experiments, check that the listed market and risk settings agree.
# With match_test=True, also check shared test identifiers and the quotes, units,
# and costs of instruments that appear in both experiments.
# Return the list of original experiment objects without changing those objects.
def _experiments(value, match_test=True):
    experiments = list(value) if isinstance(value, (list, tuple)) else [value]
    if not experiments:
        raise ValueError("Pass an experiment or a nonempty list of experiments.")
    # A single experiment does not need comparison checks against another experiment.
    if len(experiments) == 1:
        return experiments

    # Use the first experiment as the reference for the remaining comparisons.
    reference = experiments[0]
    for experiment in experiments[1:]:
        differences = [
            key for key in MATCHED_SETTINGS
            if getattr(experiment.settings, key) != getattr(reference.settings, key)
        ]
        if differences:
            raise ValueError("Different comparison settings: " + ", ".join(differences))
        # Learning curves can be drawn before test evaluation, so they skip these checks.
        if not match_test:
            continue
        # Compare stored identifiers; no test paths are generated or changed here.
        identity = reference.metadata.get("test_data_id")
        if identity is None or experiment.metadata.get("test_data_id") != identity:
            raise ValueError("Evaluate the experiments on common test paths first.")
        first_quotes = reference.metadata.get("instrument_data_ids", {})
        next_quotes = experiment.metadata.get("instrument_data_ids", {})
        # Only compare instruments both experiments contain. Their instrument sets may differ.
        for name in first_quotes.keys() & next_quotes.keys():
            if first_quotes[name] != next_quotes[name]:
                raise ValueError("Different observed quotes for " + name)
        # Equal price paths alone are not enough: contract sizes and trading costs must agree.
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


# Decide which trained objective or objectives the calling plot will display.
# objective can be CVaR, Entropic, or None; None selects the objectives present
# in every experiment. Capitalization and surrounding spaces are ignored.
# Return a list of standard names, such as ["Entropic"], or raise a clear error.
# This checks for stored runs; it does not train or evaluate missing policies.
def _objectives(experiments, objective):
    """Choose one fitted objective, or every objective shared by the experiments."""
    selected = None
    if objective is not None:
        # Translate names such as "entropic" into the spelling used by the stored runs.
        names = {name.casefold(): name for name in OBJECTIVES}
        selected = names.get(str(objective).strip().casefold())
        if selected is None:
            raise ValueError("objective must be None, 'CVaR', or 'Entropic'.")
    # Keep an objective only if every compared experiment has at least one matching run.
    common = [
        name for name in OBJECTIVES
        if all(any(run["objective"] == name for run in item.runs) for item in experiments)
    ]
    if selected is not None:
        if selected not in common:
            # Name the experiments that cannot supply the requested objective.
            missing = [
                str(item.name) for item in experiments
                if not any(run["objective"] == selected for run in item.runs)
            ]
            raise ValueError(
                f"No fitted {selected} run is available for: " + ", ".join(missing)
                + ". Available common objectives: " + (", ".join(common) or "none") + "."
            )
        return [selected]
    if not common:
        raise ValueError("No common fitted objective is available.")
    return common


# Select one objective from an experiment's stored training runs.
# Return the matching run dictionaries ordered by replicate_id, starting with
# the lowest ID. A replicate is a separate training run of the same setup.
# Sorting this new list does not reorder experiment.runs itself.
def _runs(experiment, objective):
    runs = [run for run in experiment.runs if run["objective"] == objective]
    return sorted(runs, key=lambda run: run["replicate_id"])


# Choose which training run to use when a plot needs one policy's results.
# Use the supplied replicate ID, or settings.display_replicate when it is None.
# Return an integer ID; the caller retrieves that run's losses separately.
def _replicate(experiment, replicate):
    return int(experiment.settings.display_replicate if replicate is None else replicate)


# Create a readable legend label for one experiment within a comparison.
# Start with experiment.name. If names repeat, add the training-data mode
# and, when needed, the experiment's position in the supplied list.
# Return the label string without renaming the experiment itself.
def _label(experiment, experiments):
    label = experiment.name
    matches = [item for item in experiments if item.name == label]
    if len(matches) > 1:
        mode = "fresh batches" if experiment.dynamic_data else "fixed pool"
        label += " (" + mode + ")"
        if sum(item.dynamic_data == experiment.dynamic_data for item in matches) > 1:
            label += " #" + str(experiments.index(experiment) + 1)
    return label


# Build the title text describing the objective being plotted.
# For CVaR, alpha determines the percentage of worst losses being averaged.
# For Entropic, settings.risk_aversion is the lambda shown in the title.
# Return text only; this helper does not calculate a risk score.
def _criterion(objective, settings):
    if objective == "CVaR":
        return f"CVaR: worst {100 * (1 - settings.alpha):g}% of losses"
    return rf"Entropic risk: $\lambda={settings.risk_aversion:g}$"


# Convert a loss vector into a NumPy array for plotting.
# Accept a NumPy-compatible sequence or a tensor. Tensors are detached from
# gradient tracking and moved to the CPU before conversion.
# Return a nonempty, one-dimensional float array; reject NaN and infinity.
# The returned array may share memory with its input, so callers only read it.
def _array(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty, finite, one-dimensional loss array.")
    return values


# Give standalone figures and grid panels the same axis appearance.
def _style_axis(axis, title, xlabel, ylabel):
    """Apply the common appearance to a standalone plot or a grid panel."""
    axis.set_title(title, fontsize=15, pad=18)
    axis.set_xlabel(xlabel, fontsize=13, labelpad=9)
    axis.set_ylabel(ylabel, fontsize=13, labelpad=9)
    axis.tick_params(labelsize=11)
    axis.grid(axis="y", alpha=0.18)
    axis.set_axisbelow(True)
    # Remove the top and right borders to keep the graph visually uncluttered.
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    # Show ordinary numbers rather than a separate scientific-notation offset.
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)


# Create one plotting area. The caller adds the curves, bars or points.
def _figure(title, xlabel, ylabel, dpi, figsize):
    figure = plt.figure(figsize=figsize, dpi=dpi)
    axis = figure.add_subplot(111)
    _style_axis(axis, title, xlabel, ylabel)
    return figure, axis


# Place the labels for plotted curves or markers below the graph.
# Input: the axis whose plotted items have labels. Use two columns when useful.
# Modify that axis and return None; skip the legend if there are no labels.
def _legend(axis):
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.20),
            ncol=2 if len(handles) > 1 else 1, frameon=False, fontsize=10.5,
            handlelength=2.8, columnspacing=1.6,
        )


# Apply the final layout, optionally save files, and optionally display a figure.
# filename is the export name without an extension; save_dir selects the folder.
# show controls display, while explain adds the supplied text below a shown plot.
# Use SVG in Jupyter for sharp text and curves; export both PNG and SVG if requested.
# Return (figure, axis), even after closing the figure's pyplot registration.
# The returned figure object can still be inspected or saved by the caller.
def _finish(figure, axis, filename, explanation, show, explain, save_dir):
    """Use vector display in Jupyter; keep interpretation out of the image."""
    _legend(axis)
    figure.tight_layout(pad=1.3)
    # Saving is independent of display, so show=False can still export the graph.
    if save_dir is not None:
        folder = Path(save_dir)
        folder.mkdir(parents=True, exist_ok=True)
        # PNG is a high-resolution pixel image; SVG stores scalable shapes and text.
        figure.savefig(folder / (filename + ".png"), dpi=300, bbox_inches="tight")
        figure.savefig(folder / (filename + ".svg"), format="svg", bbox_inches="tight")
    if show:
        try:
            from IPython import get_ipython
            from IPython.display import Markdown, SVG, display
            # Check for a notebook kernel, rather than assuming every IPython session is Jupyter.
            in_notebook = get_ipython() is not None and hasattr(get_ipython(), "kernel")
        except ImportError:
            in_notebook = False
        if in_notebook:
            # Build the SVG in memory so notebook display does not need a temporary image file.
            buffer = io.StringIO()
            figure.savefig(buffer, format="svg", bbox_inches="tight")
            display(SVG(buffer.getvalue()))
            if explain:
                display(Markdown("**How to read this graph.** " + explanation))
        else:
            plt.show()
            if explain:
                print(explanation)
    # Avoid leaving an active figure that Jupyter could display a second time.
    plt.close(figure)
    return figure, axis


# Build an export name from the plot kind, experiment families, and objective.
# Return lowercase text containing only letters, digits, underscores, and hyphens.
# This chooses a name only. It does not create a file or make the name unique.
def _filename(kind, experiments, objective=""):
    families = "_".join(str(item.family) for item in experiments)
    return re.sub(r"[^a-z0-9_-]+", "_", f"{kind}_{families}_{objective}".lower()).strip("_")


def _draw_learning(axis, experiments, name, replicate):
    """Draw stored validation histories on an existing axis."""
    values = []
    for experiment in experiments:
        runs = _runs(experiment, name)
        if replicate is not None:
            runs = [experiment.get_run(name, int(replicate))]
        # Organize each run as update number -> validation score so runs can be aligned.
        histories = [
            {int(row["update"]): float(row["risk"]) for row in run["history"]}
            for run in runs
        ]
        if any(not history for history in histories):
            raise ValueError("A run has an empty validation history.")
        # For this experiment, compare repeated runs only at updates they all recorded.
        common = sorted(set.intersection(*(set(history) for history in histories)))
        if not common:
            raise ValueError("The runs have no common recorded update.")
        # Each row is a run and each column is one shared update.
        # The bold curve takes the median across runs separately at each update.
        matrix = np.array([[history[step] for step in common] for history in histories])
        median, = axis.plot(
            common, np.median(matrix, axis=0), linewidth=2.4,
            marker="o" if len(common) <= 15 else None, markersize=4,
            label=_label(experiment, experiments),
        )
        # Keep individual histories visible as faint curves when there are several runs.
        for run, history in zip(runs, histories):
            updates = sorted(history)
            risks = [history[step] for step in updates]
            values.extend(risks)
            if len(runs) > 1:
                axis.plot(updates, risks, color=median.get_color(), alpha=0.28, linewidth=1)
            # Mark the checkpoint already selected during training; plotting selects no new model.
            chosen = int(run["best_update"])
            axis.scatter(
                chosen, history[chosen], s=80, facecolors="none",
                edgecolors=median.get_color(), linewidths=1.7, zorder=5,
            )
    # Use a logarithmic vertical axis when positive scores span a large range.
    # This changes the axis spacing while keeping all recorded values.
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
    return explanation


def _draw_risk(axis, experiments, name):
    """Draw the stored test scores without recomputing them."""
    reference = experiments[0]
    # Saved metric keys are lowercase: "cvar" and "entropic".
    metric = name.lower()
    all_values = []
    for index, experiment in enumerate(experiments):
        scores = np.array([run["metrics"][metric] for run in _runs(experiment, name)])
        # Spread run dots slightly left and right so repeated runs are easier to see.
        # These horizontal offsets do not change their risk scores.
        offsets = np.linspace(-0.08, 0.08, len(scores)) if len(scores) > 1 else [0]
        points = axis.scatter(
            index + np.asarray(offsets), scores, s=45, zorder=4,
            label="Individual training run" if index == 0 else None,
        )
        # The diamond is the arithmetic mean of the separate runs' risk scores.
        # It is not a confidence interval or the risk of an averaged trading strategy.
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
    # Use the same objective's stored no-hedging score as the final comparison point.
    baseline = float(reference.benchmarks["no_hedge"][metric])
    position = len(experiments)
    axis.scatter(position, baseline, s=65, marker="s", zorder=4)
    axis.annotate(
        f"{baseline:.3f}", (position, baseline), xytext=(0, 13),
        textcoords="offset points", ha="center", fontsize=11,
    )
    all_values.append(baseline)
    # Leave room above and below the points for readable numeric labels.
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
    return explanation


def _draw_loss_tails(axis, experiments, name, replicate, threshold):
    """Draw exact empirical loss tails, including all sample extremes."""
    reference = experiments[0]
    no_hedge = _array(reference.arrays["benchmark_loss_no_hedge"])
    series = [
        (_label(item, experiments), _array(item.losses(name, _replicate(item, replicate))))
        for item in experiments
    ]
    series.append(("No hedge", no_hedge))
    # All displayed strategies must have the same number of evaluation scenarios.
    if any(len(values) != len(no_hedge) for _, values in series):
        raise ValueError("Tail comparisons require the same number of evaluation scenarios.")
    for index, (label, losses) in enumerate(series):
        # Sort the losses and keep each distinct value with its first sorted position.
        # N minus that position counts every loss >= this value, including ties.
        values, first_indices = np.unique(np.sort(losses), return_index=True)
        percentages = 100 * (len(losses) - first_indices) / len(losses)
        # Draw the empirical percentages as steps; each height comes from the sample counts.
        line, = axis.step(
            values, percentages, where="pre", linewidth=2.2,
            linestyle="--" if index == len(experiments) else "-", label=label,
        )
        # Mark the largest observed loss. If several scenarios tie there, its height
        # includes all of them and can exceed the frequency of a single scenario.
        axis.scatter(
            values[-1], percentages[-1], s=50, facecolors="none",
            edgecolors=line.get_color(), linewidths=1.5, zorder=4,
        )
    # A frequency of 100/N percent corresponds to one observed scenario.
    # The log axis makes rare outcomes visible; it cannot display zero frequency.
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
    # The optional vertical line is only a reading aid; it does not cut off any losses.
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
    # Add an optional explanation of exponential weighting for entropic risk.
    # These weights describe risk sensitivity; the plotted curves remain scenario frequencies.
    if name == "Entropic":
        details = []
        lam = reference.settings.risk_aversion
        for label, losses in series[:-1]:
            # Subtract the maximum loss before exponentiation to keep positive-lambda
            # weights numerically manageable. Their normalized shares stay the same.
            weights = np.exp(lam * (losses - losses.max()))
            # One maximum-loss scenario has scaled weight 1, so its share is 100/sum.
            # With tied maxima, this reports one such scenario's share, not their combined share.
            largest_weight = 100 / weights.sum()
            details.append(
                f"{label}: maximum loss {losses.max():.3f}; one maximum-loss scenario "
                f"has {largest_weight:.1f}% of the exponential weight"
            )
        explanation += " For the displayed runs, " + "; ".join(details) + "."
    return explanation


# Show how the saved validation risk changed during training.
# Pass one experiment or a list. Choose objective="CVaR" or "Entropic",
# or leave objective=None to show every objective shared by the experiments.
# Use replicate=0 for one training run; the default includes all saved runs.
# dpi and figsize control appearance; explain=True adds reading guidance,
# and save_dir writes PNG and SVG copies. No new training is performed.
# With show=False, return {objective: (figure, axis)}; otherwise display
# the figures and return None.
def plot_learning(experiments, objective=None, replicate=None, *, dpi=DEFAULT_DPI,
                  figsize=DEFAULT_SIZE, show=True, explain=False, save_dir=None):
    """Plot validation histories, one full-size figure per objective.

    By default, show every run and its pointwise median. Set replicate=0
    to display only run zero. The recorded checkpoints are not smoothed.
    With show=False, return a dictionary of (figure, axis) pairs.
    """
    # Check compatible settings. Learning histories can be shown before test evaluation.
    experiments = _experiments(experiments, match_test=False)
    output = {}
    for name in _objectives(experiments, objective):
        figure, axis = _figure(
            "How does the network learn?\n" + _criterion(name, experiments[0].settings),
            "Updates after warm-up", "Validation risk (lower is better)", dpi, figsize,
        )
        explanation = _draw_learning(axis, experiments, name, replicate)
        output[name] = _finish(
            figure, axis, _filename("learning", experiments, name),
            explanation, show, explain, save_dir,
        )
    return output if not show else None


# Show the learning graph for one experiment and one selected training run.
# The default is CVaR and run zero; dpi controls the displayed resolution.
# Reuse plot_learning, then display the figure and return its (figure, axis) pair.
def plot_learning_simple(experiment, objective="CVaR", replicate=0, dpi=DEFAULT_DPI):
    """Backward-compatible single-run version of the earlier helper."""
    # Build the selected learning figure without displaying it yet.
    result = plot_learning(
        experiment, objective=objective, replicate=replicate, dpi=dpi, show=False,
    )
    # Take the figure from the returned dictionary and display it once here.
    figure, axis = next(iter(result.values()))
    _finish(figure, axis, "learning_simple", "", True, False, None)
    return figure, axis


# Compare the distribution of terminal test losses for the selected strategies.
# Pass one experiment or a list and optionally choose CVaR or Entropic.
# replicate selects a saved run; None uses each experiment's display_replicate.
# bins controls the histogram intervals. dpi, figsize, explain and save_dir
# control presentation and export. Existing saved loss arrays supply the data.
# Return {objective: (figure, axis)} with show=False, or display and return None.
def plot_losses(experiments, objective=None, replicate=None, *, bins=80,
                dpi=DEFAULT_DPI, figsize=DEFAULT_SIZE, show=True, explain=False, save_dir=None):
    """Plot full-range loss histograms; no tail observations are removed.

    Use objective="CVaR" or objective="Entropic" for one objective.
    With objective=None, display every common fitted objective separately.
    """
    # Require compatible experiments evaluated on common test paths.
    experiments = _experiments(experiments)
    reference = experiments[0]
    no_hedge = _array(reference.arrays["benchmark_loss_no_hedge"])
    output = {}
    for name in _objectives(experiments, objective):
        # Read one selected run per experiment; the curves do not pool repeated training runs.
        series = [
            (_label(item, experiments), _array(item.losses(name, _replicate(item, replicate))))
            for item in experiments
        ]
        # Include the stored no-hedging benchmark and the stock-delta benchmark when available.
        series.append(("No hedge", no_hedge))
        if "benchmark_loss_stock_delta" in reference.arrays:
            series.append(("Stock delta", _array(reference.arrays["benchmark_loss_stock_delta"])))
        # Use common bin edges so the same loss intervals are compared for every curve.
        edges = np.histogram_bin_edges(np.concatenate([v for _, v in series]), bins=bins)
        figure, axis = _figure(
            "What changes when we hedge?\n" + _criterion(name, reference.settings),
            "Terminal loss before premium", "Scenarios in each bin (%)", dpi, figsize,
        )
        # Give each scenario weight 100/N, so a bin height is a percentage of scenarios.
        # Solid outlines identify learned strategies; dashed outlines identify benchmarks.
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


# Compare final profits using side-by-side bars, as in the paper's histogram.
def plot_pnl_histogram(experiments, objective=None, replicate=None, *, bins=40,
                       premium=None, include_delta=True, dpi=DEFAULT_DPI,
                       figsize=DEFAULT_SIZE, show=True, explain=False, save_dir=None):
    """Draw side-by-side histograms of final profit = premium - stored loss.

    Use one experiment or a list evaluated on common test paths. By default,
    include the first experiment's stored Heston stock-delta benchmark:

        ours.evaluate(include_delta=True)
        plots.plot_pnl_histogram(ours, objective="CVaR")

    Set include_delta=False to compare only neural policies. One common
    premium shifts every series equally. For the built-in single-call
    experiment it defaults to the Heston option price; other liabilities
    require an explicit premium. No training or evaluation is performed.
    """
    experiments = _experiments(experiments)
    reference = experiments[0]
    objectives = _objectives(experiments, objective)

    # Read the benchmark prepared by evaluate(include_delta=True).
    delta_losses = None
    if include_delta:
        key = "benchmark_loss_stock_delta"
        if key not in reference.arrays:
            raise ValueError(
                "First call evaluate(include_delta=True) on the first experiment, "
                "or pass include_delta=False."
            )
        delta_losses = _array(reference.arrays[key])

    # Use the same premium for every strategy, without centering them separately.
    if premium is None:
        if (reference.metadata.get("evaluation_kind") == "historical"
                or reference.metadata.get("default_call_payoff") is False):
            raise ValueError("Pass premium=... for this liability.")
        premium = reference.metadata.get("analytic_heston_call_price")
        if premium is None:
            if not reference.metadata.get("default_call_payoff"):
                raise ValueError("Pass premium=... for this liability.")
            from scripts.general_nn import call_price
            premium = call_price(reference.settings)
    premium = float(premium)
    if not np.isfinite(premium):
        raise ValueError("premium must be a finite number.")

    output = {}
    for name in objectives:
        labels = [_label(item, experiments) for item in experiments]
        losses = [_array(item.losses(name, _replicate(item, replicate)))
                  for item in experiments]
        if delta_losses is not None:
            labels.append("Heston stock-delta hedge")
            losses.append(delta_losses)
        if any(len(values) != len(losses[0]) for values in losses):
            raise ValueError("Histogram comparisons require equal scenario counts.")

        # Shared bins make each group of bars compare the same profit interval.
        profits = [premium - values for values in losses]
        edges = np.histogram_bin_edges(np.concatenate(profits), bins=bins)
        figure, axis = _figure(
            "What is the final hedging profit?\n" + _criterion(name, reference.settings),
            "Final profit and loss (premium - loss)", "Number of scenarios", dpi, figsize,
        )
        axis.hist(profits, bins=edges, label=labels, alpha=0.6, rwidth=0.9)
        explanation = (
            f"Every strategy receives the same premium of {premium:.4f}. "
            "Positive horizontal values mean profit; negative values mean loss. "
            "Bar heights count scenarios in shared bins, including all observed extremes. "
            "Trading costs are already included in each stored loss."
        )
        if delta_losses is not None:
            explanation += " The Heston stock-delta benchmark trades only the stock."
        output[name] = _finish(
            figure, axis, _filename("pnl_histogram", experiments, name),
            explanation, show, explain, save_dir,
        )
    return output if not show else None


# Compare saved test-risk scores across experiments and repeated training runs.
# Choose objective="CVaR" or "Entropic"; None makes a separate figure for each
# objective shared by all experiments. Lower risk is better.
# Small dots show individual runs, an open diamond shows their mean, and a
# square shows no hedging. The scores are read from saved evaluation results.
# dpi, figsize, explain and save_dir control presentation and export.
# Return {objective: (figure, axis)} with show=False, or display and return None.
def plot_risk(experiments, objective=None, *, dpi=DEFAULT_DPI, figsize=DEFAULT_SIZE,
              show=True, explain=False, save_dir=None):
    """Compare objective-specific test risks and variation across training runs.

    Use objective="CVaR" or objective="Entropic" for one objective.
    With objective=None, display every common fitted objective separately.
    """
    experiments = _experiments(experiments)
    reference = experiments[0]
    output = {}
    for name in _objectives(experiments, objective):
        figure, axis = _figure(
            "Which hedge has less risk?\n" + _criterion(name, reference.settings),
            "", "Measured risk before premium (lower is better)", dpi, figsize,
        )
        explanation = _draw_risk(axis, experiments, name)
        output[name] = _finish(
            figure, axis, _filename("risk", experiments, name), explanation, show, explain, save_dir,
        )
    return output if not show else None


# Show how often each strategy has a loss at least as large as a chosen value.
# The horizontal axis lists loss thresholds; the vertical axis gives the
# percentage of test scenarios reaching each threshold, on a logarithmic scale.
# Pass one experiment or a list. Choose objective by keyword; the second
# positional argument is the optional vertical reference line threshold.
# threshold=None removes that line. replicate selects one saved training run;
# None uses each experiment's configured display run. No runs are pooled.
# dpi, figsize, explain and save_dir control presentation and export.
# Return {objective: (figure, axis)} with show=False, or display and return None.
def plot_loss_tails(experiments, threshold=5.0, *, objective=None, replicate=None,
                    dpi=DEFAULT_DPI, figsize=DEFAULT_SIZE, show=True, explain=False, save_dir=None):
    """Plot empirical P(loss >= threshold) on a logarithmic percentage axis.

    Each objective has a separate full-size figure. All sample extremes are
    retained. Open endpoint circles mark the observed maxima; no smoothing,
    binning, truncation, or extrapolation is used.

    Use objective="CVaR" or objective="Entropic" for one objective, or omit
    it for every common fitted objective. The second positional argument
    remains threshold, so pass the objective by keyword.
    """
    experiments = _experiments(experiments)
    reference = experiments[0]
    output = {}
    for name in _objectives(experiments, objective):
        figure, axis = _figure(
            "Why can a few bad hedges matter so much?\n" + _criterion(name, reference.settings),
            "Loss threshold before premium", "Scenarios with at least this loss (%)", dpi, figsize,
        )
        explanation = _draw_loss_tails(axis, experiments, name, replicate, threshold)
        output[name] = _finish(
            figure, axis, _filename("tails", experiments, name), explanation, show, explain, save_dir,
        )
    return output if not show else None


# Arrange the same learning, risk and tail plots into one figure.
# Rows are plot types; columns are common fitted objectives.
# Reuse the drawing helpers above so data checks and plotted values stay the same.
def plot_results_grid(experiments, objective=None, *, replicate=None, threshold=5.0,
                      dpi=DEFAULT_DPI, figsize=None, show=True, explain=False,
                      save_dir=None):
    """Compare learning curves, test risk and loss tails in a single grid.

    By default, CVaR and Entropic occupy separate columns (three rows, two
    columns). Set objective="CVaR" or "Entropic" for a three-row, one-column
    figure. Only objectives fitted in every experiment are included.

    replicate selects the learning and tail runs; None keeps their existing
    defaults. The risk row always displays all runs, as plot_risk does.
    threshold is the optional vertical reference line in the tail panels.
    No training, evaluation or simulation is performed.

    With show=False, return (figure, axes), with axes always two-dimensional.
    save_dir exports the complete grid as a 300-dpi PNG and a vector SVG.
    """
    experiments = _experiments(experiments)
    names = _objectives(experiments, objective)
    if figsize is None:
        figsize = (8 * len(names), 16)

    figure, axes = plt.subplots(
        3, len(names), figsize=figsize, dpi=dpi, squeeze=False,
    )
    panels = (
        ("Learning curves", "Updates after warm-up", "Validation risk (lower is better)"),
        ("Test risk", "", "Measured risk before premium (lower is better)"),
        ("Loss tails", "Loss threshold before premium", "Scenarios with at least this loss (%)"),
    )
    try:
        for column, name in enumerate(names):
            for row, (title, xlabel, ylabel) in enumerate(panels):
                _style_axis(
                    axes[row, column],
                    title + "\n" + _criterion(name, experiments[0].settings),
                    xlabel, ylabel,
                )
            _draw_learning(axes[0, column], experiments, name, replicate)
            _draw_risk(axes[1, column], experiments, name)
            _draw_loss_tails(axes[2, column], experiments, name, replicate, threshold)

        # _finish adds the first legend; prepare the remaining panel legends.
        for axis in axes.flat:
            if axis is not axes[0, 0]:
                _legend(axis)
        explanation = (
            "Rows show learning curves, test risk and loss tails; columns separate "
            "the fitted risk objectives. Lower validation and test risk is better. "
            "In the bottom row, lower curves at a given threshold mean fewer losses "
            "at least that large. Tail frequencies use a logarithmic vertical axis "
            "and retain every sample extreme. The grid uses the same recorded "
            "histories, test scores and loss arrays as the individual plotting functions."
        )
        _finish(
            figure, axes[0, 0],
            _filename("results_grid", experiments, "_".join(names)),
            explanation, show, explain, save_dir,
        )
    except Exception:
        # Failed comparisons must not leave a partially drawn figure active.
        plt.close(figure)
        raise
    return (figure, axes) if not show else None


# Display example stock paths and a second graph of the liability payments.
# Pass one experiment or a compatible list. The first experiment provides
# the displayed paths, so a list does not create a separate panel per method.
# These market paths and payments do not require an objective selection.
# dpi, figsize, explain and save_dir control presentation and export.
# Return {"paths": (figure, axis), "payoffs": (figure, axis)} with show=False;
# otherwise display both figures and return None.
def plot_paths(experiments, *, dpi=DEFAULT_DPI, figsize=DEFAULT_SIZE,
               show=True, explain=False, save_dir=None):
    """Display example stock paths, then the corresponding liability payments.

    These evaluation scenarios do not depend on the training objective.
    """
    experiments = _experiments(experiments)
    experiment = experiments[0]
    # Use at most six stored paths. A three-dimensional prices array has a final
    # instrument axis; select its first instrument for this stock-path display.
    prices = np.asarray(experiment.arrays["example_prices"], dtype=float)
    stock = prices[:, :, 0] if prices.ndim == 3 else prices
    stock = stock[:6]
    # Historical evaluations and custom liabilities use their stored payments.
    # The default simulated call can instead be illustrated from its strike.
    supplied = (
        experiment.metadata.get("evaluation_kind") == "historical"
        or not experiment.metadata.get("default_call_payoff", True)
    )
    figure, axis = _figure(
        "Example stock paths", "Trading step", "First stock price", dpi, figsize,
    )
    # Remember each path's color so its payment uses the same color in the next graph.
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
    # For a supplied liability, read the actual example payments rather than
    # reconstructing them from one stock's final price.
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
        # For the default call, draw max(final stock price - strike, 0).
        # Matching colored dots connect each displayed path to its maturity payment.
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


# Display an existing numbered figure from the source paper.
# number selects an entry in assets/paper/metadata.json. That metadata and
# its referenced image files must already be present in the project.
# This function displays stored artwork; it does not recreate paper results.
# explain=True includes the stored interpretation. save_dir copies the
# original PNG and, when present, SVG. dpi and figsize apply to the
# Matplotlib display; Jupyter uses the original SVG when available.
# Return {"published": (figure, axis)} with show=False, or display and return None.
def show_published_figure(number, *, dpi=DEFAULT_DPI, figsize=(10, 6),
                          show=True, explain=False, save_dir=None):
    """Show original paper artwork, using its original SVG in Jupyter."""
    # Read the figure title, image locations and explanation from the paper metadata.
    metadata = json.loads((PROJECT_ROOT / "assets" / "paper" / "metadata.json").read_text())
    entry = metadata["figures"][str(number)]
    image_path = PROJECT_ROOT / entry["image"]
    svg_path = PROJECT_ROOT / entry.get("original_svg", "missing.svg")
    # Copy the original artwork when export is requested, keeping its existing resolution.
    if save_dir is not None:
        import shutil
        folder = Path(save_dir)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path, folder / f"paper_figure_{int(number):02d}.png")
        if svg_path.is_file():
            shutil.copyfile(svg_path, folder / f"paper_figure_{int(number):02d}.svg")
    # In a Jupyter kernel, display the original vector SVG so lines and text stay sharp.
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
    # Use the PNG through Matplotlib when no notebook SVG display is used.
    # This path also supplies figure and axis objects for show=False.
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


def _heston_series(models):
    """Read the first experiment's stored, comparable Heston benchmark."""
    reference = models[0]
    metadata = reference.metadata.get("heston_benchmark")
    if metadata is None or "benchmark_loss_heston" not in reference.arrays:
        raise ValueError("Run evaluate(include_heston=True) on the first experiment.")
    if metadata["test_data_id"] != reference.metadata.get("test_data_id"):
        raise ValueError("Recompute the Heston benchmark on the current test set.")

    # A stock-only network needs a stock-only benchmark. A network that can
    # trade the variance claim needs the corresponding two-instrument hedge.
    for model in models:
        instruments = sorted([
            (item.name, item.kind, item.stock_index, item.multiplier,
             model.settings.cost_rate if item.cost_rate is None else item.cost_rate)
            for item in model.settings.instruments if item.tradable
        ])
        if instruments != metadata["instruments"]:
            raise ValueError("Compare models with the same enabled trading instruments.")
    return metadata["label"], _array(reference.arrays["benchmark_loss_heston"])


def plot_pnl_comparison(
    experiments, *, premium, title="Terminal P&L", objective="CVaR",
    replicate=0, bin_width=0.25, zoom=(-3, 3), include_heston=False,
):
    """Compare stored P&Ls, optionally including the classical Heston hedge.

    Use the same premium for every strategy. The right panel only zooms the
    horizontal axis: no observations are removed from the histogram counts.
    """
    models = _experiments(experiments)
    if objective is None:
        raise ValueError("Choose CVaR or Entropic.")
    objective = _objectives(models, objective)[0]
    premium, bin_width = float(premium), float(bin_width)
    if not np.isfinite(premium) or not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("Use a finite premium and a positive, finite bin width.")
    if len(zoom) != 2 or not np.isfinite(zoom).all() or zoom[0] >= zoom[1]:
        raise ValueError("zoom must contain two increasing, finite limits.")

    # Read the selected policies' existing test losses.
    profits = [
        premium - _array(m.losses(objective, replicate=replicate))
        for m in models
    ]
    labels = [
        m.name + "\n" + ("fresh batches" if m.dynamic_data else "fixed pool")
        for m in models
    ]

    # The model hedge is calculated during evaluation, never inside a plot.
    # Its holdings do not depend on the neural network's training objective.
    if include_heston:
        label, losses = _heston_series(models)
        profits.append(premium - losses)
        labels.append(label + "\nmodel based; no training")
    if len({len(values) for values in profits}) != 1:
        raise ValueError("Compare equal numbers of test scenarios.")

    # Zero-aligned bins keep the same intervals across separate calls.
    all_values = np.concatenate(profits)
    low = np.floor(all_values.min() / bin_width)
    high = max(low + 1, np.ceil(all_values.max() / bin_width))
    bins = bin_width * np.arange(low, high + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, dpi=200)

    for axis in axes:
        axis.hist(profits, bins=bins, label=labels, alpha=0.6, rwidth=0.9)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Final P&L (premium - loss)")
        axis.grid(axis="y", alpha=0.2)

    axes[0].set_title("Full observed range")
    axes[0].set_ylabel("Number of scenarios")
    axes[1].set_xlim(*zoom)
    axes[1].set_title(f"Central view only: [{zoom[0]:g}, {zoom[1]:g}]")
    figure.suptitle(title + "\nNeural training objective: "
                   + _criterion(objective, models[0].settings))

    handles, _ = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    figure.tight_layout(rect=(0, 0.15, 1, 0.90))
    plt.show()
    plt.close(figure)
    return figure, axes

def plot_memory_comparison(reports, *, save_dir=None, show=True):
    """Compare peak CPU RAM and runtime from two stored memory reports.

    Accepts the reports returned by run_memory_experiment().
    With show=False, returns (figure, axes).
    """

    # Identify the fixed-pool and fresh-batch reports.
    reports = list(reports)
    if len(reports) != 2 or {
        r["dynamic_data"] for r in reports
    } != {False, True}:
        raise ValueError(
            "Pass one fixed-pool report and one fresh-batch report."
        )

    fixed, fresh = sorted(reports, key=lambda r: r["dynamic_data"])
    reports = [fixed, fresh]

    # Require matching settings, test paths, quotes and source code.
    for key in (
        "comparison_signature",
        "test_data_id",
        "instrument_data_ids",
        "source_hashes",
    ):
        if key not in fixed or key not in fresh or fixed[key] != fresh[key]:
            raise ValueError(f"The two memory reports do not match: {key}")

    # Convert the recorded measurements to GiB and minutes.
    values = np.array([
        [r["memory"]["peak_mib"] / 1024 for r in reports],
        [r["memory"]["seconds"] / 60 for r in reports],
    ], dtype=float)

    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Memory and runtime must be positive and finite.")

    # Use the actual dataset and batch sizes in the labels.
    labels = [
        f"Fixed pool\n{fixed['settings']['static_training_size']:,} paths",
        f"Fresh batches\n{fresh['settings']['batch_size']:,} paths per update",
    ]

    panels = [
        ("Sampled peak CPU RAM", "GiB"),
        ("Data preparation, training and evaluation", "Minutes"),
    ]

    # Draw the same two panels currently shown in the notebook.
    figure, axes = plt.subplots(
        1, 2, figsize=(10, 4), dpi=DEFAULT_DPI
    )

    for axis, measured, (title, unit) in zip(axes, values, panels):
        bars = axis.bar(
            labels,
            measured,
            color=["#4c78a8", "#f58518"],
            width=0.6,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.2f}" for value in measured],
            padding=4,
        )
        axis.set_title(title, fontsize=11)
        axis.set_ylabel(unit)
        axis.set_ylim(0, measured.max() * 1.22)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)

    figure.suptitle(
        "Same portfolio and network; fixed pool versus fresh batches"
    )

    # Reuse the existing SVG display and PNG/SVG saving behavior.
    # Neither panel needs a legend: the x-axis labels identify the modes.
    _finish(
        figure, axes[0], "memory_comparison",
        "", show, False, save_dir,
    )

    return (figure, axes) if not show else None

