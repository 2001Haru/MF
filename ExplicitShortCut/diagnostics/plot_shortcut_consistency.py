import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def triple_label(triple):
    return f"t={triple['t']}, s={triple['s']}, r={triple['r']}"


def triple_key(triple):
    return f"{triple['t']}:{triple['s']}:{triple['r']}"


def plot_semigroup(summary, output_dir, metric_name, ylabel, filename):
    triples = summary["triples"]
    labels = [triple_label(tri) for tri in triples]
    means = []
    medians = []
    p90s = []
    p95s = []

    for tri in triples:
        stats = summary["semigroup"][triple_key(tri)][metric_name]
        means.append(stats["mean"])
        medians.append(stats["median"])
        p90s.append(stats["p90"])
        p95s.append(stats["p95"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=180)
    ax.plot(x, means, marker="o", linewidth=2.2, label="mean")
    ax.plot(x, medians, marker="s", linewidth=1.8, label="median")
    ax.vlines(x, p90s, p95s, linewidth=7.0, alpha=0.22, label="p90-p95")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " across semigroup triples")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.png")
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_target_fit(summary, output_dir):
    items = sorted(
        summary["target_fit"].items(),
        key=lambda kv: tuple(float(x) for x in kv[0].split("->")),
    )
    labels = [key for key, _ in items]
    means = [stats["mean"] for _, stats in items]
    medians = [stats["median"] for _, stats in items]
    p90s = [stats["p90"] for _, stats in items]
    p95s = [stats["p95"] for _, stats in items]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=180)
    ax.plot(x, means, marker="o", linewidth=2.2, label="mean")
    ax.plot(x, medians, marker="s", linewidth=1.8, label="median")
    ax.vlines(x, p90s, p95s, linewidth=7.0, alpha=0.22, label="p90-p95")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_xlabel("Shortcut interval")
    ax.set_ylabel("MSE to conditional shortcut target")
    ax.set_title("Conditional target-fit error")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "conditional_target_fit_error.png")
    fig.savefig(output_dir / "conditional_target_fit_error.pdf")
    plt.close(fig)


def plot_endpoint_comparison(summary, output_dir):
    if "endpoint_comparison" not in summary:
        return

    triples = summary["triples"]
    labels = [triple_label(tri) for tri in triples]
    keys = [triple_key(tri) for tri in triples]
    mse_one = [summary["endpoint_comparison"][key]["mse_one"]["mean"] for key in keys]
    mse_two = [summary["endpoint_comparison"][key]["mse_two"]["mean"] for key in keys]
    delta = [
        summary["endpoint_comparison"][key]["delta_two_minus_one"]["mean"]
        for key in keys
    ]
    better = [
        summary["endpoint_comparison"][key]["fraction_two_step_better"]
        for key in keys
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    ax.bar(x - width / 2, mse_one, width, label="one-step")
    ax.bar(x + width / 2, mse_two, width, label="two-step")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("MSE to data endpoint")
    ax.set_title("Does two-step improve endpoint reconstruction?")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_mse_one_vs_two.png")
    fig.savefig(output_dir / "endpoint_mse_one_vs_two.pdf")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    colors = ["#2f855a" if v < 0 else "#c53030" for v in delta]
    ax1.bar(x, delta, color=colors, alpha=0.82)
    ax1.axhline(0, color="black", linewidth=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right")
    ax1.set_ylabel("Delta MSE (two-step - one-step)")
    ax1.set_title("Endpoint gain from splitting long shortcuts")
    ax1.grid(True, axis="y", alpha=0.28)

    ax2 = ax1.twinx()
    ax2.plot(x, better, marker="o", color="#2b6cb0", linewidth=2.0)
    ax2.set_ylabel("Fraction two-step better")
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_delta_two_minus_one.png")
    fig.savefig(output_dir / "endpoint_delta_two_minus_one.pdf")
    plt.close(fig)


def plot_correlations(summary, output_dir):
    if "correlation" not in summary:
        return

    triples = summary["triples"]
    labels = [triple_label(tri) for tri in triples]
    keys = [triple_key(tri) for tri in triples]
    corr_gap = [
        summary["correlation"][key]["endpoint_disagreement_vs_gap"]["pearson"]
        for key in keys
    ]
    corr_gap_rel = [
        summary["correlation"][key]["endpoint_disagreement_vs_gap_rel"]["pearson"]
        for key in keys
    ]
    corr_delta = [
        summary["correlation"][key]["endpoint_disagreement_vs_delta"]["pearson"]
        for key in keys
    ]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=180)
    ax.plot(x, corr_gap, marker="o", linewidth=2.0, label="D_end vs gap")
    ax.plot(x, corr_gap_rel, marker="s", linewidth=2.0, label="D_end vs relative gap")
    ax.plot(x, corr_delta, marker="^", linewidth=2.0, label="D_end vs delta")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Correlation between endpoint disagreement and shortcut errors")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_disagreement_correlations.png")
    fig.savefig(output_dir / "endpoint_disagreement_correlations.pdf")
    plt.close(fig)


def main(args):
    summary_path = Path(args.summary)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_semigroup(
        summary,
        output_dir,
        "gap",
        "Semigroup gap MSE",
        "semigroup_gap",
    )
    plot_semigroup(
        summary,
        output_dir,
        "gap_rel",
        "Relative semigroup gap",
        "relative_semigroup_gap",
    )
    plot_endpoint_comparison(summary, output_dir)
    plot_correlations(summary, output_dir)
    plot_target_fit(summary, output_dir)

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    main(parser.parse_args())
