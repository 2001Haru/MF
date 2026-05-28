import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def stat_series(summary, metric_name, stat_name):
    t_values = [float(t) for t in summary["t_values"]]
    values = [
        summary["by_t"][str(t)][metric_name][stat_name]
        for t in summary["t_values"]
    ]
    return np.asarray(t_values), np.asarray(values)


def plot_endpoint_disagreement(summary, output_dir):
    t, mean = stat_series(summary, "endpoint_disagreement", "mean")
    _, median = stat_series(summary, "endpoint_disagreement", "median")
    _, p90 = stat_series(summary, "endpoint_disagreement", "p90")
    _, p95 = stat_series(summary, "endpoint_disagreement", "p95")

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=180)
    ax.plot(t, mean, marker="o", linewidth=2.2, label="mean")
    ax.plot(t, median, marker="s", linewidth=1.8, label="median")
    ax.fill_between(t, p90, p95, alpha=0.22, label="p90-p95")
    ax.set_xlabel("Noise time t")
    ax.set_ylabel("Endpoint disagreement")
    ax.set_title("Endpoint disagreement across shortcut r")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_disagreement_vs_t.png")
    fig.savefig(output_dir / "endpoint_disagreement_vs_t.pdf")
    plt.close(fig)


def plot_endpoint_mse_by_r(summary, output_dir):
    t_values = summary["t_values"]
    r_fractions = summary["r_fractions"]

    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=180)
    for t in t_values:
        means = [
            summary["by_t"][str(t)]["endpoint_mse_to_data_by_r_fraction"][str(r)]["mean"]
            for r in r_fractions
        ]
        ax.plot(r_fractions, means, marker="o", linewidth=1.9, label=f"t={t}")

    ax.set_xlabel("r / t")
    ax.set_ylabel("MSE to true data endpoint")
    ax.set_title("Endpoint extrapolation error by shortcut target r")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_mse_by_r_fraction.png")
    fig.savefig(output_dir / "endpoint_mse_by_r_fraction.pdf")
    plt.close(fig)


def plot_relative_disagreement(summary, output_dir):
    t, disagreement = stat_series(summary, "endpoint_disagreement", "mean")
    _, mse_to_data = stat_series(summary, "mean_endpoint_mse_to_data", "mean")
    relative = disagreement / np.maximum(mse_to_data, 1e-12)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=180)
    ax.plot(t, relative, marker="o", linewidth=2.2, color="#9b2c2c")
    ax.set_xlabel("Noise time t")
    ax.set_ylabel("Disagreement / endpoint MSE")
    ax.set_title("Relative endpoint instability")
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(output_dir / "relative_endpoint_disagreement.png")
    fig.savefig(output_dir / "relative_endpoint_disagreement.pdf")
    plt.close(fig)


def main(args):
    summary_path = Path(args.summary)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_endpoint_disagreement(summary, output_dir)
    plot_endpoint_mse_by_r(summary, output_dir)
    plot_relative_disagreement(summary, output_dir)

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    main(parser.parse_args())
