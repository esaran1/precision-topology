"""Generate the requested markdown summary and diagnostic figures from persisted results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .precision import DELTA_TABLE


def _seed_summary(frame: pd.DataFrame, value: str, groups: list[str]) -> pd.DataFrame:
    per_seed = frame.groupby(groups + ["seed"], dropna=False)[value].mean().reset_index()
    return per_seed.groupby(groups, dropna=False)[value].agg(["mean", "std", "count"]).reset_index()


def write_summary_table(saturation: pd.DataFrame, destination: Path) -> None:
    tanh = saturation[saturation["activation"] == "tanh"]
    labels = {
        (spec.format, spec.convention): f"{spec.format} ({spec.convention})"
        for spec in DELTA_TABLE
    }
    columns = [(spec.format, spec.convention) for spec in DELTA_TABLE]
    lines = [
        "# Tanh saturation summary",
        "",
        "Values are mean ± sample standard deviation of the paper-criterion total",
        "saturation fraction. Within each seed, accepted depth × width configurations",
        "containing that absolute layer are averaged first; the displayed mean and SD",
        "are then computed across the five seed-level averages. Failed runs are excluded.",
        "",
        "The exact-rounding criterion is retained in the CSV and Parquet artifacts and",
        "reported alongside the primary results in `FINDINGS.md`.",
    ]
    for dataset in ("linked_tori", "blobs"):
        lines.extend(["", f"## {dataset}", ""])
        subset = tanh[tanh["dataset"] == dataset]
        summary = _seed_summary(
            subset,
            "paper_total_saturation_fraction",
            ["layer", "format", "convention"],
        ).set_index(["layer", "format", "convention"])
        lines.append("| Layer | " + " | ".join(labels[column] for column in columns) + " |")
        lines.append("|---:|" + "---:|" * len(columns))
        for layer in range(1, int(subset["layer"].max()) + 1):
            cells = []
            for format_name, convention in columns:
                key = (layer, format_name, convention)
                if key not in summary.index:
                    cells.append("—")
                else:
                    row = summary.loc[key]
                    cells.append(f"{100 * row['mean']:.4f}% ± {100 * row['std']:.4f}%")
            lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_sd_by_x(frame: pd.DataFrame, value: str, x: str) -> pd.DataFrame:
    return _seed_summary(frame, value, [x]).sort_values(x)


def plot_distance_profiles(saturation: pd.DataFrame, figures: Path) -> None:
    subset = saturation[
        (saturation["activation"] == "tanh")
        & (saturation["format"] == "bfloat16")
        & (saturation["convention"] == "IEEE")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, dataset in zip(axes, ("linked_tori", "blobs")):
        data = subset[subset["dataset"] == dataset]
        for value, label in (
            ("paper_total_saturation_fraction", "paper threshold"),
            ("exact_total_saturation_fraction", "exact rounding"),
        ):
            summary = _mean_sd_by_x(data, value, "distance_from_output")
            axis.errorbar(summary["distance_from_output"], summary["mean"], yerr=summary["std"], marker="o", label=label)
        axis.set_title(dataset)
        axis.set_xlabel("remaining hidden layers (0 = final hidden)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("total saturation fraction")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures / "saturation_by_output_distance.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, dataset in zip(axes, ("linked_tori", "blobs")):
        data = subset[subset["dataset"] == dataset]
        for value, label in (
            ("vector_collision_rate", "trained"),
            ("baseline_vector_collision_rate", "initialization baseline"),
            ("excess_vector_collision_rate", "excess"),
        ):
            summary = _mean_sd_by_x(data, value, "distance_from_output")
            axis.errorbar(summary["distance_from_output"], summary["mean"], yerr=summary["std"], marker="o", label=label)
        axis.set_title(dataset)
        axis.set_xlabel("remaining hidden layers (0 = final hidden)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("bfloat16 vector collision rate")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures / "vector_collisions_by_output_distance.png", dpi=160)
    plt.close(fig)


def plot_training_dynamics(dynamics: pd.DataFrame, figures: Path) -> None:
    subset = dynamics[
        (dynamics["activation"] == "tanh")
        & (dynamics["format"] == "bfloat16")
        & (dynamics["convention"] == "IEEE")
        & (dynamics["distance_from_output"] == 0)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, dataset in zip(axes, ("linked_tori", "blobs")):
        data = subset[subset["dataset"] == dataset]
        for value, label in (
            ("paper_total_saturation_fraction", "paper saturation"),
            ("exact_total_saturation_fraction", "exact saturation"),
            ("vector_collision_rate", "vector collision"),
        ):
            summary = _mean_sd_by_x(data, value, "training_progress")
            axis.errorbar(summary["training_progress"], summary["mean"], yerr=summary["std"], marker="o", label=label)
        axis.set_title(dataset)
        axis.set_xlabel("training progress")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("final-hidden-layer fraction")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures / "training_dynamics.png", dpi=160)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    saturation = pd.read_parquet(results / "saturation.parquet")
    dynamics = pd.read_parquet(results / "training_dynamics.parquet")
    write_summary_table(saturation, results / "summary_table.md")
    plot_distance_profiles(saturation, figures)
    plot_training_dynamics(dynamics, figures)


if __name__ == "__main__":
    main()
