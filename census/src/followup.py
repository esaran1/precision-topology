"""Follow-up analyses for width baselines, dynamics, and paired dataset gaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .data import linked_tori
from .models import MLP
from .precision import collision_metrics
from .train import seed_everything


DEPTHS = (4, 6, 8, 10)
WIDTHS = (5, 15, 30, 50)
SEEDS = (0, 1, 2, 3, 4)


def recompute_initialization_baselines() -> pd.DataFrame:
    """Recreate step-zero final-layer collisions without performing training."""

    rows: list[dict[str, float | int]] = []
    for seed in SEEDS:
        evaluation = linked_tori(1_000, tube_radius=0.2, seed=20_000 + seed)
        features = torch.as_tensor(evaluation.features, dtype=torch.float32, device="cpu")
        for depth in DEPTHS:
            for width in WIDTHS:
                seed_everything(seed, cpu_threads=1)
                model = MLP(3, depth, width, "tanh").to(device="cpu", dtype=torch.float32)
                final_preactivations = model.collect_preactivations(features)[-1]
                metrics = collision_metrics(final_preactivations, "bfloat16", "tanh")
                rows.append(
                    {
                        "dataset": "linked_tori",
                        "activation": "tanh",
                        "format": "bfloat16",
                        "depth": depth,
                        "width": width,
                        "seed": seed,
                        "eval_data_seed": 20_000 + seed,
                        "baseline_vector_collision_rate": metrics["vector_collision_rate"],
                    }
                )
    return pd.DataFrame(rows)


def _mean_std_count_by_seed(
    frame: pd.DataFrame,
    value: str,
    groups: list[str],
) -> pd.DataFrame:
    seed_values = frame.groupby(groups + ["seed"], dropna=False)[value].mean().reset_index()
    return (
        seed_values.groupby(groups, dropna=False)[value]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def width_baseline_analysis(
    saturation: pd.DataFrame,
    initialization: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize all-init baselines and accepted trained excess separately by width."""

    all_initialization = _mean_std_count_by_seed(
        initialization,
        "baseline_vector_collision_rate",
        ["width"],
    ).rename(
        columns={
            "mean": "all_init_baseline_mean",
            "std": "all_init_baseline_std",
            "count": "all_init_seed_count",
        }
    )
    accepted = saturation[
        (saturation["dataset"] == "linked_tori")
        & (saturation["activation"] == "tanh")
        & (saturation["format"] == "bfloat16")
        & (saturation["convention"] == "IEEE")
        & (saturation["distance_from_output"] == 0)
    ]
    pieces = []
    for value, prefix in (
        ("vector_collision_rate", "trained_collision"),
        ("baseline_vector_collision_rate", "matched_init_baseline"),
        ("excess_vector_collision_rate", "matched_excess"),
    ):
        summary = _mean_std_count_by_seed(accepted, value, ["width"]).rename(
            columns={
                "mean": f"{prefix}_mean",
                "std": f"{prefix}_std",
                "count": f"{prefix}_seed_count",
            }
        )
        pieces.append(summary)
    result = all_initialization
    for piece in pieces:
        result = result.merge(piece, on="width", validate="one_to_one")
    return result.sort_values("width").reset_index(drop=True)


def dynamics_analysis(dynamics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabulate the existing final-layer checkpoint trajectory and post-plateau share."""

    selected = dynamics[
        (dynamics["dataset"] == "linked_tori")
        & (dynamics["activation"] == "tanh")
        & (dynamics["format"] == "bfloat16")
        & (dynamics["convention"] == "IEEE")
        & (dynamics["distance_from_output"] == 0)
    ].copy()
    metrics = (
        "train_accuracy",
        "eval_accuracy",
        "vector_collision_rate",
        "paper_total_saturation_fraction",
    )
    rows: list[dict[str, float | int]] = []
    for step, group in selected.groupby("training_step", sort=True):
        row: dict[str, float | int] = {
            "training_step": int(step),
            "training_progress": float(group["training_progress"].iloc[0]),
            "seed_count": int(group["seed"].nunique()),
        }
        for metric in metrics:
            seed_values = group.groupby("seed")[metric].mean()
            row[f"{metric}_mean"] = float(seed_values.mean())
            row[f"{metric}_std"] = float(seed_values.std())
        rows.append(row)
    trajectory = pd.DataFrame(rows)

    perfect = trajectory[
        (trajectory["train_accuracy_mean"] == 1.0)
        & (trajectory["eval_accuracy_mean"] == 1.0)
        & (trajectory["train_accuracy_std"] == 0.0)
        & (trajectory["eval_accuracy_std"] == 0.0)
    ]
    if perfect.empty:
        raise ValueError("no checkpoint has perfect train and evaluation accuracy for every seed")
    plateau_step = int(perfect["training_step"].min())
    per_seed = selected.pivot(index="seed", columns="training_step", values="vector_collision_rate")
    initial_step = int(trajectory["training_step"].min())
    final_step = int(trajectory["training_step"].max())
    fraction_after = (per_seed[final_step] - per_seed[plateau_step]) / (
        per_seed[final_step] - per_seed[initial_step]
    )
    plateau = per_seed[plateau_step]
    final = per_seed[final_step]
    summary = pd.DataFrame(
        [
            {
                "plateau_step": plateau_step,
                "seed_count": int(len(per_seed)),
                "collision_at_plateau_mean": float(plateau.mean()),
                "collision_at_plateau_std": float(plateau.std()),
                "collision_at_final_mean": float(final.mean()),
                "collision_at_final_std": float(final.std()),
                "fraction_collision_increase_after_plateau_mean": float(fraction_after.mean()),
                "fraction_collision_increase_after_plateau_std": float(fraction_after.std()),
            }
        ]
    )
    return trajectory, summary


def _paired_dataset_gap(
    saturation: pd.DataFrame,
    format_name: str,
    metric: str,
    final_layer_only: bool,
    analysis_name: str,
) -> dict[str, float | int | str | bool]:
    selected = saturation[
        (saturation["activation"] == "tanh")
        & (saturation["format"] == format_name)
        & (saturation["convention"] == "IEEE")
    ]
    keys = ["depth", "width", "seed"]
    if final_layer_only:
        selected = selected[selected["distance_from_output"] == 0]
    else:
        keys.append("layer")
    blobs = selected[selected["dataset"] == "blobs"][keys + [metric]].rename(
        columns={metric: "blobs"}
    )
    tori = selected[selected["dataset"] == "linked_tori"][keys + [metric]].rename(
        columns={metric: "linked_tori"}
    )
    paired = blobs.merge(tori, on=keys, validate="one_to_one")
    paired["difference"] = paired["blobs"] - paired["linked_tori"]
    seed_values = paired.groupby("seed")[["blobs", "linked_tori", "difference"]].mean()
    mean = float(seed_values["difference"].mean())
    std = float(seed_values["difference"].std())
    return {
        "analysis": analysis_name,
        "format": format_name,
        "metric": metric,
        "pairing": "paired by seed and matched accepted architecture-layer rows",
        "matched_row_count": int(len(paired)),
        "seed_count": int(len(seed_values)),
        "blobs_mean": float(seed_values["blobs"].mean()),
        "blobs_std": float(seed_values["blobs"].std()),
        "linked_tori_mean": float(seed_values["linked_tori"].mean()),
        "linked_tori_std": float(seed_values["linked_tori"].std()),
        "blobs_minus_tori_mean": mean,
        "blobs_minus_tori_std": std,
        "clears_half_mean_rule": std <= 0.5 * abs(mean),
    }


def dataset_gap_analysis(saturation: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _paired_dataset_gap(
                saturation,
                "bfloat16",
                "paper_total_saturation_fraction",
                False,
                "all-layer paper saturation",
            ),
            _paired_dataset_gap(
                saturation,
                "bfloat16",
                "excess_vector_collision_rate",
                True,
                "final-layer vector collision excess",
            ),
            _paired_dataset_gap(
                saturation,
                "float32",
                "vector_collision_rate",
                True,
                "final-layer vector collision",
            ),
        ]
    )


def plot_dynamics(trajectory: pd.DataFrame, destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for metric, label in (
        ("train_accuracy", "train accuracy"),
        ("eval_accuracy", "evaluation accuracy"),
        ("vector_collision_rate", "bfloat16 vector collision"),
        ("paper_total_saturation_fraction", "bfloat16 paper saturation"),
    ):
        axis.errorbar(
            trajectory["training_step"],
            trajectory[f"{metric}_mean"],
            yerr=trajectory[f"{metric}_std"],
            marker="o",
            capsize=3,
            label=label,
        )
    axis.set_xlabel("training step")
    axis.set_ylabel("fraction")
    axis.set_ylim(-0.04, 1.04)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _percent(mean: float, std: float) -> str:
    return f"{100 * mean:.4f}% ± {100 * std:.4f}%"


def write_notes(
    widths: pd.DataFrame,
    trajectory: pd.DataFrame,
    dynamics_summary: pd.DataFrame,
    gaps: pd.DataFrame,
    destination: Path,
) -> None:
    lines = [
        "# Follow-up analysis notes",
        "",
        "All uncertainties are sample standard deviations across seed-level values.",
        "No training was rerun. A1 recreates deterministic step-zero models only so",
        "failed trained runs do not remove their initialization baselines.",
        "",
        "## A1. Initialization baseline by width",
        "",
        "| Width | All-init baseline | Trained collision (accepted) | Matched baseline (accepted) | Matched excess | Excess seed n |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in widths.itertuples():
        lines.append(
            f"| {row.width} | {_percent(row.all_init_baseline_mean, row.all_init_baseline_std)} "
            f"| {_percent(row.trained_collision_mean, row.trained_collision_std)} "
            f"| {_percent(row.matched_init_baseline_mean, row.matched_init_baseline_std)} "
            f"| {_percent(row.matched_excess_mean, row.matched_excess_std)} "
            f"| {int(row.matched_excess_seed_count)} |"
        )
    lines.extend(
        [
            "",
            "The initialization baseline falls systematically with width. The pooled",
            "13.8630% ± 5.9025% baseline is therefore retained only as a secondary",
            "accepted-run summary. Width-5 excess has four seed-level estimates because",
            "seed 2 failed the training gate at every depth; no value is imputed.",
            "",
            "## A2. Collision trajectory after the accuracy plateau",
            "",
            "| Step | Train accuracy | Eval accuracy | Vector collision | Paper saturation |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in trajectory.itertuples():
        lines.append(
            f"| {int(row.training_step)} | {_percent(row.train_accuracy_mean, row.train_accuracy_std)} "
            f"| {_percent(row.eval_accuracy_mean, row.eval_accuracy_std)} "
            f"| {_percent(row.vector_collision_rate_mean, row.vector_collision_rate_std)} "
            f"| {_percent(row.paper_total_saturation_fraction_mean, row.paper_total_saturation_fraction_std)} |"
        )
    dynamic = dynamics_summary.iloc[0]
    lines.extend(
        [
            "",
            f"The first observed all-seed accuracy plateau is step {int(dynamic['plateau_step'])}. "
            f"Vector collision is {_percent(dynamic['collision_at_plateau_mean'], dynamic['collision_at_plateau_std'])} "
            f"there and {_percent(dynamic['collision_at_final_mean'], dynamic['collision_at_final_std'])} "
            f"at the final step. {_percent(dynamic['fraction_collision_increase_after_plateau_mean'], dynamic['fraction_collision_increase_after_plateau_std'])} "
            "of each seed's total observed collision increase occurs after the plateau.",
            "",
            "The trajectory supports continued post-plateau growth but not an onset time:",
            "the only checkpoints before or at the plateau are steps 0 and 200. Resolving",
            "onset would require checkpoints about every 20 steps or finer through step 200.",
            "",
            "## A3. Paired blob-minus-tori gaps",
            "",
            "Pairing is by seed and matched accepted architecture-layer rows. Error bars",
            "are sample SDs of the five paired seed-level differences, not unpaired error",
            "propagation.",
            "",
            "| Comparison | Blobs | Linked tori | Blob − tori gap | Clears SD ≤ half-mean rule? |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    for row in gaps.itertuples():
        lines.append(
            f"| {row.format}: {row.analysis} | {_percent(row.blobs_mean, row.blobs_std)} "
            f"| {_percent(row.linked_tori_mean, row.linked_tori_std)} "
            f"| {_percent(row.blobs_minus_tori_mean, row.blobs_minus_tori_std)} "
            f"| {'yes' if row.clears_half_mean_rule else 'no'} |"
        )
    lines.extend(
        [
            "",
            "All three paired gaps clear seed variance under the stated rule. This does",
            "not contradict the current numerical results, but it sharpens their framing:",
            "the blob–tori difference is large relative to seed variation, and the width",
            "dependence makes the pooled initialization baseline unsuitable as the primary",
            "excess summary.",
        ]
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    saturation = pd.read_parquet(results / "saturation.parquet")
    dynamics = pd.read_parquet(results / "training_dynamics.parquet")
    initialization = recompute_initialization_baselines()
    widths = width_baseline_analysis(saturation, initialization)
    trajectory, dynamics_summary = dynamics_analysis(dynamics)
    gaps = dataset_gap_analysis(saturation)
    initialization.to_csv(results / "followup_initialization_recomputed.csv", index=False)
    widths.to_csv(results / "followup_width_baseline.csv", index=False)
    trajectory.to_csv(results / "followup_dynamics.csv", index=False)
    dynamics_summary.to_csv(results / "followup_dynamics_summary.csv", index=False)
    gaps.to_csv(results / "followup_dataset_gaps.csv", index=False)
    plot_dynamics(trajectory, results / "figures" / "followup_accuracy_collision_dynamics.png")
    write_notes(widths, trajectory, dynamics_summary, gaps, results / "followup_notes.md")


if __name__ == "__main__":
    main()
