"""Per-project box plot of transition distance.

Three boxes per project: all non-stable mutants (grey), attributable mutants
(black), and mutation-specific mutants (blue). Non-stable distances come from
the JSON histories in data_seq: for each mutant, the S->K or K->S transition
closest to injection, measured from the bug's newest recorded revision.
Attributable and mutation-specific distances come from validated.xlsx.
mu_spec_IDs.csv is not used here; it has no distance column.

Paths are inside this directory, so the script does not depend on a
surrounding checkout. Put the inputs in RQs/data, or pass them explicitly:

  python rq_transition_timing_by_project.py \
      --data-seq data/data_seq --workbook data/validated.xlsx
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook


HERE = Path(__file__).resolve().parent
DATA_SEQ = Path("data") / "data_seq"
WORKBOOK = Path("data") / "validated.xlsx"
OUTPUT_STEM = Path("transition_timing_by_project_combined")
CROSS_TRANSITIONS = {"S->K", "K->S"}
REQUIRED_COLUMNS = {
    "project",
    "bug",
    "direction",
    "mutation_specific",
    "revisions_before_injection",
    "window_revisions",
    "relative_position",
}


@dataclass(frozen=True)
class NonstableMutant:
    project: str
    nearest_transition_distance: int
    normalized_nearest_position: float


@dataclass(frozen=True)
class AttributedMutant:
    project: str
    distance: int
    window: int
    mutation_specific: bool


def load_nonstable(data_directory: Path) -> list[NonstableMutant]:
    mutants: list[NonstableMutant] = []
    for path in sorted(data_directory.glob("*_rq1.json")):
        project = path.stem[:-4].rsplit("-", 1)[0]
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        newest = max(
            (
                int(item["index"])
                for record in data.values()
                for item in record.get("sequence") or []
            ),
            default=0,
        )
        for record in data.values():
            distances = [
                newest - int(transition["to_index"]) + 1
                for transition in record.get("transition_records") or []
                if transition.get("transition") in CROSS_TRANSITIONS
            ]
            if not distances:
                continue
            nearest = min(distances)
            mutants.append(
                NonstableMutant(
                    project=project,
                    nearest_transition_distance=nearest,
                    normalized_nearest_position=(
                        (nearest - 1) / newest if newest > 0 else 0.0
                    ),
                )
            )
    if not mutants:
        raise ValueError("No status-changing mutants in data/data_seq")
    return mutants


def load_attributed(workbook: Path) -> list[AttributedMutant]:
    worksheet = load_workbook(workbook, read_only=True, data_only=True)["data"]
    headers = [cell.value for cell in next(worksheet.iter_rows(min_row=1, max_row=1))]
    missing = REQUIRED_COLUMNS.difference(headers)
    if missing:
        raise ValueError(f"Missing required columns in the workbook: {sorted(missing)}")

    mutants: list[AttributedMutant] = []
    for row_number, values in enumerate(
        worksheet.iter_rows(min_row=2, values_only=True), start=2
    ):
        record = dict(zip(headers, values))
        distance = int(record["revisions_before_injection"])
        window = int(record["window_revisions"])
        flag = str(record["mutation_specific"]).strip().lower()
        expected_relative = (window - distance) / (window - 1)
        if not np.isclose(float(record["relative_position"]), expected_relative, atol=5e-5):
            raise ValueError(f"Relative-position mismatch on row {row_number}")
        mutants.append(
            AttributedMutant(
                project=str(record["project"]).strip(),
                distance=distance,
                window=window,
                mutation_specific=flag == "yes",
            )
        )
    if not mutants:
        raise ValueError("No mutant rows in the workbook")
    return mutants


def create_combined_boxplot(
    mutants: list[NonstableMutant],
    attributed: list[AttributedMutant],
    output_stem: Path,
) -> None:
    groups = [
        (
            "Non-stable",
            "#9A9A9A",
            mutants,
            lambda mutant: float(mutant.nearest_transition_distance),
            lambda mutant: mutant.normalized_nearest_position,
        ),
        (
            "Attributable",
            "#202020",
            attributed,
            lambda mutant: float(mutant.distance),
            lambda mutant: (mutant.distance - 1) / (mutant.window - 1),
        ),
        (
            "Mutation-specific",
            "#2D6F9E",
            [mutant for mutant in attributed if mutant.mutation_specific],
            lambda mutant: float(mutant.distance),
            lambda mutant: (mutant.distance - 1) / (mutant.window - 1),
        ),
    ]
    attributed_counts = Counter(mutant.project for mutant in attributed)
    nonstable_counts = Counter(mutant.project for mutant in mutants)
    projects = sorted(
        set(attributed_counts) | set(nonstable_counts),
        key=lambda project: (
            -attributed_counts[project],
            -nonstable_counts[project],
            project,
        ),
    )
    positions = np.arange(len(projects), 0, -1, dtype=float)
    offsets = [0.26, 0.0, -0.26]

    figure, axes = plt.subplots(1, 2, figsize=(7.45, 6.4), sharey=True)
    axis_specs = [
        (
            "Absolute transition distance",
            "Revisions before injection\n(1 = nearest)",
            (-1, 101),
        ),
        (
            "Window-normalized transition distance",
            "Normalized history position\n(0 = nearest, 1 = oldest)",
            (-0.02, 1.02),
        ),
    ]
    legend_handles = []
    for axis_index, (axis, (title, xlabel, limits)) in enumerate(zip(axes, axis_specs)):
        for (label, color, group, absolute, normalized), offset in zip(groups, offsets):
            metric = absolute if axis_index == 0 else normalized
            data = []
            group_positions = []
            for project, position in zip(projects, positions):
                values = [metric(mutant) for mutant in group if mutant.project == project]
                if values:
                    data.append(np.asarray(values))
                    group_positions.append(position + offset)

            boxes = axis.boxplot(
                data,
                positions=group_positions,
                widths=0.21,
                vert=False,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "#D55E00", "linewidth": 1.5},
                whiskerprops={"color": color, "linewidth": 1.0},
                capprops={"color": color, "linewidth": 1.0},
                flierprops={
                    "marker": "o",
                    "markersize": 2.6,
                    "markerfacecolor": "none",
                    "markeredgecolor": color,
                    "markeredgewidth": 0.6,
                    "alpha": 0.8,
                },
                boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.78},
            )
            if axis_index == 0:
                legend_handles.append((boxes["boxes"][0], label))
            for position, values in zip(group_positions, data):
                if len(values) == 1:
                    axis.scatter(
                        values,
                        [position],
                        s=24,
                        color=color,
                        edgecolors="#FFFFFF",
                        linewidths=0.5,
                        zorder=3,
                    )

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_xlim(*limits)
        axis.grid(axis="x", color="#E3E3E3", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for boundary in positions[:-1] - 0.5:
            axis.axhline(boundary, color="#EBEBEB", linewidth=0.6, zorder=0)

    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(
        [
            f"{project}\n(n={nonstable_counts[project]:,}/"
            f"{attributed_counts[project]}/"
            f"{sum(mutant.project == project for mutant in groups[2][2])})"
            for project in projects
        ]
    )
    axes[0].set_ylabel("Project")
    figure.legend(
        [handle for handle, _ in legend_handles],
        [label for _, label in legend_handles],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    figure.subplots_adjust(left=0.16, right=0.985, top=0.93, bottom=0.18, wspace=0.18)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-seq", type=Path, default=DATA_SEQ)
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--output-stem", type=Path, default=OUTPUT_STEM)
    args = parser.parse_args()
    for name in ("data_seq", "workbook", "output_stem"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, HERE / value)
    if not args.data_seq.is_dir():
        raise FileNotFoundError(
            "data/data_seq was not found. Pass --data-seq with the directory of *_rq1.json files."
        )
    if not args.workbook.is_file():
        raise FileNotFoundError(
            "data/validated.xlsx was not found. Pass --workbook with the workbook path."
        )

    mutants = load_nonstable(args.data_seq)
    attributed = load_attributed(args.workbook)
    create_combined_boxplot(mutants, attributed, args.output_stem)
    specific = sum(mutant.mutation_specific for mutant in attributed)
    print(f"Non-stable mutants: {len(mutants)}")
    print(f"Attributable mutants: {len(attributed)}")
    print(f"Mutation-specific mutants: {specific}")
    print("Wrote transition_timing_by_project_combined.pdf")


if __name__ == "__main__":
    main()
