import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
RES_DIR = Path("D://res")
PATTERN = "*_rq1.json"

CATS = [
    "S->S",
    "K->K",
    "S->K",
    "K->S",
    "S->K->S",
    "K->S->K",
    "Recurrent",
]

SAVE_FIGS = False
FIG_DIR = RES_DIR / "rq1_figs"


# ============================================================
# HELPERS
# ============================================================
def load_json(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def get_mutants(data):
    if isinstance(data, dict) and "mutants" in data and isinstance(data["mutants"], dict):
        return data["mutants"]
    return data


def get_count(m, *names):
    for name in names:
        if name in m:
            return int(m.get(name) or 0)
    return 0


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def cell(a, b):
    return f"{a} ({pct(a, b):.1f}%)"


def instance_from_path(path):
    name = path.name
    if name.endswith("_rq1.json"):
        return name[:-len("_rq1.json")]
    return path.stem


def project_from_instance(instance):
    m = re.match(r"(.+)-\d+b$", instance)
    if m:
        return m.group(1)
    return instance


def latex_escape(s):
    return str(s).replace("_", r"\_")


def latex_pattern(cat):
    return "$" + cat.replace("->", r"{\to}") + "$"


def stat_summary(xs):
    xs = np.array(xs, dtype=float)

    if len(xs) == 0:
        return 0, 0, 0, 0, 0, 0, 0

    mean = float(np.mean(xs))
    sd = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    median = float(np.median(xs))
    q1 = float(np.percentile(xs, 25))
    q3 = float(np.percentile(xs, 75))
    mn = float(np.min(xs))
    mx = float(np.max(xs))

    return mean, sd, median, q1, q3, mn, mx


def classify_mutant(m):
    """
    Non-overlapping classification.

    Priority:
      1. Recurrent: more than one length-3 oscillation pattern overall
      2. Exactly one S->K->S
      3. Exactly one K->S->K
      4. One-way S->K
      5. One-way K->S
      6. Stable S->S
      7. Stable K->K
      8. Noisy
    """

    ss = get_count(m, "S_to_S", "S->S")
    kk = get_count(m, "K_to_K", "K->K")
    sk = get_count(m, "S_to_K", "S->K")
    ks = get_count(m, "K_to_S", "K->S")

    num_sks = get_count(m, "num_SKS", "S_K_S", "S->K->S")
    num_ksk = get_count(m, "num_KSK", "K_S_K", "K->S->K")

    valid = ss + kk + sk + ks

    if valid == 0:
        return "Noisy"

    # Three or more status changes.
    # Examples:
    # S->K->S->K has num_SKS=1 and num_KSK=1.
    # K->S->K->S has num_KSK=1 and num_SKS=1.
    if num_sks + num_ksk >= 2:
        return "Recurrent"

    # Exactly two status changes.
    if num_sks == 1 and num_ksk == 0:
        return "S->K->S"

    if num_ksk == 1 and num_sks == 0:
        return "K->S->K"

    # Exactly one direction of change.
    if sk > 0 and ks == 0:
        return "S->K"

    if ks > 0 and sk == 0:
        return "K->S"

    # Stable patterns.
    if ss > 0 and kk == 0:
        return "S->S"

    if kk > 0 and ss == 0:
        return "K->K"

    return "Noisy"


# ============================================================
# LOAD INSTANCE FILES
# ============================================================
files = sorted(RES_DIR.glob(PATTERN))

instance_rows = []

for path in files:
    data = load_json(path)
    mutants = get_mutants(data)

    counts = Counter()

    for _, m in mutants.items():
        counts[classify_mutant(m)] += 1

    n_all = len(mutants)
    n_class = sum(counts[c] for c in CATS)

    instance = instance_from_path(path)
    project = project_from_instance(instance)

    row = {
        "Instance": instance,
        "Project": project,
        "# All": n_all,
        "# Classifiable": n_class,
        "Noisy": counts["Noisy"],
    }

    for c in CATS:
        row[c] = counts[c]
        row[f"{c}_pct"] = pct(counts[c], n_class)

    instance_rows.append(row)

if not instance_rows:
    raise RuntimeError(f"No files found in {RES_DIR} matching {PATTERN}")


# ============================================================
# AGGREGATE BY PROJECT
# ============================================================
project_groups = defaultdict(list)

for r in instance_rows:
    project_groups[r["Project"]].append(r)

project_rows = []

for project, group in sorted(project_groups.items()):
    agg = Counter()
    agg["# Instances"] = len(group)

    for r in group:
        agg["# All"] += r["# All"]
        agg["# Classifiable"] += r["# Classifiable"]
        agg["Noisy"] += r["Noisy"]

        for c in CATS:
            agg[c] += r[c]

    row = {
        "Project": project,
        "# Instances": agg["# Instances"],
        "# All": agg["# All"],
        "# Classifiable": agg["# Classifiable"],
        "Noisy": agg["Noisy"],
    }

    for c in CATS:
        row[c] = agg[c]
        row[f"{c}_pct"] = pct(agg[c], agg["# Classifiable"])

    project_rows.append(row)


# ============================================================
# TOTAL
# ============================================================
total = Counter()

for r in project_rows:
    total["# Instances"] += r["# Instances"]
    total["# All"] += r["# All"]
    total["# Classifiable"] += r["# Classifiable"]
    total["Noisy"] += r["Noisy"]

    for c in CATS:
        total[c] += r[c]


# ============================================================
# PRINT PROJECT-LEVEL TABLE
# ============================================================
print("\nRQ1 Pattern Prevalence Aggregated by Project")
print("=" * 190)

print(
    f"{'Project':<18} "
    f"{'#Inst':>6} "
    f"{'#All':>8} "
    f"{'#Class':>8} "
    f"{'S->S':>16} "
    f"{'K->K':>16} "
    f"{'S->K':>16} "
    f"{'K->S':>16} "
    f"{'S->K->S':>16} "
    f"{'K->S->K':>16} "
    f"{'Recurrent':>16} "
    f"{'Noisy':>8}"
)

print("-" * 190)

for r in project_rows:
    d = r["# Classifiable"]

    print(
        f"{r['Project']:<18} "
        f"{r['# Instances']:>6} "
        f"{r['# All']:>8} "
        f"{r['# Classifiable']:>8} "
        f"{cell(r['S->S'], d):>16} "
        f"{cell(r['K->K'], d):>16} "
        f"{cell(r['S->K'], d):>16} "
        f"{cell(r['K->S'], d):>16} "
        f"{cell(r['S->K->S'], d):>16} "
        f"{cell(r['K->S->K'], d):>16} "
        f"{cell(r['Recurrent'], d):>16} "
        f"{r['Noisy']:>8}"
    )

print("-" * 190)

d = total["# Classifiable"]

print(
    f"{'Total':<18} "
    f"{total['# Instances']:>6} "
    f"{total['# All']:>8} "
    f"{total['# Classifiable']:>8} "
    f"{cell(total['S->S'], d):>16} "
    f"{cell(total['K->K'], d):>16} "
    f"{cell(total['S->K'], d):>16} "
    f"{cell(total['K->S'], d):>16} "
    f"{cell(total['S->K->S'], d):>16} "
    f"{cell(total['K->S->K'], d):>16} "
    f"{cell(total['Recurrent'], d):>16} "
    f"{total['Noisy']:>8}"
)


# ============================================================
# PRINT ACROSS-PROJECT SUMMARY STATS
# ============================================================
print("\nAcross-Project Percentage Statistics")
print("=" * 110)

print(
    f"{'Category':<12} "
    f"{'Mean ± SD':>20} "
    f"{'Median':>10} "
    f"{'IQR':>18} "
    f"{'Min':>8} "
    f"{'Max':>8}"
)

print("-" * 110)

for c in CATS:
    vals = [r[f"{c}_pct"] for r in project_rows if r["# Classifiable"] > 0]
    mean, sd, med, q1, q3, mn, mx = stat_summary(vals)

    print(
        f"{c:<12} "
        f"{mean:>8.2f} ± {sd:<8.2f} "
        f"{med:>10.2f} "
        f"[{q1:>5.2f}, {q3:>5.2f}] "
        f"{mn:>8.2f} "
        f"{mx:>8.2f}"
    )


# ============================================================
# PRINT ACROSS-INSTANCE SUMMARY STATS
# ============================================================
print("\nAcross-Instance Percentage Statistics")
print("=" * 110)

print(
    f"{'Category':<12} "
    f"{'Mean ± SD':>20} "
    f"{'Median':>10} "
    f"{'IQR':>18} "
    f"{'Min':>8} "
    f"{'Max':>8}"
)

print("-" * 110)

for c in CATS:
    vals = [r[f"{c}_pct"] for r in instance_rows if r["# Classifiable"] > 0]
    mean, sd, med, q1, q3, mn, mx = stat_summary(vals)

    print(
        f"{c:<12} "
        f"{mean:>8.2f} ± {sd:<8.2f} "
        f"{med:>10.2f} "
        f"[{q1:>5.2f}, {q3:>5.2f}] "
        f"{mn:>8.2f} "
        f"{mx:>8.2f}"
    )


# ============================================================
# PRINT LATEX TABLE: MAIN PAPER
# ============================================================
print("\nLaTeX Table: Main Paper")
print("=" * 150)

print(r"\begin{table*}[t]")
print(r"\centering")
print(
    r"\caption{Prevalence of longitudinal mutant-status patterns by project. "
    r"Each cell reports the number of mutants and the percentage among classifiable mutants "
    r"after pooling all bug instances of the project.}"
)
print(r"\label{tab:rq1_pattern_prevalence}")
print(r"\begin{tabular}{lrrrrrrrrrr}")
print(r"\toprule")
print(
    r"Project & \# Inst. & \# All & \# Classifiable & "
    r"$S{\to}S$ & $K{\to}K$ & $S{\to}K$ & $K{\to}S$ & "
    r"$S{\to}K{\to}S$ & $K{\to}S{\to}K$ & Recurrent \\"
)
print(r"\midrule")

for r in project_rows:
    d = r["# Classifiable"]

    print(
        f"{latex_escape(r['Project'])} & "
        f"{r['# Instances']} & "
        f"{r['# All']} & "
        f"{r['# Classifiable']} & "
        f"{cell(r['S->S'], d)} & "
        f"{cell(r['K->K'], d)} & "
        f"{cell(r['S->K'], d)} & "
        f"{cell(r['K->S'], d)} & "
        f"{cell(r['S->K->S'], d)} & "
        f"{cell(r['K->S->K'], d)} & "
        f"{cell(r['Recurrent'], d)} \\\\"
    )

print(r"\midrule")

d = total["# Classifiable"]

print(
    f"Total & "
    f"{total['# Instances']} & "
    f"{total['# All']} & "
    f"{total['# Classifiable']} & "
    f"{cell(total['S->S'], d)} & "
    f"{cell(total['K->K'], d)} & "
    f"{cell(total['S->K'], d)} & "
    f"{cell(total['K->S'], d)} & "
    f"{cell(total['S->K->S'], d)} & "
    f"{cell(total['K->S->K'], d)} & "
    f"{cell(total['Recurrent'], d)} \\\\"
)

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table*}")


# ============================================================
# PRINT LATEX SUMMARY STATS TABLE
# ============================================================
print("\nLaTeX Table: Summary Statistics")
print("=" * 130)

print(r"\begin{table}[t]")
print(r"\centering")
print(
    r"\caption{Across-project distribution of pattern prevalence. "
    r"Values are percentages among classifiable mutants.}"
)
print(r"\label{tab:rq1_across_project_stats}")
print(r"\begin{tabular}{lrrrr}")
print(r"\toprule")
print(r"Pattern & Mean $\pm$ SD & Median & IQR & Range \\")
print(r"\midrule")

for c in CATS:
    vals = [r[f"{c}_pct"] for r in project_rows if r["# Classifiable"] > 0]
    mean, sd, med, q1, q3, mn, mx = stat_summary(vals)

    print(
        f"{latex_pattern(c)} & "
        f"{mean:.1f} $\\pm$ {sd:.1f} & "
        f"{med:.1f} & "
        f"[{q1:.1f}, {q3:.1f}] & "
        f"[{mn:.1f}, {mx:.1f}] \\\\"
    )

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")


# ============================================================
# FIGURE SETTINGS
# ============================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

if SAVE_FIGS:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIGURE 1: PROJECT-LEVEL HEATMAP
# ============================================================
projects = [r["Project"] for r in project_rows]

heat = np.array([
    [r[f"{c}_pct"] for c in CATS]
    for r in project_rows
])

fig, ax = plt.subplots(figsize=(12.5, max(4.5, 0.48 * len(projects))))

im = ax.imshow(heat, aspect="auto", cmap="Blues", vmin=0, vmax=100)

ax.set_xticks(np.arange(len(CATS)))
ax.set_xticklabels(CATS, rotation=25, ha="right")
ax.set_yticks(np.arange(len(projects)))
ax.set_yticklabels(projects)

ax.set_title("Pattern prevalence by project")
ax.set_xlabel("Longitudinal pattern")
ax.set_ylabel("Project")

for i in range(len(projects)):
    for j in range(len(CATS)):
        val = heat[i, j]
        text_color = "white" if val >= 55 else "black"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color, fontsize=9)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("% among classifiable mutants")

fig.tight_layout()

if SAVE_FIGS:
    fig.savefig(FIG_DIR / "rq1_project_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "rq1_project_heatmap.png", dpi=300, bbox_inches="tight")

plt.show()


# ============================================================
# FIGURE 2: PROJECT-LEVEL STACKED BAR
# ============================================================
projects = [r["Project"] for r in project_rows]
x = np.arange(len(projects))
bottom = np.zeros(len(projects))

fig, ax = plt.subplots(figsize=(max(11.0, 0.9 * len(projects)), 5.5))

for c in CATS:
    vals = np.array([r[f"{c}_pct"] for r in project_rows])
    ax.bar(x, vals, bottom=bottom, label=c)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(projects, rotation=35, ha="right")
ax.set_ylabel("% among classifiable mutants")
ax.set_ylim(0, 100)
ax.set_title("Pattern prevalence aggregated by project")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.24), frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

if SAVE_FIGS:
    fig.savefig(FIG_DIR / "rq1_project_stacked_bar.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "rq1_project_stacked_bar.png", dpi=300, bbox_inches="tight")

plt.show()


# ============================================================
# FIGURE 3: INSTANCE-LEVEL DOT PLOT GROUPED BY PROJECT
# ============================================================
fig, axes = plt.subplots(
    1,
    len(CATS),
    figsize=(21, max(4.5, 0.48 * len(projects))),
    sharey=True,
)

if len(CATS) == 1:
    axes = [axes]

project_to_y = {p: i for i, p in enumerate(projects)}
rng = np.random.default_rng(0)

for ax, cat in zip(axes, CATS):
    for p in projects:
        group = project_groups[p]
        y_base = project_to_y[p]

        vals = [
            r[f"{cat}_pct"]
            for r in group
            if r["# Classifiable"] > 0
        ]

        if not vals:
            continue

        jitter = rng.normal(0, 0.045, size=len(vals))
        y = np.full(len(vals), y_base) + jitter

        ax.scatter(vals, y, alpha=0.75, s=35)

        med = np.median(vals)
        ax.plot([med, med], [y_base - 0.25, y_base + 0.25], linewidth=2)

    ax.set_title(cat)
    ax.set_xlim(-2, 102)
    ax.set_xlabel("%")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_yticks(np.arange(len(projects)))
axes[0].set_yticklabels(projects)
axes[0].set_ylabel("Project")

fig.suptitle("Instance-level variation within each project", y=1.02)
fig.tight_layout()

if SAVE_FIGS:
    fig.savefig(FIG_DIR / "rq1_instance_dotplot_by_project.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "rq1_instance_dotplot_by_project.png", dpi=300, bbox_inches="tight")

plt.show()


# ============================================================
# FIGURE 4: INSTANCE-LEVEL STACKED BAR
# ============================================================
instance_rows_sorted = sorted(instance_rows, key=lambda r: (r["Project"], r["Instance"]))

instances = [r["Instance"] for r in instance_rows_sorted]
projects_for_instances = [r["Project"] for r in instance_rows_sorted]

x = np.arange(len(instances))
bottom = np.zeros(len(instances))

fig, ax = plt.subplots(figsize=(max(12, 0.28 * len(instances)), 5.5))

for c in CATS:
    vals = np.array([r[f"{c}_pct"] for r in instance_rows_sorted])
    ax.bar(x, vals, bottom=bottom, label=c, width=0.85)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(instances, rotation=90)
ax.set_ylabel("% among classifiable mutants")
ax.set_ylim(0, 100)
ax.set_title("Pattern prevalence by bug instance")

last_project = None

for i, p in enumerate(projects_for_instances):
    if last_project is not None and p != last_project:
        ax.axvline(i - 0.5, color="black", linewidth=0.8, alpha=0.35)
    last_project = p

ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.24), frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

if SAVE_FIGS:
    fig.savefig(FIG_DIR / "rq1_instance_stacked_bar.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "rq1_instance_stacked_bar.png", dpi=300, bbox_inches="tight")

plt.show()