"""Per-project regional prevalence of attributable and mutation-specific mutants.

Every categorized mutant in data_seq (stable S, stable K, or status-transitioning)
is assigned to r1-r4 with the same analysis as rq2_region_assign.py. Attributable
mutants are those listed in mu_region.csv; mutation-specific ones are the subset
with mutation_specific == "yes".

Both tables use the same denominator, the categorized mutants in the
project-region pair. The pattern table restricts the numerator to one
mutant-status pattern:

  rq2_region_table.tex          attributable (or mutation-specific) / categorized
  rq2_region_pattern_table.tex  attributable with pattern h / categorized

Usage (from py-tracker/RQs):  python rq2_region_table.py
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict

from rq2_region_assign import DATA_DIR, DATA_SEQ, HERE, OUT_CSV, analyze_bug, warnings

REGIONS = ["r1", "r2", "r3", "r4"]
CATEGORIES = [("attr", "Attrib."), ("spec", "Mut.-spec.")]
OUT_ALL = DATA_DIR / "mu_region_all.csv"
OUT_TEX = HERE / "rq2_region_table.tex"
OUT_PATTERN_TEX = HERE / "rq2_region_pattern_table.tex"

# Set to False to report S→K→S, K→S→K and Recur. as separate rows.
COMBINE_FLUCTUATING = True
FLUCTUATING = ("S→K→S", "K→S→K", "Recur.")
PATTERNS = (["S→K", "K→S", "Fluct."] if COMBINE_FLUCTUATING
            else ["S→K", "K→S", *FLUCTUATING])
PATTERN_TEX = {
    "S→K": r"S$\rightarrow$K",
    "K→S": r"K$\rightarrow$S",
    "S→K→S": r"S$\rightarrow$K$\rightarrow$S",
    "K→S→K": r"K$\rightarrow$S$\rightarrow$K",
    "Recur.": "Recur.",
    "Fluct.": "Fluct.",
}

# Reference "Trans." percentages from the paper's RQ2 table, used only to
# sanity-check the region assignment.
PAPER_TRANS = {
    "black": [11.4, 6.5, 6.3, 0.3],
    "cookiecutter": [0.0, 0.0, 5.9, 0.0],
    "fastapi": [0.0, 3.3, 1.6, 2.1],
    "httpie": [38.1, 35.8, 1.4, 7.0],
    "sanic": [0.0, None, 23.7, None],
    "thefuck": [12.0, 4.3, 5.2, 7.2],
    "tqdm": [25.1, 41.9, 37.5, 29.7],
    "youtube-dl": [8.0, 1.9, 2.3, None],
}

CAPTION_DETAIL = (
    "Percentages are computed independently within each project-region pair. "
    "Boldface marks, for each project and category, the region with the highest percentage."
)


def is_categorized(m):
    return m["has_status_transition"] or m["S_to_S"] > 0 or m["K_to_K"] > 0


def status_pattern(m):
    """RQ1 pattern of a transitioning mutant; None for stable mutants."""
    if not m["has_status_transition"]:
        return None
    n = m["S_to_K"] + m["K_to_S"]
    if n == 1:
        pat = "S→K" if m["S_to_K"] else "K→S"
    elif n == 2:
        pat = "S→K→S" if m["num_SKS"] else "K→S→K"
    else:
        pat = "Recur."
    return "Fluct." if COMBINE_FLUCTUATING and pat in FLUCTUATING else pat


def table_region(res):
    """Fold mutants outside r1-r4 into the nearest region."""
    region = res["region"]
    if region in REGIONS:
        return region
    if region == "module-level" and "file is fix-modified" in res["note"]:
        return "r3"
    if region in ("outside", "module-level"):
        return "r4"
    if res["note"].startswith("file missing"):
        return "r4"
    return None


def fmt(v):
    return "NA" if v is None else f"{v:.2f}\\%"


def tex_row(lead, values, bold=True):
    """values[region][category] -> float or None; bold the max region per category."""
    best = {}
    for key, _ in CATEGORIES:
        vals = [values[reg][key] for reg in REGIONS if values[reg][key] is not None]
        top = max(vals) if vals and bold else None
        best[key] = top if top else None
    out = [lead + " "]
    for i, reg in enumerate(REGIONS):
        cells = []
        for key, _ in CATEGORIES:
            v = values[reg][key]
            s = fmt(v)
            if v is not None and best[key] is not None and round(v, 2) == round(best[key], 2):
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        out.append("& " + " & ".join(cells) + (r" \\" if i == 3 else " "))
    return out


def tex_header(caption, label, lead_cols):
    """lead_cols: list of row-header column titles (Project, [Pattern])."""
    n_lead = len(lead_cols)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{",
        caption,
        r"}",
        f"\\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        f"\\begin{{tabular}}{{{'l' * n_lead} *{{8}}{{r}}}}",
        r"\toprule",
        " & ".join(f"\\multirow{{2}}{{*}}{{\\textbf{{{c}}}}}" for c in lead_cols) + " ",
    ]
    lines += [f"& \\multicolumn{{2}}{{c}}{{\\textbf{{Region {i} ($r_{i}$)}}}} "
              + (r"\\" if i == 4 else "") for i in range(1, 5)]
    lines.append(" ".join(f"\\cmidrule(lr){{{n_lead + 2 * i + 1}-{n_lead + 2 * i + 2}}}"
                          for i in range(4)))
    header = [("& " * (n_lead - 1) if i == 0 else "")
              + f"& \\textbf{{{CATEGORIES[0][1]}}} & \\textbf{{{CATEGORIES[1][1]}}} "
              for i in range(4)]
    header[-1] += r"\\ "
    return lines + header + [r"\midrule", ""]


TEX_FOOTER = [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]

PATTERN_HEAD = {"S→K": r"$\mathbf{SK}$", "K→S": r"$\mathbf{KS}$",
                "S→K→S": r"$\mathbf{SKS}$", "K→S→K": r"$\mathbf{KSK}$",
                "Recur.": r"\textbf{Rec.}", "Fluct.": r"\textbf{Fluct.}"}


def pattern_table(caption, projects, values, mean):
    """Patterns as columns: region > category > pattern. values[p][reg][(key, pat)]."""
    n = len(PATTERNS)
    per_reg = n * len(CATEGORIES)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{",
        caption,
        r"}",
        r"\label{tab:rq2_attributable_pattern}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{l *{{{len(REGIONS)}}}{{{'r' * n}|{'r' * n}}}}}",
        r"\toprule",
        r"\multirow{3}{*}{\textbf{Project}} ",
    ]
    lines += [f"& \\multicolumn{{{per_reg}}}{{c}}{{\\textbf{{Region {i} ($r_{i}$)}}}} "
              + (r"\\" if i == len(REGIONS) else "") for i in range(1, len(REGIONS) + 1)]
    lines.append(" ".join(f"\\cmidrule(lr){{{2 + i * per_reg}-{1 + (i + 1) * per_reg}}}"
                          for i in range(len(REGIONS))))
    cat_head = (f"& \\multicolumn{{{n}}}{{c|}}{{\\textbf{{{CATEGORIES[0][1]}}}}} "
                f"& \\multicolumn{{{n}}}{{c}}{{\\textbf{{{CATEGORIES[1][1]}}}}}")
    lines.append(" ".join([cat_head] * len(REGIONS)) + r" \\")
    lines.append(" ".join(f"\\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}"
                          for i in range(len(REGIONS) * len(CATEGORIES))))
    pat_head = " & ".join(PATTERN_HEAD[p] for p in PATTERNS)
    lines += [f"& {pat_head} & {pat_head}" + (r" \\" if i == len(REGIONS) - 1 else "")
              for i in range(len(REGIONS))]
    lines += [r"\midrule", ""]

    def row(label, vals, bold=True):
        best = {}
        for cell in vals[REGIONS[0]]:
            got = [vals[reg][cell] for reg in REGIONS if vals[reg][cell] is not None]
            top = max(got) if got and bold else None
            best[cell] = top if top else None
        out = [label + " "]
        for i, reg in enumerate(REGIONS):
            cells = []
            for key, _ in CATEGORIES:
                for pat in PATTERNS:
                    v = vals[reg][(key, pat)]
                    s = "NA" if v is None else f"{v:.2f}\\%"
                    shown = None if v is None else round(v, 2)
                    top = best[(key, pat)]
                    if shown and top is not None and shown == round(top, 2):
                        s = f"\\textbf{{{s}}}"
                    cells.append(s)
            out.append("& " + " & ".join(cells) + (r" \\" if i == len(REGIONS) - 1 else ""))
        return out + [""]

    for p in projects:
        lines += row(p, values[p])
    lines += [r"\midrule", ""]
    lines += row("Proj. mean", mean, bold=False)
    return lines


def project_mean(per_project, projects):
    mean = {}
    for reg in REGIONS:
        mean[reg] = {}
        for key, _ in CATEGORIES:
            vals = [per_project[p][reg][key] for p in projects
                    if per_project[p][reg][key] is not None]
            mean[reg][key] = sum(vals) / len(vals) if vals else None
    return mean


def main():
    attributable = {}
    for r in csv.DictReader(OUT_CSV.open()):
        attributable[(r["bug"], r["mutant_id"])] = r["mutation_specific"] == "yes"

    all_rows = []
    for path in sorted(DATA_SEQ.glob("*_rq1.json")):
        bug = path.name[: -len("_rq1.json")]
        project = re.sub(r"-\d+b$", "", bug)
        data = json.loads(path.read_text())
        mutants = [{"project": project, "bug": bug, "mutant_id": mid,
                    "transition": data[mid]["has_status_transition"],
                    "pattern": status_pattern(data[mid]) or ""}
                   for mid in data if is_categorized(data[mid])]
        print(f"analyzing {bug} ({len(mutants)} categorized mutants)", file=sys.stderr)
        results, _ = analyze_bug(project, bug, mutants)
        for res in results:
            key = (bug, res["mutant_id"])
            res["table_region"] = table_region(res)
            res["attributable"] = key in attributable
            res["mutation_specific"] = attributable.get(key, False)
            all_rows.append(res)

    fields = ["project", "bug", "mutant_id", "file", "method", "region",
              "table_region", "transition", "pattern", "attributable",
              "mutation_specific", "scope_match", "note"]
    with OUT_ALL.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    missing = set(attributable) - {(r["bug"], r["mutant_id"]) for r in all_rows}
    dropped = [r for r in all_rows if r["table_region"] is None]
    stable_attr = [r for r in all_rows if r["attributable"] and not r["pattern"]]

    counts = defaultdict(Counter)  # (project, region) -> Counter
    for r in all_rows:
        reg = r["table_region"]
        if reg is None:
            continue
        c = counts[(r["project"], reg)]
        c["cat"] += 1
        c["trans"] += r["transition"]
        c["attr"] += r["attributable"]
        c["spec"] += r["mutation_specific"]
        if r["pattern"]:
            pat = r["pattern"]
            c[f"cat:{pat}"] += 1
            c[f"attr:{pat}"] += r["attributable"]
            c[f"spec:{pat}"] += r["mutation_specific"]

    projects = sorted({r["project"] for r in all_rows})

    def pct(project, region, key, denom="cat"):
        c = counts.get((project, region))
        if not c or not c[denom]:
            return None
        return 100.0 * c[key] / c[denom]

    print("\nSanity check: Trans. % per region (ours vs. paper)")
    for p in projects:
        ours = [pct(p, reg, "trans") for reg in REGIONS]
        cells = []
        for o, ref in zip(ours, PAPER_TRANS.get(p, [None] * 4)):
            cells.append(f"{'NA' if o is None else f'{o:.1f}':>5}/{'NA' if ref is None else ref:>5}")
        print(f"  {p:<13}" + "  ".join(cells))

    print("\nCounts per region (categorized / attributable / mutation-specific)")
    for p in projects:
        for pat in [None, *PATTERNS]:
            sfx = f":{pat}" if pat else ""
            cells = []
            for reg in REGIONS:
                c = counts.get((p, reg), Counter())
                cells.append(f"{c['cat' + sfx]:>6}/{c['attr' + sfx]:>3}/{c['spec' + sfx]:>3}")
            print(f"  {p:<13}{pat or 'all':<8}" + "  ".join(cells))

    # --- overall table ---
    lines = tex_header(
        "Per-project regional prevalence of attributable and mutation-specific mutants. "
        "Each cell reports the percentage of categorized mutants in the corresponding "
        "project-region pair assigned to the category. " + CAPTION_DETAIL,
        "tab:rq2_attributable", ["Project"])
    per_project = {p: {reg: {key: pct(p, reg, key) for key, _ in CATEGORIES}
                       for reg in REGIONS} for p in projects}
    for p in projects:
        lines += tex_row(p, per_project[p]) + [""]
    lines += [r"\midrule", ""]
    lines += tex_row("Project mean", project_mean(per_project, projects), bold=False) + [""]
    OUT_TEX.write_text("\n".join(lines + TEX_FOOTER))

    # --- per-pattern table ---
    # Denominator is every categorized mutant in (p, r), the same base as the
    # reported regional table, so a cell is the share of that pair.
    by_pattern = {p: {reg: {(key, pat): pct(p, reg, f"{key}:{pat}")
                            for key, _ in CATEGORIES for pat in PATTERNS}
                      for reg in REGIONS} for p in projects}
    mean = {reg: {} for reg in REGIONS}
    for reg in REGIONS:
        for cell in by_pattern[projects[0]][reg]:
            vals = [by_pattern[p][reg][cell] for p in projects
                    if by_pattern[p][reg][cell] is not None]
            mean[reg][cell] = sum(vals) / len(vals) if vals else None
    fluct_note = (" Fluct.\\ combines the S$\\rightarrow$K$\\rightarrow$S, "
                  "K$\\rightarrow$S$\\rightarrow$K, and recurrent patterns."
                  if COMBINE_FLUCTUATING else "")
    caption = (
        "Per-project regional prevalence of attributable (Attrib.) and mutation-specific "
        "(Mut.-spec.) mutants by mutant-status pattern. Each cell reports the percentage of "
        "categorized mutants in the corresponding project-region pair that are assigned to "
        "the category and have the given pattern. Percentages are computed independently "
        "within each project-region pair. Boldface marks, for each project, category, and "
        "pattern, the region with the highest percentage. NA indicates that no categorized "
        "mutants were observed in the region." + fluct_note
    )
    OUT_PATTERN_TEX.write_text("\n".join(
        pattern_table(caption, projects, by_pattern, mean) + TEX_FOOTER).replace(
        r"\end{table}", r"\end{table*}"))

    print(f"\nWrote {OUT_ALL.name} ({len(all_rows)} categorized mutants), "
          f"{OUT_TEX.name}, {OUT_PATTERN_TEX.name}")
    print(f"attributable mutants without data_seq entry: {len(missing)} "
          f"({Counter(b for b, _ in missing)})")
    print(f"attributable mutants that are stable in data_seq (no pattern row): "
          f"{len(stable_attr)} ({Counter(r['bug'] for r in stable_attr)})")
    print(f"mutants with no region (dropped): {len(dropped)}")
    print(f"warnings: {len(warnings)}")


if __name__ == "__main__":
    main()
