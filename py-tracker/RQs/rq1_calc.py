import json
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIG
# ============================================================
pid, bid = "black", "16"

BASE = Path(f"D://data_from_hpc//{pid}-{bid}b")
WITH_LBL_DIR = BASE / "with_lbl"
COMMITS_PATH = BASE / "commits_to_run.txt"
MUTANT_HISTORY_PATH = BASE / "mutant_history.json"

OUT_JSON = Path(f"D://res") / f"{pid}-{bid}b_rq1.json"

REQUIRE_PROGRAM_EXITSTATUS_0_OR_1 = True
REMOVE_UNINFORMATIVE_COMMITS = True


# ============================================================
# HELPERS
# ============================================================
def load_json(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def result_commit_id(path, data):
    commit = data.get("program", {}).get("meta", {}).get("commit")
    if commit:
        return commit

    name = path.name
    if name.startswith("results_") and name.endswith(".json"):
        return name[len("results_"):-len(".json")]

    return None


def is_usable_result(data):
    exitstatus = data.get("program", {}).get("exitstatus")
    if REQUIRE_PROGRAM_EXITSTATUS_0_OR_1:
        return exitstatus in (0, 1)
    return True


def mutant_status_from_tests(minfo):
    counts = Counter()

    for _, tinfo in minfo.get("tests", {}).items():
        st = (tinfo.get("status") or "unknown").lower().strip()
        counts[st] += 1

    if counts["killed"] > 0:
        return "K"
    elif counts["survived"] > 0:
        return "S"
    elif counts["anomalous"] > 0:
        return "A"
    elif counts["incompetent"] > 0:
        return "I"
    else:
        return "U"


def status_label(mu_hist, mid, commit):
    status = mu_hist[mid]["dict_history"][commit].get("status")

    if status is None:
        raise KeyError(f"No status for mutant={mid}, commit={commit}")

    if isinstance(status, str):
        return status

    if isinstance(status, dict):
        return status.get("label", "U")

    if isinstance(status, (list, tuple)) and len(status) >= 1:
        return "K" if status[0] > 0 else "S"

    return "U"


def remove_consecutive_duplicates(seq):
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def count_pattern(seq, pattern):
    n = len(pattern)
    return sum(1 for i in range(len(seq) - n + 1) if seq[i:i+n] == pattern)


# ============================================================
# LOAD MUTANT HISTORY
# ============================================================
mu_hist = load_json(MUTANT_HISTORY_PATH)

for mid in mu_hist:
    mu_hist[mid]["dict_history"] = {
        item["commit"]: item
        for item in mu_hist[mid].get("history", [])
    }

print("Mutants in mutant_history.json:", len(mu_hist))


# ============================================================
# LOAD COMMITS
# ============================================================
with open(COMMITS_PATH, "r", encoding="utf-8", errors="replace") as f:
    commits_newest_to_oldest = [line.strip() for line in f if line.strip()]

commits_oldest_to_newest = list(reversed(commits_newest_to_oldest))


# ============================================================
# LOAD RESULT FILES
# ============================================================
results_by_commit = {}

for commit in commits_newest_to_oldest:
    path = WITH_LBL_DIR / f"results_{commit}.json"

    if not path.exists():
        continue

    data = load_json(path)
    real_commit = result_commit_id(path, data)

    if real_commit is None:
        continue

    if not is_usable_result(data):
        continue

    results_by_commit[real_commit] = data

print("Usable result files:", len(results_by_commit))


# ============================================================
# ADD STATUS TO MUTANT HISTORY
# ============================================================
for commit, data in results_by_commit.items():
    for mid, minfo in data.get("mutants", {}).items():
        if mid not in mu_hist:
            continue

        if commit not in mu_hist[mid]["dict_history"]:
            continue

        mu_hist[mid]["dict_history"][commit]["status"] = mutant_status_from_tests(minfo)


# ============================================================
# OPTIONAL: REMOVE UNINFORMATIVE COMMITS
# ============================================================
ordered_result_commits = [
    c for c in commits_oldest_to_newest
    if c in results_by_commit
]

if REMOVE_UNINFORMATIVE_COMMITS:
    filtered_results_by_commit = {}

    for commit in ordered_result_commits:
        labels = []

        for mid in results_by_commit[commit].get("mutants", {}):
            try:
                labels.append(status_label(mu_hist, mid, commit))
            except Exception:
                continue

        if not labels:
            continue

        counter = Counter(labels)

        if counter["K"] == len(labels):
            continue
        if counter["S"] == len(labels):
            continue
        if counter["A"] == len(labels):
            continue
        if counter["I"] == len(labels):
            continue
        if counter["U"] == len(labels):
            continue

        if counter["K"] + counter["S"] == 0:
            continue

        filtered_results_by_commit[commit] = results_by_commit[commit]

    results_by_commit = filtered_results_by_commit

ordered_commits = [
    c for c in commits_oldest_to_newest
    if c in results_by_commit
]

print("Informative commits:", len(ordered_commits))


# ============================================================
# PER-MUTANT COUNTS
# ============================================================
mutants_out = {}

global_counts = Counter()

for mid, minfo in mu_hist.items():
    ks_sequence = []

    for commit in ordered_commits:
        if mid not in results_by_commit[commit].get("mutants", {}):
            continue

        try:
            st = status_label(mu_hist, mid, commit)
        except Exception:
            continue

        # Ignore A/I/U entirely.
        # This makes S -> A -> K become S -> K.
        if st in {"K", "S"}:
            ks_sequence.append(st)

    counts = Counter({
        "S->K": 0,
        "K->S": 0,
        "K->K": 0,
        "S->S": 0,
    })

    for i in range(len(ks_sequence) - 1):
        tr = f"{ks_sequence[i]}->{ks_sequence[i + 1]}"
        counts[tr] += 1
        global_counts[tr] += 1

    # For SKS/KSK, remove only repeated same K/S runs.
    # Example: S S A K K S -> K/S-only sequence S S K K S
    # collapsed pattern sequence: S K S
    collapsed_ks_sequence = remove_consecutive_duplicates(ks_sequence)

    num_SKS = count_pattern(collapsed_ks_sequence, ["S", "K", "S"])
    num_KSK = count_pattern(collapsed_ks_sequence, ["K", "S", "K"])

    metadata = {
        k: v
        for k, v in minfo.items()
        if k not in {"history", "dict_history"}
    }

    mutants_out[mid] = {
        **metadata,
        "S_to_K": counts["S->K"],
        "K_to_S": counts["K->S"],
        "K_to_K": counts["K->K"],
        "S_to_S": counts["S->S"],
        "num_SKS": num_SKS,
        "num_KSK": num_KSK,
    }


# ============================================================
# SAVE
# ============================================================
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(mutants_out, f, indent=2, ensure_ascii=False)

print("Saved:", OUT_JSON)
print("Mutants saved:", len(mutants_out))
print("Global counts:", dict(global_counts))