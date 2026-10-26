"""Turn the saved results into README tables and figures, so the prose cannot drift from the data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from recsys import data  # noqa: E402

ORDER = ["random", "popularity", "itemknn", "ease", "als", "twotower_nofeat", "twotower"]
LABEL = {
    "random": "Random", "popularity": "Popularity", "itemknn": "Item-kNN", "ease": "EASE", "als": "ALS (implicit)",
    "twotower_nofeat": "Two-tower (history only)", "twotower": "Two-tower (+ user/item features)",
}
NON_NEURAL = ("popularity", "itemknn", "ease", "als")


def _present(summary: dict) -> list[str]:
    return [m for m in ORDER if m in summary["models"]]


def main_table(summary: dict) -> str:
    rows = ["| model | NDCG@10 (95% CI) | Recall@10 | Hit@10 | NDCG@20 | item coverage@10 | mean popularity percentile of recs@10 |", "|---|---|---|---|---|---|---|"]
    for m in _present(summary):
        d = summary["models"][m]
        n10, r10, h10, n20 = (d["metrics"]["10"]["ndcg"], d["metrics"]["10"]["recall"], d["metrics"]["10"]["hit"], d["metrics"]["20"]["ndcg"])
        b = d["beyond"]["10"]
        rows.append(f"| {LABEL[m]} | {n10[0]:.4f} ({n10[1]:.4f}-{n10[2]:.4f}) | {r10[0]:.4f} | {h10[0]:.3f} | {n20[0]:.4f} | {b['coverage']:.1%} | {b['mean_pop_percentile']:.2f} |")
    return "\n".join(rows)


def baseline_name(summary: dict) -> str:
    """The comparison partner is the best non-neural model on VALIDATION, so it is picked without touching test."""
    tuned = summary["tuning"]["models"]
    scored = {m: tuned[m]["val_ndcg"] for m in tuned if m in NON_NEURAL}
    return max(scored, key=scored.get)


def significance_table(summary: dict, per_user: dict) -> str:
    from recsys.evaluate import paired_bootstrap

    base = baseline_name(summary)
    rows, pvals, names = [], [], []
    for m in _present(summary):
        if m in (base, "random"):
            continue
        d, lo, hi, p = paired_bootstrap(per_user[f"{m}.10.ndcg"], per_user[f"{base}.10.ndcg"])
        rows.append((m, d, lo, hi, p))
        pvals.append(p)
        names.append(m)
    order = np.argsort(pvals)
    holm, running = {}, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        holm[names[i]] = running
    lines = [f"Paired bootstrap over users, difference in NDCG@10 versus **{LABEL[base]}** (the best non-neural model on validation). "
             f"Holm-adjusted across the {len(rows)} comparisons.", "",
             "| model | difference | 95% CI | adjusted p |", "|---|---|---|---|"]
    for m, d, lo, hi, _p in rows:
        lines.append(f"| {LABEL[m]} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {holm[m]:.3f} |")
    return "\n".join(lines)


def bucket_table(summary: dict, per_user: dict, history_len: np.ndarray) -> str:
    edges = [(5, 20), (21, 60), (61, 150), (151, 10**9)]
    heads = ["5-20", "21-60", "61-150", "151+"]
    users = per_user["users"]
    n = history_len[users]
    lines = ["| model | " + " | ".join(f"history {h} (n={int(((n >= lo) & (n <= hi)).sum())})" for h, (lo, hi) in zip(heads, edges)) + " |",
             "|---|" + "---|" * len(edges)]
    for m in _present(summary):
        v = per_user[f"{m}.10.ndcg"]
        lines.append(f"| {LABEL[m]} | " + " | ".join(f"{v[(n >= lo) & (n <= hi)].mean():.4f}" for lo, hi in edges) + " |")
    return "\n".join(lines)


def tuning_table(summary: dict) -> str:
    t = summary["tuning"]["models"]
    lines = ["| model | configurations tried | chosen | validation NDCG@10 |", "|---|---|---|---|"]
    for m in _present(summary):
        if m in t:
            cfg = ", ".join(f"{k}={v}" for k, v in t[m]["config"].items())
            if "epochs" in t[m]:
                cfg += f", epochs={t[m]['epochs']}"
            lines.append(f"| {LABEL[m]} | {t[m]['trials']} | {cfg} | {t[m]['val_ndcg']:.4f} |")
    return "\n".join(lines)


def plot_main(summary: dict, path: Path) -> None:
    names = [m for m in _present(summary) if m != "random"]
    vals = [summary["models"][m]["metrics"]["10"]["ndcg"] for m in names]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(names))
    ax.barh(y, [v[0] for v in vals], xerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], color="#4c78a8", capsize=3)
    ax.set_yticks(y, [LABEL[m] for m in names])
    ax.invert_yaxis()
    ax.set_xlabel("NDCG@10 on the held-out final 10% of time (95% bootstrap CI over users)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build(results: Path, readme: Path) -> None:
    summary = json.loads((results / "test_summary.json").read_text(encoding="utf-8"))
    per_user = dict(np.load(results / "per_user.npz"))
    split = data.make_split(data.load_raw())
    hist_len = np.diff((split.train + split.val).tocsr().indptr)
    plot_main(summary, results / "ndcg_at_10.png")
    c = summary["cuts"]
    sections = {
        "setup": (f"Positives (rating >= 4): {c['positives']:,}. Train {c['train']:,} | validation {c['val']:,} | test {c['test']:,} interactions "
                  f"(future positives on items never seen earlier were dropped: {c['val_dropped_cold_item']} validation, {c['test_dropped_cold_item']} test). "
                  f"Evaluated on **{summary['test_users']:,} test users** (those with at least {data.MIN_HISTORY} earlier positives and at least one future one) "
                  f"and **{summary['tuning']['validation_users']} validation users**."),
        "main-table": main_table(summary),
        "significance": significance_table(summary, per_user),
        "buckets": bucket_table(summary, per_user, hist_len),
        "tuning": tuning_table(summary),
    }
    text = readme.read_text(encoding="utf-8")
    for name, body in sections.items():
        start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
        i, j = text.find(start), text.find(end)
        if i == -1 or j < i:
            raise ValueError(f"README is missing the {name} markers")
        text = text[: i + len(start)] + "\n" + body.strip("\n") + "\n" + text[j:]
    readme.write_text(text, encoding="utf-8")
