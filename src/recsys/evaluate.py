"""Full-ranking evaluation.

Every model ranks *all* catalogue items for each user (minus what the user already has), and the
metric is computed on that ranking. The popular alternative, ranking the positive against ~100
sampled negatives, changes model orderings and is not a faithful estimate of full-ranking quality
(Krichene & Rendle, 2020, "On sampled metrics for item recommendation"), so it is not used.

Per-user metric values are kept (not just means) so differences between models can be tested
with a paired bootstrap over users.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

BATCH = 512


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first."""
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def user_metrics(ranked: np.ndarray, relevant: sp.csr_matrix, k: int) -> dict[str, np.ndarray]:
    """recall@k, ndcg@k, hit@k for each row of `ranked` against the matching rows of `relevant`."""
    n = ranked.shape[0]
    rel = np.zeros((n, k), dtype=np.float32)
    for r in range(n):
        row = relevant.getrow(r)
        rel[r] = np.isin(ranked[r], row.indices)
    n_rel = np.diff(relevant.indptr).astype(np.float32)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = rel @ discounts
    ideal = np.array([discounts[: int(min(k, m))].sum() for m in n_rel], dtype=np.float32)
    return {
        "recall": rel.sum(1) / n_rel,
        "ndcg": dcg / ideal,
        "hit": (rel.sum(1) > 0).astype(np.float32),
    }


def evaluate(model, history: sp.csr_matrix, targets: sp.csr_matrix, users: np.ndarray, ks=(10, 20)) -> dict:
    """Returns per-user metrics for each k plus the recommended lists (for coverage/popularity analysis)."""
    kmax = max(ks)
    per_user = {k: {m: [] for m in ("recall", "ndcg", "hit")} for k in ks}
    lists = []
    for start in range(0, len(users), BATCH):
        batch = users[start : start + BATCH]
        scores = np.asarray(model.score(batch, history), dtype=np.float32)
        seen = history[batch].tocoo()
        scores[seen.row, seen.col] = -np.inf  # never recommend what the user already has
        ranked = topk(scores, kmax)
        lists.append(ranked)
        rel = targets[batch]
        for k in ks:
            for name, vals in user_metrics(ranked[:, :k], rel, k).items():
                per_user[k][name].append(vals)
    return {
        "users": users,
        "metrics": {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in per_user.items()},
        "lists": np.concatenate(lists),
    }


def beyond_accuracy(lists: np.ndarray, k: int, n_items: int, popularity: np.ndarray) -> dict[str, float]:
    """Catalogue coverage and how popular the recommendations are (mean percentile of item popularity)."""
    top = lists[:, :k]
    rank_pct = popularity.argsort().argsort() / max(n_items - 1, 1)  # 0 = least popular, 1 = most
    return {"coverage": float(len(np.unique(top)) / n_items), "mean_pop_percentile": float(rank_pct[top].mean())}


def bootstrap_ci(values: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float, float, float]:
    """Mean difference a - b over the same users, its 95% CI, and a two-sided bootstrap p-value."""
    if a.shape != b.shape:
        raise ValueError("per-user arrays must be aligned")
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(reps)])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(min(1.0, max(p, 1.0 / reps)))
