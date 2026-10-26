from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys import data, evaluate
from recsys.models import classic, twotower


def synthetic_raw(n_users=60, n_items=40, per_user=30, seed=0) -> data.Raw:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        items = rng.choice(np.arange(1, n_items + 1), per_user, replace=False)
        ts = np.sort(rng.integers(1_000_000, 2_000_000, per_user))
        for i, t in zip(items, ts):
            rows.append((u, int(i), int(rng.integers(4, 6)), int(t)))  # all positives
    ratings = pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])
    users = pd.DataFrame({"user": range(1, n_users + 1), "gender": ["M", "F"] * (n_users // 2), "age": [25] * n_users, "occupation": [1] * n_users})
    movies = pd.DataFrame({"item": range(1, n_items + 1), "title": [f"Film {i} (199{i % 10})" for i in range(1, n_items + 1)], "genres": ["Drama|Comedy"] * n_items})
    return data.Raw(ratings, users, movies)


def test_split_is_strictly_time_ordered_and_disjoint():
    raw = synthetic_raw()
    s = data.make_split(raw)
    assert s.cuts["t_val"] < s.cuts["t_test"]
    pos = raw.ratings
    uix = {u: k for k, u in enumerate(s.user_ids)}
    iix = {i: k for k, i in enumerate(s.item_ids)}
    for name, mat in (("train", s.train), ("val", s.val), ("test", s.test)):
        coo = mat.tocoo()
        for u, i in zip(coo.row[:200], coo.col[:200]):
            ts = pos[(pos.user == s.user_ids[u]) & (pos.item == s.item_ids[i])]["ts"].iloc[0]
            if name == "train":
                assert ts < s.cuts["t_val"]
            elif name == "val":
                assert s.cuts["t_val"] <= ts < s.cuts["t_test"]
            else:
                assert ts >= s.cuts["t_test"]
    assert s.train.multiply(s.val).nnz == 0 and s.train.multiply(s.test).nnz == 0
    assert uix and iix


def test_cold_items_are_dropped_from_targets_and_counted():
    raw = synthetic_raw()
    s = data.make_split(raw)
    warm_val = np.asarray(s.train.sum(0)).ravel() > 0
    assert np.asarray(s.val.sum(0)).ravel()[~warm_val].sum() == 0
    warm_test = np.asarray((s.train + s.val).sum(0)).ravel() > 0
    assert np.asarray(s.test.sum(0)).ravel()[~warm_test].sum() == 0
    assert s.cuts["test_dropped_cold_item"] >= 0


def test_eval_users_need_history_and_a_future():
    h = sp.csr_matrix(np.array([[1, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=np.float32))
    t = sp.csr_matrix(np.array([[0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=np.float32))
    assert data.eval_users(h, t).tolist() == [0]


def test_metrics_hand_checked():
    ranked = np.array([[2, 0, 1], [0, 1, 2]])
    rel = sp.csr_matrix(np.array([[0, 0, 1], [0, 0, 0]], dtype=np.float32) + np.array([[0, 0, 0], [1, 1, 0]], dtype=np.float32))
    m = evaluate.user_metrics(ranked, rel, 3)
    assert m["recall"].tolist() == [1.0, 1.0]
    assert m["ndcg"][0] == pytest.approx(1.0)  # single relevant item ranked first
    ideal = 1 + 1 / np.log2(3)
    assert m["ndcg"][1] == pytest.approx(ideal / ideal)
    m2 = evaluate.user_metrics(np.array([[1, 2, 0]]), sp.csr_matrix(np.array([[0, 0, 1]], dtype=np.float32)), 3)
    assert m2["ndcg"][0] == pytest.approx(1 / np.log2(3))
    assert m2["hit"][0] == 1.0


def test_evaluation_never_recommends_history():
    class Echo:  # scores exactly the items the user already has, highest
        def score(self, users, history):
            return np.asarray(history[users].todense()) * 10 + np.random.default_rng(0).random((len(users), history.shape[1]))

    h = sp.csr_matrix(np.eye(6, 8, dtype=np.float32) + np.eye(6, 8, k=1, dtype=np.float32))
    t = sp.csr_matrix(np.eye(6, 8, k=3, dtype=np.float32))
    res = evaluate.evaluate(Echo(), h, t, np.arange(6), ks=(3,))
    for u in range(6):
        assert set(res["lists"][u, :3]).isdisjoint(h[u].indices)


def test_paired_bootstrap_detects_a_shift_and_is_calibrated_on_ties():
    rng = np.random.default_rng(0)
    a = rng.random(400)
    assert evaluate.paired_bootstrap(a + 0.1, a)[3] < 0.01
    d, lo, hi, p = evaluate.paired_bootstrap(a, a.copy())
    assert (d, p) == (0.0, 1.0) and lo == hi == 0.0


def test_ease_and_itemknn_recommend_the_cooccurring_item():
    # items 0 and 1 always together, 2 and 3 always together
    rows = []
    for u in range(20):
        rows += [(u, 0), (u, 1)] if u < 10 else [(u, 2), (u, 3)]
    m = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), tuple(zip(*rows))), shape=(20, 4))
    q = sp.csr_matrix(np.array([[1, 0, 0, 0]], dtype=np.float32))
    full = sp.vstack([m, q]).tocsr()
    for model in (classic.EASE(reg=1.0), classic.ItemKNN(k=3)):
        model.fit(m)
        s = model.score(np.array([20]), full)[0]
        assert np.argmax(s[1:]) + 1 == 1


def test_als_reconstructs_block_structure():
    rows = [(u, i) for u in range(15) for i in (range(0, 5) if u < 8 else range(5, 10))]
    m = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), tuple(zip(*rows))), shape=(15, 10))
    model = classic.ALS(factors=4, reg=0.1, alpha=10, iters=8).fit(m)
    s = model.score(np.array([0, 14]), m)
    assert s[0, :5].mean() > s[0, 5:].mean()
    assert s[1, 5:].mean() > s[1, :5].mean()


def test_twotower_history_sampling_never_contains_the_target():
    raw = synthetic_raw()
    s = data.make_split(raw)
    model = twotower.TwoTower(epochs=1)
    model.n_items = s.n_items
    coo = s.train.tocoo()
    rng = np.random.default_rng(0)
    hist = model._sample_history(s.train, coo.row[:300], coo.col[:300], rng)
    for row, target in zip(hist, coo.col[:300]):
        assert target not in row[row != s.n_items]


def test_twotower_trains_and_scores_without_nans():
    raw = synthetic_raw()
    s = data.make_split(raw)
    model = twotower.TwoTower(dim=16, epochs=2, batch=64, threads=1).fit(s.train, s)
    assert np.isfinite(model.history_log).all()
    scores = model.score(np.arange(5), s.train)
    assert scores.shape == (5, s.n_items) and np.isfinite(scores).all()
    assert (np.abs(scores) <= 1.0001).all()  # cosine similarity


def test_twotower_user_vector_is_inductive():
    """An unseen user is scored from a history alone, with no per-user parameters involved."""
    raw = synthetic_raw()
    s = data.make_split(raw)
    model = twotower.TwoTower(dim=16, epochs=1, batch=64, threads=1).fit(s.train, s)
    fresh = sp.csr_matrix(np.eye(1, s.n_items, k=3, dtype=np.float32) + np.eye(1, s.n_items, k=7, dtype=np.float32))
    padded = sp.vstack([s.train, fresh]).tocsr()
    model.split.user_feats = np.vstack([model.split.user_feats, [[0, 0, 0]]])
    v = model.user_vectors(np.array([s.n_users]), padded)
    assert v.shape == (1, 16) and np.isfinite(v).all()
