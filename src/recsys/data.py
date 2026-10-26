"""MovieLens-1M loading and the time-respecting split.

Why a *global time* split rather than the common per-user leave-last-out: leave-one-out trains on
other users' interactions from after a given user's test event, so a model can learn tomorrow's
taste from today's evaluation target (Ji et al., 2023, "A critical study on data leakage in
recommender system offline evaluation"). Here everything before the cut is training data and
everything after it is the future, as in production.

The dataset is licensed for research use and may not be redistributed, so it is never committed:
`recsys fetch` downloads it from GroupLens and verifies the published checksum.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MD5 = "c4d9eecfca2ab87c1945afe126590906"  # published at URL + ".md5"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
POSITIVE = 4  # a rating >= 4 counts as "liked"
MIN_HISTORY = 5  # users need this many earlier positives to be evaluated (cold users are reported, not hidden)
GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def fetch(dest: Path = DATA_DIR / "ml-1m.zip") -> Path:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    blob = httpx.get(URL, timeout=120, follow_redirects=True).raise_for_status().content
    got = hashlib.md5(blob).hexdigest()  # noqa: S324 - integrity check against the published checksum, not security
    if got != MD5:
        raise ValueError(f"checksum mismatch: expected {MD5}, got {got}")
    dest.write_bytes(blob)
    return dest


@dataclass
class Raw:
    ratings: pd.DataFrame  # user, item, rating, ts  (ids are the original MovieLens ids)
    users: pd.DataFrame  # user, gender, age, occupation
    movies: pd.DataFrame  # item, title, genres


def load_raw(path: Path = DATA_DIR / "ml-1m.zip") -> Raw:
    with zipfile.ZipFile(path) as z:
        def read(name, cols):
            with z.open(f"ml-1m/{name}") as f:
                return pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"), sep="::", engine="python", names=cols, usecols=range(len(cols)))
        ratings = read("ratings.dat", ["user", "item", "rating", "ts"])
        users = read("users.dat", ["user", "gender", "age", "occupation"])
        movies = read("movies.dat", ["item", "title", "genres"])
    return Raw(ratings, users, movies)


@dataclass
class Split:
    """Interaction matrices (users x items, binary CSR) for each phase, plus side information.
    Indices are dense 0..n-1; `user_ids`/`item_ids` map back to MovieLens ids."""

    n_users: int
    n_items: int
    user_ids: np.ndarray
    item_ids: np.ndarray
    train: sp.csr_matrix
    val: sp.csr_matrix
    test: sp.csr_matrix
    user_feats: np.ndarray  # (n_users, 3) ints: gender, age bucket, occupation
    item_genres: np.ndarray  # (n_items, 18) multi-hot
    item_year: np.ndarray  # (n_items,) release-year bucket, 0 = unknown
    titles: list[str]
    cuts: dict  # timestamps and bookkeeping: what was dropped and why

    def history(self, phase: str) -> sp.csr_matrix:
        """Everything known before `phase` starts."""
        return {"val": self.train, "test": (self.train + self.val).tocsr()}[phase]

    def targets(self, phase: str) -> sp.csr_matrix:
        return {"val": self.val, "test": self.test}[phase]


def _csr(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    m = sp.csr_matrix((np.ones(len(df), dtype=np.float32), (df["u"].to_numpy(), df["i"].to_numpy())), shape=(n_users, n_items))
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def make_split(raw: Raw, val_frac: float = 0.1, test_frac: float = 0.1) -> Split:
    pos = raw.ratings[raw.ratings["rating"] >= POSITIVE].sort_values("ts", kind="stable").reset_index(drop=True)
    n = len(pos)
    t_val = pos["ts"].iloc[int(n * (1 - val_frac - test_frac))]
    t_test = pos["ts"].iloc[int(n * (1 - test_frac))]
    phase = np.where(pos["ts"] < t_val, 0, np.where(pos["ts"] < t_test, 1, 2))

    user_ids = np.sort(raw.users["user"].unique())
    item_ids = np.sort(raw.movies["item"].unique())
    uix = pd.Series(np.arange(len(user_ids)), index=user_ids)
    iix = pd.Series(np.arange(len(item_ids)), index=item_ids)
    pos["u"], pos["i"] = uix.loc[pos["user"]].to_numpy(), iix.loc[pos["item"]].to_numpy()
    nu, ni = len(user_ids), len(item_ids)

    train_df, val_df, test_df = pos[phase == 0], pos[phase == 1], pos[phase == 2]
    warm_train = set(train_df["i"])
    val_keep = val_df["i"].isin(warm_train)
    warm_val = warm_train | set(val_df["i"])
    test_keep = test_df["i"].isin(warm_val)
    cuts = {
        "t_val": int(t_val), "t_test": int(t_test), "positives": int(n),
        "train": int(len(train_df)), "val": int(val_keep.sum()), "test": int(test_keep.sum()),
        "val_dropped_cold_item": int((~val_keep).sum()), "test_dropped_cold_item": int((~test_keep).sum()),
    }
    train, val, test = _csr(train_df, nu, ni), _csr(val_df[val_keep], nu, ni), _csr(test_df[test_keep], nu, ni)

    u = raw.users.set_index("user").loc[user_ids]
    age_levels = sorted(raw.users["age"].unique())
    user_feats = np.stack([
        (u["gender"] == "M").astype(int).to_numpy(),
        u["age"].map({a: k for k, a in enumerate(age_levels)}).to_numpy(),
        u["occupation"].to_numpy(),
    ], axis=1).astype(np.int64)

    m = raw.movies.set_index("item").loc[item_ids]
    genres = np.zeros((ni, len(GENRES)), dtype=np.float32)
    for r, g in enumerate(m["genres"]):
        for name in g.split("|"):
            if name in GENRES:
                genres[r, GENRES.index(name)] = 1.0
    years = pd.to_numeric(m["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    year = ((years.fillna(1900) - 1900) // 5 + 1).clip(lower=0).astype(int).to_numpy().copy()
    year[years.isna().to_numpy()] = 0
    return Split(nu, ni, user_ids, item_ids, train, val, test, user_feats, genres, year, m["title"].tolist(), cuts)


def eval_users(history: sp.csr_matrix, targets: sp.csr_matrix, min_history: int = MIN_HISTORY) -> np.ndarray:
    """Users with at least `min_history` known positives and at least one future positive."""
    h, t = np.diff(history.indptr), np.diff(targets.indptr)
    return np.flatnonzero((h >= min_history) & (t >= 1))
