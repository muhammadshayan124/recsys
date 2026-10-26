"""Non-neural recommenders. Each implements fit(history) and score(users, history) -> dense scores.

`score` takes the history matrix explicitly so a model fitted on the training window can be
asked about users using *later* history only where that is intended (it never is here: scoring
uses the same history the model was fitted on).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class Random:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def fit(self, history, split=None):
        return self

    def score(self, users, history):
        return self.rng.random((len(users), history.shape[1]), dtype=np.float32)


class Popularity:
    """Recommend what the most people liked in the training window. The baseline everything must beat."""

    name = "popularity"

    def fit(self, history, split=None):
        self.pop = np.asarray(history.sum(0)).ravel().astype(np.float32)
        return self

    def score(self, users, history):
        return np.broadcast_to(self.pop, (len(users), len(self.pop))).copy()


class ItemKNN:
    """Item-item cosine similarity on the binary matrix, keeping each item's `k` nearest neighbours."""

    name = "itemknn"

    def __init__(self, k: int = 100, shrink: float = 0.0):
        self.k, self.shrink = k, shrink

    def fit(self, history, split=None):
        x = history.tocsc().astype(np.float32)
        gram = (x.T @ x).toarray()
        norm = np.sqrt(np.diag(gram)) + 1e-9
        sim = gram / (norm[:, None] * norm[None, :] + self.shrink)
        np.fill_diagonal(sim, 0.0)
        if self.k < sim.shape[0]:
            drop = np.argpartition(-sim, self.k, axis=1)[:, self.k :]
            np.put_along_axis(sim, drop, 0.0, axis=1)
        self.sim = sim.astype(np.float32)
        return self

    def score(self, users, history):
        return np.asarray(history[users] @ self.sim)


class EASE:
    """Embarrassingly Shallow Autoencoder (Steck, 2019): ridge regression of every item on all the
    others with a zero diagonal, in closed form. Linear, no latent factors, and still one of the
    hardest baselines to beat on this kind of data, which is why it is included."""

    name = "ease"

    def __init__(self, reg: float = 500.0):
        self.reg = reg

    def fit(self, history, split=None):
        x = history.astype(np.float64)
        g = (x.T @ x).toarray()
        g[np.diag_indices_from(g)] += self.reg
        p = np.linalg.inv(g)
        b = p / (-np.diag(p))[None, :]
        b[np.diag_indices_from(b)] = 0.0
        self.b = b.astype(np.float32)
        return self

    def score(self, users, history):
        return np.asarray(history[users] @ self.b)


class ALS:
    """Implicit-feedback matrix factorisation by alternating least squares (Hu, Koren, Volinsky 2008).
    Observed pairs get confidence 1 + alpha, unobserved pairs confidence 1 and preference 0."""

    name = "als"

    def __init__(self, factors: int = 64, reg: float = 10.0, alpha: float = 20.0, iters: int = 15, seed: int = 0):
        self.factors, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed

    def _solve(self, fixed: np.ndarray, mat: sp.csr_matrix) -> np.ndarray:
        f = fixed.shape[1]
        ftf = fixed.T @ fixed + self.reg * np.eye(f, dtype=np.float32)
        out = np.zeros((mat.shape[0], f), dtype=np.float32)
        for u in range(mat.shape[0]):
            idx = mat.indices[mat.indptr[u] : mat.indptr[u + 1]]
            if idx.size == 0:
                continue
            y = fixed[idx]
            a = ftf + self.alpha * (y.T @ y)
            out[u] = np.linalg.solve(a, (1.0 + self.alpha) * y.sum(0))
        return out

    def fit(self, history, split=None):
        rng = np.random.default_rng(self.seed)
        users, items = history.shape
        self.item_f = (rng.standard_normal((items, self.factors)) * 0.01).astype(np.float32)
        hist_t = history.T.tocsr()
        for _ in range(self.iters):
            self.user_f = self._solve(self.item_f, history)
            self.item_f = self._solve(self.user_f, hist_t)
        return self

    def score(self, users, history):
        # user vectors were solved on exactly this history during fit
        return self.user_f[users] @ self.item_f.T
