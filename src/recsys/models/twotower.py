"""Two-tower retrieval model in PyTorch.

  user tower:  mean of the embeddings of the user's history items  (+ gender, age, occupation)  -> MLP -> L2-norm
  item tower:  item-id embedding + genres + release-year bucket                                -> MLP -> L2-norm
  score:       cosine similarity / temperature

The user tower is built from the user's *history*, not a per-user ID embedding. That makes it an
inductive model: it can score any user for whom a few item interactions are known, including users
who were never seen in training, which is how retrieval towers are deployed.

Training: in-batch sampled softmax with logQ correction (Yi et al., 2019). Other rows' positives
act as negatives; because popular items appear as in-batch negatives far more often, each logit is
reduced by log(sampling probability of that item), which removes the bias against popular items.
Accidental hits (the same item as someone else's positive) are masked.

Leakage control: when a training pair (u, i) is built, the history fed to the user tower is a
random subset of u's training items **excluding i**, otherwise the tower could just copy i.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

MAX_HIST = 50


def _mlp(d_in: int, d_out: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, d_out * 2), nn.ReLU(), nn.Linear(d_out * 2, d_out))


class Towers(nn.Module):
    def __init__(self, n_items: int, item_genres: np.ndarray, item_year: np.ndarray, user_card: tuple[int, int, int], dim: int, use_features: bool, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.use_features = use_features
        self.hist_emb = nn.Embedding(n_items + 1, dim, padding_idx=n_items)  # input embeddings for the history bag
        self.item_emb = nn.Embedding(n_items, dim)
        self.register_buffer("genres", torch.as_tensor(item_genres))
        self.register_buffer("year", torch.as_tensor(item_year, dtype=torch.long))
        self.year_emb = nn.Embedding(int(item_year.max()) + 1, 8)
        self.gender, self.age, self.occ = (nn.Embedding(c, 8) for c in user_card)
        self.user_mlp = _mlp(dim + (24 if use_features else 0), dim)
        self.item_mlp = _mlp(dim + (item_genres.shape[1] + 8 if use_features else 0), dim)
        self.pad = n_items
        for e in (self.hist_emb, self.item_emb):
            nn.init.normal_(e.weight, std=0.1)

    def user(self, hist: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        mask = (hist != self.pad).float().unsqueeze(-1)
        bag = (self.hist_emb(hist) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        bag = self.drop(bag)
        x = torch.cat([bag, self.gender(feats[:, 0]), self.age(feats[:, 1]), self.occ(feats[:, 2])], 1) if self.use_features else bag
        return F.normalize(self.user_mlp(x), dim=-1)

    def items(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        if idx is None:
            idx = torch.arange(self.item_emb.num_embeddings, device=self.item_emb.weight.device)
        x = self.drop(self.item_emb(idx))
        if self.use_features:
            x = torch.cat([x, self.genres[idx], self.year_emb(self.year[idx])], 1)
        return F.normalize(self.item_mlp(x), dim=-1)


class TwoTower:
    name = "twotower"

    def __init__(self, dim: int = 64, temperature: float = 0.07, epochs: int = 12, batch: int = 1024, lr: float = 2e-3,
                 weight_decay: float = 1e-6, use_features: bool = True, seed: int = 0, threads: int = 4, dropout: float = 0.0):
        self.dim, self.temperature, self.epochs, self.batch = dim, temperature, epochs, batch
        self.lr, self.weight_decay, self.use_features, self.seed, self.threads = lr, weight_decay, use_features, seed, threads
        self.dropout = dropout
        self.history_log: list[float] = []

    # -- training ------------------------------------------------------------------------------

    def _sample_history(self, hist: sp.csr_matrix, users: np.ndarray, items: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = np.full((len(users), MAX_HIST), self.n_items, dtype=np.int64)
        for r, (u, i) in enumerate(zip(users, items)):
            row = hist.indices[hist.indptr[u] : hist.indptr[u + 1]]
            row = row[row != i]  # never show the target to the user tower
            if row.size > MAX_HIST:
                row = rng.choice(row, MAX_HIST, replace=False)
            out[r, : row.size] = row
        return out

    def fit(self, history: sp.csr_matrix, split, on_epoch=None):
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        rng = np.random.default_rng(self.seed)
        self.n_items = history.shape[1]
        self.split = split
        self.net = Towers(self.n_items, split.item_genres, split.item_year, tuple(int(split.user_feats[:, c].max()) + 1 for c in range(3)),
                          self.dim, self.use_features, self.dropout)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        coo = history.tocoo()
        pairs_u, pairs_i = coo.row.astype(np.int64), coo.col.astype(np.int64)
        pop = np.asarray(history.sum(0)).ravel() + 1.0
        log_q = torch.as_tensor(np.log(pop / pop.sum()), dtype=torch.float32)
        feats = torch.as_tensor(split.user_feats)
        self.history = history

        for epoch in range(self.epochs):
            self.net.train()
            order = rng.permutation(len(pairs_u))
            total, steps = 0.0, 0
            for s in range(0, len(order) - self.batch + 1, self.batch):  # drop the ragged tail
                b = order[s : s + self.batch]
                u, i = pairs_u[b], pairs_i[b]
                hist = torch.as_tensor(self._sample_history(history, u, i, rng))
                zu = self.net.user(hist, feats[u])
                it = torch.as_tensor(i)
                zi = self.net.items(it)
                logits = zu @ zi.T / self.temperature - log_q[it][None, :]  # logQ correction
                same = it[:, None] == it[None, :]
                logits = logits.masked_fill(same & ~torch.eye(len(b), dtype=torch.bool), -1e9)  # accidental hits
                loss = F.cross_entropy(logits, torch.arange(len(b)))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, steps = total + float(loss.detach()), steps + 1
            self.history_log.append(total / max(steps, 1))
            if on_epoch is not None:
                self.net.eval()
                on_epoch(epoch, self)
        self.net.eval()
        return self

    # -- inference -----------------------------------------------------------------------------

    @torch.no_grad()
    def user_vectors(self, users: np.ndarray, history: sp.csr_matrix) -> np.ndarray:
        """All of a user's known items go into the bag at inference (a mean, so length does not matter)."""
        self.net.eval()
        width = max(1, int(np.diff(history.indptr)[users].max()))
        hist = np.full((len(users), width), self.n_items, dtype=np.int64)
        for r, u in enumerate(users):
            row = history.indices[history.indptr[u] : history.indptr[u + 1]]
            hist[r, : row.size] = row
        return self.net.user(torch.as_tensor(hist), torch.as_tensor(self.split.user_feats[users])).numpy()

    @torch.no_grad()
    def item_vectors(self) -> np.ndarray:
        self.net.eval()
        return self.net.items().numpy()

    def score(self, users, history):
        return self.user_vectors(users, history) @ self.item_vectors().T
