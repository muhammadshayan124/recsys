"""Tune on the validation window, refit on train+val, evaluate once on the test window.

Protocol, in order:
  1. split by time (data.make_split)
  2. tune every model on VALIDATION (fit on train, score the next 10% of time)
  3. refit each model with its chosen setting on train+val
  4. evaluate ONCE on TEST, the final 10% of time, which no choice has seen

The baseline used for significance tests is the best *non-neural* model by validation NDCG, so the
comparison partner is chosen without looking at the test set.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from recsys import data, evaluate
from recsys.models import classic, twotower

K = 10
GRIDS = {
    "itemknn": [dict(k=k, shrink=s) for k, s in itertools.product([20, 50, 100, 200, 10_000], [0.0, 10.0])],
    "ease": [dict(reg=r) for r in (100.0, 300.0, 1000.0, 3000.0, 10_000.0, 30_000.0, 100_000.0, 300_000.0, 1_000_000.0)],
    "als": [dict(factors=f, reg=r, alpha=a, iters=10) for f, r, a in itertools.product([32, 64], [50.0, 200.0, 500.0], [1.0, 5.0])],
}
TOWER_GRID = [
    dict(dim=64, temperature=0.07, lr=2e-3, weight_decay=1e-6),
    dict(dim=64, temperature=0.10, lr=1e-3, weight_decay=1e-5),
    dict(dim=128, temperature=0.10, lr=1e-3, weight_decay=1e-5),
    dict(dim=64, temperature=0.05, lr=5e-4, weight_decay=1e-4),
]
TOWER_EPOCHS = 8
MAKE = {"itemknn": classic.ItemKNN, "ease": classic.EASE, "als": classic.ALS}


def _ndcg(model, history, targets, users) -> float:
    return float(evaluate.evaluate(model, history, targets, users, ks=(K,))["metrics"][K]["ndcg"].mean())


def tune(split: data.Split, log=print, only: tuple[str, ...] | None = None) -> dict:
    h, t = split.history("val"), split.targets("val")
    users = data.eval_users(h, t)
    log(f"validation: {len(users)} users")
    best: dict = {"validation_users": int(len(users)), "models": {}}
    for name, grid in GRIDS.items():
        if only and name not in only:
            continue
        trials = []
        for cfg in grid:
            t0 = time.time()
            score = _ndcg(MAKE[name](**cfg).fit(h), h, t, users)
            trials.append({"config": cfg, "val_ndcg": score})
            log(f"  {name} {cfg} -> {score:.4f} ({time.time() - t0:.0f}s)")
        top = max(trials, key=lambda r: r["val_ndcg"])
        best["models"][name] = {"config": top["config"], "val_ndcg": top["val_ndcg"], "trials": len(trials)}
    for variant, feats in (("twotower", True), ("twotower_nofeat", False)):
        if only and variant not in only:
            continue
        trials = []
        for cfg in TOWER_GRID:
            curve = []
            model = twotower.TwoTower(epochs=TOWER_EPOCHS, use_features=feats, **cfg)
            model.fit(h, split, on_epoch=lambda e, m, curve=curve: curve.append(_ndcg(m, h, t, users)))
            trials.append({"config": cfg, "val_ndcg": max(curve), "epochs": int(np.argmax(curve)) + 1, "curve": curve})
            log(f"  {variant} {cfg} -> best {max(curve):.4f} at epoch {int(np.argmax(curve)) + 1}; curve {np.round(curve, 4).tolist()}")
        top = max(trials, key=lambda r: r["val_ndcg"])
        best["models"][variant] = {"config": top["config"], "epochs": top["epochs"], "val_ndcg": top["val_ndcg"], "trials": len(trials)}
    return best


def _build(name: str, spec: dict):
    if name.startswith("twotower"):
        return twotower.TwoTower(epochs=spec["epochs"], use_features=(name == "twotower"), **spec["config"])
    return MAKE[name](**spec["config"])


def final(split: data.Split, chosen: dict, log=print) -> dict:
    """Refit on train+val and evaluate on test. Returns per-user metrics for every model."""
    h, t = split.history("test"), split.targets("test")
    users = data.eval_users(h, t)
    pop = np.asarray(h.sum(0)).ravel()
    log(f"test: {len(users)} users, {int(t.nnz)} future positives")
    out = {"users": users, "models": {}, "n_items": split.n_items}
    models = {"random": classic.Random(), "popularity": classic.Popularity()}
    for name, spec in chosen["models"].items():
        models[name] = _build(name, spec)
    for name, model in models.items():
        t0 = time.time()
        model.fit(h, split) if name.startswith("twotower") else model.fit(h)
        res = evaluate.evaluate(model, h, t, users)
        res["beyond"] = {k: evaluate.beyond_accuracy(res["lists"], k, split.n_items, pop) for k in (10, 20)}
        out["models"][name] = res
        log(f"  {name}: ndcg@10 {res['metrics'][10]['ndcg'].mean():.4f} ({time.time() - t0:.0f}s)")
    return out


def save(results: dict, chosen: dict, split: data.Split, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / "per_user.npz", users=results["users"],
        **{f"{n}.{k}.{m}": v for n, r in results["models"].items() for k, d in r["metrics"].items() for m, v in d.items()},
    )
    summary = {
        "cuts": split.cuts, "test_users": int(len(results["users"])), "tuning": chosen,
        "models": {
            n: {
                "metrics": {str(k): {m: evaluate.bootstrap_ci(v) for m, v in d.items()} for k, d in r["metrics"].items()},
                "beyond": {str(k): v for k, v in r["beyond"].items()},
            }
            for n, r in results["models"].items()
        },
    }
    (outdir / "test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
