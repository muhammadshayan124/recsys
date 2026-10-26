"""Side experiment: does dropout fix the two-tower's overfitting (validation NDCG@10 peaks after one epoch)?
Writes results/dropout_check.json. Validation window only; the test set is not touched."""

import json

from recsys import data, evaluate
from recsys.models import twotower

split = data.make_split(data.load_raw())
h, t = split.history("val"), split.targets("val")
users = data.eval_users(h, t)
out = {}
for dropout in (0.0, 0.3):
    curve = []
    model = twotower.TwoTower(dim=64, temperature=0.07, lr=1e-3, weight_decay=1e-5, epochs=6, dropout=dropout)
    model.fit(h, split, lambda e, m, curve=curve: curve.append(round(float(evaluate.evaluate(m, h, t, users, ks=(10,))["metrics"][10]["ndcg"].mean()), 4)))
    out[str(dropout)] = curve
    print(dropout, curve)
json.dump(out, open("results/dropout_check.json", "w"), indent=2)
