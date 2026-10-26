from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from recsys.models.twotower import Towers
from recsys.serve import create_app

N_ITEMS = 25


@pytest.fixture()
def client(tmp_path):
    torch.manual_seed(0)
    genres = np.zeros((N_ITEMS, 18), dtype=np.float32)
    year = np.arange(N_ITEMS) % 5
    net = Towers(N_ITEMS, genres, year, (2, 7, 21), 8, True)
    torch.save(net.state_dict(), tmp_path / "model.pt")
    np.savez(tmp_path / "item_side.npz", genres=genres, year=year)
    meta = {"item_ids": list(range(100, 100 + N_ITEMS)), "titles": [f"Film {i}" for i in range(N_ITEMS)], "dim": 8,
            "use_features": True, "user_card": [2, 7, 21]}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return TestClient(create_app(tmp_path))


def test_health_and_lazy_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": False}


def test_recommend_excludes_history_and_respects_k(client):
    r = client.post("/recommend", json={"history": [100, 101, 102], "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 5
    assert {b["movie_id"] for b in body}.isdisjoint({100, 101, 102})
    assert [b["score"] for b in body] == sorted((b["score"] for b in body), reverse=True)
    assert client.get("/health").json()["model_loaded"] is True


def test_unknown_history_is_a_clear_422_and_bad_input_is_rejected(client):
    assert client.post("/recommend", json={"history": [9999]}).status_code == 422
    assert client.post("/recommend", json={"history": []}).status_code == 422
    assert client.post("/recommend", json={"history": [100], "k": 1000}).status_code == 422
    assert client.post("/recommend", json={"history": [100], "gender": "X"}).status_code == 422


def test_similar_excludes_itself_and_404s_on_unknown(client):
    body = client.get("/similar/105?k=4").json()
    assert len(body) == 4 and 105 not in {b["movie_id"] for b in body}
    assert client.get("/similar/1").status_code == 404


def test_missing_artifacts_is_503_not_a_crash(tmp_path):
    c = TestClient(create_app(tmp_path / "nothing"))
    assert c.post("/recommend", json={"history": [1]}).status_code == 503
