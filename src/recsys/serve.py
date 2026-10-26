"""FastAPI service around the two-tower model.

Scoring is exact (a matrix product against every item vector). With a 3.7k-item catalogue that is
well under a millisecond, so an approximate-nearest-neighbour index would add complexity and
error for nothing; it becomes worthwhile in the hundreds of thousands of items.

The user tower works from a *history*, so the API recommends for any list of watched movies,
including people who were never in the training data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from recsys.models.twotower import Towers

ARTIFACT_DIR = Path(os.environ.get("RECSYS_ARTIFACTS", Path(__file__).resolve().parents[2] / "artifacts"))


class Recommendation(BaseModel):
    movie_id: int
    title: str
    score: float


class RecommendRequest(BaseModel):
    history: list[int] = Field(min_length=1, max_length=2000, description="MovieLens movie ids the person liked")
    gender: str = Field("M", pattern="^[MF]$")
    age_bucket: int = Field(2, ge=0, description="index into the MovieLens age brackets (0 = under 18)")
    occupation: int = Field(0, ge=0, le=20)
    k: int = Field(10, ge=1, le=100)


class Service:
    def __init__(self, directory: Path):
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        arrays = np.load(directory / "item_side.npz")
        self.item_ids: list[int] = meta["item_ids"]
        self.titles: list[str] = meta["titles"]
        self.index = {m: i for i, m in enumerate(self.item_ids)}
        self.net = Towers(len(self.item_ids), arrays["genres"], arrays["year"], tuple(meta["user_card"]), meta["dim"], meta["use_features"])
        self.net.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        self.net.eval()
        with torch.no_grad():
            self.item_vecs = self.net.items().numpy()
        self.pad = len(self.item_ids)

    def recommend(self, req: RecommendRequest) -> list[Recommendation]:
        known = [self.index[m] for m in req.history if m in self.index]
        if not known:
            raise HTTPException(status_code=422, detail="none of the supplied movie ids are in the catalogue")
        hist = torch.as_tensor([known])
        feats = torch.as_tensor([[1 if req.gender == "M" else 0, req.age_bucket, req.occupation]])
        with torch.no_grad():
            user = self.net.user(hist, feats).numpy()[0]
        scores = self.item_vecs @ user
        scores[known] = -np.inf  # do not recommend what they already like
        top = np.argsort(-scores)[: req.k]
        return [Recommendation(movie_id=self.item_ids[i], title=self.titles[i], score=float(scores[i])) for i in top]

    def similar(self, movie_id: int, k: int) -> list[Recommendation]:
        if movie_id not in self.index:
            raise HTTPException(status_code=404, detail="unknown movie id")
        i = self.index[movie_id]
        sims = self.item_vecs @ self.item_vecs[i]
        sims[i] = -np.inf
        top = np.argsort(-sims)[:k]
        return [Recommendation(movie_id=self.item_ids[j], title=self.titles[j], score=float(sims[j])) for j in top]


def create_app(directory: Path | None = None) -> FastAPI:
    app = FastAPI(title="recsys", version="0.1.0")
    state: dict[str, Service] = {}

    def service() -> Service:
        if "s" not in state:
            try:
                state["s"] = Service(directory or ARTIFACT_DIR)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail="model artifacts not found; run `recsys export`") from exc
        return state["s"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": "s" in state}

    @app.post("/recommend", response_model=list[Recommendation])
    def recommend(req: RecommendRequest):
        return service().recommend(req)

    @app.get("/similar/{movie_id}", response_model=list[Recommendation])
    def similar(movie_id: int, k: int = 10):
        return service().similar(movie_id, max(1, min(k, 100)))

    return app


app = create_app()
