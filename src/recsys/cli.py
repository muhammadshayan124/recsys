from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from recsys import data, experiment

app = typer.Typer(help="Recommender benchmark on MovieLens-1M and a small serving API.", no_args_is_help=True)
console = Console(highlight=False)
RESULTS = Path("results")


@app.command()
def fetch() -> None:
    """Download MovieLens-1M from GroupLens and verify its checksum."""
    console.print(f"saved {data.fetch()}")


@app.command()
def tune(
    out: Annotated[Path, typer.Option(help="Results directory")] = RESULTS,
    only: Annotated[str, typer.Option(help="Comma-separated model names to (re)tune; others keep their saved result")] = "",
) -> None:
    """Tune models on the validation window."""
    split = data.make_split(data.load_raw())
    names = tuple(x for x in only.split(",") if x)
    chosen = experiment.tune(split, log=console.print, only=names or None)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "tuning.json"
    if names and path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous["models"].update(chosen["models"])
        chosen = previous
    path.write_text(json.dumps(chosen, indent=2), encoding="utf-8")


@app.command("evaluate")
def evaluate_cmd(out: Annotated[Path, typer.Option(help="Results directory")] = RESULTS) -> None:
    """Refit on train+validation with the tuned settings and evaluate once on the test window."""
    split = data.make_split(data.load_raw())
    chosen = json.loads((out / "tuning.json").read_text(encoding="utf-8"))
    results = experiment.final(split, chosen, log=console.print)
    experiment.save(results, chosen, split, out)
    with (out / "_final_results.pkl").open("wb") as f:  # for `recsys report`; not committed
        pickle.dump({"users": results["users"], "metrics": {n: r["metrics"] for n, r in results["models"].items()}}, f)
    console.print(f"wrote {out}/test_summary.json")


@app.command("report")
def report_cmd(
    results: Annotated[Path, typer.Option(help="Results directory")] = RESULTS,
    readme: Annotated[Path, typer.Option(help="README with <!-- name:start/end --> markers")] = Path("README.md"),
) -> None:
    """Fill the README tables and figure from the saved results."""
    from recsys import report

    report.build(results, readme)
    console.print(f"updated {readme}")


@app.command()
def export(
    out: Annotated[Path, typer.Option(help="Results directory holding tuning.json")] = RESULTS,
    artifacts: Annotated[Path, typer.Option(help="Where to write the serving artifacts")] = Path("artifacts"),
    model: Annotated[str, typer.Option(help="twotower or twotower_nofeat")] = "twotower",
) -> None:
    """Fit the chosen two-tower on ALL positives and write what the API needs."""
    import numpy as np
    import torch

    from recsys.models.twotower import TwoTower

    raw = data.load_raw()
    split = data.make_split(raw)
    spec = json.loads((out / "tuning.json").read_text(encoding="utf-8"))["models"][model]
    full = (split.train + split.val + split.test).tocsr()
    tt = TwoTower(epochs=spec["epochs"], use_features=(model == "twotower"), **spec["config"]).fit(full, split)
    artifacts.mkdir(parents=True, exist_ok=True)
    torch.save(tt.net.state_dict(), artifacts / "model.pt")
    np.savez(artifacts / "item_side.npz", genres=split.item_genres, year=split.item_year)
    meta = {
        "item_ids": [int(i) for i in split.item_ids], "titles": split.titles, "dim": spec["config"]["dim"],
        "use_features": model == "twotower", "user_card": [int(split.user_feats[:, c].max()) + 1 for c in range(3)],
    }
    (artifacts / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    console.print(f"wrote {artifacts}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API."""
    import uvicorn

    uvicorn.run("recsys.serve:app", host=host, port=port)


if __name__ == "__main__":
    app()
