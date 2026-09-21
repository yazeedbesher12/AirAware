from __future__ import annotations

import argparse
from pathlib import Path

from app.ml import MODEL_NAMES
from app.ml.dataset import HORIZONS
from app.ml.trainer import Phase4Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AirAware Phase 4A models independently.")
    parser.add_argument("--model", choices=[*MODEL_NAMES, "all"], default="all")
    parser.add_argument("--horizon", choices=[*HORIZONS, "all"], default="all")
    parser.add_argument("--sensor", type=int)
    parser.add_argument("--target", choices=["pm25"], default="pm25")
    args = parser.parse_args()
    models = list(MODEL_NAMES) if args.model == "all" else [args.model]
    horizons = list(HORIZONS) if args.horizon == "all" else [args.horizon]
    Phase4Trainer(Path(__file__).resolve().parent).run(models, horizons, args.sensor, args.target)


if __name__ == "__main__":
    main()
