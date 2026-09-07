#!/usr/bin/env python3
"""Example deterministic LightGBM fit using a previously selected configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_parameters(path: Path) -> dict:
    record = json.loads(path.read_text())
    configuration = record.get("selected_parameters", record)
    parameters = dict(configuration)
    if "log_max_bin" in parameters:
        parameters["max_bin"] = 2 ** int(parameters.pop("log_max_bin")) - 1
    for name in ("num_leaves", "n_estimators", "min_child_samples", "max_depth"):
        if name in parameters:
            parameters[name] = int(parameters[name])
    return parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("parameters", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--site-column", default="site_id")
    parser.add_argument("--categorical", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=33)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table = read_table(args.table)
    required = set(args.features + [args.target, args.site_column])
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if table[list(required)].isna().any().any():
        raise ValueError("the modeling table contains missing values")

    features = table[args.features].copy()
    for column in args.categorical:
        categories = sorted(features[column].unique())
        features[column] = pd.Categorical(features[column], categories=categories)
    numeric = [column for column in args.features if column not in args.categorical]
    if numeric and not np.isfinite(features[numeric].to_numpy(float)).all():
        raise ValueError("numeric features contain non-finite values")

    counts = table.groupby(args.site_column)[args.site_column].transform("size")
    weights = (1.0 / counts).to_numpy(float)
    parameters = load_parameters(args.parameters)
    parameters.update(
        objective="regression",
        subsample_freq=1,
        random_state=args.seed,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )
    model = lgb.LGBMRegressor(**parameters)
    model.fit(
        features,
        table[args.target].to_numpy(float),
        sample_weight=weights,
        categorical_feature=args.categorical,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.output))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
