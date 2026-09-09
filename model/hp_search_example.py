#!/usr/bin/env python3
"""Example LightGBM hyperparameter search with spatially blocked cross-validation.

The input is a quality-controlled development table with any independent outer
holdout removed. The example focuses on spatial cross-validation and model
selection.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message="flaml.automl is not available.*", category=UserWarning
    )
    from flaml import tune
    from flaml.tune.searcher.blendsearch import CFO


SEARCH_SPACE = {
    "num_leaves": tune.randint(7, 128),
    "n_estimators": tune.randint(200, 1501),
    "learning_rate": tune.loguniform(0.01, 0.2),
    "min_child_samples": tune.randint(20, 501),
    "max_depth": tune.randint(3, 13),
    "colsample_bytree": tune.uniform(0.6, 1.0),
    "subsample": tune.uniform(0.6, 1.0),
    "reg_alpha": tune.uniform(0.0, 20.0),
    "reg_lambda": tune.uniform(0.0, 20.0),
    "log_max_bin": tune.randint(5, 9),
}


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def add_spatial_blocks(
    table: pd.DataFrame,
    site_column: str,
    latitude_column: str,
    longitude_column: str,
    block_size_degrees: float,
) -> pd.DataFrame:
    coordinates = table.groupby(site_column)[[latitude_column, longitude_column]].nunique()
    if (coordinates > 1).any().any():
        raise ValueError("each station must have one latitude and one longitude")
    if not np.isfinite(table[[latitude_column, longitude_column]].to_numpy(float)).all():
        raise ValueError("station coordinates must be finite")

    result = table.copy()
    lat_bin = np.floor((result[latitude_column] + 90.0) / block_size_degrees).astype(int)
    lon_bin = np.floor((result[longitude_column] + 180.0) / block_size_degrees).astype(int)
    result["spatial_block"] = lat_bin.astype(str) + "_" + lon_bin.astype(str)
    return result


def assign_blocked_folds(
    table: pd.DataFrame,
    site_column: str,
    target_column: str,
    n_folds: int,
) -> np.ndarray:
    """Assign whole spatial blocks while balancing station count and target terciles."""
    stations = (
        table.groupby(["spatial_block", site_column], as_index=False)[target_column]
        .mean()
        .rename(columns={target_column: "station_mean"})
    )
    stations["stratum"] = pd.qcut(
        stations["station_mean"], 3, labels=False, duplicates="drop"
    ).fillna(1).astype(int)

    block_counts = stations.groupby("spatial_block")[site_column].nunique()
    strata = sorted(stations["stratum"].unique())
    block_strata = (
        stations.groupby(["spatial_block", "stratum"])[site_column]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=strata, fill_value=0)
    )
    if len(block_counts) < n_folds:
        raise ValueError("the number of occupied spatial blocks is smaller than n_folds")

    total_load = np.zeros(n_folds, dtype=float)
    stratum_load = np.zeros((n_folds, len(strata)), dtype=float)
    target_total = block_counts.sum() / n_folds
    target_strata = block_strata.sum(axis=0).to_numpy(float) / n_folds
    block_to_fold: dict[str, int] = {}

    ordered_blocks = sorted(block_counts.index, key=lambda b: (-block_counts[b], str(b)))
    for block in ordered_blocks:
        count = float(block_counts[block])
        composition = block_strata.loc[block].to_numpy(float)
        scores = []
        for fold in range(n_folds):
            candidate_total = total_load.copy()
            candidate_strata = stratum_load.copy()
            candidate_total[fold] += count
            candidate_strata[fold] += composition
            size_cost = np.square((candidate_total - target_total) / max(target_total, 1)).sum()
            stratum_cost = np.square(
                (candidate_strata - target_strata) / np.maximum(target_strata, 1)
            ).sum()
            scores.append((size_cost + stratum_cost, candidate_total[fold], fold))
        chosen = min(scores)[2]
        block_to_fold[str(block)] = chosen
        total_load[chosen] += count
        stratum_load[chosen] += composition

    folds = table["spatial_block"].astype(str).map(block_to_fold)
    if folds.isna().any() or sorted(folds.unique()) != list(range(n_folds)):
        raise RuntimeError("invalid fold assignment")
    if table.assign(_fold=folds).groupby(site_column)["_fold"].nunique().max() != 1:
        raise RuntimeError("a station spans validation folds")
    if table.assign(_fold=folds).groupby("spatial_block")["_fold"].nunique().max() != 1:
        raise RuntimeError("a spatial block spans validation folds")
    return folds.to_numpy(int)


def encode_features(
    table: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    features = table[feature_columns].copy()
    for column in categorical_columns:
        categories = sorted(features[column].dropna().unique())
        features[column] = pd.Categorical(features[column], categories=categories)
    numeric = [c for c in feature_columns if c not in categorical_columns]
    if features[feature_columns].isna().any().any():
        raise ValueError("features contain missing values")
    if numeric and not np.isfinite(features[numeric].to_numpy(float)).all():
        raise ValueError("numeric features contain non-finite values")
    return features


def resolve_parameters(configuration: dict, n_jobs: int, seed: int) -> dict:
    parameters = dict(configuration)
    parameters["max_bin"] = 2 ** int(parameters.pop("log_max_bin")) - 1
    for name in ("num_leaves", "n_estimators", "min_child_samples", "max_depth"):
        parameters[name] = int(parameters[name])
    parameters.update(
        objective="regression",
        subsample_freq=1,
        n_jobs=n_jobs,
        random_state=seed,
        verbosity=-1,
    )
    return parameters


def station_equal_weights(table: pd.DataFrame, site_column: str) -> np.ndarray:
    counts = table.groupby(site_column)[site_column].transform("size")
    weights = (1.0 / counts).to_numpy(float)
    sums = pd.Series(weights, index=table.index).groupby(table[site_column]).sum()
    if not np.allclose(sums.to_numpy(), 1.0):
        raise RuntimeError("station weights do not sum to one")
    return weights


def paired_block_standard_error(
    candidate: dict,
    best: dict,
    site_to_block: dict,
    n_bootstrap: int,
    seed: int,
) -> float:
    sites = sorted(best["site_mse"])
    if set(candidate["site_mse"]) != set(sites):
        raise RuntimeError("candidate configurations were not scored on the same stations")
    blocks = sorted({site_to_block[site] for site in sites})
    block_sites = {block: [s for s in sites if site_to_block[s] == block] for block in blocks}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(blocks), size=len(blocks))
        drawn_sites = [site for i in sampled for site in block_sites[blocks[i]]]
        candidate_risk = np.sqrt(np.mean([candidate["site_mse"][s] for s in drawn_sites]))
        best_risk = np.sqrt(np.mean([best["site_mse"][s] for s in drawn_sites]))
        differences.append(candidate_risk - best_risk)
    return float(np.std(differences, ddof=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--site-column", default="site_id")
    parser.add_argument("--latitude-column", default="latitude")
    parser.add_argument("--longitude-column", default="longitude")
    parser.add_argument("--categorical", nargs="*", default=[])
    parser.add_argument("--block-size-degrees", type=float, default=3.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--time-budget", type=float, default=3600.0)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--clip", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = set(args.features + [
        args.target,
        args.site_column,
        args.latitude_column,
        args.longitude_column,
    ])
    table = read_table(args.table)
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    table = table.dropna(subset=list(required)).reset_index(drop=True)
    table = add_spatial_blocks(
        table,
        args.site_column,
        args.latitude_column,
        args.longitude_column,
        args.block_size_degrees,
    )
    folds = assign_blocked_folds(table, args.site_column, args.target, args.folds)
    features = encode_features(table, args.features, args.categorical)
    target = table[args.target].to_numpy(float)
    weights = station_equal_weights(table, args.site_column)
    site_to_block = (
        table.drop_duplicates(args.site_column)
        .set_index(args.site_column)["spatial_block"]
        .to_dict()
    )
    records: dict[tuple, dict] = {}

    def evaluate(configuration: dict) -> dict:
        normalized = {
            key: float(value) if isinstance(value, (float, np.floating)) else int(value)
            for key, value in configuration.items()
        }
        key = tuple(sorted(normalized.items()))
        if key in records:
            return {"risk": records[key]["risk"]}

        predictions = np.full(len(table), np.nan)
        realized_leaves = []
        for fold in range(args.folds):
            validation = folds == fold
            model = lgb.LGBMRegressor(
                **resolve_parameters(normalized, args.jobs, args.seed)
            )
            model.fit(
                features.loc[~validation],
                target[~validation],
                sample_weight=weights[~validation],
                categorical_feature=args.categorical,
            )
            predictions[validation] = model.predict(features.loc[validation])
            realized_leaves.append(
                sum(tree["num_leaves"] for tree in model.booster_.dump_model()["tree_info"])
            )

        if args.clip:
            predictions = np.clip(predictions, args.clip[0], args.clip[1])
        squared_error = np.square(predictions - target)
        site_mse = (
            pd.DataFrame({"site": table[args.site_column], "error": squared_error})
            .groupby("site")["error"]
            .mean()
            .to_dict()
        )
        risk = float(np.sqrt(np.mean(list(site_mse.values()))))
        records[key] = {
            "configuration": normalized,
            "risk": risk,
            "realized_leaves": float(np.mean(realized_leaves)),
            "site_mse": site_mse,
        }
        return {"risk": risk}

    search = CFO(
        metric="risk",
        mode="min",
        space=SEARCH_SPACE,
        time_budget_s=args.time_budget,
        num_samples=-1,
        seed=args.seed,
    )
    tune.run(
        evaluate,
        config=SEARCH_SPACE,
        metric="risk",
        mode="min",
        time_budget_s=args.time_budget,
        num_samples=-1,
        search_alg=search,
        verbose=1,
    )
    trials = list(records.values())
    if not trials:
        raise RuntimeError("the search evaluated no configurations")
    best = min(trials, key=lambda record: record["risk"])
    eligible = []
    for candidate in trials:
        standard_error = paired_block_standard_error(
            candidate,
            best,
            site_to_block,
            args.bootstrap_draws,
            args.seed,
        )
        if candidate["risk"] - best["risk"] <= standard_error:
            eligible.append(candidate)
    selected = min(
        eligible,
        key=lambda record: (
            record["realized_leaves"],
            -record["configuration"]["min_child_samples"],
            -record["configuration"]["reg_lambda"],
        ),
    )
    result = {
        "selected_parameters": selected["configuration"],
        "cross_validated_risk": selected["risk"],
        "evaluated_configurations": len(trials),
        "folds": args.folds,
        "selection_rule": "paired block-bootstrap 1-SE, then smallest realized leaf count",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
