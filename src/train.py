"""Reproducible comparison with a fixed train/validation/test split."""
import argparse
import itertools
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("x1", "x2", "x3", "x4")


def evaluate(model, data, features):
    actual = data["y"]
    predicted = model.predict(data[list(features)])
    return {"rmse": float(mean_squared_error(actual, predicted) ** 0.5),
            "r2": float(r2_score(actual, predicted))}


def run(count):
    data = pd.read_csv(ROOT / "data" / "sampregdata.csv")
    if data[list(CANDIDATES) + ["y"]].isna().any().any():
        raise ValueError("Input has missing predictors or target")
    development_idx, test_idx = train_test_split(data.index, test_size=0.2,
                                                   random_state=42)
    train_idx, validation_idx = train_test_split(development_idx, test_size=0.25,
                                                  random_state=42)
    train, validation, test = data.loc[train_idx], data.loc[validation_idx], data.loc[test_idx]
    ranking = []
    for features in itertools.combinations(CANDIDATES, count):
        candidate = LinearRegression().fit(train[list(features)], train["y"])
        ranking.append({"features": list(features), "validation": evaluate(candidate, validation, features)})
    ranking.sort(key=lambda row: row["validation"]["rmse"])
    best_features = ranking[0]["features"]
    # Refit with the validation data after feature selection; test stays untouched.
    final = LinearRegression().fit(data.loc[development_idx, best_features],
                                   data.loc[development_idx, "y"])
    report = {"features": best_features, "n_rows": len(data),
              "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
              "candidate_ranking": ranking, "test": evaluate(final, test, best_features),
              "intercept": float(final.intercept_),
              "coefficients": dict(zip(best_features, map(float, final.coef_)))}
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": best_features}, ROOT / "models" / "current.joblib")
    (ROOT / "results" / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, choices=(1, 2), default=2)
    run(parser.parse_args().features)
