"""Compare binary classifiers and evaluate the selected pipeline once."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.preprocess import CATEGORICAL, NON_FEATURES, TEST_NAME, TRAIN_NAME, prepare
from src.utils import MODEL_DIR, PROCESSED_DIR, REPORT_DIR


SEED = 42
ARTIFACTS = {
    "Logistic Regression": "nids_logistic_regression.pkl",
    "Random Forest": "nids_random_forest.pkl",
    "Gradient Boosting": "nids_gradient_boosting.pkl",
}


def make_pipeline(name: str, numeric: list[str]) -> Pipeline:
    if name == "Gradient Boosting":
        prep = ColumnTransformer(
            [
                ("categorical", Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]), CATEGORICAL),
                ("numeric", SimpleImputer(strategy="median"), numeric),
            ],
            sparse_threshold=0,
        )
        estimator = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
            categorical_features=list(range(len(CATEGORICAL))), random_state=SEED,
        )
    else:
        numerical_steps = [("impute", SimpleImputer(strategy="median"))]
        if name == "Logistic Regression":
            numerical_steps.append(("scale", StandardScaler()))
        prep = ColumnTransformer(
            [
                ("numeric", Pipeline(numerical_steps), numeric),
                ("categorical", Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore")),
                ]), CATEGORICAL),
            ]
        )
        if name == "Logistic Regression":
            estimator = LogisticRegression(max_iter=1500, random_state=SEED)
        else:
            estimator = RandomForestClassifier(
                n_estimators=120, min_samples_leaf=2, n_jobs=-1, random_state=SEED
            )
    return Pipeline([("preprocess", prep), ("model", estimator)])


def scores(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> tuple[dict, list[list[int]]]:
    probability = model.predict_proba(x)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return scores(y, prediction, probability), confusion_matrix(y, prediction, labels=[0, 1]).tolist()


def main() -> None:
    train_path = PROCESSED_DIR / TRAIN_NAME
    test_path = PROCESSED_DIR / TEST_NAME
    if not train_path.exists() or not test_path.exists():
        prepare()
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    features = [col for col in train if col not in NON_FEATURES]
    numeric = [col for col in features if col not in CATEGORICAL]
    x_train, x_val, y_train, y_val = train_test_split(
        train[features], train["label"], test_size=0.20,
        stratify=train["label"], random_state=SEED,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = []
    for name in ARTIFACTS:
        print(f"Fitting {name} for validation...", flush=True)
        model = make_pipeline(name, numeric)
        model.fit(x_train, y_train)
        metrics, _ = evaluate(model, x_val, y_val)
        comparison.append({"model": name, **metrics})
        print(f"  recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}", flush=True)
    comparison.sort(
        key=lambda row: (row["recall"], row["f1"], row["model"] == "Random Forest"),
        reverse=True,
    )
    selected = comparison[0]["model"]
    print(f"Selected {selected}; fitting on all official training rows...", flush=True)
    final_model = make_pipeline(selected, numeric)
    final_model.fit(train[features], train["label"])
    selected_artifact = ARTIFACTS[selected]
    joblib.dump(final_model, MODEL_DIR / selected_artifact, compress=3)
    if selected != "Random Forest":
        print("Saving full-training Random Forest comparison artifact...", flush=True)
        forest = make_pipeline("Random Forest", numeric)
        forest.fit(train[features], train["label"])
        joblib.dump(forest, MODEL_DIR / ARTIFACTS["Random Forest"], compress=3)
    test_metrics, matrix = evaluate(final_model, test[features], test["label"])
    print(f"Test recall={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}", flush=True)

    # Permute original input columns, so categories and numerical features remain readable.
    sample = x_val.sample(n=min(1200, len(x_val)), random_state=SEED)
    importance = permutation_importance(
        final_model, sample, y_val.loc[sample.index], scoring="recall",
        n_repeats=2, random_state=SEED, n_jobs=1,
    )
    pd.DataFrame({"feature": features, "importance": importance.importances_mean})\
        .sort_values("importance", ascending=False).to_csv(
            REPORT_DIR / "feature_importance.csv", index=False
        )
    pd.DataFrame(comparison).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(matrix, index=["Actual normal", "Actual attack"],
                 columns=["Predicted normal", "Predicted attack"]).to_csv(
        REPORT_DIR / "confusion_matrix.csv"
    )
    evaluation = {
        "selected_model": selected,
        "artifact": selected_artifact,
        "selection_rule": "Highest validation attack recall; F1 breaks ties; threshold 0.5",
        "validation_rows": len(x_val),
        "test_rows": len(test),
        "validation_comparison": comparison,
        "test_metrics": test_metrics,
        "confusion_matrix": matrix,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
    }
    (REPORT_DIR / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    quantiles = {
        col: float(train[col].quantile(0.95))
        for col in ["rate", "spkts", "dpkts", "sbytes", "dbytes", "dur"]
    }
    metadata = {
        "model_name": selected,
        "artifact": selected_artifact,
        "features": features,
        "categorical": CATEGORICAL,
        "numeric": numeric,
        "training_p95": quantiles,
        "decision_threshold": 0.5,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
