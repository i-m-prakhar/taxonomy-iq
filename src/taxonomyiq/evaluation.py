"""Repeatable holdout evaluation (template-family separated)."""
import csv
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, f1_score
from .data import generate, load
from .model import make_model, save_model
from .taxonomy import LABELS


def run(data_dir: Path, model_dir: Path, reports: Path) -> dict:
    if not (data_dir / "train.csv").exists():
        generate(data_dir)
    x_train, y_train = load(data_dir / "train.csv")
    x_val, y_val = load(data_dir / "validation.csv")
    x_test, y_test = load(data_dir / "test.csv")
    reports.mkdir(parents=True, exist_ok=True)
    candidates = {}
    fits = {}
    for kind in ("word", "hybrid"):
        model = make_model(kind)
        t0 = time.perf_counter(); model.fit(x_train, y_train)
        elapsed = time.perf_counter() - t0
        pred = model.predict(x_val)
        candidates[kind] = {"validation_macro_f1": round(float(f1_score(y_val, pred, average="macro")), 5),
                            "fit_seconds": round(elapsed, 4)}
        fits[kind] = model
    chosen = sorted(candidates, key=lambda k: (-candidates[k]["validation_macro_f1"], k))[0]
    model = fits[chosen]
    t0 = time.perf_counter(); probs = model.predict_proba(x_test)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / len(x_test)
    pred = model.classes_[np.argmax(probs, axis=1)]
    mx = np.max(probs, axis=1)
    cm = confusion_matrix(y_test, pred, labels=LABELS)
    with (reports / "confusion_matrix.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["actual/predicted", *LABELS])
        for name, row in zip(LABELS, cm): w.writerow([name, *row.tolist()])
    with (reports / "error_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["description", "expected", "predicted", "predicted_confidence"])
        for description, actual, predicted, confidence in zip(x_test, y_test, pred, mx):
            if actual != predicted:
                writer.writerow([description, actual, predicted, round(float(confidence), 6)])
    calibration_bins = []
    for lower in np.linspace(0, 1, 11)[:-1]:
        mask = (mx >= lower) & (mx < lower + 0.1 if lower < 0.9 else mx <= 1)
        if mask.any():
            calibration_bins.append({"min_confidence": round(float(lower), 2),
                                     "count": int(mask.sum()),
                                     "accuracy": round(float(np.mean(pred[mask] == np.array(y_test)[mask])), 4),
                                     "avg_confidence": round(float(mx[mask].mean()), 4)})
    ece = sum(b["count"] / len(y_test) * abs(b["accuracy"] - b["avg_confidence"])
              for b in calibration_bins)
    report = {
        "dataset": "synthetic; NOT statutory tax labels or real catalog data",
        "split": "template-family disjoint 70/10/20; 6 variants/template; seed=41",
        "train_count": len(x_train), "validation_count": len(x_val), "test_count": len(x_test),
        "candidate_models": candidates, "selected_by_validation": chosen,
        "test_accuracy": round(float(accuracy_score(y_test, pred)), 5),
        "test_macro_f1": round(float(f1_score(y_test, pred, average="macro")), 5),
        "test_log_loss": round(float(log_loss(y_test, probs, labels=list(model.classes_))), 5),
        "expected_calibration_error": round(float(ece), 5),
        "mean_batch_inference_ms_per_item": round(elapsed_ms, 5),
        "coverage_at_0_70": round(float((mx >= .7).mean()), 4),
        "accuracy_at_0_70": round(float(np.mean(pred[mx >= .7] == np.array(y_test)[mx >= .7])), 4)
                               if (mx >= .7).any() else None,
        "classification_report": classification_report(y_test, pred, labels=LABELS, output_dict=True, zero_division=0),
        "python_version": platform.python_version(),
        "limitations": "Tiny synthetic holdout; no real-world effectiveness, legal tax accuracy or production SLO claims."
    }
    metadata = {"model_version": f"calibrated-{chosen}-v1", "feature_type": chosen,
                "labels": LABELS, "training_rows": len(x_train),
                "reference_label_distribution": {k: v/len(y_train) for k, v in Counter(y_train).items()},
                "training_data": "synthetic seed=41; template-family split"}
    save_model(model, model_dir, metadata)
    (reports / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    # Optional experiment tracking, not needed for the local zero-service path.
    if os.getenv("TAXONOMYIQ_TRACK_MLFLOW") == "1":
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("Install the optional extra: uv sync --extra mlops") from exc
        mlflow.set_tracking_uri((reports.parent / "mlruns").resolve().as_uri())
        mlflow.set_experiment("taxonomyiq-synthetic-baseline")
        with mlflow.start_run(run_name=metadata["model_version"]):
            mlflow.log_params({"training_data": "synthetic", "selected_model": chosen, "seed": 41})
            mlflow.log_metrics({"test_macro_f1": report["test_macro_f1"],
                                "test_accuracy": report["test_accuracy"],
                                "expected_calibration_error": report["expected_calibration_error"]})
            mlflow.log_artifact(str(reports / "evaluation.json"))
    return report
