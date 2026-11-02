"""CLI entrypoints for reproducible dataset, training, batch and drift."""
import argparse
import csv
import json
from pathlib import Path
from .config import DB_PATH, MODEL_DIR, ROOT
from .data import generate
from .evaluation import run
from .model import load_model
from .monitor import population_stability_index
from .storage import Store


def main():
    p = argparse.ArgumentParser(description="TaxonomyIQ")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("generate-data"); sub.add_parser("train-evaluate"); sub.add_parser("monitor")
    batch = sub.add_parser("batch"); batch.add_argument("input_csv"); batch.add_argument("output_csv")
    args = p.parse_args()
    data = ROOT / "data" / "synthetic"
    if args.command == "generate-data": print(json.dumps(generate(data), indent=2))
    elif args.command == "train-evaluate": print(json.dumps(run(data, MODEL_DIR, ROOT / "reports"), indent=2))
    elif args.command == "monitor":
        meta = load_model(MODEL_DIR).metadata
        print(json.dumps(population_stability_index(meta["reference_label_distribution"], Store(DB_PATH).distribution()), indent=2))
    elif args.command == "batch":
        with Path(args.input_csv).open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if any(not (r.get("description") or "").strip() for r in rows):
            raise SystemExit("Each row requires a nonempty 'description' column")
        out = load_model(MODEL_DIR).predict([r["description"] for r in rows])
        with Path(args.output_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["description", "category", "confidence", "needs_review", "model_version"])
            writer.writeheader()
            for item, prediction in zip(rows, out):
                writer.writerow({"description": item["description"], **{k: prediction[k] for k in writer.fieldnames if k != "description"}})
        print(f"Wrote {len(rows)} predictions to {args.output_csv}")
if __name__ == "__main__": main()
