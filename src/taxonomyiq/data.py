"""Deterministic, explicitly synthetic dataset with template-family holdout splits."""
import csv
import random
from collections import Counter
from pathlib import Path
from .taxonomy import EXAMPLES, LABELS

PREFIX = ["", "premium", "standard", "bulk", "new", "eco friendly", "store brand", "imported"]
SUFFIX = ["", "pack of 2", "single unit", "family size", "2026 edition", "retail box", "500g", "value bundle"]

def generate(folder: Path, seed: int = 41) -> dict:
    """Hold out entire base templates so near-duplicate templates do not cross splits."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for label in LABELS:
        templates = list(EXAMPLES[label])
        rng.shuffle(templates)
        for i, template in enumerate(templates):
            split = "train" if i < 7 else "validation" if i == 7 else "test"
            for variant in range(6):
                text = f"{rng.choice(PREFIX)} {template} {rng.choice(SUFFIX)}".strip()
                splits[split].append({"description": " ".join(text.split()),
                                      "label": label, "template_family": template,
                                      "variant": variant})
    for split, rows in splits.items():
        rng.shuffle(rows)
        with (folder / f"{split}.csv").open("w", newline="", encoding="utf-8") as f:
            out = csv.DictWriter(f, fieldnames=["description", "label", "template_family", "variant"])
            out.writeheader(); out.writerows(rows)
    return {split: {"rows": len(rows), "classes": dict(Counter(x["label"] for x in rows))}
            for split, rows in splits.items()}

def load(path: Path) -> tuple[list[str], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["description"] for r in rows], [r["label"] for r in rows]
