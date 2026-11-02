import json
from taxonomyiq.model import load_model
from taxonomyiq.monitor import population_stability_index

def test_trained_model_and_report(trained):
    model, _ = trained
    p = load_model(model)
    pred = p.predict(["wireless bluetooth gaming earbuds"], threshold=1.0)[0]
    assert pred["category"] in p.metadata["labels"]
    assert pred["needs_review"]
    assert len(pred["alternatives"]) == 3
    report = json.loads((model.parent / "reports" / "evaluation.json").read_text())
    assert report["test_count"] == 96
    assert 0 <= report["test_macro_f1"] <= 1

def test_drift_empty_and_positive():
    assert population_stability_index({"a": .5, "b": .5}, {})["psi"] is None
    assert population_stability_index({"a": .5, "b": .5}, {"a": 100})["psi"] > .2
