import csv
from taxonomyiq.data import generate
from taxonomyiq.taxonomy import LABELS

def test_repeatable_and_template_family_isolation(tmp_path):
    first = generate(tmp_path / "a")
    second = generate(tmp_path / "b")
    assert first == second
    families = []
    for split in ("train", "validation", "test"):
        content_a = (tmp_path / "a" / f"{split}.csv").read_text()
        assert content_a == (tmp_path / "b" / f"{split}.csv").read_text()
        rows = list(csv.DictReader(content_a.splitlines()))
        assert set(r["label"] for r in rows) == set(LABELS)
        families.append({r["template_family"] for r in rows})
    assert not (families[0] & families[1] | families[0] & families[2] | families[1] & families[2])
