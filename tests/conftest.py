import pytest
from taxonomyiq.data import generate
from taxonomyiq.evaluation import run

@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("trained")
    data = root / "data"; model = root / "model"; reports = root / "reports"
    generate(data)
    run(data, model, reports)
    return model, root / "service.sqlite3"
