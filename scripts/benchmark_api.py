import json
import statistics
import time
from fastapi.testclient import TestClient
from taxonomyiq.api import create_app
from taxonomyiq.config import ROOT

with TestClient(create_app()) as client:
    for _ in range(10): client.post("/v1/predict", json={"description": "wireless bluetooth earbud headphones"})
    durations=[]
    for _ in range(100):
        start=time.perf_counter()
        result=client.post("/v1/predict",json={"description":"wireless bluetooth earbud headphones"})
        result.raise_for_status(); durations.append((time.perf_counter()-start)*1000)
report={"mode":"sequential in-process TestClient, warm model, local SQLite, 100 requests",
        "count":len(durations), "p50_ms":round(statistics.median(durations),3),
        "p95_ms":round(sorted(durations)[94],3),
        "max_ms":round(max(durations),3),
        "limitations":"Not network load testing; machine-dependent; no production SLO inference."}
(ROOT / "reports" / "api_benchmark.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
