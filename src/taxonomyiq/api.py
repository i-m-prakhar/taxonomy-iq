"""Versioned FastAPI serving: real-time, bounded batch, feedback, monitoring."""
import hmac
import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field, field_validator
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from .config import API_KEY, DB_PATH, MODEL_DIR, REVIEW_THRESHOLD
from .model import load_model
from .monitor import population_stability_index
from .storage import Store
from .taxonomy import LABELS

logger = logging.getLogger(__name__)
REQUESTS = Counter("taxonomyiq_requests_total", "Prediction requests", ["result"])
LATENCY = Histogram("taxonomyiq_predict_seconds", "Prediction handler latency")

class Item(BaseModel):
    description: str = Field(min_length=3, max_length=2000)
    request_id: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("description")
    @classmethod
    def nonblank(cls, v):
        if not v.strip(): raise ValueError("description cannot be blank")
        return v.strip()

class Batch(BaseModel):
    items: list[Item] = Field(min_length=1, max_length=200)

class Feedback(BaseModel):
    request_id: str = Field(min_length=1, max_length=100)
    corrected_category: str
    reviewer_note: str = Field(default="", max_length=250)

    @field_validator("corrected_category")
    @classmethod
    def known_label(cls, v):
        if v not in LABELS: raise ValueError("unknown category")
        return v


def require_api_key(x_api_key: str | None = Header(None)):
    if API_KEY and (x_api_key is None or not hmac.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(401, "API key required")


def create_app(model_dir=None, db_path=None):
    model_path = model_dir or MODEL_DIR
    path = db_path or DB_PATH
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = load_model(model_path)
        app.state.store = Store(path)
        yield
    app = FastAPI(title="TaxonomyIQ", version="1.0.0", lifespan=lifespan,
                  description="Experimental product-category classification; not a tax decision engine")

    @app.get("/health/live")
    def live(): return {"status": "alive"}

    @app.get("/health/ready")
    def ready(): return {"status": "ready", "model_version": app.state.predictor.metadata["model_version"]}

    @app.get("/metrics")
    def metrics(): return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/model", dependencies=[Depends(require_api_key)])
    def model(): return app.state.predictor.metadata

    def infer(items: list[Item]):
        started = time.perf_counter()
        with LATENCY.time():
            outputs = app.state.predictor.predict([i.description for i in items], REVIEW_THRESHOLD)
        for item, output in zip(items, outputs):
            output["request_id"] = item.request_id or str(uuid4())
            app.state.store.record(output["request_id"], output)
        REQUESTS.labels(result="ok").inc(len(items))
        logger.info("prediction_completed count=%d duration_ms=%.3f", len(items), (time.perf_counter()-started)*1000)
        return outputs

    @app.post("/v1/predict", dependencies=[Depends(require_api_key)])
    def predict(item: Item): return infer([item])[0]

    @app.post("/v1/predict/batch", dependencies=[Depends(require_api_key)])
    def predict_batch(batch: Batch): return {"items": infer(batch.items), "count": len(batch.items)}

    @app.post("/v1/feedback", dependencies=[Depends(require_api_key)])
    def feedback(item: Feedback):
        if not app.state.store.feedback(item.request_id, item.corrected_category, item.reviewer_note):
            raise HTTPException(404, "prediction request_id not found")
        return {"status": "recorded", "request_id": item.request_id}

    @app.get("/v1/monitor/drift", dependencies=[Depends(require_api_key)])
    def drift():
        return population_stability_index(app.state.predictor.metadata["reference_label_distribution"],
                                          app.state.store.distribution())
    return app

app = create_app()
