'''
This script performs:

1) Loads a persisted Spark PipelineModel and its feature schema
2) Exposes /schema for required feature names
3) Exposes /predict for single-sample inference (JSON)
4) Exposes /predict_batch for multi-sample (batch) inference
5) Applies the best operating threshold from metrics.json (fallback=0.5)

'''
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array


MODEL_DIR = Path("data/models/latest_model")
PROCESSED_DIR = Path("data/processed")
METRICS_PATH = MODEL_DIR / "metrics.json"

app = FastAPI(title="Retention Pipeline Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

spark: Optional[SparkSession] = None
model: Optional[PipelineModel] = None
feature_cols: List[str] = []
best_threshold: float = 0.5


class PredictItem(BaseModel):
    """
    Payload schema for single prediction.
    'features' must include numeric values keyed by feature name.
    Missing features are imputed as 0.0 server-side.
    """
    features: Dict[str, float]


class PredictBatch(BaseModel):
    """
    Payload schema for batch prediction.
    'items' is a list of PredictItem objects.
    """
    items: List[PredictItem]


def init_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("serve-online-retention")
        .master("local[*]")
        .getOrCreate()
    )


def load_artifacts() -> None:
    '''
    Load the persisted PipelineModel and supporting artifacts:
    - VectorAssembler input columns (feature_cols) inferred from the pipeline
    or from the processed features parquet as a fallback
    - Best operating threshold from metrics.json
    '''
    global model, feature_cols, best_threshold
    
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")

    # Load model
    m = PipelineModel.load(str(MODEL_DIR))

    # Try to read assembler inputCols from pipeline
    # stages: [Imputer, VectorAssembler, StandardScaler, LR]
    try:
        assembler = m.stages[1]
        feature_cols = list(assembler.getInputCols())
    except Exception:
        # fallback: read from features parquet (schema)
        feats_path = PROCESSED_DIR / "features"
        if feats_path.exists():
            df = spark.read.parquet(str(feats_path)).limit(1)
            all_cols = [c for c in df.columns if c not in {"CustomerID", "label"}]
            feature_cols = all_cols
        else:
            raise RuntimeError("Cannot infer feature columns (no assembler and no features parquet).")

    # Load best threshold
    if METRICS_PATH.exists():
        try:
            data = json.loads(METRICS_PATH.read_text())
            best_threshold = float(data.get("best_threshold", {}).get("threshold", 0.5))
        except Exception:
            best_threshold = 0.5
    else:
        best_threshold = 0.5

    return m


def dicts_to_spark(df_dicts: List[Dict[str, Any]]) -> DataFrame:
    """
    List of Python dict -> Spark DataFrame
    - unknown features are ignored
    - missing features are filled with 0.0
    - casts everything to double
    """
    # normalize: keep only known features
    rows = []
    for d in df_dicts:
        row = {}
        for f in feature_cols:
            val = d.get(f, 0.0)
            try:
                row[f] = float(val)
            except Exception:
                row[f] = 0.0
        rows.append(row)

    sdf = spark.createDataFrame(rows)

    # enforce double
    for f in feature_cols:
        sdf = sdf.withColumn(f, F.col(f).cast("double"))
    return sdf


def run_inference(sdf: DataFrame, thr: float) -> List[Dict[str, Any]]:
    '''
    Run the full pipeline transform and apply the decision threshold.
    '''
    pred = (
        model.transform(sdf)
        .withColumn("p1", vector_to_array(F.col("probability")).getItem(1))
        .withColumn("pred_label", (F.col("p1") >= F.lit(thr)).cast("double"))
        .select(*feature_cols, "p1", "pred_label")
    )

    # Convert to pandas for lightweight response formatting.
    # NOTE: For very large batches, consider writing to parquet/DB instead.
    out = pred.toPandas()
    results = []
    for _, r in out.iterrows():
        results.append({
            "probability": float(r["p1"]),
            "prediction": int(r["pred_label"]),
        })
    return results


@app.on_event("startup")
def _startup():
    """FastAPI startup hook: initialize Spark and load model artifacts once."""
    global spark, model
    try:
        spark = init_spark()
        model = load_artifacts()
        print(f"[API] Loaded model from: {MODEL_DIR}")
        print(f"[API] Feature columns: {feature_cols}")
        print(f"[API] Best threshold: {best_threshold}")
    except Exception as e:
        raise RuntimeError(f"Startup failed: {e}") from e


@app.get("/health")
def health():
    """
    Returns the current feature schema and threshold for quick sanity checks.
    """
    return {"status": "ok", "features": feature_cols, "threshold": best_threshold}


@app.get("/schema")
def schema():
    """
    Return the required feature names in the exact order expected by the model.
    """
    return {"required_features": feature_cols}


@app.post("/predict")
def predict(payload: PredictItem):
    """
    Single-sample prediction. Accepts one feature dict and returns one prediction result.
    """
    if model is None or spark is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    sdf = dicts_to_spark([payload.features])
    results = run_inference(sdf, best_threshold)
    
    return {"n": 1, "threshold": best_threshold, "results": results}


@app.post("/predict_batch")
def predict_batch(payload: PredictBatch):
    """
    Batch prediction. Accepts a list of feature dicts and returns a list of results in order.
    """
    if model is None or spark is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not payload.items:
        return {"n": 0, "threshold": best_threshold, "results": []}
    
    sdf = dicts_to_spark([it.features for it in payload.items])
    results = run_inference(sdf, best_threshold)
    
    return {"n": len(results), "threshold": best_threshold, "results": results}


# To run locally:
#   uvicorn serve_api:app --reload --port 8000