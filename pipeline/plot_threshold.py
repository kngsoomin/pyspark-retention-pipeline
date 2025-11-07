import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array


def get_spark(app_name="plot-threshold"):
    return SparkSession.builder.appName(app_name).getOrCreate()


def sweep_thresholds(pdf: pd.DataFrame, steps: int = 101) -> pd.DataFrame:
    thr_grid = np.linspace(0.0, 1.0, steps)
    rows = []
    for t in thr_grid:
        pred = (pdf["p1"] >= t).astype(int)
        tp = int(((pdf["label"] == 1) & (pred == 1)).sum())
        fp = int(((pdf["label"] == 0) & (pred == 1)).sum())
        tn = int(((pdf["label"] == 0) & (pred == 0)).sum())
        fn = int(((pdf["label"] == 1) & (pred == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        rows.append({"threshold": float(t), "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="processed dir with parquet splits")
    ap.add_argument("--model_dir", required=True, help="saved PipelineModel dir")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", default="docs/threshold_curve.png")
    ap.add_argument("--csv_out", default="docs/threshold_curve.csv")
    args = ap.parse_args()

    spark = get_spark()
    model = PipelineModel.load(args.model_dir)

    sdf = spark.read.parquet(str(Path(args.in_dir) / args.split))
    pred = (model.transform(sdf)
                 .withColumn("p1", vector_to_array(F.col("probability")).getItem(1))
                 .select("label", "p1"))
    pdf = pred.toPandas()

    curve = sweep_thresholds(pdf, steps=101)
    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.csv_out, index=False)

    best_thr = None
    metrics_path = Path(args.model_dir) / "metrics.json"
    if metrics_path.exists():
        try:
            best_thr = float(json.loads(metrics_path.read_text())["best_threshold"]["threshold"])
        except Exception:
            best_thr = None

    plt.figure(figsize=(7, 5))
    plt.plot(curve["threshold"], curve["f1"], label="F1")
    plt.plot(curve["threshold"], curve["precision"], label="Precision")
    plt.plot(curve["threshold"], curve["recall"], label="Recall")
    
    if best_thr is not None:
        plt.axvline(best_thr, linestyle="--", label=f"Best thr = {best_thr:.2f}")
    
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    
    plt.title(f"Threshold Optimization Curve ({args.split})")
    
    plt.legend()
    
    plt.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"[PLOT] Saved curve PNG as {args.out}")
    print(f"[PLOT] Saved curve CSV as {args.csv_out}")


if __name__ == "__main__":
    main()
