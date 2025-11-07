'''
This script performs:

1) Loads the persisted Spark PipelineModel from --model_dir
2) Loads the held-out test split from --in_dir/test
3) Computes AUC-ROC and AUC-PR on the test set
4) Reads the best operating threshold from metrics.json (fallback = 0.5)
5) Applies the threshold to produce a confusion matrix and point metrics
   (precision / recall / F1 / positive rate)

'''
import argparse, json
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def get_spark(app_name="eval-online-retail"):
    return (SparkSession.builder.appName(app_name).getOrCreate())

def main():
    '''
    Steps:
        1. Parse CLI args for processed dir and model dir.
        2. Load test parquet, cache to stabilize timings.
        3. Load PipelineModel and compute prediction probabilities.
        4. Compute AUC-ROC and AUC-PR using BinaryClassificationEvaluator.
        5. Load best threshold from metrics.json.
        6. Apply threshold to get confusion matrix and point metrics.
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="processed dir with test split")
    ap.add_argument("--model_dir", required=True, help="saved PipelineModel dir (latest_model)")
    ap.add_argument("--metrics_path", default=None, help="metrics.json (optional; defaults to model_dir/metrics.json)")
    args = ap.parse_args()

    spark = get_spark()
    in_dir = Path(args.in_dir); model_dir = Path(args.model_dir)
    metrics_path = Path(args.metrics_path) if args.metrics_path else (model_dir / "metrics.json")

    # Load the held-out test split produced by ETL.
    test = spark.read.parquet(str(in_dir / "test")).cache()
    _ = test.count() # materialize cache

    # Load the trained PipelineModel (Imputer -> Assembler -> Scaler -> Classifier).
    model = PipelineModel.load(str(model_dir))

    # Run inference on test and extract p(y=1) from the probability vector.
    pred = (
        model.transform(test)
             .withColumn("p1", vector_to_array("probability")[1])
             .cache()
    )
    _ = pred.count()

    # AUCs (evaluator is stateless)
    auc_roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                            metricName="areaUnderROC").evaluate(pred)
    auc_pr  = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                            metricName="areaUnderPR").evaluate(pred)

    # Default to 0.5; if metrics.json exists, prefer the best validation threshold
    # chosen during training (e.g., best F1).
    thr = 0.5
    try:
        with open(metrics_path) as f:
            thr = float(json.load(f)["best_threshold"]["threshold"])
    except Exception:
        pass

    # Confusion matrix & point metrics
    scored = pred.withColumn("pred_label", (F.col("p1") >= F.lit(thr)).cast("double"))
    tp = scored.filter("label=1.0 and pred_label=1.0").count()
    fp = scored.filter("label=0.0 and pred_label=1.0").count()
    tn = scored.filter("label=0.0 and pred_label=0.0").count()
    fn = scored.filter("label=1.0 and pred_label=0.0").count()

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    pos_rate  = test.filter("label=1.0").count() / max(test.count(), 1)

    print("[EVAL] Test AUC-ROC = %.4f  AUC-PR = %.4f" % (auc_roc, auc_pr))
    print("[EVAL] Threshold = %.2f" % thr)
    print("[EVAL] Confusion Matrix (label=1 is positive):")
    print("       TP=%d  FP=%d  TN=%d  FN=%d" % (tp, fp, tn, fn))
    print("[EVAL] Precision=%.4f  Recall=%.4f  F1=%.4f  PosRate=%.4f" % (precision, recall, f1, pos_rate))

if __name__ == "__main__":
    main()
