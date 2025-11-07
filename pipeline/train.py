'''
This script performs:

1) Loading preprocessed parquet splits (train/val)
2) Applying class weights to mitigate imbalance
3) Building a Spark ML pipeline: Imputer > Assembler > Scaler > LogisticRegression
4) Cross-validation for hyperparameter tuning
5) Validation performance evaluation (AUC-ROC / AUC-PR)
6) Threshold search for operational optimization (F1 / recall / precision)
7) Export of model, metrics, and feature coefficients

'''
import argparse, csv, json
from pathlib import Path
from typing import List, Dict, Any

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, Imputer, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.functions import vector_to_array


def get_spark(app_name: str = "train-online-retail") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def read_split(spark: SparkSession, base: Path, name: str) -> DataFrame:
    return spark.read.parquet(str(base / name))


def feature_columns(df: DataFrame) -> List[str]:
    exclude = {"CustomerID", "label"} # to avoid data leaking
    return [c for c in df.columns if c not in exclude]


def add_class_weight(df: DataFrame) -> DataFrame:
    '''
    Add a 'class_weight' column to handle label imbalance.

    Weighting scheme:
        - positive class (label=1): w_pos = N_neg / N_pos
        - negative class (label=0): w_neg = 1.0

    '''
    cls = df.groupBy("label").count().collect()
    counts = {row["label"]: row["count"] for row in cls}
    pos = float(counts.get(1.0, 1.0)); neg = float(counts.get(0.0, 1.0))
    w_pos = neg / max(pos, 1.0)
    w_neg = 1.0
    
    return df.withColumn("class_weight", F.when(F.col("label") == 1.0, w_pos).otherwise(w_neg))


def find_best_threshold(pred_df: DataFrame, metric: str = "f1") -> Dict[str, float]:
    '''
    Grid-search decision threshold for operational optimization.

    Steps:
        1. Extract P(y=1) from the 'probability' vector.
        2. For thresholds in [0.00, 1.00] (0.01 step), compute precision, recall, F1.
        3. Select the threshold that maximizes the chosen metric (F1, recall, or precision).

    '''
    base = pred_df.withColumn("p1", vector_to_array(F.col("probability")).getItem(1)).cache()
    _ = base.count()
    
    steps = [round(x / 100, 2) for x in range(0, 101)]

    best = {
        "threshold" : 0.5,
        "value"     : -1.0,
        "precision" : 0.0,
        "recall"    : 0.0,
        "f1"        : 0.0
    }

    for t in steps:
        pd = base.withColumn("pred", (F.col("p1") >= F.lit(t)).cast("double"))

        tp = pd.filter((F.col("label") == 1.0) & (F.col("pred") == 1.0)).count()
        fp = pd.filter((F.col("label") == 0.0) & (F.col("pred") == 1.0)).count()
        fn = pd.filter((F.col("label") == 1.0) & (F.col("pred") == 0.0)).count()
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        val = f1 if metric == "f1" else (recall if metric == "recall" else precision)
        if val > best["value"]:
            best = {
                "threshold" : float(t),
                "value"     : float(val),
                "precision" : float(precision),
                "recall"    : float(recall),
                "f1"        : float(f1)
            }

    return best


def save_metrics(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def export_coefficients(model: PipelineModel, feat_names: List[str], out_csv: Path) -> None:
    lr_model = model.stages[-1]
    coefs = lr_model.coefficients.toArray().tolist()

    rows = [{"feature": f, "coefficient": c, "abs_coeff": abs(c)}
            for f, c in zip(feat_names, coefs)]
    rows.sort(key=lambda x: x["abs_coeff"], reverse=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature","coefficient","abs_coeff"])
        writer.writeheader(); writer.writerows(rows)


def main():
    '''
    Workflow:
        1. Parse CLI args (input/output dirs, metrics, CV settings)
        2. Load train/val splits
        3. Apply class weights
        4. Define ML pipeline
        5. Perform cross-validation with hyperparameter grid
        6. Evaluate on validation data (AUC-ROC / AUC-PR)
        7. Search best threshold
        8. Export model, metrics, coefficients
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cv_parallelism", type=int, default=4)
    ap.add_argument("--metric", choices=["aucroc","aucpr"], default="aucpr")
    ap.add_argument("--thr_metric", choices=["f1","recall","precision"], default="f1")
    args = ap.parse_args()

    spark = get_spark()
    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    train = read_split(spark, in_dir, "train").cache(); _=train.count()
    val = read_split(spark, in_dir, "val").cache(); _=val.count()

    feats = feature_columns(train)
    train_w = add_class_weight(train)
    val_w = add_class_weight(val)

    imputer = Imputer(inputCols=feats, outputCols=feats)
    assembler = VectorAssembler(inputCols=feats, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="features_scaled", withMean=True, withStd=True)
    lr = LogisticRegression(featuresCol="features_scaled", labelCol="label", weightCol="class_weight", maxIter=80)
    pipeline = Pipeline(stages=[imputer, assembler, scaler, lr])

    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam,[0.0, 0.01, 0.1])
        .addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])
        .build()
    )

    metric_name = "areaUnderROC" if args.metric == "aucroc" else "areaUnderPR"
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName=metric_name)
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator,
                        numFolds=3, parallelism=args.cv_parallelism, seed=42)

    cv_model = cv.fit(train_w)
    best_model = cv_model.bestModel

    out_dir.mkdir(parents=True, exist_ok=True)
    best_model.write().overwrite().save(str(out_dir))

    pred_val = best_model.transform(val_w).cache()
    _ = pred_val.count()
    
    auc_roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                            metricName="areaUnderROC").evaluate(pred_val)
    auc_pr  = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                            metricName="areaUnderPR").evaluate(pred_val)

    best_thr = find_best_threshold(pred_val, metric=args.thr_metric)
    coef_csv = out_dir / "coefficients.csv"
    export_coefficients(best_model, feats, coef_csv)

    metrics = {"cv_metric": args.metric, "val_auc_roc": auc_roc, "val_auc_pr": auc_pr,
               "best_threshold": best_thr, "model_dir": str(out_dir), "coefficients_csv": str(coef_csv)}
    save_metrics(out_dir / "metrics.json", metrics)

    print(f"[TRAIN] Saved best model to: {out_dir}")
    print(f"[TRAIN] Validation AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}")
    print(f"[TRAIN] Best threshold ({args.thr_metric}) → {best_thr}")
    print(f"[TRAIN] Coefficients exported to: {coef_csv}")
    print(f"[TRAIN] Metrics saved to: {out_dir/'metrics.json'}")

if __name__ == "__main__":
    main()
