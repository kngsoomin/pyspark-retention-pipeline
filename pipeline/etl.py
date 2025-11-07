import argparse
from pathlib import Path
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F


def get_spark(app_name: str = "etl-online-retail") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def read_csv_any(spark: SparkSession, raw_dir: Path) -> DataFrame:

    df = (spark.read.option("header", True).csv(str(raw_dir / "*.csv")))
    
    # Column Standardization: 'Customer ID' -> 'CustomerID', 'Price' -> 'UnitPrice'
    df = df.withColumnRenamed("Customer ID", "CustomerID")
    df = df.withColumnRenamed("Price", "UnitPrice")
    df = df.withColumnRenamed("InvoiceNo", "Invoice")

    # Typecast
    df = (
        df
        .withColumn("InvoiceDate", F.to_timestamp("InvoiceDate"))
        .withColumn("Quantity", F.col("Quantity").cast("int"))
        .withColumn("UnitPrice", F.col("UnitPrice").cast("double"))
        .withColumn("CustomerID", F.col("CustomerID").cast("string"))
    )
    return df


def build_features_and_labels(df_lines: DataFrame, cutoff: datetime, lookahead_days: int) -> DataFrame:
    # valid row: notnull custID, qty/price > 0 (exclude return/outlier)
    lines = df_lines.filter(
        (F.col("CustomerID").isNotNull()) &
        (F.col("Quantity") > 0) &
        (F.col("UnitPrice") > 0)
    ).withColumn("Amount", F.col("Quantity") * F.col("UnitPrice"))

    # Aggregate by order-level
    orders = (
        lines.groupBy("Invoice")
        .agg(
            F.max("InvoiceDate").alias("OrderDate"),
            F.first("CustomerID").alias("CustomerID"),
            F.sum("Amount").alias("OrderAmount"),
            F.countDistinct("StockCode").alias("OrderDistinctProducts"),
            F.sum("Quantity").alias("OrderQty")
        )
    )

    T = cutoff
    T1 = cutoff + timedelta(days=lookahead_days)
    T_lit, T1_lit = F.lit(T), F.lit(T1)

    # Split data by cutoff date - hist/lookahead
    hist_orders = orders.filter(F.col("OrderDate") <= T_lit)
    look_orders = (
        orders.filter((F.col("OrderDate") > T_lit) & (F.col("OrderDate") <= T1_lit))
        .select("CustomerID").distinct()
        .withColumn("has_future_purchase", F.lit(1.0))
    )

    # Customer history feature
    agg = (
        hist_orders.groupBy("CustomerID")
        .agg(
            F.count("*").alias("total_orders"),
            F.sum("OrderQty").alias("total_qty"),
            F.sum("OrderAmount").alias("total_amount"),
            F.avg("OrderAmount").alias("avg_order_amount"),
            F.max("OrderDate").alias("last_purchase_ts"),
        )
    )

    # customerID-level: number of unique items (hist)
    distinct_prod = (
        lines.filter(F.col("InvoiceDate") <= T_lit)
             .groupBy("CustomerID")
             .agg(F.countDistinct("StockCode").alias("distinct_products"))
    )

    feats = agg.join(distinct_prod, "CustomerID", "left")

    # recent 90days
    recent_start = T - timedelta(days=90)
    recent = (
        hist_orders.filter((F.col("OrderDate") > F.lit(recent_start)) & (F.col("OrderDate") <= T_lit))
                   .groupBy("CustomerID")
                   .agg(
                       F.count("*").alias("recent90_orders"),
                       F.sum("OrderAmount").alias("recent90_amount"),
                   )
    )
    feats = feats.join(recent, "CustomerID", "left")

    # recency 
    feats = feats.withColumn("recency_days", F.datediff(T_lit, F.col("last_purchase_ts")).cast("int"))
    
    # monetary log transforms
    feats = feats.withColumn("total_amount_log", F.log1p(F.col("total_amount")))
    feats = feats.withColumn("recent90_amount_log", F.log1p(F.col("recent90_amount")))

    # null handling
    for c in [
        "total_orders",
        "total_qty",
        "total_amount",
        "avg_order_amount",
        "distinct_products",
        "recent90_orders",
        "recent90_amount",
        "recency_days",
        "total_amount_log",
        "recent90_amount_log",
    ]:
        feats = feats.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

    # label: if has purchase during lookahead 0, otherwise 1(churn)
    feats = (feats.join(look_orders, "CustomerID", "left")
                  .withColumn("has_future_purchase", F.coalesce(F.col("has_future_purchase"), F.lit(0.0)))
                  .withColumn("label", F.when(F.col("has_future_purchase")==1.0, F.lit(0.0)).otherwise(F.lit(1.0)))
                  .drop("has_future_purchase"))

    # limit cohort: minimum 2 orders or ordered within (cutoff~180 days)
    feats = feats.filter( (F.col("total_orders") >= 2) | (F.col("recency_days") <= 180) )

    # feature selection
    selected = feats.select(
        "CustomerID",
        "total_orders",
        "total_qty",
        "avg_order_amount",
        "distinct_products",
        "recent90_orders",
        "recency_days",
        "total_amount_log",
        "recent90_amount_log",
        "label"
    )
    return selected


def split_and_save(features: DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    features.write.mode("overwrite").parquet(str(out_dir / "features"))
    train, val, test = features.randomSplit([0.7, 0.15, 0.15], seed=42)
    train.write.mode("overwrite").parquet(str(out_dir / "train"))
    val.write.mode("overwrite").parquet(str(out_dir / "val"))
    test.write.mode("overwrite").parquet(str(out_dir / "test"))


def main():
    ap = argparse.ArgumentParser(description="ETL for Online Retail II (single CSV) retention pipeline")
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cutoff", required=True, help="YYYY-MM-DD")
    ap.add_argument("--lookahead_days", type=int, default=180)
    args = ap.parse_args()

    spark = get_spark()
    raw = Path(args.raw_dir); out = Path(args.out_dir)
    cutoff = datetime.fromisoformat(args.cutoff).replace(hour=23, minute=59, second=59, microsecond=0)

    df = read_csv_any(spark, raw)
    selected = build_features_and_labels(df, cutoff, args.lookahead_days)
    split_and_save(selected, out)

    print("[ETL] Saved features & splits to", out)


if __name__ == "__main__":
    main()
