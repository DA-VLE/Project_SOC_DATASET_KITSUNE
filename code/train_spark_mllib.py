import argparse
import json
import os
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def build_schema(n_features: int = 115) -> StructType:
    fields = [StructField(f"f{i}", DoubleType(), True) for i in range(n_features)]
    fields += [
        StructField("label", IntegerType(), True),         # 0/1
        StructField("attack_name", StringType(), True),    # Mirai / Fuzzing / ...
    ]
    return StructType(fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--limit_rows", type=int, default=0)
    parser.add_argument("--num_trees", type=int, default=80)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("kitsune_train_rf_mllib")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )

    n_features = 115
    feature_cols = [f"f{i}" for i in range(n_features)]
    schema = build_schema(n_features)

    df = (
        spark.read
        .option("header", "true")
        .schema(schema)
        .csv(args.input_csv)
    )

    # Basic cleaning
    df = df.fillna(0.0, subset=feature_cols)
    df = df.fillna("unknown", subset=["attack_name"])

    # IMPORTANT: build supervised class name with benign as explicit class
    df = df.withColumn(
        "class_name",
        when(col("label") == lit(0), lit("benign")).otherwise(col("attack_name"))
    )

    if args.sample_frac and args.sample_frac < 1.0:
        df = df.sample(withReplacement=False, fraction=args.sample_frac, seed=args.seed)

    if args.limit_rows and args.limit_rows > 0:
        df = df.limit(args.limit_rows)

    # Index labels (benign + attack types)
    label_indexer = StringIndexer(
        inputCol="class_name",
        outputCol="label_idx",
        handleInvalid="keep"
    ).fit(df)

    labels = label_indexer.labels
    if "benign" not in labels:
        raise RuntimeError(
            f"'benign' not found in labels: {labels}. "
            f"Check that your global_dataset.csv has label==0 rows."
        )

    df2 = label_indexer.transform(df)
    train_df, test_df = df2.randomSplit([0.8, 0.2], seed=args.seed)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")

    # RF doesn't require scaling, but keeping scaler is fine and CV-friendly
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label_idx",
        numTrees=args.num_trees,
        maxDepth=args.max_depth,
        seed=args.seed
    )

    pipeline = Pipeline(stages=[assembler, scaler, rf])
    model = pipeline.fit(train_df)

    pred = model.transform(test_df)

    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label_idx", predictionCol="prediction", metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label_idx", predictionCol="prediction", metricName="f1"
    )
    acc = acc_eval.evaluate(pred)
    f1 = f1_eval.evaluate(pred)

    os.makedirs(args.model_dir, exist_ok=True)
    pipeline_out = os.path.join(args.model_dir, "pipeline")
    model.write().overwrite().save(pipeline_out)

    with open(os.path.join(args.model_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, ensure_ascii=False, indent=2)

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input_csv": args.input_csv,
        "sample_frac": args.sample_frac,
        "limit_rows": args.limit_rows,
        "num_trees": args.num_trees,
        "max_depth": args.max_depth,
        "seed": args.seed,
        "metrics": {"accuracy": acc, "f1": f1},
        "n_features": n_features,
        "spark_version": spark.version
    }
    with open(os.path.join(args.model_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n=== TRAIN DONE ===")
    print(f"Model saved to: {pipeline_out}")
    print(f"Labels: {labels}")
    print(f"Metrics: accuracy={acc:.4f}, f1={f1:.4f}\n")

    spark.stop()


if __name__ == "__main__":
    main()
