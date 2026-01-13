#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
spark_stream_predict_mllib.py

Kafka (topic_in) -> Spark Structured Streaming -> MLlib PipelineModel -> Kafka (topic_out)
+ (optionnel) agrégations FP/FN/TP/TN par fenêtre -> Kafka (topic_metrics)

Attendu en entrée (JSON Kafka value):
- f0..f114 (float)
- ts (float) optionnel
- label (int) optionnel (0=benign, 1=attack)
- attack_name (str) optionnel

Sortie (topic_out):
- processing_ts, event_id, topic, kafka_ts, partition, offset, raw_input
- model_version, predicted_attack, confidence, p_benign, anomaly_score, suspect, rule_alert, severity
- label, attack_name, true_class, is_attack_true, is_attack_pred, fp, fn, tp, tn

Sortie metrics (topic_metrics):
- window_start, window_end, fp, fn, tp, tn, total, fp_rate, fn_rate, attack_true_rate, attack_pred_rate
"""

import argparse
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, current_timestamp, lit, struct, to_json,
    array, element_at, when, expr, concat_ws, coalesce,
    window, sum as Fsum, count as Fcount
)
from pyspark.sql.types import (
    StructType, StructField, DoubleType, IntegerType, StringType
)


def build_schema(n_features: int = 115) -> StructType:
    fields = [StructField(f"f{i}", DoubleType(), True) for i in range(n_features)]
    fields += [
        StructField("label", IntegerType(), True),        # 0/1
        StructField("attack_name", StringType(), True),   # "mirai", "fuzzing", ...
        StructField("ts", DoubleType(), True),            # epoch float (optionnel)
    ]
    return StructType(fields)


def main():
    parser = argparse.ArgumentParser()

    # Kafka
    parser.add_argument("--bootstrap_servers", default="kafka:29092")
    parser.add_argument("--topic_in", default="kitsune")
    parser.add_argument("--topic_out", default="kitsune_enriched")

    # Model + checkpoints
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)

    # Offsets
    parser.add_argument("--starting_offsets", default="latest", choices=["latest", "earliest"])
    parser.add_argument("--max_offsets_per_trigger", type=int, default=20000)

    # Thresholds (SOC rules)
    parser.add_argument("--anomaly_threshold", type=float, default=0.70,
                        help="suspect if anomaly_score = 1 - P(benign) >= this")
    parser.add_argument("--confidence_threshold", type=float, default=0.70,
                        help="high confidence if max_prob >= this")

    parser.add_argument("--model_version", default="kitsune_rf_v1")

    # FP/FN logic
    parser.add_argument("--pred_logic", default="predicted_attack",
                        choices=["predicted_attack", "suspect"],
                        help="Define is_attack_pred from predicted_attack!=benign OR suspect==true")

    # Metrics stream (optional)
    parser.add_argument("--enable_metrics", action="store_true",
                        help="If set, compute FP/FN/TP/TN aggregated per window and publish to topic_metrics")
    parser.add_argument("--topic_metrics", default="kitsune_metrics")
    parser.add_argument("--metrics_window", default="1 minute")
    parser.add_argument("--watermark", default="5 minutes")

    args = parser.parse_args()

    spark = SparkSession.builder.appName("kitsune_stream_predict_rf_mllib").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    n_features = 115
    feature_cols = [f"f{i}" for i in range(n_features)]
    json_schema = build_schema(n_features)

    # Load model + labels
    from pyspark.ml import PipelineModel
    model = PipelineModel.load(f"{args.model_dir}/pipeline")

    with open(f"{args.model_dir}/labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    if "benign" not in labels:
        raise RuntimeError(f"'benign' not found in labels.json: {labels}")

    benign_idx = labels.index("benign")
    labels_arr = array([lit(x) for x in labels])  # mapping prediction -> class string

    # Kafka source
    df = (spark.readStream
          .format("kafka")
          .option("kafka.bootstrap.servers", args.bootstrap_servers)
          .option("subscribe", args.topic_in)
          .option("startingOffsets", args.starting_offsets)
          .option("maxOffsetsPerTrigger", str(args.max_offsets_per_trigger))
          .option("failOnDataLoss", "false")
          .load())

    parsed = (
        df.select(
            col("topic"),
            col("timestamp").alias("kafka_ts"),
            col("partition"),
            col("offset"),
            col("value").cast("string").alias("json_str"),
        )
        .select(
            "topic", "kafka_ts", "partition", "offset", "json_str",
            from_json(col("json_str"), json_schema).alias("j")
        )
        .select("topic", "kafka_ts", "partition", "offset", "json_str", "j.*")
        .fillna(0.0, subset=feature_cols)
    )

    # Inference
    scored = model.transform(parsed)

    # Convert probability vector -> array
    try:
        from pyspark.ml.functions import vector_to_array
        scored = scored.withColumn("prob_arr", vector_to_array(col("probability")))
        scored = scored.withColumn("max_prob", expr("array_max(prob_arr)"))
    except Exception:
        # Fallback: keep probability as-is; compute max_prob as best effort
        scored = scored.withColumn("prob_arr", col("probability"))
        scored = scored.withColumn("max_prob", lit(None).cast("double"))

    # Predicted class string
    scored = scored.withColumn(
        "predicted_attack",
        element_at(labels_arr, col("prediction").cast("int") + lit(1))  # element_at is 1-based
    )

    # Option A anomaly score = 1 - P(benign)
    scored = scored.withColumn("p_benign", col("prob_arr")[lit(benign_idx)])
    scored = scored.withColumn("anomaly_score", lit(1.0) - col("p_benign"))
    scored = scored.withColumn("suspect", col("anomaly_score") >= lit(args.anomaly_threshold))

    # Rule layer (simple SOC-style)
    scored = scored.withColumn(
        "rule_alert",
        when(col("suspect") & (col("max_prob") >= lit(args.confidence_threshold)), lit("ALERT_HIGH"))
        .when(col("suspect") & (col("max_prob") < lit(args.confidence_threshold)), lit("ALERT_LOW"))
        .otherwise(lit("BENIGN"))
    )

    scored = scored.withColumn(
        "severity",
        when(col("rule_alert") == lit("ALERT_HIGH"), lit("high"))
        .when(col("rule_alert") == lit("ALERT_LOW"), lit("medium"))
        .otherwise(lit("none"))
    )

    # Stable event_id for dedup/debug
    scored = scored.withColumn(
        "event_id",
        concat_ws(":", col("topic"), col("partition").cast("string"), col("offset").cast("string"))
    )

    scored = scored.withColumn("raw_input", col("json_str"))

    # --- Ground truth + confusion matrix (if label exists) ---
    # true_class: if label=0 => benign else attack_name (fallback "attack")
    scored = scored.withColumn(
        "true_class",
        when(col("label") == lit(0), lit("benign"))
        .otherwise(coalesce(col("attack_name"), lit("attack")))
    )

    scored = scored.withColumn("is_attack_true", (col("label") == lit(1)))

    if args.pred_logic == "suspect":
        scored = scored.withColumn("is_attack_pred", (col("suspect") == lit(True)))
    else:
        scored = scored.withColumn("is_attack_pred", (col("predicted_attack") != lit("benign")))

    scored = scored.withColumn("fp", (col("is_attack_pred") & ~col("is_attack_true")).cast("int"))
    scored = scored.withColumn("fn", (~col("is_attack_pred") & col("is_attack_true")).cast("int"))
    scored = scored.withColumn("tp", (col("is_attack_pred") & col("is_attack_true")).cast("int"))
    scored = scored.withColumn("tn", (~col("is_attack_pred") & ~col("is_attack_true")).cast("int"))

    # Output (events)
    out = scored.select(
        current_timestamp().alias("processing_ts"),
        col("event_id"),
        col("topic"),
        col("kafka_ts"),
        col("partition"),
        col("offset"),
        col("raw_input"),

        lit(args.model_version).alias("model_version"),
        col("predicted_attack"),
        col("max_prob").alias("confidence"),
        col("p_benign"),
        col("anomaly_score"),
        col("suspect"),
        col("rule_alert"),
        col("severity"),

        col("label"),
        col("attack_name"),
        col("true_class"),
        col("is_attack_true"),
        col("is_attack_pred"),
        col("fp"), col("fn"), col("tp"), col("tn"),
    )

    kafka_out = out.select(
        col("event_id").cast("string").alias("key"),
        to_json(struct(
            "processing_ts",
            "event_id",
            "topic",
            "kafka_ts",
            "partition",
            "offset",
            "raw_input",
            "model_version",
            "predicted_attack",
            "confidence",
            "p_benign",
            "anomaly_score",
            "suspect",
            "rule_alert",
            "severity",
            "label",
            "attack_name",
            "true_class",
            "is_attack_true",
            "is_attack_pred",
            "fp", "fn", "tp", "tn"
        )).alias("value")
    )

    q_events = (kafka_out.writeStream
                .format("kafka")
                .option("kafka.bootstrap.servers", args.bootstrap_servers)
                .option("topic", args.topic_out)
                .option("checkpointLocation", args.checkpoint_dir.rstrip("/") + "/events")
                .outputMode("append")
                .start())

    # Output (metrics) optional
    if args.enable_metrics:
        # watermark on kafka_ts for window aggregation stability
        metrics_base = (out
                        .withWatermark("kafka_ts", args.watermark)
                        .groupBy(window(col("kafka_ts"), args.metrics_window))
                        .agg(
                            Fsum(col("fp")).alias("fp"),
                            Fsum(col("fn")).alias("fn"),
                            Fsum(col("tp")).alias("tp"),
                            Fsum(col("tn")).alias("tn"),
                            Fcount(lit(1)).alias("total"),
                            Fsum(col("is_attack_true").cast("int")).alias("attack_true_cnt"),
                            Fsum(col("is_attack_pred").cast("int")).alias("attack_pred_cnt"),
                        )
                        )

        metrics = (metrics_base
                   .withColumn("window_start", col("window.start"))
                   .withColumn("window_end", col("window.end"))
                   .drop("window")
                   .withColumn("fp_rate", (col("fp") / col("total")))
                   .withColumn("fn_rate", (col("fn") / col("total")))
                   .withColumn("attack_true_rate", (col("attack_true_cnt") / col("total")))
                   .withColumn("attack_pred_rate", (col("attack_pred_cnt") / col("total")))
                   )

        metrics_kafka = metrics.select(
            to_json(struct(
                "window_start", "window_end",
                "fp", "fn", "tp", "tn",
                "total",
                "fp_rate", "fn_rate",
                "attack_true_rate", "attack_pred_rate"
            )).alias("value")
        )

        q_metrics = (metrics_kafka.writeStream
                     .format("kafka")
                     .option("kafka.bootstrap.servers", args.bootstrap_servers)
                     .option("topic", args.topic_metrics)
                     .option("checkpointLocation", args.checkpoint_dir.rstrip("/") + "/metrics")
                     .outputMode("append")
                     .start())

        spark.streams.awaitAnyTermination()
    else:
        q_events.awaitTermination()


if __name__ == "__main__":
    main()
