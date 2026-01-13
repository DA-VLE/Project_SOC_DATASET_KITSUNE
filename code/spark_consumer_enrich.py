from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType

import os

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
TOPIC_IN = os.getenv("TOPIC_IN", "kitsune")
TOPIC_OUT = os.getenv("TOPIC_OUT", "kitsune_enriched")

ART_DIR = os.getenv("ART_DIR", "/opt/project/artifacts/v1")
SCALER_PATH = os.path.join(ART_DIR, "scaler.joblib")
ISO_PATH    = os.path.join(ART_DIR, "iso_forest.joblib")
LE_PATH     = os.path.join(ART_DIR, "label_encoder.joblib")
XGB_PATH    = os.path.join(ART_DIR, "attack_classifier_xgb.joblib")
LR_PATH     = os.path.join(ART_DIR, "attack_classifier_lr.joblib")

CHECKPOINT = os.getenv("CHECKPOINT", "/tmp/spark_checkpoints/kitsune_enriched")

spark = (
    SparkSession.builder
    .appName("SparkKafkaConsumerEnrich")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# -------- Schema JSON entrant: f0..f114
schema = StructType([StructField(f"f{i}", DoubleType(), True) for i in range(115)])

# -------- Read Kafka
df_raw = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", TOPIC_IN)
    .option("startingOffsets", "latest")  # "earliest" si tu veux tout relire
    .load()
)

df_json = df_raw.select(
    F.col("timestamp").alias("kafka_ts"),
    F.col("topic").alias("kafka_topic"),
    F.col("partition").alias("kafka_partition"),
    F.col("offset").alias("kafka_offset"),
    F.col("key").cast("string").alias("kafka_key"),
    F.col("value").cast("string").alias("json_str"),
)

df_parsed = (
    df_json
    .select(
        "kafka_ts","kafka_topic","kafka_partition","kafka_offset","kafka_key",
        F.from_json(F.col("json_str"), schema).alias("data")
    )
    .select("kafka_ts","kafka_topic","kafka_partition","kafka_offset","kafka_key","data.*")
)

# -------- Build features array column
feature_cols = [F.col(f"f{i}") for i in range(115)]
df_feat = df_parsed.withColumn("features", F.array(*feature_cols))

# -------- Simple rule-based enrich (à ajuster)
df_feat = df_feat.withColumn(
    "rule_alert",
    F.expr("CASE WHEN f1 > 1400 OR f2 > 100000 THEN 1 ELSE 0 END")
)

# ============================================================
# Pandas UDFs: scaler + iso + classifiers (LR/XGB)
# ============================================================
from pyspark.sql.functions import pandas_udf
import pandas as pd
import numpy as np
import joblib

# Lazy globals (chargés une seule fois par worker Python)
_G = {"loaded": False, "scaler": None, "iso": None, "lr": None, "xgb": None, "le": None}

def _load_models_once():
    if _G["loaded"]:
        return
    _G["scaler"] = joblib.load(SCALER_PATH)
    _G["iso"] = joblib.load(ISO_PATH)

    if os.path.exists(LR_PATH):
        _G["lr"] = joblib.load(LR_PATH)
    if os.path.exists(XGB_PATH):
        _G["xgb"] = joblib.load(XGB_PATH)
    if os.path.exists(LE_PATH):
        _G["le"] = joblib.load(LE_PATH)

    _G["loaded"] = True

def _to_numpy(features_series: pd.Series) -> np.ndarray:
    # features_series contient des listes/arrays de longueur 115
    X = np.array(features_series.tolist(), dtype=np.float32)
    # safety: NaN/inf -> 0
    X[~np.isfinite(X)] = 0.0
    return X

def _decode_pred(pred):
    # si le modèle renvoie déjà des strings => ok
    if pred is None:
        return None
    if isinstance(pred, (list, tuple, np.ndarray)):
        p0 = pred[0]
    else:
        p0 = pred
    return pred

@pandas_udf("double")
def iso_score_udf(features: pd.Series) -> pd.Series:
    _load_models_once()
    X = _to_numpy(features)
    Xs = _G["scaler"].transform(X)
    # plus grand = plus "normal"
    s = _G["iso"].decision_function(Xs).astype(np.float64)
    return pd.Series(s)

@pandas_udf("int")
def iso_is_anom_udf(features: pd.Series) -> pd.Series:
    _load_models_once()
    X = _to_numpy(features)
    Xs = _G["scaler"].transform(X)
    p = _G["iso"].predict(Xs)  # -1 anomaly, +1 normal
    is_anom = (p == -1).astype(np.int32)
    return pd.Series(is_anom)

@pandas_udf("string")
def pred_lr_udf(features: pd.Series) -> pd.Series:
    _load_models_once()
    if _G["lr"] is None:
        return pd.Series([None] * len(features), dtype="object")

    X = _to_numpy(features)
    Xs = _G["scaler"].transform(X)
    pred = _G["lr"].predict(Xs)

    # si tu utilises un LabelEncoder (classes numériques)
    if _G["le"] is not None and np.issubdtype(np.array(pred).dtype, np.integer):
        pred = _G["le"].inverse_transform(pred)

    return pd.Series(pred.astype(str))

@pandas_udf("string")
def pred_xgb_udf(features: pd.Series) -> pd.Series:
    _load_models_once()
    if _G["xgb"] is None:
        return pd.Series([None] * len(features), dtype="object")

    X = _to_numpy(features)
    Xs = _G["scaler"].transform(X)
    pred = _G["xgb"].predict(Xs)

    if _G["le"] is not None and np.issubdtype(np.array(pred).dtype, np.integer):
        pred = _G["le"].inverse_transform(pred)

    return pd.Series(pred.astype(str))

# -------- Apply UDFs
df_scored = (
    df_feat
    .withColumn("iso_score", iso_score_udf(F.col("features")))
    .withColumn("iso_is_anom", iso_is_anom_udf(F.col("features")))
    .withColumn("pred_lr", pred_lr_udf(F.col("features")))
    .withColumn("pred_xgb", pred_xgb_udf(F.col("features")))
)

# -------- Choix final (logique simple)
# - si pas d'anom et pas de règle => benign
# - sinon: priorité XGB si dispo, sinon LR, sinon "attack"
df_scored = df_scored.withColumn(
    "pred_final",
    F.when((F.col("iso_is_anom") == 0) & (F.col("rule_alert") == 0), F.lit("benign"))
     .otherwise(
         F.when(F.col("pred_xgb").isNotNull(), F.col("pred_xgb"))
          .when(F.col("pred_lr").isNotNull(), F.col("pred_lr"))
          .otherwise(F.lit("attack"))
     )
)

# -------- Output JSON vers Kafka
out_cols = [
    "kafka_ts","kafka_topic","kafka_partition","kafka_offset","kafka_key",
    "rule_alert","iso_score","iso_is_anom","pred_lr","pred_xgb","pred_final"
]

df_out = df_scored.select(F.to_json(F.struct(*[F.col(c) for c in out_cols])).alias("value"))

query = (
    df_out.writeStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("topic", TOPIC_OUT)
    .option("checkpointLocation", CHECKPOINT)
    .outputMode("append")
    .start()
)

query.awaitTermination()