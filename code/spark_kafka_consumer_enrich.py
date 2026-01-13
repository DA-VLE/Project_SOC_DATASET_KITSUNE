from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, when, lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

KAFKA_BOOTSTRAP = "kafka:9092"
TOPIC = "kitsune"

spark = (
    SparkSession.builder
    .appName("SparkKafkaConsumerEnrich")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Schéma du message JSON envoyé par le producer (ex: {"f0":..., "f1":..., ...})
schema = StructType([
    StructField("f0", StringType(), True),
    StructField("f1", StringType(), True),
    StructField("f2", StringType(), True),
    StructField("raw", ArrayType(StringType()), True),
])
parsed = df.selectExpr("CAST(value AS STRING) AS json") \
    .select(from_json(col("json"), schema).alias("d")) \
    .select("d.*") \
    .withColumn("f0", col("f0").cast("double")) \
    .withColumn("f1", col("f1").cast("double")) \
    .withColumn("f2", col("f2").cast("double"))

df_kafka = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "kafka:29092")
    .option("subscribe", "kitsune")
    .option("startingOffsets", "earliest")

    .load()
)

df_json = (
    df_kafka.select(from_json(col("value").cast("string"), schema).alias("data"))
    .select("data.*")
)

# Enrichissement SOC simple : rule_alert = 1 si f1 > 1500
df_enriched = df_json.withColumn(
    "rule_alert",
    when(col("f1") > lit(1500.0), lit(1)).otherwise(lit(0))
)

query = (
    df_enriched.writeStream
    .outputMode("append")
    .format("console")
    .option("truncate", "false")
    .start()
)

query.awaitTermination()
PY
