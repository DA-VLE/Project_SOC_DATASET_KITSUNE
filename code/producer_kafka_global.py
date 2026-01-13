import json
import time
import pandas as pd
from kafka import KafkaProducer

DATA = r"D:\Project_SOC_Kitsune\data\global_dataset.csv"
BOOTSTRAP = "localhost:9092"   # si Kafka local Windows
TOPIC = "kitsune"
CHUNK = 10_000
SLEEP = 0.001  # ralentir un peu (optionnel)

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    acks=1,
    linger_ms=20,
    batch_size=64 * 1024,
    compression_type="lz4",      # ou "gzip"
    retries=5,
    request_timeout_ms=30000,
)

cols = pd.read_csv(DATA, nrows=0).columns.tolist()
features = [c for c in cols if c.startswith("f")]  # f0..f114

for chunk in pd.read_csv(DATA, usecols=features, chunksize=CHUNK):
    for row in chunk.itertuples(index=False):
        event = {f"f{i}": float(row[i]) for i in range(115)}
        producer.send(TOPIC, value=event)
        if SLEEP:
            time.sleep(SLEEP)

producer.flush()
print("DONE producer")
