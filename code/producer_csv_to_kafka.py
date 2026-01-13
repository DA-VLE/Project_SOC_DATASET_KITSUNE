import os, json, time
import pandas as pd
from kafka import KafkaProducer

BOOTSTRAP = os.getenv("BOOTSTRAP", "kafka:29092")
TOPIC = os.getenv("TOPIC", "kitsune")
CSV_PATH = os.getenv("CSV_PATH", "/opt/project/data/global_dataset.csv")

SLEEP_SEC = float(os.getenv("SLEEP_SEC", "0"))   # 0 = max speed
CHUNK = int(os.getenv("CHUNK", "5000"))          # batch read pour pas exploser la RAM

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    acks=1,
    linger_ms=10,
)

cols = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
features = [c for c in cols if c.startswith("f")]

LABEL_COL = "label" if "label" in cols else None
ATTACK_COL = "attack_name" if "attack_name" in cols else None

if len(features) != 115:
    raise ValueError(f"Expected 115 feature columns, got {len(features)}")

# IMPORTANT: lire aussi label/attack_name
usecols = features + [c for c in [LABEL_COL, ATTACK_COL] if c]

for chunk in pd.read_csv(CSV_PATH, usecols=usecols, chunksize=CHUNK):
    # numeric uniquement sur les features
    chunk[features] = chunk[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    for row in chunk.itertuples(index=False, name=None):
        msg = {features[i]: float(row[i]) for i in range(len(features))}
        idx = len(features)

        if LABEL_COL:
            msg["label"] = int(row[idx]); idx += 1
        if ATTACK_COL:
            msg["attack_name"] = str(row[idx])

        msg["ts"] = time.time()
        producer.send(TOPIC, msg)