import socket, time, os

host = os.getenv("KAFKA_HOST", "kafka")
port = int(os.getenv("KAFKA_PORT", "29092"))

for i in range(60):
    try:
        s = socket.create_connection((host, port), 2)
        s.close()
        print("Kafka OK")
        break
    except Exception as e:
        print(f"waiting for kafka... ({i+1}/60) {e}")
        time.sleep(1)
else:
    raise SystemExit("Kafka not reachable")
