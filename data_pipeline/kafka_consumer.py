from kafka import KafkaConsumer
import json
from uuid import UUID
from datetime import datetime
from cassandra_utils import get_cassandra_session

# For logging artifacts
import mlflow

TOPIC = "agent-observations"
session = get_cassandra_session()

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id="marl-consumer-group"
)

print("Listening for messages...")

# Log to file for artifact
log_file = "kafka_to_cassandra_log.jsonl"
with open(log_file, "w") as fout:
    for msg in consumer:
        data = msg.value
        print(f"Received: {data}")

        fout.write(json.dumps(data) + "\n")
        fout.flush()

        session.execute("""
            INSERT INTO observations (id, agent_id, timestamp, state, action, reward)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            UUID(data["id"]),
            data["agent_id"],
            datetime.fromisoformat(data["timestamp"]),
            data["state"],
            data["action"],
            float(data["reward"])
        ))

# In thực tế bạn nên upload file này lên MLflow ở phần train.py để tracking, hoặc thêm một bước upload ở đây nếu MLflow server luôn chạy.
