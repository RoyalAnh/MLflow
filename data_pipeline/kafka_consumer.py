from kafka import KafkaConsumer
import json
from uuid import UUID
from datetime import datetime
from cassandra_utils import get_cassandra_session

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

for msg in consumer:
    data = msg.value
    print(f"Received: {data}")

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
