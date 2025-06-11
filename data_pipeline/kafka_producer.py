from kafka import KafkaProducer
import json
import time
import uuid
from datetime import datetime
import random
from datetime import datetime, timezone
import argparse

TOPIC = "agent-observations"

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",  
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

AGENT_IDS = ["agent_A", "agent_B", "agent_C"]

def generate_fake_observation():
    return {
        "id": str(uuid.uuid4()),
        "agent_id": random.choice(AGENT_IDS),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state": f"state_{random.randint(1,5)}",
        "action": f"action_{random.randint(1,4)}",
        "reward": round(random.uniform(-1, 1), 3)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="Number of observations to send", default=0)
    args = parser.parse_args()

    i = 0
    while args.n == 0 or i < args.n:
        obs = generate_fake_observation()
        producer.send(TOPIC, value=obs)
        print(f"Sent: {obs}")
        time.sleep(1)
        i += 1

# python producer.py --n 10  # Gửi 10 bản ghi rồi dừng
