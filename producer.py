import pandas as pd
import json
import time
from kafka import KafkaProducer

TOPIC_NAME = "houses"
BOOTSTRAP_SERVERS = "localhost:9092"

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

df = pd.read_csv("data/house_data.csv")

# Model ile aynı 4 feature
feature_cols = [
    "bathrooms",
    "floors",
    "lat",
    "long"
]

df = df[feature_cols].dropna()

def main():
    for _, row in df.iterrows():
        record = row.to_dict()
        producer.send(TOPIC_NAME, record)
        print("Sent:", record)
        time.sleep(0.5)

    producer.flush()
    print("All messages sent.")

if __name__ == "__main__":
    main()