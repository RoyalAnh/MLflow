from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

def get_cassandra_session(keyspace="marl_data", host="localhost", port=9042):
    cluster = Cluster([host], port=port)
    session = cluster.connect()
    session.execute(f"""
        CREATE KEYSPACE IF NOT EXISTS {keyspace}
        WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}};
    """)
    session.set_keyspace(keyspace)

    session.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id UUID PRIMARY KEY,
            agent_id TEXT,
            timestamp TIMESTAMP,
            state TEXT,
            action TEXT,
            reward FLOAT
        );
    """)

    return session
