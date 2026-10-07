"""SQLite-backed feature store: idempotent upserts + SQL aggregations."""
import sqlite3, pandas as pd

DDL = """CREATE TABLE IF NOT EXISTS txns(txn_id INTEGER PRIMARY KEY, customer_id INTEGER, ts TEXT,
         amount REAL, category TEXT, channel TEXT);
         CREATE INDEX IF NOT EXISTS ix_cust ON txns(customer_id);"""

FEATURE_SQL = """
SELECT customer_id,
       COUNT(*) AS txn_count,
       AVG(amount) AS avg_amount,
       MAX(amount) AS max_amount,
       COUNT(DISTINCT category) AS n_categories,
       AVG(CASE WHEN CAST(strftime('%H', ts) AS INT) < 5 THEN 1.0 ELSE 0.0 END) AS night_ratio,
       AVG(CASE WHEN channel = 'web' THEN 1.0 ELSE 0.0 END) AS web_ratio
FROM txns GROUP BY customer_id"""

class FeatureStore:
    def __init__(self, path=":memory:"):
        self.con = sqlite3.connect(path); self.con.executescript(DDL)
    def upsert(self, df):
        d = df.copy(); d["ts"] = d["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
        rows = list(d[["txn_id","customer_id","ts","amount","category","channel"]].itertuples(index=False, name=None))
        with self.con:
            self.con.executemany("INSERT OR REPLACE INTO txns VALUES (?,?,?,?,?,?)", rows)
    def count(self): return self.con.execute("SELECT COUNT(*) FROM txns").fetchone()[0]
    def features(self): return pd.read_sql_query(FEATURE_SQL, self.con)
