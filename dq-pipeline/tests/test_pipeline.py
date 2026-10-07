import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.generate_data import make_transactions
from src.contracts import validate
from src.feature_store import FeatureStore
from src.pipeline import ingest, train, data_hash

DF, LAB = make_transactions(n_customers=300, n_rows=6000, seed=1)
MAXTS = pd.Timestamp("2026-06-30")

class TestContracts(unittest.TestCase):
    def test_clean_passes(self): self.assertEqual(validate(DF, DF["amount"], MAXTS), [])
    def test_missing_column(self): self.assertIn("schema_missing_column", validate(DF.drop(columns=["amount"])))
    def test_type_change(self):
        d = DF.copy(); d["amount"] = d["amount"].astype(str)
        self.assertIn("schema_type:amount", validate(d))
    def test_nulls(self):
        d = DF.copy(); d.loc[0, "amount"] = np.nan
        self.assertIn("nulls", validate(d))
    def test_negative(self):
        d = DF.copy(); d.loc[0, "amount"] = -5
        self.assertIn("range_amount", validate(d))
    def test_duplicate(self):
        self.assertIn("duplicate_key", validate(pd.concat([DF, DF.head(5)])))
    def test_unknown_category(self):
        d = DF.copy(); d.loc[0, "category"] = "crypto"
        self.assertIn("category:category", validate(d))
    def test_future_ts(self):
        d = DF.copy(); d.loc[0, "ts"] = pd.Timestamp("2030-01-01")
        self.assertIn("future_timestamp", validate(d, max_ts=MAXTS))
    def test_drift(self):
        d = DF.copy(); d["amount"] = d["amount"] * 2
        self.assertIn("drift_amount", validate(d, DF["amount"]))

class TestStoreAndPipeline(unittest.TestCase):
    def test_upsert_idempotent(self):
        s = FeatureStore(); s.upsert(DF); n = s.count(); s.upsert(DF)
        self.assertEqual(s.count(), n)
    def test_sql_features_match_pandas(self):
        s = FeatureStore(); s.upsert(DF); f = s.features().set_index("customer_id")
        exp = DF.groupby("customer_id")["amount"].mean()
        self.assertTrue(np.allclose(f["avg_amount"].sort_index(), exp.sort_index()))
    def test_bad_batch_rejected_good_accepted(self):
        bad = DF.copy(); bad.loc[0, "amount"] = -1
        s = FeatureStore(); a, r = ingest(s, [DF.iloc[:3000], bad.iloc[:3000]], DF["amount"], MAXTS)
        self.assertEqual((a, len(r)), (1, 1))
    def test_hash_deterministic(self): self.assertEqual(data_hash(DF), data_hash(DF.copy()))
    def test_train_writes_versioned_artifact(self):
        s = FeatureStore(); s.upsert(DF)
        with tempfile.TemporaryDirectory() as d:
            ver, meta = train(s, LAB, d)
            self.assertTrue(os.path.exists(os.path.join(d, ver, "model.joblib")))
            self.assertGreater(meta["auc"], 0.5)

if __name__ == "__main__": unittest.main()
