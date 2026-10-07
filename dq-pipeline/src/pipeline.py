"""Gate -> upsert -> SQL features -> train -> versioned artifact with lineage."""
import hashlib, json, os, time, joblib, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from .contracts import validate
from .feature_store import FeatureStore

def data_hash(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]

def ingest(store, batches, reference_amount, max_ts):
    accepted, rejected = 0, []
    for i, b in enumerate(batches):
        viol = validate(b, reference_amount, max_ts)
        if viol: rejected.append((i, viol))
        else: store.upsert(b); accepted += 1
    return accepted, rejected

def train(store, labels, out_dir, seed=7):
    feats = store.features().merge(labels, on="customer_id")
    X, y = feats.drop(columns=["customer_id", "high_risk"]), feats["high_risk"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    m = GradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    meta = {"data_hash": data_hash(feats), "n_rows": int(len(feats)), "auc": round(float(auc), 4),
            "features": list(X.columns), "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    ver = f"v_{meta['data_hash']}"
    os.makedirs(os.path.join(out_dir, ver), exist_ok=True)
    joblib.dump(m, os.path.join(out_dir, ver, "model.joblib"))
    with open(os.path.join(out_dir, ver, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return ver, meta
