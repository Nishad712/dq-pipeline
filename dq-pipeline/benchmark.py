"""Reproducible benchmark: python benchmark.py  (prints measured numbers, writes artifacts/results.json)."""
import json, time, numpy as np, pandas as pd
from src.generate_data import make_transactions
from src.contracts import validate
from src.feature_store import FeatureStore
from src.pipeline import ingest, train

SEED = 7
df, labels = make_transactions(n_customers=5000, n_rows=200_000, seed=SEED)
MAXTS = pd.Timestamp("2026-06-30")
ref = df["amount"].sample(20_000, random_state=SEED)
BATCH = 5000
batches = [df.iloc[i:i + BATCH].reset_index(drop=True) for i in range(0, len(df), BATCH)]  # 40 clean batches
rng = np.random.default_rng(SEED)

# ---- fault injection ----
def f_nulls(b):  b = b.copy(); b.loc[rng.integers(0, len(b), 5), "amount"] = np.nan; return b
def f_neg(b):    b = b.copy(); b.loc[rng.integers(0, len(b), 5), "amount"] = -1.0; return b
def f_dup(b):    return pd.concat([b, b.sample(5, random_state=1)])
def f_col(b):    return b.drop(columns=["channel"])
def f_type(b):   b = b.copy(); b["amount"] = b["amount"].astype(str); return b
def f_cat(b):    b = b.copy(); b.loc[rng.integers(0, len(b), 5), "category"] = "crypto"; return b
def f_future(b): b = b.copy(); b.loc[rng.integers(0, len(b), 5), "ts"] = pd.Timestamp("2031-01-01"); return b
def f_shift(k):
    def g(b): b = b.copy(); b["amount"] = b["amount"] * k; return b
    return g
faults = {"nulls": f_nulls, "negative_amount": f_neg, "duplicate_key": f_dup, "missing_column": f_col,
          "type_change": f_type, "unknown_category": f_cat, "future_timestamp": f_future,
          "drift_x1.05": f_shift(1.05), "drift_x1.15": f_shift(1.15), "drift_x1.5": f_shift(1.5)}

res = {"rows": len(df), "customers": 5000, "clean_batches": len(batches)}
t = time.perf_counter()
fp = sum(1 for b in batches if validate(b, ref, MAXTS))
dt = time.perf_counter() - t
res["false_positives_on_clean"] = fp
res["validation_rows_per_sec"] = round(len(df) / dt)
per = {}
for name, fn in faults.items():
    det = sum(1 for b in batches if validate(fn(b), ref, MAXTS))
    per[name] = f"{det}/{len(batches)}"
res["fault_detection"] = per
hard = [k for k in per if not k.startswith("drift_x1.05") and not k.startswith("drift_x1.15")]
res["structural_faults_detected"] = sum(int(per[k].split("/")[0]) for k in hard)
res["structural_faults_injected"] = len(hard) * len(batches)

# ---- gated ingest with 4 poisoned batches ----
mixed = list(batches); poison_idx = [3, 11, 20, 33]
for i, fn in zip(poison_idx, [f_neg, f_dup, f_nulls, f_cat]): mixed[i] = fn(mixed[i])
store = FeatureStore(); t = time.perf_counter()
acc, rej = ingest(store, mixed, ref, MAXTS)
res["gated_ingest"] = {"accepted": acc, "rejected": len(rej), "rejected_idx": [r[0] for r in rej],
                       "rows_in_store": store.count(), "seconds": round(time.perf_counter() - t, 2)}
# idempotency
n = store.count(); store.upsert(batches[0]); res["idempotent_replay_row_delta"] = store.count() - n

# ---- SQL features + train ----
t = time.perf_counter(); feats = store.features(); res["sql_feature_seconds"] = round(time.perf_counter() - t, 3)
res["feature_rows"] = len(feats)
t = time.perf_counter(); ver, meta = train(store, labels, "artifacts"); res["train_seconds"] = round(time.perf_counter() - t, 2)
res["model_version"], res["auc_gated_data"] = ver, meta["auc"]

# ---- ungated baseline: same poisoned stream, no gate ----
store2 = FeatureStore()
for b in mixed:
    try: store2.upsert(b)
    except Exception as e: res.setdefault("ungated_errors", []).append(type(e).__name__)
f2 = store2.features().merge(labels, on="customer_id")
res["ungated_rows_in_store"] = store2.count()
res["ungated_null_or_negative_amounts"] = int(store2.con.execute(
    "SELECT COUNT(*) FROM txns WHERE amount IS NULL OR amount<=0").fetchone()[0])
res["ungated_unknown_category_rows"] = int(store2.con.execute(
    "SELECT COUNT(*) FROM txns WHERE category NOT IN ('grocery','fuel','travel','electronics','dining','utilities','fashion')").fetchone()[0])
json.dump(res, open("artifacts/results.json", "w"), indent=2)
print(json.dumps(res, indent=2))
