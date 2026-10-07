"""Data-contract validation: schema, nulls, ranges, uniqueness, categories, freshness, drift (KS)."""
import pandas as pd
from scipy.stats import ks_2samp

SCHEMA = {"txn_id": "int", "customer_id": "int", "ts": "datetime", "amount": "float",
          "category": "str", "channel": "str"}
ALLOWED = {"category": {"grocery", "fuel", "travel", "electronics", "dining", "utilities", "fashion"},
           "channel": {"pos", "web", "app"}}

def _kind(s):
    if pd.api.types.is_datetime64_any_dtype(s): return "datetime"
    if pd.api.types.is_integer_dtype(s): return "int"
    if pd.api.types.is_float_dtype(s): return "float"
    return "str"

def validate(df, reference_amount=None, max_ts=None, ks_alpha=0.001):
    """Return list of violation codes (empty list == batch passes the gate)."""
    v = []
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        return ["schema_missing_column"]
    for c, k in SCHEMA.items():
        got = _kind(df[c])
        if got != k and not (k == "float" and got == "int"):
            v.append(f"schema_type:{c}")
    if v:
        return v
    if df[list(SCHEMA)].isna().any().any(): v.append("nulls")
    if df["txn_id"].duplicated().any(): v.append("duplicate_key")
    if (df["amount"] <= 0).any(): v.append("range_amount")
    for c, ok in ALLOWED.items():
        if not set(df[c].dropna().unique()) <= ok: v.append(f"category:{c}")
    if max_ts is not None and (df["ts"] > max_ts).any(): v.append("future_timestamp")
    if reference_amount is not None and len(df) >= 200:
        if ks_2samp(df["amount"].dropna(), reference_amount).pvalue < ks_alpha: v.append("drift_amount")
    return v
