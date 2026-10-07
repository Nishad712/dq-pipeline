"""Synthetic transaction generator (seeded, reproducible)."""
import numpy as np, pandas as pd

CATEGORIES = ["grocery", "fuel", "travel", "electronics", "dining", "utilities", "fashion"]
CHANNELS = ["pos", "web", "app"]

def make_transactions(n_customers=5000, n_rows=200_000, seed=7):
    rng = np.random.default_rng(seed)
    risk = rng.normal(0, 1, n_customers)
    cust = rng.integers(0, n_customers, n_rows)
    base = np.exp(rng.normal(3.2 + 0.35 * risk[cust], 0.8))
    day = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 90, n_rows), unit="D")
    night_p = 1 / (1 + np.exp(-(risk[cust] - 1.0)))
    hour = np.where(rng.random(n_rows) < night_p * 0.5, rng.integers(0, 5, n_rows), rng.integers(6, 23, n_rows))
    ts = day + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(rng.integers(0, 3600, n_rows), unit="s")
    df = pd.DataFrame({
        "txn_id": np.arange(n_rows),
        "customer_id": cust,
        "ts": ts,
        "amount": base.round(2),
        "category": rng.choice(CATEGORIES, n_rows, p=[.3, .15, .08, .1, .2, .12, .05]),
        "channel": rng.choice(CHANNELS, n_rows, p=[.5, .3, .2]),
    })
    p = 1 / (1 + np.exp(-(1.6 * risk - 0.4)))
    labels = pd.DataFrame({"customer_id": np.arange(n_customers), "high_risk": (rng.random(n_customers) < p).astype(int)})
    return df, labels
