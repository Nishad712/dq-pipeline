# Data-Contract-Gated Feature Pipeline with SQL Feature Store

Ingest -> data-quality gate -> idempotent SQL feature store -> training -> versioned artifact with data-hash lineage.
All data is synthetic and seeded (seed=7), so every number below is reproducible: `python benchmark.py`.

Measured results (200,000 rows, 5,000 customers, 40 batches of 5,000):
- 0 false positives on 40 clean batches; 562K rows/s validation throughput
- 320/320 injected structural faults caught (8 fault types x 40 batches)
- Mean-shift drift detection (KS test): 6/40 at 5% shift, 40/40 at 15% and 50%
- Gated ingest of 4 poisoned batches: all 4 rejected, 36 accepted, 180,000 rows stored; ungated ingest stored 15 bad rows
- Idempotent replay: 0 duplicate rows; SQL feature build 0.31 s; training 0.92 s; held-out AUC 0.775
- 14 unit tests (`python -m unittest discover -s tests`)

Not yet run by me: Dockerfile build and the GitHub Actions workflow (no Docker/network in my environment). Run both before listing them.
