# Results Summary

Replace placeholder values after running the pipeline on the final dataset.

| Metric | Target | Final Value |
|---|---:|---:|
| Support Macro-F1 | >= 0.80 | TBD |
| Evidence Recall@5 | >= 0.85 | TBD |
| Answerability F1 | >= 0.80 | TBD |
| Calibration Error | <= 0.10 | TBD |
| Latency p95 | <= 900 ms | TBD |

## Ablations

| Experiment | Expected Direction |
|---|---|
| Base model | strongest balanced result |
| Top-k 3 | faster, possible lower recall |
| Top-k 8 | higher recall, possible slower latency |
| No calibration | weaker confidence reliability |
| No fusion features | weaker support classification |
