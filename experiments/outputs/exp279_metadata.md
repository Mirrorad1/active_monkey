# Exp 279 execution and runtime metadata

Raw JSON SHA256: bc3bf6de70e4753fbad7429104cbfaebca285cbe65b13c0666bb898e1a1c7981.
Raw data are immutable. Four executed-source hashes are in `source_hashes`; the enclosing
atomic experiment commit pins the executed runner. Prior framework base: f387c2c.

Pre-run gate used source-matched excluded seed 9299 in exp279_smoke_v3.json:
83.93782375007868 seconds *16/2 =671.5025900006294 projected seconds. Actual full batch:
1091.0693069999106 seconds, below the pre-run 1200-second ceiling, with 9478.602752 MB
peak RSS and 38.654705664 GB physical memory. The forecast underestimated runtime.

Bug: full JSON `preflight.projected_seconds_2_workers=8728.554455999285` multiplies the
already-full elapsed time by 16/2. It is a mislabeled derived field, not the forecast
consumed by the gate. No scientific values or gate decision depend on that full-run field.
It must not be interpreted as the pre-run forecast or silently repaired in historical raw.
Future runner reporting is corrected separately with a regression guard; the historical
run is reproducible from the atomic experiment commit/source hashes rather than new code.

First smoke failed on adapter shape before any encounter. Three successful excluded-seed
smokes retain engineering versions and hashes. Only v3 is source-matched preflight for the
confirmation. Earlier transient smoke source versions were not committed; those are retained
diagnostics, not exact replayable scientific evidence. No outcome-driven tuning occurred.

## Reporting-only follow-up

Executed source commit: eb26ff5aa6e5bd705e97ac82f58978bf0753b15f. It matches every raw source hash.
A later commit adds `runtime_metadata` and a regression test distinguishing pilot time
from full-batch time. Future full reports use the supplied excluded-seed pilot for
forecast fields and retain actual elapsed separately. No scientific functions, seeds,
thresholds, raw data or historical receipts change. The old pilot becomes source-stale
for the new runner; any new scientific execution must obtain a fresh pilot. This metadata
fix was unit-tested without repeating the completed scientific batch.
