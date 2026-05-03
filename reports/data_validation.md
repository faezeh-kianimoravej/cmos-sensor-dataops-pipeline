# Data Validation Report

Overall status: **PASS**

| Check | Status | Details |
|---|---|---|
| required-files-exist | PASS | All required files are present. |
| expected-columns-processed_timeseries | PASS | Columns match a supported schema for processed_timeseries. |
| expected-columns-windowed_timeseries | PASS | All expected columns exist in windowed_timeseries. |
| expected-columns-features_timeseries | PASS | Columns match a supported schema for features_timeseries. |
| missing-values-processed_timeseries | PASS | No unexpected missing values in processed_timeseries. |
| missing-values-windowed_timeseries | PASS | No unexpected missing values in windowed_timeseries. |
| missing-values-features_timeseries | PASS | No unexpected missing values in features_timeseries. |
| sensor-value-ranges | PASS | Normalized sensor values are within [0, 1]. |
| train-test-leakage | PASS | No train/test leakage detected. |
