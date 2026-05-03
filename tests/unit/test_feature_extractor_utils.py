from __future__ import annotations

import numpy as np
import pytest

from src.features.extractor import build_feature_table, extract_temporal_features

pytestmark = pytest.mark.unit


def test_extract_temporal_features_handles_short_signal() -> None:
    features = extract_temporal_features(np.array([1.0]))
    assert features["first_diff_mean"] == 0.0
    assert features["sign_changes"] == 0.0


def test_build_feature_table_contains_expected_columns(synthetic_windowed_df) -> None:
    table = build_feature_table(synthetic_windowed_df)
    for col in ["run_id", "trend_slope", "mean", "energy", "relative_peak_position"]:
        assert col in table.columns
    assert len(table) == len(synthetic_windowed_df)
