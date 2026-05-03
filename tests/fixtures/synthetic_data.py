from __future__ import annotations

import numpy as np
import pandas as pd


def make_processed_timeseries(
    run_id: str = "run_1", experiment: str = "Toluene", n_samples: int = 100
) -> pd.DataFrame:
    signal = np.linspace(0.0, 1.0, n_samples)
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "experiment": experiment,
                "experiment_folder": experiment.lower(),
                "run_folder": f"folder_{run_id}",
                "repeat_index": 1,
                "time": np.arange(n_samples, dtype=float).tolist(),
                "signal": signal.tolist(),
                "signal_smoothed": signal.tolist(),
                "signal_normalized": signal.tolist(),
                "n_samples": n_samples,
            }
        ]
    )
