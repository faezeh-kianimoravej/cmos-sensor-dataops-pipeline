"""
Toluene Concentration Calibration Model

This module provides calibration and prediction functionality for the
ZIF-8 MOF CMOS sensor toluene detection system.

Based on the experimental protocol:
- Concentration sequence: 500 → 1000 → 3000 → 5000 → 7000 → 9000 → 7000 → 5000 → 3000 → 1000 → 500 ppm
- Each concentration held for ~10 minutes (600 seconds)
- N₂ baseline between each toluene exposure
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class ConcentrationPeriod:
    """Represents a period of known toluene concentration."""

    start_time: datetime
    end_time: datetime
    concentration_ppm: int
    peak_delta_adc: float = 0.0
    mean_delta_adc: float = 0.0


@dataclass
class CalibrationResult:
    """Result of calibration fitting."""

    slope: float  # ΔADC per ppm
    intercept: float
    r_squared: float
    concentrations: list[int]
    measured_delta_adc: list[float]

    def predict_concentration(self, delta_adc: float) -> float:
        """Predict concentration from ΔADC value."""
        if self.slope == 0:
            return 0.0
        return max(0, (delta_adc - self.intercept) / self.slope)

    def predict_delta_adc(self, concentration_ppm: float) -> float:
        """Predict ΔADC from concentration."""
        return self.slope * concentration_ppm + self.intercept


class TolueneCalibrationModel:
    """
    Calibration model for ZIF-8 MOF CMOS toluene sensor.

    Uses the known concentration sequence to build a linear calibration
    curve relating ΔADC to toluene concentration in ppm.
    """

    # Known concentration sequence from the experiment protocol
    CONCENTRATION_SEQUENCE = [
        500,
        1000,
        3000,
        5000,
        7000,
        9000,  # Ascending
        7000,
        5000,
        3000,
        1000,
        500,  # Descending
    ]

    # Approximate duration of each concentration period (seconds)
    PERIOD_DURATION = 600  # 10 minutes

    # Approximate duration of N₂ baseline between concentrations
    BASELINE_DURATION = 600  # 10 minutes

    def __init__(self):
        self.calibration: Optional[CalibrationResult] = None
        self.concentration_periods: list[ConcentrationPeriod] = []

    def detect_concentration_periods(
        self, timestamps: list[datetime], delta_adc: list[float], threshold: float = 0.3
    ) -> list[ConcentrationPeriod]:
        """
        Detect concentration periods from ΔADC data using peak detection.

        Args:
            timestamps: List of timestamps
            delta_adc: List of ΔADC values
            threshold: Minimum ΔADC to consider as toluene exposure

        Returns:
            List of detected concentration periods
        """
        if not timestamps or not delta_adc:
            return []

        periods = []
        in_peak = False
        peak_start_idx = 0
        concentration_idx = 0

        for i, (ts, adc) in enumerate(zip(timestamps, delta_adc)):
            if adc > threshold and not in_peak:
                # Start of a peak
                in_peak = True
                peak_start_idx = i
            elif adc <= threshold and in_peak:
                # End of a peak
                in_peak = False
                if concentration_idx < len(self.CONCENTRATION_SEQUENCE):
                    peak_values = delta_adc[peak_start_idx:i]
                    period = ConcentrationPeriod(
                        start_time=timestamps[peak_start_idx],
                        end_time=timestamps[i - 1],
                        concentration_ppm=self.CONCENTRATION_SEQUENCE[
                            concentration_idx
                        ],
                        peak_delta_adc=max(peak_values) if peak_values else 0,
                        mean_delta_adc=np.mean(peak_values) if peak_values else 0,
                    )
                    periods.append(period)
                    concentration_idx += 1

        # Handle case where last peak extends to end of data
        if in_peak and concentration_idx < len(self.CONCENTRATION_SEQUENCE):
            peak_values = delta_adc[peak_start_idx:]
            period = ConcentrationPeriod(
                start_time=timestamps[peak_start_idx],
                end_time=timestamps[-1],
                concentration_ppm=self.CONCENTRATION_SEQUENCE[concentration_idx],
                peak_delta_adc=max(peak_values) if peak_values else 0,
                mean_delta_adc=np.mean(peak_values) if peak_values else 0,
            )
            periods.append(period)

        self.concentration_periods = periods
        return periods

    def fit_calibration(
        self, timestamps: list[datetime], delta_adc: list[float], use_peak: bool = True
    ) -> CalibrationResult:
        """
        Fit a linear calibration model from the experimental data.

        Args:
            timestamps: List of timestamps
            delta_adc: List of ΔADC values
            use_peak: If True, use peak ΔADC; if False, use mean ΔADC

        Returns:
            CalibrationResult with fitted parameters
        """
        # First detect concentration periods if not already done
        if not self.concentration_periods:
            self.detect_concentration_periods(timestamps, delta_adc)

        if not self.concentration_periods:
            # No periods detected, return default calibration
            self.calibration = CalibrationResult(
                slope=0.0002,  # Approximate from reference chart
                intercept=0.0,
                r_squared=0.0,
                concentrations=[],
                measured_delta_adc=[],
            )
            return self.calibration

        # Extract concentration and ΔADC pairs
        concentrations = []
        measured_adc = []

        for period in self.concentration_periods:
            concentrations.append(period.concentration_ppm)
            if use_peak:
                measured_adc.append(period.peak_delta_adc)
            else:
                measured_adc.append(period.mean_delta_adc)

        # Fit linear model: ΔADC = slope * concentration + intercept
        x = np.array(concentrations)
        y = np.array(measured_adc)

        # Linear regression
        n = len(x)
        if n < 2:
            slope = 0.0002
            intercept = 0.0
            r_squared = 0.0
        else:
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_xx = np.sum(x * x)

            denom = n * sum_xx - sum_x * sum_x
            if denom != 0:
                slope = (n * sum_xy - sum_x * sum_y) / denom
                intercept = (sum_y - slope * sum_x) / n
            else:
                slope = 0.0002
                intercept = 0.0

            # Calculate R²
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        self.calibration = CalibrationResult(
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            concentrations=concentrations,
            measured_delta_adc=measured_adc,
        )

        return self.calibration

    def predict(self, delta_adc: float) -> dict:
        """
        Predict toluene concentration from a ΔADC reading.

        Args:
            delta_adc: The ΔADC value to convert

        Returns:
            Dictionary with predicted concentration and confidence info
        """
        if self.calibration is None:
            # Use default calibration from reference chart
            # Approximate: 9000 ppm ≈ 1.8 ΔADC, so slope ≈ 0.0002
            concentration = delta_adc / 0.0002
        else:
            concentration = self.calibration.predict_concentration(delta_adc)

        # Determine confidence based on calibration quality
        if self.calibration and self.calibration.r_squared > 0.9:
            confidence = "high"
        elif self.calibration and self.calibration.r_squared > 0.7:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "concentration_ppm": max(0, concentration),
            "delta_adc": delta_adc,
            "confidence": confidence,
            "r_squared": self.calibration.r_squared if self.calibration else 0.0,
        }

    def get_concentration_labels(
        self, timestamps: list[datetime], delta_adc: list[float]
    ) -> list[dict]:
        """
        Get concentration labels for plotting on the chart.

        Returns list of dicts with x (timestamp), y (ΔADC), and text (concentration label).
        """
        if not self.concentration_periods:
            self.detect_concentration_periods(timestamps, delta_adc)

        labels = []
        for period in self.concentration_periods:
            # Find the peak position within this period
            mid_time = period.start_time + (period.end_time - period.start_time) / 2
            labels.append(
                {
                    "x": mid_time,
                    "y": period.peak_delta_adc,
                    "text": f"{period.concentration_ppm} ppm",
                }
            )

        return labels


def create_default_calibration() -> TolueneCalibrationModel:
    """
    Create a calibration model with default values from reference chart.

    Based on the reference chart:
    - 9000 ppm → ~1.8 ΔADC
    - Linear relationship
    """
    model = TolueneCalibrationModel()

    # Set up default calibration from reference chart observations
    # Approximate values: slope ≈ 0.0002 ΔADC per ppm
    model.calibration = CalibrationResult(
        slope=0.0002,
        intercept=0.0,
        r_squared=0.95,  # Reference chart shows good linearity
        concentrations=[500, 1000, 3000, 5000, 7000, 9000],
        measured_delta_adc=[0.1, 0.2, 0.6, 1.0, 1.4, 1.8],
    )

    return model


if __name__ == "__main__":
    # Test the calibration model
    import sys
    from pathlib import Path

    # Add parent directory
    sys.path.insert(0, str(Path(__file__).parent))
    from analyse_cmos import load_frames

    data_file = Path(__file__).parent / "dataset" / "tolouene_aug_2024.dat"

    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        sys.exit(1)

    print("Loading frames...")
    frames = load_frames(data_file)

    # Calculate ΔADC
    timestamps = [f.timestamp for f in frames]
    adc_sums = [int(f.grids[1].sum()) for f in frames]
    baseline = float(np.percentile(adc_sums, 10))
    delta_adc = [(s - baseline) / 1000.0 for s in adc_sums]

    print(f"Loaded {len(frames)} frames")
    print(f"Baseline ADC: {baseline:,.0f}")
    print(f"ΔADC range: {min(delta_adc):.2f} to {max(delta_adc):.2f}")

    # Create and fit calibration model
    model = TolueneCalibrationModel()
    periods = model.detect_concentration_periods(timestamps, delta_adc)

    print(f"\nDetected {len(periods)} concentration periods:")
    for i, period in enumerate(periods, 1):
        print(
            f"  {i}. {period.concentration_ppm} ppm: "
            f"{period.start_time.strftime('%H:%M:%S')} - {period.end_time.strftime('%H:%M:%S')} "
            f"(peak ΔADC: {period.peak_delta_adc:.2f})"
        )

    # Fit calibration
    result = model.fit_calibration(timestamps, delta_adc)

    print("\nCalibration Results:")
    print(f"  Slope: {result.slope:.6f} ΔADC/ppm")
    print(f"  Intercept: {result.intercept:.4f}")
    print(f"  R²: {result.r_squared:.4f}")

    # Test predictions
    print("\nTest Predictions:")
    for test_adc in [0.5, 1.0, 1.5, 2.0, 3.0]:
        pred = model.predict(test_adc)
        print(
            f"  ΔADC {test_adc:.1f} → {pred['concentration_ppm']:.0f} ppm ({pred['confidence']} confidence)"
        )
