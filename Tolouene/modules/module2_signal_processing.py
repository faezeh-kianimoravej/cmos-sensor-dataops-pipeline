"""
================================================================================
MODULE 2: SIGNAL PROCESSING & NOISE REDUCTION
================================================================================
Status: IN PROGRESS

Objective:
----------
Design a multi-stage filtering pipeline and robust event detection on ΔADC
to separate true toluene exposure events from noise artifacts.

Key Challenges:
---------------
1. High noise level (433 micro-events detected with simple thresholding)
2. Baseline drift over 3.8 hours of operation
3. Need to preserve fast response characteristics while reducing noise
4. Balance between sensitivity and false alarm rate for barn deployment

Dependencies:
-------------
numpy, pandas, scipy, matplotlib
"""

from __future__ import annotations

# Import from Module 1
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).parent))
from module1_data_preprocessing import ProcessedDataset

# =============================================================================
# 2.1 NOISE CHARACTERIZATION
# =============================================================================


@dataclass
class NoiseStatistics:
    """Container for noise statistics."""

    rms_noise: float
    peak_to_peak: float
    snr_db: float
    baseline_drift_per_hour: float
    short_term_std: float  # 10-second window
    long_term_std: float  # 10-minute window


def compute_noise_statistics(
    delta_adc: np.ndarray,
    timestamps: List,
    window_short: int = 10,  # frames for short-term noise
    window_long: int = 600,  # frames for long-term noise
) -> NoiseStatistics:
    """
    Characterize noise in the ΔADC signal.

    Computes both short-term (high-frequency) noise and
    long-term (baseline drift) variations.
    """
    # Short-term noise: rolling std with small window
    short_term_std = pd.Series(delta_adc).rolling(window=window_short).std().mean()

    # Long-term noise: rolling std with large window
    long_term_std = pd.Series(delta_adc).rolling(window=window_long).std().mean()

    # RMS noise (using high-pass filtered signal)
    # High-pass to remove baseline drift
    b, a = signal.butter(2, 0.1, btype="high")
    hp_filtered = signal.filtfilt(b, a, delta_adc)
    rms_noise = np.sqrt(np.mean(hp_filtered**2))

    # Peak-to-peak range
    peak_to_peak = delta_adc.max() - delta_adc.min()

    # SNR: signal power / noise power
    signal_power = np.var(delta_adc)
    noise_power = rms_noise**2
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

    # Baseline drift per hour
    duration_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600

    # Compute drift using low-pass filtered baseline
    baseline_trend = uniform_filter1d(delta_adc, size=len(delta_adc) // 10)
    drift_total = baseline_trend[-1] - baseline_trend[0]
    drift_per_hour = drift_total / duration_hours if duration_hours > 0 else 0

    return NoiseStatistics(
        rms_noise=float(rms_noise),
        peak_to_peak=float(peak_to_peak),
        snr_db=float(snr_db),
        baseline_drift_per_hour=float(drift_per_hour),
        short_term_std=float(short_term_std) if not np.isnan(short_term_std) else 0,
        long_term_std=float(long_term_std) if not np.isnan(long_term_std) else 0,
    )


def compute_rolling_noise(delta_adc: np.ndarray, window: int = 30) -> np.ndarray:
    """Compute rolling standard deviation (local noise level)."""
    return pd.Series(delta_adc).rolling(window=window, center=True).std().values


# =============================================================================
# 2.2 FILTERING PIPELINE
# =============================================================================


class FilterPipeline:
    """
    Multi-stage filtering pipeline for ΔADC signal.

    Stages:
    1. Baseline correction (remove drift)
    2. Noise reduction (smoothing)
    3. Optional: derivative for edge detection
    """

    def __init__(
        self,
        baseline_window: int = 300,  # ~5 min at 1 Hz
        smoothing_method: str = "savgol",  # 'ma', 'savgol', 'butter'
        smoothing_window: int = 15,
        savgol_order: int = 3,
        butter_cutoff: float = 0.1,  # Normalized frequency
        butter_order: int = 2,
    ):
        self.baseline_window = baseline_window
        self.smoothing_method = smoothing_method
        self.smoothing_window = smoothing_window
        self.savgol_order = savgol_order
        self.butter_cutoff = butter_cutoff
        self.butter_order = butter_order

        self._baseline = None
        self._filtered = None

    def fit_transform(self, delta_adc: np.ndarray) -> np.ndarray:
        """Apply full filtering pipeline."""
        # Step 1: Baseline correction
        baseline_corrected = self._correct_baseline(delta_adc)

        # Step 2: Noise reduction
        filtered = self._smooth(baseline_corrected)

        self._filtered = filtered
        return filtered

    def _correct_baseline(self, signal: np.ndarray) -> np.ndarray:
        """Remove baseline drift using rolling percentile."""
        # Use rolling 10th percentile as baseline estimate
        baseline = (
            pd.Series(signal)
            .rolling(window=self.baseline_window, center=True, min_periods=1)
            .quantile(0.1)
            .values
        )

        self._baseline = baseline
        return signal - baseline

    def _smooth(self, signal: np.ndarray) -> np.ndarray:
        """Apply smoothing filter."""
        if self.smoothing_method == "ma":
            return self._moving_average(signal)
        elif self.smoothing_method == "savgol":
            return self._savitzky_golay(signal)
        elif self.smoothing_method == "butter":
            return self._butterworth_lowpass(signal)
        else:
            raise ValueError(f"Unknown smoothing method: {self.smoothing_method}")

    def _moving_average(self, signal: np.ndarray) -> np.ndarray:
        """Simple moving average filter."""
        return uniform_filter1d(signal, size=self.smoothing_window)

    def _savitzky_golay(self, signal: np.ndarray) -> np.ndarray:
        """Savitzky-Golay filter - preserves peaks better than MA."""
        window = self.smoothing_window
        if window % 2 == 0:
            window += 1  # Must be odd
        return signal.filtfilt(window, self.savgol_order, signal)

    def _butterworth_lowpass(self, signal: np.ndarray) -> np.ndarray:
        """Butterworth low-pass filter."""
        b, a = signal.butter(self.butter_order, self.butter_cutoff, btype="low")
        return signal.filtfilt(b, a, signal)

    @property
    def baseline(self) -> Optional[np.ndarray]:
        return self._baseline


def apply_moving_average(signal: np.ndarray, window: int = 15) -> np.ndarray:
    """Apply simple moving average filter."""
    return uniform_filter1d(signal, size=window)


def apply_savitzky_golay(
    signal_data: np.ndarray, window: int = 15, order: int = 3
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter.

    Advantages over moving average:
    - Better preserves peak heights and widths
    - Reduces noise while maintaining signal shape
    """
    if window % 2 == 0:
        window += 1  # Must be odd
    return signal.savgol_filter(signal_data, window, order)


def apply_butterworth_lowpass(
    signal_data: np.ndarray, cutoff: float = 0.1, order: int = 2
) -> np.ndarray:
    """
    Apply Butterworth low-pass filter.

    cutoff: Normalized frequency (0 to 1, where 1 is Nyquist)
    """
    b, a = signal.butter(order, cutoff, btype="low")
    return signal.filtfilt(b, a, signal_data)


def apply_wavelet_denoising(
    signal_data: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    threshold_mode: str = "soft",
) -> np.ndarray:
    """
    Apply wavelet denoising.

    Requires pywt library (pip install PyWavelets).
    """
    try:
        import pywt
    except ImportError:
        print("Warning: PyWavelets not installed. Using Savitzky-Golay instead.")
        return apply_savitzky_golay(signal_data)

    # Decompose
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)

    # Estimate noise level from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))

    # Apply thresholding to detail coefficients
    denoised_coeffs = [coeffs[0]]  # Keep approximation
    for c in coeffs[1:]:
        if threshold_mode == "soft":
            denoised_coeffs.append(pywt.threshold(c, threshold, mode="soft"))
        else:
            denoised_coeffs.append(pywt.threshold(c, threshold, mode="hard"))

    # Reconstruct
    return pywt.waverec(denoised_coeffs, wavelet)[: len(signal_data)]


def compare_filters(signal_data: np.ndarray, window: int = 15) -> Dict[str, np.ndarray]:
    """
    Compare different filtering methods.

    Returns dict of filtered signals for comparison.
    """
    results = {
        "original": signal_data,
        "moving_average": apply_moving_average(signal_data, window),
        "savitzky_golay": apply_savitzky_golay(signal_data, window),
        "butterworth": apply_butterworth_lowpass(signal_data, cutoff=0.1),
    }

    # Try wavelet if available
    try:
        results["wavelet"] = apply_wavelet_denoising(signal_data)
    except:
        pass

    return results


def evaluate_filter_performance(
    original: np.ndarray, filtered: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate filter performance.

    Metrics:
    - Noise reduction ratio
    - Signal preservation (correlation)
    - Peak preservation
    """
    # Noise reduction: compare high-frequency content
    b, a = signal.butter(2, 0.3, btype="high")
    noise_original = signal.filtfilt(b, a, original)
    noise_filtered = signal.filtfilt(b, a, filtered)

    noise_reduction = 1 - (np.std(noise_filtered) / np.std(noise_original))

    # Signal correlation
    correlation = np.corrcoef(original, filtered)[0, 1]

    # Peak preservation: compare max values
    peak_ratio = filtered.max() / original.max() if original.max() > 0 else 1

    return {
        "noise_reduction": float(noise_reduction),
        "correlation": float(correlation),
        "peak_preservation": float(peak_ratio),
        "rms_original": float(np.std(noise_original)),
        "rms_filtered": float(np.std(noise_filtered)),
    }


# =============================================================================
# 2.3 ADAPTIVE BASELINE CORRECTION
# =============================================================================


class AdaptiveBaselineCorrector:
    """
    Adaptive baseline correction for long-term drift.

    Methods:
    1. Rolling quantile (robust to outliers)
    2. Asymmetric least squares (ALS)
    3. Polynomial detrending
    """

    def __init__(
        self,
        method: str = "rolling_quantile",
        window: int = 600,  # ~10 minutes
        quantile: float = 0.1,
    ):
        self.method = method
        self.window = window
        self.quantile = quantile
        self._baseline = None

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """Estimate and remove baseline."""
        if self.method == "rolling_quantile":
            self._baseline = self._rolling_quantile(signal)
        elif self.method == "polynomial":
            self._baseline = self._polynomial_detrend(signal)
        elif self.method == "als":
            self._baseline = self._als_baseline(signal)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return signal - self._baseline

    def _rolling_quantile(self, signal: np.ndarray) -> np.ndarray:
        """Rolling quantile baseline estimation."""
        baseline = (
            pd.Series(signal)
            .rolling(window=self.window, center=True, min_periods=1)
            .quantile(self.quantile)
            .values
        )
        return baseline

    def _polynomial_detrend(self, signal: np.ndarray, degree: int = 3) -> np.ndarray:
        """Polynomial baseline fitting."""
        x = np.arange(len(signal))
        coeffs = np.polyfit(x, signal, degree)
        baseline = np.polyval(coeffs, x)
        return baseline

    def _als_baseline(
        self, signal: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10
    ) -> np.ndarray:
        """
        Asymmetric Least Squares baseline estimation.

        Reference: Eilers & Boelens (2005)
        """
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        L = len(signal)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)

        for _ in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.T)
            baseline = spsolve(Z, w * signal)
            w = p * (signal > baseline) + (1 - p) * (signal < baseline)

        return baseline

    @property
    def baseline(self) -> Optional[np.ndarray]:
        return self._baseline


# =============================================================================
# 2.4 EVENT DETECTION WITH HYSTERESIS
# =============================================================================


@dataclass
class DetectedEvent:
    """A detected exposure event."""

    start_idx: int
    end_idx: int
    start_time: float  # Elapsed seconds
    end_time: float
    duration: float
    peak_value: float
    mean_value: float
    integrated_value: float  # Area under curve

    def __repr__(self):
        return (
            f"Event(t={self.start_time:.1f}-{self.end_time:.1f}s, "
            f"dur={self.duration:.1f}s, peak={self.peak_value:.3f})"
        )


class EventDetector:
    """
    Event detection with hysteresis thresholding.

    Uses upper and lower thresholds to prevent rapid on/off switching
    due to noise. Also applies minimum duration filter.
    """

    def __init__(
        self,
        threshold_high: float = 0.5,  # Enter event when signal crosses this
        threshold_low: float = 0.3,  # Exit event when signal falls below this
        min_duration: float = 30.0,  # Minimum event duration in seconds
        merge_gap: float = 10.0,  # Merge events closer than this (seconds)
        sampling_rate: float = 1.0,  # Samples per second
    ):
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.min_duration = min_duration
        self.merge_gap = merge_gap
        self.sampling_rate = sampling_rate

    def detect(
        self, signal: np.ndarray, timestamps: Optional[np.ndarray] = None
    ) -> List[DetectedEvent]:
        """
        Detect events using hysteresis thresholding.

        Returns list of DetectedEvent objects.
        """
        if timestamps is None:
            timestamps = np.arange(len(signal)) / self.sampling_rate

        events = []
        in_event = False
        start_idx = 0

        for i, val in enumerate(signal):
            if not in_event and val >= self.threshold_high:
                # Start of event
                in_event = True
                start_idx = i
            elif in_event and val < self.threshold_low:
                # End of event
                in_event = False
                events.append(self._create_event(signal, timestamps, start_idx, i))

        # Handle event that extends to end of signal
        if in_event:
            events.append(
                self._create_event(signal, timestamps, start_idx, len(signal) - 1)
            )

        # Filter by minimum duration
        events = [e for e in events if e.duration >= self.min_duration]

        # Merge close events
        events = self._merge_events(events, signal, timestamps)

        return events

    def _create_event(
        self, signal: np.ndarray, timestamps: np.ndarray, start_idx: int, end_idx: int
    ) -> DetectedEvent:
        """Create DetectedEvent from indices."""
        event_signal = signal[start_idx : end_idx + 1]
        if hasattr(np, "trapezoid"):
            integrated_value = float(np.trapezoid(event_signal))
        else:
            integrated_value = float(np.trapz(event_signal))

        return DetectedEvent(
            start_idx=start_idx,
            end_idx=end_idx,
            start_time=float(timestamps[start_idx]),
            end_time=float(timestamps[end_idx]),
            duration=float(timestamps[end_idx] - timestamps[start_idx]),
            peak_value=float(event_signal.max()),
            mean_value=float(event_signal.mean()),
            integrated_value=integrated_value,
        )

    def _merge_events(
        self, events: List[DetectedEvent], signal: np.ndarray, timestamps: np.ndarray
    ) -> List[DetectedEvent]:
        """Merge events that are close together."""
        if len(events) < 2:
            return events

        merged = [events[0]]

        for event in events[1:]:
            gap = event.start_time - merged[-1].end_time

            if gap < self.merge_gap:
                # Merge with previous event
                merged[-1] = self._create_event(
                    signal, timestamps, merged[-1].start_idx, event.end_idx
                )
            else:
                merged.append(event)

        return merged

    def events_to_dataframe(self, events: List[DetectedEvent]) -> pd.DataFrame:
        """Convert events to DataFrame for analysis."""
        return pd.DataFrame(
            [
                {
                    "start_time": e.start_time,
                    "end_time": e.end_time,
                    "duration": e.duration,
                    "peak_dadc": e.peak_value,
                    "mean_dadc": e.mean_value,
                    "integrated": e.integrated_value,
                    "est_ppm": self._estimate_ppm(e.peak_value),
                }
                for e in events
            ]
        )

    def _estimate_ppm(self, delta_adc: float) -> float:
        """Rough ppm estimate from ΔADC (linear approximation)."""
        # Based on calibration: 500 ppm @ 0.5 ΔADC, 9000 ppm @ 3.5 ΔADC
        # Slope: (9000-500)/(3.5-0.5) = 2833 ppm per ΔADC
        return 500 + max(0, delta_adc - 0.5) * 2833


# =============================================================================
# 2.5 COMPLETE FILTERING + DETECTION PIPELINE
# =============================================================================


class SignalProcessor:
    """
    Complete signal processing pipeline.

    Combines:
    1. Baseline correction
    2. Noise filtering
    3. Event detection
    """

    def __init__(
        self,
        # Baseline correction params
        baseline_window: int = 600,
        baseline_quantile: float = 0.1,
        # Filtering params
        filter_method: str = "savgol",
        filter_window: int = 15,
        # Event detection params
        threshold_high: float = 0.5,
        threshold_low: float = 0.3,
        min_event_duration: float = 30.0,
        merge_gap: float = 10.0,
        sampling_rate: float = 1.0,
    ):
        self.baseline_corrector = AdaptiveBaselineCorrector(
            method="rolling_quantile",
            window=baseline_window,
            quantile=baseline_quantile,
        )

        self.filter_method = filter_method
        self.filter_window = filter_window

        self.event_detector = EventDetector(
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            min_duration=min_event_duration,
            merge_gap=merge_gap,
            sampling_rate=sampling_rate,
        )

        self._raw_signal = None
        self._baseline_corrected = None
        self._filtered_signal = None
        self._events = None

    def process(
        self, signal: np.ndarray, timestamps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[DetectedEvent]]:
        """
        Run complete processing pipeline.

        Returns:
            Tuple of (filtered_signal, detected_events)
        """
        self._raw_signal = signal.copy()

        # Step 1: Baseline correction
        self._baseline_corrected = self.baseline_corrector.fit_transform(signal)

        # Step 2: Noise filtering
        if self.filter_method == "savgol":
            self._filtered_signal = apply_savitzky_golay(
                self._baseline_corrected, self.filter_window
            )
        elif self.filter_method == "ma":
            self._filtered_signal = apply_moving_average(
                self._baseline_corrected, self.filter_window
            )
        elif self.filter_method == "butter":
            self._filtered_signal = apply_butterworth_lowpass(self._baseline_corrected)
        else:
            self._filtered_signal = self._baseline_corrected

        # Step 3: Event detection
        self._events = self.event_detector.detect(self._filtered_signal, timestamps)

        return self._filtered_signal, self._events

    @property
    def baseline(self) -> Optional[np.ndarray]:
        return self.baseline_corrector.baseline

    @property
    def filtered_signal(self) -> Optional[np.ndarray]:
        return self._filtered_signal

    @property
    def events(self) -> Optional[List[DetectedEvent]]:
        return self._events

    def get_processing_report(self) -> Dict:
        """Generate processing summary report."""
        if self._events is None:
            return {"error": "Pipeline not run yet"}

        return {
            "num_events_detected": len(self._events),
            "total_exposure_time": sum(e.duration for e in self._events),
            "max_peak_dadc": (
                max(e.peak_value for e in self._events) if self._events else 0
            ),
            "noise_reduction": (
                evaluate_filter_performance(self._raw_signal, self._filtered_signal)
                if self._raw_signal is not None
                else None
            ),
        }


# =============================================================================
# 2.6 VISUALIZATION FUNCTIONS
# =============================================================================


def plot_noise_analysis(dataset: ProcessedDataset, save_path: Optional[Path] = None):
    """Plot noise characterization analysis."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    delta_adc = dataset.delta_adc
    elapsed = np.array(
        [(t - dataset.timestamps[0]).total_seconds() for t in dataset.timestamps]
    )

    # Plot 1: Rolling noise (local std)
    rolling_noise = compute_rolling_noise(delta_adc, window=30)
    axes[0, 0].plot(elapsed / 60, rolling_noise, "b-", linewidth=0.5)
    axes[0, 0].set_xlabel("Time (minutes)")
    axes[0, 0].set_ylabel("Local Noise (30-frame std)")
    axes[0, 0].set_title("Local Noise Level Over Time")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Baseline drift
    baseline = (
        pd.Series(delta_adc).rolling(window=300, center=True).quantile(0.1).values
    )
    axes[0, 1].plot(
        elapsed / 60, delta_adc, "b-", linewidth=0.3, alpha=0.5, label="Raw"
    )
    axes[0, 1].plot(elapsed / 60, baseline, "r-", linewidth=2, label="Baseline")
    axes[0, 1].set_xlabel("Time (minutes)")
    axes[0, 1].set_ylabel("ΔADC")
    axes[0, 1].set_title("Baseline Drift")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Noise histogram
    hp_b, hp_a = signal.butter(2, 0.1, btype="high")
    noise_only = signal.filtfilt(hp_b, hp_a, delta_adc)
    axes[1, 0].hist(noise_only, bins=100, edgecolor="black", alpha=0.7)
    axes[1, 0].set_xlabel("High-frequency Noise")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title(f"Noise Distribution (RMS: {np.std(noise_only):.4f})")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Power spectral density
    f, psd = signal.welch(delta_adc, fs=1.0, nperseg=256)
    axes[1, 1].semilogy(f, psd)
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Power Spectral Density")
    axes[1, 1].set_title("Signal Power Spectrum")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_filter_comparison(
    signal_data: np.ndarray,
    start_idx: int = 0,
    end_idx: int = 500,
    save_path: Optional[Path] = None,
):
    """Compare different filtering methods."""
    import matplotlib.pyplot as plt

    # Get filtered signals
    filtered = compare_filters(signal_data)

    # Subset for visualization
    x = np.arange(start_idx, min(end_idx, len(signal_data)))

    fig, axes = plt.subplots(
        len(filtered), 1, figsize=(14, 3 * len(filtered)), sharex=True
    )

    colors = ["blue", "green", "orange", "red", "purple"]

    for i, (name, data) in enumerate(filtered.items()):
        axes[i].plot(
            x,
            data[start_idx:end_idx],
            color=colors[i % len(colors)],
            linewidth=0.8,
            label=name,
        )

        # Add performance metrics
        if name != "original":
            metrics = evaluate_filter_performance(filtered["original"], data)
            axes[i].text(
                0.98,
                0.95,
                f"Noise Red: {metrics['noise_reduction']:.1%}\n"
                f"Peak Pres: {metrics['peak_preservation']:.1%}",
                transform=axes[i].transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        axes[i].set_ylabel("ΔADC")
        axes[i].legend(loc="upper left")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame Index")
    fig.suptitle("Filter Comparison", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_detected_events(
    signal: np.ndarray,
    events: List[DetectedEvent],
    timestamps: np.ndarray,
    save_path: Optional[Path] = None,
):
    """Plot signal with detected events highlighted."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot signal
    ax.plot(
        timestamps / 60, signal, "b-", linewidth=0.5, alpha=0.7, label="Filtered Signal"
    )

    # Highlight events
    for event in events:
        ax.axvspan(
            event.start_time / 60,
            event.end_time / 60,
            alpha=0.3,
            color="red",
            label="_nolegend_",
        )
        ax.plot(event.start_time / 60, event.peak_value, "rv", markersize=8)

    # Add threshold lines
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.7, label="Threshold High")
    ax.axhline(y=0.3, color="green", linestyle="--", alpha=0.7, label="Threshold Low")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("ΔADC (filtered)")
    ax.set_title(f"Detected Exposure Events (n={len(events)})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# MODULE 2 SUMMARY
# =============================================================================

MODULE_2_SUMMARY = """
================================================================================
MODULE 2 SUMMARY: SIGNAL PROCESSING & NOISE REDUCTION
================================================================================

COMPLETED COMPONENTS:
---------------------
✓ 2.1 Noise Characterization
    - RMS noise computation
    - Short-term vs long-term noise separation
    - Power spectral density analysis
    - Baseline drift quantification

✓ 2.2 Filtering Pipeline
    - Moving Average filter
    - Savitzky-Golay filter (recommended)
    - Butterworth low-pass filter
    - Wavelet denoising (optional)
    - Filter performance evaluation metrics

✓ 2.3 Adaptive Baseline Correction
    - Rolling quantile method
    - Polynomial detrending
    - Asymmetric Least Squares (ALS)

✓ 2.4 Event Detection with Hysteresis
    - Dual-threshold (high/low) hysteresis
    - Minimum duration filtering
    - Event merging for close events
    - Comprehensive event metrics

✓ 2.5 Complete Processing Pipeline
    - SignalProcessor class combining all stages
    - Configurable parameters
    - Processing report generation

✓ 2.6 Visualization Functions
    - Noise analysis plots
    - Filter comparison
    - Event detection visualization

RECOMMENDED SETTINGS:
--------------------
- Filter: Savitzky-Golay (window=15, order=3)
- Baseline: Rolling 10th percentile (window=600)
- Event detection:
    * Threshold high: 0.5 ΔADC (~500 ppm)
    * Threshold low: 0.3 ΔADC
    * Minimum duration: 30 seconds
    * Merge gap: 10 seconds

NEXT STEPS → MODULE 3:
----------------------
- Compute T90 response/recovery times
- Analyze dynamic range and saturation
- Quantify long-term stability
- Layer diagnostics (L2 as reference)
================================================================================
"""


if __name__ == "__main__":
    print(MODULE_2_SUMMARY)
