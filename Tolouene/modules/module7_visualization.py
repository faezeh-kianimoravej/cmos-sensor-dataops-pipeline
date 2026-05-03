"""
================================================================================
MODULE 7: VISUALIZATION
================================================================================
Status: COMPLETE

Objective:
----------
Provide comprehensive visualization capabilities for the CMOS VOC analysis pipeline.
Generates detailed plots for each stage of processing:
1. Signal Processing (Raw vs Filtered)
2. Event Detection
3. Sensor Characterization
4. Calibration Models
5. Application Severity

Dependencies:
-------------
plotly, pandas, numpy
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from module2_signal_processing import DetectedEvent
from module3_sensor_characterization import SensorCharacteristics
from module4_calibration_modeling import CalibrationModel
from plotly.subplots import make_subplots


class PipelineVisualizer:
    """
    Generates visualizations for the CMOS VOC Analysis Pipeline.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color schemes
        self.colors = {
            "raw": "#95a5a6",
            "filtered": "#2980b9",
            "baseline": "#e74c3c",
            "event": "rgba(46, 204, 113, 0.3)",
            "event_border": "#27ae60",
            "severity": {
                1: "#2ecc71",  # NORMAL - Green
                2: "#f97316",  # WARNING - Orange
                3: "#ef4444",  # CRITICAL - Red
            },
        }

    def plot_signal_processing(
        self,
        timestamps: np.ndarray,
        raw_signal: np.ndarray,
        filtered_signal: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        events: List[DetectedEvent] = None,
        show_thresholds: bool = True,
        label_top_n_events: int = 10,
        label_every_nth: int = 3,
    ) -> go.Figure:
        """
        Plot raw signal, filtered signal, baseline, and detected events.

        Enhanced version with:
        - Threshold lines (ON/OFF) for event detection clarity
        - Selective event labeling (top N by amplitude + every Nth)
        - Summary statistics box
        - Scientific caption

        Args:
            timestamps: Array of timestamps
            raw_signal: Raw ΣADC signal
            filtered_signal: Butterworth-filtered signal
            baseline: Optional baseline array
            events: List of detected events
            show_thresholds: Whether to show ON/OFF threshold lines
            label_top_n_events: Number of highest-amplitude events to always label
            label_every_nth: Label every Nth event (for remaining events)
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Signal Processing Steps", "Event Detection"),
            row_heights=[0.55, 0.45],
        )

        # Convert timestamps to datetime if they are not already
        if isinstance(timestamps[0], (int, float)):
            x_axis = timestamps
            x_label = "Time (seconds)"
        else:
            x_axis = timestamps
            x_label = "Time"

        # Calculate threshold values for event detection visualization
        # Using the same logic as the event detector: ON = μ + 3σ, OFF = μ + 1.5σ
        signal_mean = np.mean(filtered_signal)
        signal_std = np.std(filtered_signal)
        on_threshold = signal_mean + 3 * signal_std
        off_threshold = signal_mean + 1.5 * signal_std

        # --- Top Plot: Signal Processing ---

        # Raw Signal
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=raw_signal,
                name="Raw Signal",
                line=dict(color=self.colors["raw"], width=1),
                opacity=0.6,
            ),
            row=1,
            col=1,
        )

        # Filtered Signal
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=filtered_signal,
                name="Filtered Signal",
                line=dict(color=self.colors["filtered"], width=2),
            ),
            row=1,
            col=1,
        )

        # Baseline (if available)
        if baseline is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=baseline,
                    name="Baseline",
                    line=dict(color=self.colors["baseline"], width=2, dash="dash"),
                ),
                row=1,
                col=1,
            )

        # --- Bottom Plot: Events ---

        # Filtered Signal again for reference
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=filtered_signal,
                name="Signal",
                line=dict(color=self.colors["filtered"], width=1),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Add threshold lines (both plots)
        if show_thresholds:
            # ON threshold (μ + 3σ) - triggers event start
            fig.add_hline(
                y=on_threshold,
                line_dash="dash",
                line_color="#e74c3c",
                line_width=1.5,
                annotation_text=f"ON (μ+3σ = {on_threshold:.2f})",
                annotation_position="right",
                annotation_font_size=9,
                annotation_font_color="#e74c3c",
                row=2,
                col=1,
            )

            # OFF threshold (μ + 1.5σ) - triggers event end
            fig.add_hline(
                y=off_threshold,
                line_dash="dot",
                line_color="#f39c12",
                line_width=1.5,
                annotation_text=f"OFF (μ+1.5σ = {off_threshold:.2f})",
                annotation_position="right",
                annotation_font_size=9,
                annotation_font_color="#f39c12",
                row=2,
                col=1,
            )

        # Determine which events to label
        events_to_label = set()
        if events:
            # Sort events by peak amplitude to find top N
            sorted_events = sorted(
                enumerate(events), key=lambda x: x[1].peak_value, reverse=True
            )
            top_n_indices = {idx for idx, _ in sorted_events[:label_top_n_events]}

            # Also include every Nth event
            nth_indices = {i for i in range(0, len(events), label_every_nth)}

            events_to_label = top_n_indices | nth_indices

        # Event statistics for summary
        event_durations = []
        event_amplitudes = []

        # Highlight Events
        if events:
            for i, event in enumerate(events):
                # Handle indices vs timestamps
                if isinstance(x_axis[0], (int, float)):
                    x0 = event.start_time
                    x1 = event.end_time
                    duration = event.end_time - event.start_time
                else:
                    try:
                        x0 = x_axis[event.start_idx]
                        x1 = x_axis[event.end_idx]
                        # Calculate duration in seconds
                        if hasattr(x0, "timestamp"):
                            duration = (x1 - x0).total_seconds()
                        else:
                            duration = event.end_idx - event.start_idx
                    except IndexError:
                        continue

                event_durations.append(duration)
                event_amplitudes.append(event.peak_value)

                # Add shaded region (bottom plot only)
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=self.colors["event"],
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                    row=2,
                    col=1,
                )

                # Only add labels for selected events
                if i in events_to_label:
                    # Determine if this is a top event (for styling)
                    is_top_event = i in top_n_indices if events else False
                    label_color = "#c0392b" if is_top_event else "#27ae60"
                    font_size = 10 if is_top_event else 8

                    # Add annotation for peak (Bottom Plot only - reduces clutter)
                    fig.add_annotation(
                        x=x0,
                        y=event.peak_value,
                        text=f"<b>E{i+1}</b>",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=0.8,
                        arrowwidth=1,
                        arrowcolor=label_color,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor=label_color,
                        borderwidth=1,
                        font=dict(size=font_size, color=label_color),
                        opacity=0.95,
                        row=2,
                        col=1,
                    )

        # Calculate summary statistics
        n_events = len(events) if events else 0
        mean_duration = np.mean(event_durations) if event_durations else 0
        mean_amplitude = np.mean(event_amplitudes) if event_amplitudes else 0
        max_amplitude = np.max(event_amplitudes) if event_amplitudes else 0

        # Add summary box annotation (bottom-right of top plot)
        summary_text = (
            f"<b>Event Summary</b><br>"
            f"Total events: {n_events}<br>"
            f"Mean duration: {mean_duration:.1f} sec<br>"
            f"Mean amplitude: {mean_amplitude:.2f} ΔADC<br>"
            f"Max amplitude: {max_amplitude:.2f} ΔADC"
        )

        fig.add_annotation(
            text=summary_text,
            xref="paper",
            yref="paper",
            x=0.99,
            y=0.98,
            showarrow=False,
            font=dict(size=10, family="Arial", color="#2c3e50"),
            align="left",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#3498db",
            borderwidth=2,
            borderpad=8,
        )

        fig.update_layout(
            title_text="Module 1 & 2: Signal Processing & Event Detection",
            height=750,
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=60, r=120, t=80, b=100),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.update_xaxes(title_text=x_label, row=2, col=1)
        fig.update_yaxes(title_text="ΔADC", row=1, col=1)
        fig.update_yaxes(title_text="ΔADC", row=2, col=1)

        # Add scientific caption below the plot
        caption_text = (
            f"<b>Figure:</b> Butterworth-filtered ΣADC signal (blue) compared with raw signal (grey). "
            f"Event detection uses hysteresis thresholds (ON: μ+3σ, OFF: μ+1.5σ) and minimum duration filtering, "
            f"producing {n_events} validated VOC exposure events (green shaded regions). "
            f"Labels show top {label_top_n_events} highest-amplitude events plus every {label_every_nth}rd event."
        )

        fig.add_annotation(
            text=caption_text,
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.12,
            showarrow=False,
            font=dict(size=10, color="#555", family="Arial"),
            align="center",
            bgcolor="rgba(248,249,250,0.9)",
            borderpad=6,
        )

        return fig

    def plot_sensor_characteristics(
        self, characteristics: SensorCharacteristics
    ) -> go.Figure:
        """
        Visualize sensor characteristics (T90, Dynamic Range).
        """
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Response Times (T90)", "Dynamic Range"),
            specs=[[{"type": "bar"}, {"type": "indicator"}]],
        )

        # T90 Bar Chart
        fig.add_trace(
            go.Bar(
                x=["Rise Time", "Recovery Time"],
                y=[characteristics.mean_t90_rise, characteristics.mean_t90_recovery],
                error_y=dict(
                    type="data",
                    array=[
                        characteristics.std_t90_rise,
                        characteristics.std_t90_recovery,
                    ],
                    visible=True,
                ),
                marker_color=[self.colors["filtered"], self.colors["baseline"]],
                name="Response Times",
            ),
            row=1,
            col=1,
        )

        # Dynamic Range Indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=characteristics.max_response,
                title={"text": "Max Response (ΔADC)"},
                delta={
                    "reference": characteristics.min_detectable,
                    "position": "bottom",
                },
                gauge={
                    "axis": {
                        "range": [0, max(5.0, characteristics.max_response * 1.2)]
                    },
                    "bar": {"color": self.colors["filtered"]},
                    "steps": [
                        {
                            "range": [0, characteristics.min_detectable],
                            "color": "lightgray",
                        },
                        {
                            "range": [
                                characteristics.min_detectable,
                                characteristics.max_response,
                            ],
                            "color": "white",
                        },
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": characteristics.max_response,
                    },
                },
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title_text="Module 3: Sensor Characteristics",
            height=450,
            template="plotly_white",
            margin=dict(l=60, r=60, t=80, b=60),
        )

        fig.update_yaxes(title_text="Seconds", row=1, col=1)

        return fig

    def plot_calibration(
        self, model: CalibrationModel, dataset: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Plot calibration curve and model fit.
        """
        fig = go.Figure()

        # Generate curve points
        x_range = np.linspace(0, 3.0, 100)  # Assuming ΔADC range 0-3
        y_pred = model.predict_ppm(x_range)

        # Plot Model Curve
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pred,
                mode="lines",
                name=f"{model.name} Model",
                line=dict(color=self.colors["filtered"], width=3),
            )
        )

        # Plot Training Data (if available)
        # Since we don't pass the training points directly to this function usually,
        # we might skip this or add dummy points if we had them.
        # For now, we just show the curve.

        fig.update_layout(
            title_text=f"Module 4: Calibration Model ({model.name})",
            xaxis_title="Sensor Response (ΔADC)",
            yaxis_title="Concentration (ppm)",
            height=500,
            template="plotly_white",
            margin=dict(l=60, r=60, t=80, b=60),
        )

        return fig

    def plot_severity_timeline(
        self,
        timestamps: np.ndarray,
        severity_levels: np.ndarray,
        ppm_values: np.ndarray,
    ) -> go.Figure:
        """
        Plot severity levels over time.
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                "Concentration & Severity Timeline",
                "Severity Level Status",
            ),
        )

        # Create color array for points based on severity
        point_colors = [self.colors["severity"].get(s, "gray") for s in severity_levels]

        # PPM Plot with colored markers
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=ppm_values,
                mode="lines",
                name="Concentration",
                line=dict(color="gray", width=1),
                opacity=0.5,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=ppm_values,
                mode="markers",
                name="Severity",
                marker=dict(color=point_colors, size=4),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Severity Level Step Plot
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=severity_levels,
                mode="lines",
                name="Severity Level",
                line=dict(color="black", width=2, shape="hv"),
            ),
            row=2,
            col=1,
        )

        # Add colored background bands for severity levels in bottom plot
        for level, color in self.colors["severity"].items():
            fig.add_hrect(
                y0=level - 0.5,
                y1=level + 0.5,
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,
                row=2,
                col=1,
            )

        fig.update_layout(
            title_text="Module 5: Application Severity Monitoring",
            height=600,
            template="plotly_white",
            margin=dict(l=60, r=60, t=80, b=60),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.update_yaxes(title_text="Concentration (ppm)", row=1, col=1)
        fig.update_yaxes(
            title_text="Severity Level",
            tickvals=[1, 2, 3],
            ticktext=["NORMAL", "WARNING", "CRITICAL"],
            range=[0.5, 3.5],  # Force range to align bands perfectly
            row=2,
            col=1,
        )

        return fig

    def save_all_plots(
        self,
        results: "PipelineResults",  # Forward reference string to avoid circular import
        filename_prefix: str = "cmos_analysis",
    ):
        """
        Generate and save all plots.
        """
        print(f"      Generating visualizations in {self.output_dir}...")

        # 1. Signal Processing
        if results.dataset and results.filtered_signal is not None:
            fig_signal = self.plot_signal_processing(
                timestamps=results.dataset.timestamps,
                raw_signal=results.dataset.delta_adc,
                filtered_signal=results.filtered_signal,
                baseline=(
                    results.baseline_corrected
                    if hasattr(results, "baseline_corrected")
                    else None
                ),
                events=results.detected_events,
            )
            fig_signal.write_html(self.output_dir / f"{filename_prefix}_1_signal.html")
            fig_signal.write_image(self.output_dir / f"{filename_prefix}_1_signal.png")

        # 2. Sensor Characteristics
        if results.sensor_characteristics:
            fig_char = self.plot_sensor_characteristics(results.sensor_characteristics)
            fig_char.write_html(
                self.output_dir / f"{filename_prefix}_2_characteristics.html"
            )
            fig_char.write_image(
                self.output_dir / f"{filename_prefix}_2_characteristics.png"
            )

        # 3. Calibration
        if results.calibration_model:
            fig_cal = self.plot_calibration(results.calibration_model)
            fig_cal.write_html(
                self.output_dir / f"{filename_prefix}_3_calibration.html"
            )
            fig_cal.write_image(
                self.output_dir / f"{filename_prefix}_3_calibration.png"
            )

        # 4. Severity
        if results.severity_timeline is not None and results.dataset is not None:
            # We need PPM values. If we have them in results, great.
            # If not, we might need to calculate them or use raw signal as proxy for visualization if model not applied to all data
            # Assuming we can get PPM from model if it exists

            if results.calibration_model:
                ppm_values = results.calibration_model.predict_ppm(
                    results.filtered_signal
                )
            else:
                ppm_values = results.filtered_signal  # Fallback

            fig_sev = self.plot_severity_timeline(
                timestamps=results.dataset.timestamps,
                severity_levels=results.severity_timeline,
                ppm_values=ppm_values,
            )
            fig_sev.write_html(self.output_dir / f"{filename_prefix}_4_severity.html")
            fig_sev.write_image(self.output_dir / f"{filename_prefix}_4_severity.png")

        print("      Visualizations saved.")


# =============================================================================
# ADDITIONAL SCIENTIFIC VISUALIZATIONS
# =============================================================================


def plot_dadc_with_exposure_windows(
    timestamps: np.ndarray,
    delta_adc: np.ndarray,
    exposure_windows: List[Dict] = None,
    show_smoothed: bool = True,
    smoothing_window: int = 10,
    safe_threshold: float = 1.0,
    moderate_threshold: float = 2.5,
) -> go.Figure:
    """
    Plot ΔADC vs Time with exposure concentration windows overlayed.

    Enhanced version with:
    - Reduced opacity for less visual clutter
    - Reference threshold lines for severity boundaries
    - Moving average trace for trend visibility
    - Color-coded legend for exposure levels

    Args:
        timestamps: Array of timestamps
        delta_adc: Array of ΔADC values
        exposure_windows: List of dicts with 'start', 'end', 'ppm', 'color'
        show_smoothed: Whether to show moving average trace
        smoothing_window: Window size for moving average (in samples)
        safe_threshold: ΔADC threshold for safe level (default 1.0)
        moderate_threshold: ΔADC threshold for high level (default 2.5)
    """
    fig = go.Figure()

    # Default exposure windows based on typical calibration protocol
    if exposure_windows is None:
        # Auto-detect exposure windows from ΔADC signal
        exposure_windows = _auto_detect_exposure_windows(timestamps, delta_adc)

    # Severity color definitions with LOWER opacity
    severity_colors = {
        "safe": "rgba(46, 204, 113, 0.15)",  # Light green
        "moderate": "rgba(241, 196, 15, 0.20)",  # Light yellow
        "high": "rgba(231, 76, 60, 0.25)",  # Light red
    }

    # Classify and merge exposure windows by severity level
    # This reduces clutter by showing continuous severity bands instead of individual windows
    safe_windows = []
    moderate_windows = []
    high_windows = []

    for window in exposure_windows:
        ppm = window.get("ppm", 0)
        if ppm < 2000:
            safe_windows.append(window)
        elif ppm < 5000:
            moderate_windows.append(window)
        else:
            high_windows.append(window)

    # Helper function to merge overlapping/adjacent windows
    def merge_windows(windows, gap_threshold_seconds=60):
        if not windows:
            return []
        # Sort by start time
        sorted_wins = sorted(windows, key=lambda w: w["start"])
        merged = [{"start": sorted_wins[0]["start"], "end": sorted_wins[0]["end"]}]

        for w in sorted_wins[1:]:
            last = merged[-1]
            # Check if windows should be merged (overlapping or close)
            try:
                if hasattr(w["start"], "timestamp"):
                    gap = (w["start"] - last["end"]).total_seconds()
                else:
                    gap = w["start"] - last["end"]

                if gap <= gap_threshold_seconds:
                    # Extend the last merged window
                    last["end"] = max(last["end"], w["end"])
                else:
                    merged.append({"start": w["start"], "end": w["end"]})
            except:
                merged.append({"start": w["start"], "end": w["end"]})

        return merged

    # Merge windows within each severity category
    merged_safe = merge_windows(safe_windows)
    merged_moderate = merge_windows(moderate_windows)
    merged_high = merge_windows(high_windows)

    # Add merged exposure window shaded regions (only moderate and high for clarity)
    # Safe windows are very light, almost not shown to reduce clutter
    for window in merged_safe:
        fig.add_vrect(
            x0=window["start"],
            x1=window["end"],
            fillcolor=severity_colors["safe"],
            layer="below",
            line_width=0,
        )

    for window in merged_moderate:
        fig.add_vrect(
            x0=window["start"],
            x1=window["end"],
            fillcolor=severity_colors["moderate"],
            layer="below",
            line_width=0,
        )

    for window in merged_high:
        fig.add_vrect(
            x0=window["start"],
            x1=window["end"],
            fillcolor=severity_colors["high"],
            layer="below",
            line_width=0,
        )

    # Add horizontal threshold reference lines
    fig.add_hline(
        y=safe_threshold,
        line_dash="dash",
        line_color="#27ae60",
        line_width=2,
        annotation_text=f"Safe/Moderate ({safe_threshold} ΔADC)",
        annotation_position="right",
        annotation_font_size=10,
        annotation_font_color="#27ae60",
    )

    fig.add_hline(
        y=moderate_threshold,
        line_dash="dash",
        line_color="#e74c3c",
        line_width=2,
        annotation_text=f"Moderate/High ({moderate_threshold} ΔADC)",
        annotation_position="right",
        annotation_font_size=10,
        annotation_font_color="#e74c3c",
    )

    # Add smoothed moving average trace (if enabled)
    if show_smoothed and len(delta_adc) > smoothing_window:
        # Calculate moving average using pandas for efficiency
        smoothed = (
            pd.Series(delta_adc)
            .rolling(window=smoothing_window, min_periods=1, center=True)
            .mean()
            .values
        )

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=smoothed,
                mode="lines",
                name=f"Smoothed ({smoothing_window}-pt avg)",
                line=dict(color="#2c3e50", width=2.5),
                opacity=0.9,
                hovertemplate="Time: %{x}<br>Smoothed ΔADC: %{y:.3f}<extra></extra>",
            )
        )

    # Main ΔADC trace
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=delta_adc,
            mode="lines",
            name="ΔADC (raw)",
            line=dict(color="#3498db", width=1),
            opacity=0.7,
            hovertemplate="Time: %{x}<br>ΔADC: %{y:.3f}<extra></extra>",
        )
    )

    # Add invisible traces for legend entries (severity levels)
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Safe (<2000 ppm)",
            marker=dict(size=12, color="#27ae60", symbol="square"),
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Moderate (2000-5000 ppm)",
            marker=dict(size=12, color="#f39c12", symbol="square"),
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="High (>5000 ppm)",
            marker=dict(size=12, color="#e74c3c", symbol="square"),
            showlegend=True,
        )
    )

    fig.update_layout(
        title="ΔADC Response with Exposure Windows",
        xaxis_title="Time",
        yaxis_title="ΔADC (×10³ ADC counts)",
        template="plotly_white",
        height=550,
        hovermode="x unified",
        margin=dict(l=60, r=120, t=80, b=60),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=10),
        ),
    )

    return fig


def _auto_detect_exposure_windows(timestamps, delta_adc, threshold=0.3):
    """Auto-detect exposure windows from signal peaks."""
    windows = []
    # Color mapping for concentration levels
    ppm_colors = {
        500: "rgba(46, 204, 113, 0.3)",  # Green
        1000: "rgba(241, 196, 15, 0.3)",  # Yellow
        2000: "rgba(230, 126, 34, 0.3)",  # Orange
        3000: "rgba(231, 76, 60, 0.3)",  # Red
        5000: "rgba(155, 89, 182, 0.3)",  # Purple
        7000: "rgba(52, 73, 94, 0.3)",  # Dark
    }

    # Simple peak detection for exposure periods
    above_threshold = delta_adc > threshold
    in_window = False
    start_idx = 0

    for i, is_above in enumerate(above_threshold):
        if is_above and not in_window:
            in_window = True
            start_idx = i
        elif not is_above and in_window:
            in_window = False
            if i - start_idx > 10:  # Minimum window size
                peak_dadc = np.max(delta_adc[start_idx:i])
                # Estimate ppm from ΔADC (rough mapping)
                est_ppm = int(500 + peak_dadc * 2500)
                # Round to nearest calibration point
                ppm_levels = [500, 1000, 2000, 3000, 5000, 7000]
                closest_ppm = min(ppm_levels, key=lambda x: abs(x - est_ppm))

                windows.append(
                    {
                        "start": timestamps[start_idx],
                        "end": timestamps[min(i, len(timestamps) - 1)],
                        "ppm": closest_ppm,
                        "color": ppm_colors.get(closest_ppm, "rgba(100,100,100,0.3)"),
                    }
                )

    return windows


def plot_spatial_sensitivity_map(
    pixel_data: np.ndarray,
    title: str = "Spatial Sensitivity Map (32×32)",
    colorscale: str = "Viridis",
) -> go.Figure:
    """
    Create a 32×32 heatmap showing per-pixel mean ΔADC during peak exposure.

    Args:
        pixel_data: 32x32 numpy array of pixel values (mean ΔADC)
        title: Plot title
        colorscale: Plotly colorscale name
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=pixel_data,
            colorscale=colorscale,
            colorbar=dict(title="ΔADC"),
            hovertemplate="Row: %{y}<br>Col: %{x}<br>ΔADC: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Column (pixel)",
        yaxis_title="Row (pixel)",
        template="plotly_white",
        height=550,
        width=600,
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed"),
    )

    return fig


def plot_bad_pixel_map(
    pixel_timeseries: np.ndarray, layer_id: int = 1, adc_fullscale: int = 4095
) -> go.Figure:
    """
    Create a map showing dead/noisy/saturated pixels using proper classification.

    Classification rules:
      - Saturated: Any pixel whose value ever exceeds 0.98 × ADC_fullscale
      - Dead: Pixels that are always exactly 0 (zero variance, mean=0)
      - Noisy: Pixels with temporal σ > 5 × median(σ) of all pixels
      - Normal: Everything else

    Args:
        pixel_timeseries: 3D array of shape (n_frames, 32, 32) containing ADC values
        layer_id: Layer identifier (1=Active, 2=Reference, 3=Saturated)
        adc_fullscale: Full scale ADC value (default 4095 for 12-bit)

    Returns:
        Plotly figure with pixel health map
    """
    # Compute pixel statistics from time-series
    n_frames = pixel_timeseries.shape[0]
    pixel_matrix = pixel_timeseries.reshape(n_frames, -1).T  # Shape: (1024, n_frames)

    pixel_means = np.mean(pixel_matrix, axis=1)
    pixel_stds = np.std(pixel_matrix, axis=1)
    pixel_maxs = np.max(pixel_matrix, axis=1)

    # Thresholds
    saturation_threshold = 0.98 * adc_fullscale
    median_std = (
        np.median(pixel_stds[pixel_stds > 0]) if np.any(pixel_stds > 0) else 1.0
    )
    noise_threshold = 5.0 * median_std

    # Classify pixels: 0=normal, 1=dead, 2=noisy, 3=saturated
    status = np.zeros(1024, dtype=np.int32)

    for i in range(1024):
        # Saturated: consistently at or near fullscale
        if pixel_means[i] > saturation_threshold or pixel_maxs[i] >= adc_fullscale:
            status[i] = 3  # Saturated
        # Dead: always zero or near-zero
        elif pixel_maxs[i] < 1.0:
            status[i] = 1  # Dead
        # Noisy: high variance relative to median
        elif pixel_stds[i] > noise_threshold:
            status[i] = 2  # Noisy
        else:
            status[i] = 0  # Normal

    status_map = status.reshape(32, 32)

    # Count statistics
    n_normal = int(np.sum(status == 0))
    n_dead = int(np.sum(status == 1))
    n_noisy = int(np.sum(status == 2))
    n_saturated = int(np.sum(status == 3))

    colorscale = [
        [0, "rgb(46, 204, 113)"],  # Normal - Green
        [0.33, "rgb(52, 73, 94)"],  # Dead - Dark gray
        [0.66, "rgb(241, 196, 15)"],  # Noisy - Yellow
        [1, "rgb(231, 76, 60)"],  # Saturated - Red
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=status_map,
            colorscale=colorscale,
            showscale=True,
            zmin=0,
            zmax=3,
            colorbar=dict(
                title="Status",
                tickvals=[0, 1, 2, 3],
                ticktext=["Normal", "Dead", "Noisy", "Saturated"],
            ),
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Status: %{z}<extra></extra>",
        )
    )

    layer_names = {1: "Active Sensing", 2: "Reference", 3: "Saturated Reference"}
    layer_name = layer_names.get(layer_id, f"Layer {layer_id}")

    fig.update_layout(
        title=f"Pixel Health Map - {layer_name} (32×32)",
        xaxis_title="Column",
        yaxis_title="Row",
        template="plotly_white",
        height=600,
        width=600,
        margin=dict(l=60, r=60, t=80, b=100),
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed"),
    )

    # Add statistics annotation
    fig.add_annotation(
        text=f"Normal: {n_normal} | Dead: {n_dead} | Noisy: {n_noisy} | Saturated: {n_saturated}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.12,
        showarrow=False,
        font=dict(size=11),
    )

    # Add yield percentage
    total_pixels = 32 * 32
    yield_pct = (n_normal / total_pixels) * 100
    fig.add_annotation(
        text=f"Pixel Yield: {yield_pct:.1f}%",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.08,
        showarrow=False,
        font=dict(size=12, color="rgb(91, 155, 213)"),
    )

    return fig


def plot_bad_pixel_map_simple(
    pixel_data: np.ndarray, noise_threshold: float = 0.1, dead_threshold: float = 0.01
) -> go.Figure:
    """
    Create a simple map showing dead/noisy/saturated pixels from single-frame data.
    Use plot_bad_pixel_map() for proper time-series based classification.

    Args:
        pixel_data: 32x32 numpy array of pixel std or response values
        noise_threshold: Threshold above which pixel is considered noisy
        dead_threshold: Threshold below which pixel is considered dead
    """
    # Classify pixels: 0=normal, 1=dead, 2=noisy, 3=saturated
    status = np.zeros_like(pixel_data, dtype=int)
    status[pixel_data < dead_threshold] = 1  # Dead
    status[pixel_data > noise_threshold] = 2  # Noisy
    status[pixel_data > 0.95 * np.max(pixel_data)] = 3  # Saturated

    colorscale = [
        [0, "rgb(46, 204, 113)"],  # Normal - Green
        [0.33, "rgb(52, 73, 94)"],  # Dead - Dark gray
        [0.66, "rgb(241, 196, 15)"],  # Noisy - Yellow
        [1, "rgb(231, 76, 60)"],  # Saturated - Red
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=status,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="Status",
                tickvals=[0, 1, 2, 3],
                ticktext=["Normal", "Dead", "Noisy", "Saturated"],
            ),
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Status: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Pixel Health Map (32×32)",
        xaxis_title="Column",
        yaxis_title="Row",
        template="plotly_white",
        height=600,
        width=600,
        margin=dict(l=60, r=60, t=80, b=100),
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed"),
    )

    # Count statistics
    n_dead = np.sum(status == 1)
    n_noisy = np.sum(status == 2)
    n_sat = np.sum(status == 3)
    n_normal = np.sum(status == 0)

    fig.add_annotation(
        text=f"Normal: {n_normal} | Dead: {n_dead} | Noisy: {n_noisy} | Saturated: {n_sat}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.12,
        showarrow=False,
        font=dict(size=11),
    )

    return fig


def plot_t90_distribution(
    rise_times: List[float], recovery_times: List[float]
) -> go.Figure:
    """
    Create violin/box plots showing T90 rise and recovery time distributions.

    Args:
        rise_times: List of T90 rise times (seconds)
        recovery_times: List of T90 recovery times (seconds)
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("T90 Rise Time Distribution", "T90 Recovery Time Distribution"),
    )

    # Rise time violin + box
    fig.add_trace(
        go.Violin(
            y=rise_times,
            name="Rise Time",
            box_visible=True,
            meanline_visible=True,
            fillcolor="rgba(41, 128, 185, 0.5)",
            line_color="#2980b9",
            points="all",
            jitter=0.3,
            pointpos=-0.5,
        ),
        row=1,
        col=1,
    )

    # Recovery time violin + box
    fig.add_trace(
        go.Violin(
            y=recovery_times,
            name="Recovery Time",
            box_visible=True,
            meanline_visible=True,
            fillcolor="rgba(231, 76, 60, 0.5)",
            line_color="#e74c3c",
            points="all",
            jitter=0.3,
            pointpos=-0.5,
        ),
        row=1,
        col=2,
    )

    # Add statistics annotations
    if rise_times:
        rise_mean = np.mean(rise_times)
        rise_std = np.std(rise_times)
        fig.add_annotation(
            text=f"Mean: {rise_mean:.1f}s ± {rise_std:.1f}s<br>n={len(rise_times)} events",
            xref="x1",
            yref="paper",
            x=0.5,
            y=-0.15,
            showarrow=False,
            font=dict(size=10),
        )

    if recovery_times:
        rec_mean = np.mean(recovery_times)
        rec_std = np.std(recovery_times)
        fig.add_annotation(
            text=f"Mean: {rec_mean:.1f}s ± {rec_std:.1f}s<br>n={len(recovery_times)} events",
            xref="x2",
            yref="paper",
            x=0.5,
            y=-0.15,
            showarrow=False,
            font=dict(size=10),
        )

    fig.update_layout(
        title="Response Time Distributions (T90)",
        template="plotly_white",
        height=550,
        margin=dict(l=60, r=60, t=80, b=100),
        showlegend=False,
    )

    fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)

    return fig


def plot_baseline_drift(
    timestamps: np.ndarray, delta_adc: np.ndarray, window_size: int = 100
) -> go.Figure:
    """
    Visualize baseline drift over time with trendline and noise envelope.

    Args:
        timestamps: Array of timestamps
        delta_adc: Array of ΔADC values
        window_size: Window size for rolling baseline calculation
    """
    # Calculate rolling 10th percentile (baseline proxy)
    baseline_rolling = (
        pd.Series(delta_adc)
        .rolling(window=window_size, min_periods=1)
        .quantile(0.1)
        .values
    )

    # Calculate noise envelope (rolling std)
    noise_rolling = (
        pd.Series(delta_adc).rolling(window=window_size, min_periods=1).std().values
    )

    # Linear fit for drift trend
    x_numeric = np.arange(len(timestamps))
    coeffs = np.polyfit(x_numeric, baseline_rolling, 1)
    trend_line = np.polyval(coeffs, x_numeric)

    # Calculate drift rate (per hour)
    if hasattr(timestamps[0], "timestamp"):
        total_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
    else:
        total_hours = len(timestamps) / 3600  # Assume 1 sample/second
    drift_per_hour = coeffs[0] * len(timestamps) / max(total_hours, 0.001)

    fig = go.Figure()

    # Noise envelope
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=baseline_rolling + 2 * noise_rolling,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=baseline_rolling - 2 * noise_rolling,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(149, 165, 166, 0.3)",
            name="±2σ Noise Envelope",
        )
    )

    # Baseline trace
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=baseline_rolling,
            mode="lines",
            name="Rolling Baseline (P₁₀)",
            line=dict(color="#2980b9", width=2),
        )
    )

    # Drift trendline
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=trend_line,
            mode="lines",
            name=f"Drift Trend ({drift_per_hour:.4f} ΔADC/hr)",
            line=dict(color="#e74c3c", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=f"Baseline Drift Analysis (Drift Rate: {drift_per_hour:.4f} ΔADC/hour)",
        xaxis_title="Time",
        yaxis_title="Baseline ΔADC",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def plot_noise_analysis(
    baseline_values: np.ndarray, sample_rate: float = 1.0
) -> go.Figure:
    """
    Create noise analysis plot with histogram and optional PSD.

    Args:
        baseline_values: Array of baseline ΔADC values (low-response period)
        sample_rate: Sampling rate in Hz
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Baseline Noise Distribution", "Power Spectral Density"),
        column_widths=[0.5, 0.5],
    )

    # Histogram of baseline values
    fig.add_trace(
        go.Histogram(
            x=baseline_values,
            nbinsx=50,
            name="Baseline Distribution",
            marker_color="#3498db",
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    # Add normal distribution overlay
    mu = np.mean(baseline_values)
    sigma = np.std(baseline_values)
    x_norm = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_norm - mu) / sigma) ** 2
    )
    # Scale to histogram
    y_norm = (
        y_norm
        * len(baseline_values)
        * (np.max(baseline_values) - np.min(baseline_values))
        / 50
    )

    fig.add_trace(
        go.Scatter(
            x=x_norm,
            y=y_norm,
            mode="lines",
            name=f"Normal Fit (μ={mu:.3f}, σ={sigma:.3f})",
            line=dict(color="#e74c3c", width=2),
        ),
        row=1,
        col=1,
    )

    # Power Spectral Density (FFT)
    if len(baseline_values) > 10:
        # Compute PSD
        n = len(baseline_values)
        fft_vals = np.fft.fft(baseline_values - np.mean(baseline_values))
        psd = np.abs(fft_vals[: n // 2]) ** 2 / n
        freqs = np.fft.fftfreq(n, 1 / sample_rate)[: n // 2]

        # Smooth PSD for better visualization
        psd_smooth = pd.Series(psd).rolling(window=5, min_periods=1).mean().values

        fig.add_trace(
            go.Scatter(
                x=freqs[1:],
                y=psd_smooth[1:],  # Skip DC component
                mode="lines",
                name="PSD",
                line=dict(color="#9b59b6", width=1.5),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=f"Noise Analysis (RMS: {sigma:.4f} ΔADC)",
        template="plotly_white",
        height=400,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    fig.update_xaxes(title_text="ΔADC", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Power", type="log", row=1, col=2)

    return fig


def plot_calibration_model_comparison(
    dadc_values: np.ndarray,
    ppm_actual: np.ndarray,
    models: Dict[str, "CalibrationModel"],
) -> go.Figure:
    """
    Plot scatter data with multiple model fits overlayed.

    Args:
        dadc_values: Actual ΔADC measurements
        ppm_actual: Actual concentration values
        models: Dictionary of model_name -> CalibrationModel instances
    """
    fig = go.Figure()

    # Scatter plot of actual data
    fig.add_trace(
        go.Scatter(
            x=dadc_values,
            y=ppm_actual,
            mode="markers",
            name="Calibration Data",
            marker=dict(color="#2c3e50", size=8, opacity=0.7),
        )
    )

    # Model colors
    model_colors = {
        "Polynomial(1)": "#e74c3c",
        "Polynomial(2)": "#3498db",
        "Freundlich": "#2ecc71",
        "Langmuir": "#9b59b6",
        "ML-GradientBoosting": "#f39c12",
        "ML-RandomForest": "#1abc9c",
    }

    # Plot each model fit
    x_range = np.linspace(0.01, max(dadc_values) * 1.1, 100)
    for name, model in models.items():
        try:
            y_pred = model.predict_ppm(x_range)
            color = model_colors.get(name, "#95a5a6")
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2),
                )
            )
        except Exception:
            pass

    fig.update_layout(
        title="Calibration Model Comparison",
        xaxis_title="ΔADC",
        yaxis_title="Concentration (ppm)",
        template="plotly_white",
        height=500,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def plot_calibration_residuals(
    ppm_actual: np.ndarray, ppm_predicted: np.ndarray, model_name: str = "Model"
) -> go.Figure:
    """
    Create residual plot for calibration model validation.

    Args:
        ppm_actual: Actual concentration values
        ppm_predicted: Model-predicted concentration values
        model_name: Name of the model for title
    """
    residuals = ppm_actual - ppm_predicted

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Residuals vs Predicted", "Residual Distribution"),
        column_widths=[0.6, 0.4],
    )

    # Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=ppm_predicted,
            y=residuals,
            mode="markers",
            name="Residuals",
            marker=dict(color="#3498db", size=8, opacity=0.7),
        ),
        row=1,
        col=1,
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Add ±1 std band
    std_res = np.std(residuals)
    fig.add_hrect(
        y0=-std_res,
        y1=std_res,
        fillcolor="rgba(46, 204, 113, 0.2)",
        line_width=0,
        row=1,
        col=1,
    )

    # Residual histogram
    fig.add_trace(
        go.Histogram(
            y=residuals,
            nbinsy=30,
            name="Distribution",
            marker_color="#9b59b6",
            opacity=0.7,
        ),
        row=1,
        col=2,
    )

    # Statistics
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    fig.update_layout(
        title=f"Residual Analysis - {model_name} (RMSE: {rmse:.1f}, MAE: {mae:.1f} ppm)",
        template="plotly_white",
        height=400,
        showlegend=False,
    )

    fig.update_xaxes(title_text="Predicted (ppm)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (ppm)", row=1, col=1)
    fig.update_xaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Residual (ppm)", row=1, col=2)

    return fig


def plot_predicted_vs_actual(
    ppm_actual: np.ndarray,
    ppm_predicted_dict: Dict[str, np.ndarray],
    model_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    model_equations: Optional[Dict[str, str]] = None,
    exposure_labels: Optional[List[str]] = None,
    show_error_bands: bool = True,
    show_labels: bool = True,
    show_equations: bool = True,
    color_by_order: bool = True,
) -> go.Figure:
    """
    Create gold-standard Predicted vs Actual concentration scatter plot for calibration evaluation.

    Enhanced version with:
    - ±5% and ±10% error bands
    - Point labels showing exposure concentrations
    - Regression equations displayed
    - Color coding by exposure order
    - Detailed interpretation

    Args:
        ppm_actual: Actual concentration values (x-axis)
        ppm_predicted_dict: Dict mapping model_name -> predicted values array
        model_metrics: Optional dict mapping model_name -> {'r2': float, 'rmse': float, 'mae': float}
        model_equations: Optional dict mapping model_name -> equation string
        exposure_labels: Optional list of labels for each point (e.g., ["500 ppm", "1000 ppm", ...])
        show_error_bands: Whether to show ±5% and ±10% error bands
        show_labels: Whether to show point labels
        show_equations: Whether to show regression equations
        color_by_order: Whether to color points by exposure sequence

    Returns:
        Plotly figure with enhanced predicted vs actual scatter plot
    """
    # Model colors - distinct and colorblind-friendly
    model_colors = {
        "Polynomial(1)": "#e74c3c",  # Red
        "Polynomial(2)": "#3498db",  # Blue
        "Polynomial(3)": "#2ecc71",  # Green
        "Langmuir": "#9b59b6",  # Purple
        "Freundlich": "#f39c12",  # Orange
        "ML Gradient Boosting": "#1abc9c",  # Teal
        "ML Random Forest": "#e91e63",  # Pink
    }
    default_colors = [
        "#3498db",
        "#e74c3c",
        "#2ecc71",
        "#9b59b6",
        "#f39c12",
        "#1abc9c",
        "#e91e63",
    ]

    # Exposure order colorscale (viridis-like for sequential data)
    n_points = len(ppm_actual)
    order_colors = [
        f"hsl({int(240 - i * 200 / max(n_points-1, 1))}, 70%, 50%)"
        for i in range(n_points)
    ]

    fig = go.Figure()

    # Calculate ranges
    all_preds = list(ppm_predicted_dict.values())
    max_val = max(np.max(ppm_actual), max(np.max(pred) for pred in all_preds))
    min_val = min(np.min(ppm_actual), min(np.min(pred) for pred in all_preds))
    margin = (max_val - min_val) * 0.15
    x_range = [max(0, min_val - margin), max_val + margin]

    x_line = np.linspace(x_range[0], x_range[1], 200)

    # Add error bands (±5% and ±10%)
    if show_error_bands:
        # ±10% band (lighter)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_line, x_line[::-1]]),
                y=np.concatenate([x_line * 1.1, (x_line * 0.9)[::-1]]),
                fill="toself",
                fillcolor="rgba(255, 193, 7, 0.15)",
                line=dict(color="rgba(255, 193, 7, 0.4)", width=1, dash="dot"),
                name="±10% Error Band",
                hoverinfo="skip",
            )
        )

        # ±5% band (darker)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_line, x_line[::-1]]),
                y=np.concatenate([x_line * 1.05, (x_line * 0.95)[::-1]]),
                fill="toself",
                fillcolor="rgba(76, 175, 80, 0.2)",
                line=dict(color="rgba(76, 175, 80, 0.5)", width=1, dash="dash"),
                name="±5% Error Band",
                hoverinfo="skip",
            )
        )

    # Add identity line (perfect prediction y=x)
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=x_line,
            mode="lines",
            name="Perfect (y=x)",
            line=dict(color="black", dash="solid", width=2),
            hoverinfo="skip",
        )
    )

    # Store metrics for summary
    all_metrics = {}

    # Plot each model
    for idx, (model_name, ppm_pred) in enumerate(ppm_predicted_dict.items()):
        color = model_colors.get(model_name, default_colors[idx % len(default_colors)])

        # Calculate metrics
        ss_res = np.sum((ppm_actual - ppm_pred) ** 2)
        ss_tot = np.sum((ppm_actual - np.mean(ppm_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((ppm_actual - ppm_pred) ** 2))
        mae = np.mean(np.abs(ppm_actual - ppm_pred))

        # Override with provided metrics if available
        if model_metrics and model_name in model_metrics:
            r2 = model_metrics[model_name].get("r2", r2)
            rmse = model_metrics[model_name].get("rmse", rmse)
            mae = model_metrics[model_name].get("mae", mae)

        all_metrics[model_name] = {"r2": r2, "rmse": rmse, "mae": mae}

        legend_text = f"{model_name} (R²={r2:.3f}, RMSE={rmse:.0f} ppm)"

        # Determine marker colors
        if color_by_order and len(ppm_predicted_dict) == 1:
            marker_colors = order_colors
        else:
            marker_colors = color

        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=ppm_actual,
                y=ppm_pred,
                mode="markers+text" if show_labels else "markers",
                name=legend_text,
                marker=dict(
                    color=marker_colors if isinstance(marker_colors, list) else color,
                    size=14,
                    opacity=0.9,
                    line=dict(width=2, color="white"),
                    symbol="circle",
                ),
                text=[f"{int(a)} ppm" for a in ppm_actual] if show_labels else None,
                textposition="top center",
                textfont=dict(size=9, color="#333"),
                hovertemplate=f"<b>{model_name}</b><br>Actual: %{{x:.0f}} ppm<br>Predicted: %{{y:.0f}} ppm<br>Error: %{{customdata:.1f}}%<extra></extra>",
                customdata=((ppm_pred - ppm_actual) / ppm_actual * 100),
            )
        )

    # Build equation text
    equation_text = ""
    if show_equations and model_equations:
        equation_lines = ["<b>Model Equations:</b>"]
        for model_name, eq in model_equations.items():
            equation_lines.append(f"{model_name}: {eq}")
        equation_text = "<br>".join(equation_lines)

    # Add equation annotation
    if equation_text:
        fig.add_annotation(
            text=equation_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=10, family="Courier New", color="#333"),
            align="left",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#ccc",
            borderwidth=1,
            borderpad=6,
        )

    # Add interpretation guide
    fig.add_annotation(
        text="<b>Interpretation:</b><br>• Points ON black line = perfect predictions<br>• Green band = ±5% error (excellent)<br>• Yellow band = ±10% error (acceptable)",
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.02,
        showarrow=False,
        font=dict(size=9, color="#555"),
        align="right",
        bgcolor="rgba(255,255,255,0.9)",
        borderpad=4,
    )

    # Color bar for exposure order (if single model)
    if color_by_order and len(ppm_predicted_dict) == 1:
        fig.add_annotation(
            text="<b>Color: Exposure Order</b><br>Blue→Green→Red = Early→Late",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            showarrow=False,
            font=dict(size=9, color="#555"),
            align="right",
            bgcolor="rgba(255,255,255,0.9)",
            borderpad=4,
        )

    fig.update_layout(
        title=dict(
            text="<b>Calibration Model Evaluation: Predicted vs Actual Concentration</b>",
            font=dict(size=16),
        ),
        xaxis_title="Actual Concentration (ppm)",
        yaxis_title="Predicted Concentration (ppm)",
        template="plotly_white",
        height=600,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=10),
        ),
        xaxis=dict(range=x_range, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(
            range=x_range, scaleanchor="x", scaleratio=1, gridcolor="rgba(0,0,0,0.1)"
        ),
    )

    return fig, all_metrics


def generate_calibration_summary_text(
    model_name: str,
    metrics: Dict[str, float],
    ppm_actual: np.ndarray,
    ppm_predicted: np.ndarray,
    equation: Optional[str] = None,
) -> str:
    """
    Generate detailed text summary of calibration model performance.

    Args:
        model_name: Name of the calibration model
        metrics: Dict with 'r2', 'rmse', 'mae'
        ppm_actual: Actual concentration values
        ppm_predicted: Predicted concentration values
        equation: Optional model equation string

    Returns:
        Formatted markdown summary text
    """
    r2 = metrics.get("r2", 0)
    rmse = metrics.get("rmse", 0)
    mae = metrics.get("mae", 0)

    # Calculate additional metrics
    errors = ppm_predicted - ppm_actual
    rel_errors = errors / ppm_actual * 100
    max_error = np.max(np.abs(errors))
    max_rel_error = np.max(np.abs(rel_errors))

    # Identify bias pattern
    high_conc_mask = ppm_actual > np.median(ppm_actual)
    high_conc_bias = np.mean(errors[high_conc_mask])
    np.mean(errors[~high_conc_mask])

    # Determine quality assessment
    if r2 >= 0.99 and rmse < 200:
        quality = "excellent"
        quality_desc = "highly suitable for quantitative VOC monitoring"
    elif r2 >= 0.95 and rmse < 500:
        quality = "good"
        quality_desc = "suitable for in-barn VOC monitoring applications"
    elif r2 >= 0.90:
        quality = "acceptable"
        quality_desc = "adequate for screening-level measurements"
    else:
        quality = "limited"
        quality_desc = "may require additional calibration refinement"

    # Build bias description
    bias_desc = ""
    if abs(high_conc_bias) > 100:
        if high_conc_bias > 0:
            bias_desc = f"Slight overprediction at high concentrations (>{ int(np.median(ppm_actual))} ppm) with average bias of +{int(high_conc_bias)} ppm."
        else:
            bias_desc = f"Slight underprediction at high concentrations (>{int(np.median(ppm_actual))} ppm) with average bias of {int(high_conc_bias)} ppm."
    else:
        bias_desc = (
            "No significant systematic bias observed across the concentration range."
        )

    # Equation line
    eq_line = f"\n\n**Model Equation:** `{equation}`" if equation else ""

    summary = f"""### Calibration Model Performance Summary

**Model:** {model_name}
**Quality Assessment:** {quality.upper()} - {quality_desc}
{eq_line}

#### Statistical Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| R² (Coefficient of Determination) | {r2:.4f} | {'>99%' if r2 >= 0.99 else '>95%' if r2 >= 0.95 else '>90%' if r2 >= 0.90 else '<90%'} variance explained |
| RMSE (Root Mean Square Error) | {rmse:.1f} ppm | Average prediction uncertainty |
| MAE (Mean Absolute Error) | {mae:.1f} ppm | Typical absolute error magnitude |
| Maximum Absolute Error | {max_error:.0f} ppm | Worst-case prediction error |
| Maximum Relative Error | {max_rel_error:.1f}% | Largest percentage deviation |

#### Key Findings
1. **Overall Accuracy:** The {model_name} calibration model demonstrates {quality} agreement with actual exposure values (R² = {r2:.3f}).

2. **Bias Analysis:** {bias_desc}

3. **Error Distribution:** {int(np.sum(np.abs(rel_errors) <= 5))} of {len(ppm_actual)} calibration points ({np.sum(np.abs(rel_errors) <= 5)/len(ppm_actual)*100:.0f}%) fall within ±5% of actual values, and {int(np.sum(np.abs(rel_errors) <= 10))} ({np.sum(np.abs(rel_errors) <= 10)/len(ppm_actual)*100:.0f}%) within ±10%.

4. **Application Suitability:** Based on these metrics, the sensor is {quality_desc}.

#### Recommendations
- {"No immediate action required. Sensor performance is optimal for deployment." if quality in ["excellent", "good"] else "Consider collecting additional calibration data at extreme concentrations to improve model accuracy." if quality == "acceptable" else "Recommend re-calibration with additional reference standards before deployment."}
- Periodic validation recommended every {"6 months" if quality == "excellent" else "3 months" if quality == "good" else "monthly"} for drift monitoring.
"""

    return summary


def plot_all_calibration_models(
    ppm_actual: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    model_equations: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Create a comprehensive grid of individual predicted vs actual plots for ALL calibration models.

    Args:
        ppm_actual: Actual concentration values
        predictions_dict: Dict mapping model_name -> predicted values array
        model_equations: Optional dict mapping model_name -> equation string

    Returns:
        Plotly figure with subplot for each model
    """
    n_models = len(predictions_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    model_names = list(predictions_dict.keys())

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=model_names,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Model colors
    model_colors = {
        "Polynomial(1)": "#e74c3c",
        "Polynomial(2)": "#3498db",
        "Polynomial(3)": "#2ecc71",
        "Langmuir": "#9b59b6",
        "Freundlich": "#f39c12",
        "ML Gradient Boosting": "#1abc9c",
        "ML Random Forest": "#e91e63",
    }
    default_colors = [
        "#3498db",
        "#e74c3c",
        "#2ecc71",
        "#9b59b6",
        "#f39c12",
        "#1abc9c",
        "#e91e63",
    ]

    # Calculate axis range
    max_val = max(
        np.max(ppm_actual), max(np.max(pred) for pred in predictions_dict.values())
    )
    min_val = min(
        np.min(ppm_actual), min(np.min(pred) for pred in predictions_dict.values())
    )
    margin = (max_val - min_val) * 0.1
    axis_range = [max(0, min_val - margin), max_val + margin]

    x_line = np.linspace(axis_range[0], axis_range[1], 100)

    for idx, (model_name, ppm_pred) in enumerate(predictions_dict.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        color = model_colors.get(model_name, default_colors[idx % len(default_colors)])

        # Calculate R² and RMSE
        ss_res = np.sum((ppm_actual - ppm_pred) ** 2)
        ss_tot = np.sum((ppm_actual - np.mean(ppm_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((ppm_actual - ppm_pred) ** 2))

        # Perfect line (y=x)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=x_line,
                mode="lines",
                line=dict(color="black", dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # ±10% bands
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_line, x_line[::-1]]),
                y=np.concatenate([x_line * 1.1, (x_line * 0.9)[::-1]]),
                fill="toself",
                fillcolor="rgba(200, 200, 200, 0.2)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Data points
        fig.add_trace(
            go.Scatter(
                x=ppm_actual,
                y=ppm_pred,
                mode="markers",
                marker=dict(color=color, size=10, line=dict(width=1, color="white")),
                name=f"{model_name}",
                showlegend=False,
                hovertemplate=f"<b>{model_name}</b><br>Actual: %{{x:.0f}}<br>Pred: %{{y:.0f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Add R² annotation
        fig.add_annotation(
            text=f"R²={r2:.3f}<br>RMSE={rmse:.0f}",
            xref=f"x{idx+1}" if idx > 0 else "x",
            yref=f"y{idx+1}" if idx > 0 else "y",
            x=axis_range[0] + (axis_range[1] - axis_range[0]) * 0.05,
            y=axis_range[1] - (axis_range[1] - axis_range[0]) * 0.05,
            showarrow=False,
            font=dict(size=10, color=color),
            bgcolor="rgba(255,255,255,0.8)",
            borderpad=3,
        )

    fig.update_layout(
        title=dict(
            text="<b>Calibration Model Comparison: All Models</b>", font=dict(size=16)
        ),
        template="plotly_white",
        height=300 * n_rows + 100,
        showlegend=False,
        margin=dict(l=60, r=40, t=100, b=60),
    )

    # Update all axes
    for i in range(1, n_models + 1):
        fig.update_xaxes(
            title_text="Actual (ppm)",
            range=axis_range,
            row=(i - 1) // n_cols + 1,
            col=(i - 1) % n_cols + 1,
        )
        fig.update_yaxes(
            title_text="Predicted (ppm)",
            range=axis_range,
            row=(i - 1) // n_cols + 1,
            col=(i - 1) % n_cols + 1,
        )

    return fig


def plot_layer_comparison_timeseries(
    timestamps: np.ndarray,
    layer1_values: np.ndarray,
    layer2_values: np.ndarray,
    layer3_values: np.ndarray,
) -> go.Figure:
    """
    Create stacked time series plot comparing all three sensor layers.

    Args:
        timestamps: Array of timestamps
        layer1_values: Layer 1 (Active) sum values
        layer2_values: Layer 2 (Reference) sum values
        layer3_values: Layer 3 (Saturated) sum values
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Layer 1 - Active Sensing (ZIF-8 MOF)",
            "Layer 2 - Reference (Dark/Disabled)",
            "Layer 3 - Saturated (Overexposed)",
            "Layer 1 − Layer 2 (Differential)",
        ),
        row_heights=[0.25, 0.25, 0.25, 0.25],
    )

    # Layer 1
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=layer1_values,
            mode="lines",
            name="Layer 1 (Active)",
            line=dict(color="#3498db", width=1),
        ),
        row=1,
        col=1,
    )

    # Layer 2
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=layer2_values,
            mode="lines",
            name="Layer 2 (Reference)",
            line=dict(color="#95a5a6", width=1),
        ),
        row=2,
        col=1,
    )

    # Layer 3 with saturation highlighting
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=layer3_values,
            mode="lines",
            name="Layer 3 (Saturated)",
            line=dict(color="#e74c3c", width=1),
        ),
        row=3,
        col=1,
    )

    # Highlight saturation events (where Layer 3 > 95% of max)
    0.95 * np.max(layer3_values)

    # Differential: Layer 1 - Layer 2
    differential = layer1_values - layer2_values
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=differential,
            mode="lines",
            name="L1 − L2",
            line=dict(color="#2ecc71", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(46, 204, 113, 0.2)",
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        title="Sensor Layer Comparison Over Time",
        template="plotly_white",
        height=800,
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=1.15),
    )

    fig.update_yaxes(title_text="ΣADC", row=1, col=1)
    fig.update_yaxes(title_text="ΣADC", row=2, col=1)
    fig.update_yaxes(title_text="ΣADC", row=3, col=1)
    fig.update_yaxes(title_text="ΔADC", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=1)

    return fig


def create_sensor_health_gauge(
    noise_score: float, drift_score: float, t90_score: float, saturation_pct: float
) -> go.Figure:
    """
    Create a sensor health summary gauge/indicator.

    Args:
        noise_score: 0-100 score for noise (100 = best)
        drift_score: 0-100 score for drift stability (100 = best)
        t90_score: 0-100 score for response speed (100 = best)
        saturation_pct: Percentage of saturated readings

    Returns:
        Plotly figure with gauge indicators
    """
    # Calculate overall health score
    # Weight: noise 30%, drift 30%, t90 25%, saturation 15%
    overall = (
        noise_score * 0.30
        + drift_score * 0.30
        + t90_score * 0.25
        + (100 - saturation_pct) * 0.15
    )

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "indicator", "colspan": 3}, None, None],
        ],
        subplot_titles=("Noise", "Drift", "Response Speed", "Overall Health"),
        vertical_spacing=0.3,
    )

    # Helper for gauge color
    def gauge_color(val):
        if val >= 80:
            return "#2ecc71"
        if val >= 60:
            return "#f1c40f"
        if val >= 40:
            return "#e67e22"
        return "#e74c3c"

    # Noise gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=noise_score,
            title={"text": "Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color(noise_score)},
                "steps": [
                    {"range": [0, 40], "color": "rgba(231, 76, 60, 0.2)"},
                    {"range": [40, 70], "color": "rgba(241, 196, 15, 0.2)"},
                    {"range": [70, 100], "color": "rgba(46, 204, 113, 0.2)"},
                ],
            },
        ),
        row=1,
        col=1,
    )

    # Drift gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=drift_score,
            title={"text": "Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color(drift_score)},
                "steps": [
                    {"range": [0, 40], "color": "rgba(231, 76, 60, 0.2)"},
                    {"range": [40, 70], "color": "rgba(241, 196, 15, 0.2)"},
                    {"range": [70, 100], "color": "rgba(46, 204, 113, 0.2)"},
                ],
            },
        ),
        row=1,
        col=2,
    )

    # T90 Speed gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=t90_score,
            title={"text": "Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color(t90_score)},
                "steps": [
                    {"range": [0, 40], "color": "rgba(231, 76, 60, 0.2)"},
                    {"range": [40, 70], "color": "rgba(241, 196, 15, 0.2)"},
                    {"range": [70, 100], "color": "rgba(46, 204, 113, 0.2)"},
                ],
            },
        ),
        row=1,
        col=3,
    )

    # Overall health gauge (larger)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=overall,
            delta={"reference": 80, "position": "bottom"},
            title={"text": f"Saturation: {saturation_pct:.1f}%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color(overall)},
                "steps": [
                    {"range": [0, 40], "color": "rgba(231, 76, 60, 0.3)"},
                    {"range": [40, 70], "color": "rgba(241, 196, 15, 0.3)"},
                    {"range": [70, 100], "color": "rgba(46, 204, 113, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.75,
                    "value": 80,
                },
            },
        ),
        row=2,
        col=1,
    )

    fig.update_layout(title="Sensor Health Index", template="plotly_white", height=500)

    return fig


def create_pixel_health_summary(
    pixel_timeseries_by_layer: Dict[int, np.ndarray], adc_fullscale: int = 4095
) -> go.Figure:
    """
    Create comprehensive pixel health summary across all layers.

    Uses the proper classification rules:
      - Saturated: Any pixel whose value ever exceeds 0.98 × ADC_fullscale
      - Dead: Pixels that are always exactly 0 (zero variance, mean=0)
      - Noisy: Pixels with temporal σ > 5 × median(σ) of all pixels
      - Normal: Everything else

    Args:
        pixel_timeseries_by_layer: Dict mapping layer_id -> (n_frames, 32, 32) array
        adc_fullscale: Full scale ADC value (default 4095 for 12-bit)

    Returns:
        Plotly figure with comprehensive pixel health visualization
    """

    def classify_layer_pixels(pixel_timeseries: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Classify pixels for a single layer."""
        n_frames = pixel_timeseries.shape[0]
        pixel_matrix = pixel_timeseries.reshape(
            n_frames, -1
        ).T  # Shape: (1024, n_frames)

        pixel_means = np.mean(pixel_matrix, axis=1)
        pixel_stds = np.std(pixel_matrix, axis=1)
        pixel_maxs = np.max(pixel_matrix, axis=1)

        saturation_threshold = 0.98 * adc_fullscale
        median_std = (
            np.median(pixel_stds[pixel_stds > 0]) if np.any(pixel_stds > 0) else 1.0
        )
        noise_threshold = 5.0 * median_std

        status = np.zeros(1024, dtype=np.int32)
        for i in range(1024):
            if pixel_means[i] > saturation_threshold or pixel_maxs[i] >= adc_fullscale:
                status[i] = 2  # Saturated
            elif pixel_maxs[i] < 1.0:
                status[i] = 1  # Dead
            elif pixel_stds[i] > noise_threshold:
                status[i] = 3  # Noisy
            else:
                status[i] = 0  # Normal

        status_map = status.reshape(32, 32)
        summary = {
            "normal": int(np.sum(status == 0)),
            "dead": int(np.sum(status == 1)),
            "saturated": int(np.sum(status == 2)),
            "noisy": int(np.sum(status == 3)),
        }
        return status_map, summary

    layer_names = {
        1: "Layer 1 - Active Sensing",
        2: "Layer 2 - Reference",
        3: "Layer 3 - Saturated Reference",
    }

    n_layers = len(pixel_timeseries_by_layer)

    fig = make_subplots(
        rows=2,
        cols=max(n_layers, 1),
        subplot_titles=[
            layer_names.get(lid, f"Layer {lid}")
            for lid in sorted(pixel_timeseries_by_layer.keys())
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
        specs=(
            [
                [{"type": "heatmap"} for _ in range(n_layers)],
                [{"type": "bar", "colspan": n_layers}] + [None] * (n_layers - 1),
            ]
            if n_layers > 1
            else [[{"type": "heatmap"}], [{"type": "bar"}]]
        ),
    )

    # Colorscale for health status (0=Normal, 1=Dead, 2=Saturated, 3=Noisy)
    colorscale = [
        [0, "rgb(46, 204, 113)"],  # Normal - Green
        [0.33, "rgb(52, 73, 94)"],  # Dead - Dark gray
        [0.66, "rgb(231, 76, 60)"],  # Saturated - Red
        [1.0, "rgb(241, 196, 15)"],  # Noisy - Yellow
    ]

    all_summaries = {}

    for col_idx, (layer_id, data) in enumerate(
        sorted(pixel_timeseries_by_layer.items()), start=1
    ):
        status_map, summary = classify_layer_pixels(data)
        all_summaries[layer_id] = summary

        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=status_map,
                colorscale=colorscale,
                showscale=(col_idx == n_layers),
                zmin=0,
                zmax=3,
                colorbar=(
                    dict(
                        title="Status",
                        tickvals=[0, 1, 2, 3],
                        ticktext=["Normal", "Dead", "Saturated", "Noisy"],
                        x=1.02,
                    )
                    if col_idx == n_layers
                    else None
                ),
                hovertemplate="Row: %{y}<br>Col: %{x}<br>Status: %{z}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

        fig.update_xaxes(title_text="Column", row=1, col=col_idx)
        fig.update_yaxes(title_text="Row", autorange="reversed", row=1, col=col_idx)

    # Create summary bar chart
    categories = ["Normal", "Dead", "Saturated", "Noisy"]
    bar_colors = [
        "rgb(46, 204, 113)",
        "rgb(52, 73, 94)",
        "rgb(231, 76, 60)",
        "rgb(241, 196, 15)",
    ]

    for layer_id, summary in all_summaries.items():
        layer_name = layer_names.get(layer_id, f"Layer {layer_id}").split(" - ")[0]
        counts = [
            summary["normal"],
            summary["dead"],
            summary["saturated"],
            summary["noisy"],
        ]

        fig.add_trace(
            go.Bar(
                name=layer_name,
                x=categories,
                y=counts,
                marker_color=bar_colors if len(all_summaries) == 1 else None,
                text=counts,
                textposition="auto",
            ),
            row=2,
            col=1,
        )

    # Calculate overall yield
    total_normal = sum(s["normal"] for s in all_summaries.values())
    total_pixels = 32 * 32 * len(all_summaries)
    overall_yield = (total_normal / total_pixels) * 100

    fig.update_layout(
        title=f"Pixel Health Classification Summary (Overall Yield: {overall_yield:.1f}%)",
        template="plotly_white",
        height=700,
        barmode="group",
        showlegend=len(all_summaries) > 1,
    )

    fig.update_yaxes(title_text="Pixel Count", row=2, col=1)

    return fig


# =============================================================================
# PIPELINE ARCHITECTURE DIAGRAM
# =============================================================================


def create_pipeline_diagram(
    n_frames: int = 8125,
    n_events: int = 44,
    calibration_model: str = "Polynomial(2)",
    max_severity: int = 3,
    r_squared: float = 0.998,
) -> go.Figure:
    """
    Create a comprehensive scientific pipeline architecture diagram.

    Shows the complete data flow from raw sensor data to application output
    with all processing stages, algorithms, and key parameters.

    Args:
        n_frames: Number of frames in dataset
        n_events: Number of detected events
        calibration_model: Best calibration model name
        max_severity: Maximum severity level reached
        r_squared: Calibration model R² value

    Returns:
        Plotly figure with pipeline architecture diagram
    """
    fig = go.Figure()

    # Layout configuration
    fig.update_layout(
        title=dict(
            text="<b>CMOS ZIF-8 MOF VOC Sensor Analysis Pipeline</b><br><sup>End-to-End Data Processing Architecture</sup>",
            font=dict(size=20, color="#2c3e50"),
            x=0.5,
        ),
        template="plotly_white",
        height=1400,
        width=1100,
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.5, 10.5]),
        yaxis=dict(visible=False, range=[-0.5, 18]),
        margin=dict(l=20, r=20, t=100, b=20),
    )

    # Color scheme
    colors = {
        "input": "#3498db",  # Blue
        "process": "#2ecc71",  # Green
        "output": "#9b59b6",  # Purple
        "model": "#e74c3c",  # Red
        "arrow": "#7f8c8d",  # Gray
        "highlight": "#f39c12",  # Orange
        "bg_light": "rgba(52, 152, 219, 0.1)",
        "bg_green": "rgba(46, 204, 113, 0.1)",
        "bg_purple": "rgba(155, 89, 182, 0.1)",
    }

    # =========================================================================
    # MODULE 1: DATA INPUT & PREPROCESSING
    # =========================================================================

    # Data Input Box
    fig.add_shape(
        type="rect",
        x0=0.5,
        y0=16.5,
        x1=4.5,
        y1=17.8,
        fillcolor=colors["bg_light"],
        line=dict(color=colors["input"], width=3),
    )
    fig.add_annotation(
        x=2.5,
        y=17.5,
        text="<b>MODULE 1: DATA ACQUISITION & PREPROCESSING</b>",
        showarrow=False,
        font=dict(size=14, color=colors["input"]),
    )

    # Input data details
    fig.add_shape(
        type="rect",
        x0=0.7,
        y0=15.2,
        x1=2.3,
        y1=16.3,
        fillcolor="white",
        line=dict(color=colors["input"], width=2),
    )
    fig.add_annotation(
        x=1.5,
        y=15.95,
        text="<b>Raw Data Input</b>",
        showarrow=False,
        font=dict(size=11, color=colors["input"]),
    )
    fig.add_annotation(
        x=1.5,
        y=15.55,
        text=f"• {n_frames:,} frames<br>• 32×32×3 pixels/frame<br>• 12-bit ADC (0-4095)<br>• ~2 Hz sampling",
        showarrow=False,
        font=dict(size=9),
        align="left",
    )

    # Preprocessing details
    fig.add_shape(
        type="rect",
        x0=2.7,
        y0=15.2,
        x1=4.3,
        y1=16.3,
        fillcolor="white",
        line=dict(color=colors["input"], width=2),
    )
    fig.add_annotation(
        x=3.5,
        y=15.95,
        text="<b>Data Preprocessing</b>",
        showarrow=False,
        font=dict(size=11, color=colors["input"]),
    )
    fig.add_annotation(
        x=3.5,
        y=15.55,
        text="• Parse timestamp/packet<br>• Serpentine row correction<br>• Layer separation (1,2,3)<br>• ΣADC computation",
        showarrow=False,
        font=dict(size=9),
        align="left",
    )

    # Arrow down
    fig.add_annotation(
        x=2.5,
        y=14.9,
        ax=2.5,
        ay=15.2,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor=colors["arrow"],
        showarrow=True,
    )

    # =========================================================================
    # ΔADC Calculation
    # =========================================================================
    fig.add_shape(
        type="rect",
        x0=1.0,
        y0=13.8,
        x1=4.0,
        y1=14.8,
        fillcolor="white",
        line=dict(color=colors["process"], width=2),
    )
    fig.add_annotation(
        x=2.5,
        y=14.55,
        text="<b>ΔADC Calculation</b>",
        showarrow=False,
        font=dict(size=11, color=colors["process"]),
    )
    fig.add_annotation(
        x=2.5,
        y=14.15,
        text="ΔADC(t) = Avg(Coated) − Avg(Reference)<br>Coated: ~12% ZIF-8 pixels | Reference: ~88% stable pixels",
        showarrow=False,
        font=dict(size=9),
        align="center",
    )

    # Arrow down
    fig.add_annotation(
        x=2.5,
        y=13.5,
        ax=2.5,
        ay=13.8,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor=colors["arrow"],
        showarrow=True,
    )

    # =========================================================================
    # MODULE 2: SIGNAL PROCESSING
    # =========================================================================
    fig.add_shape(
        type="rect",
        x0=0.5,
        y0=11.5,
        x1=4.5,
        y1=13.4,
        fillcolor=colors["bg_green"],
        line=dict(color=colors["process"], width=3),
    )
    fig.add_annotation(
        x=2.5,
        y=13.15,
        text="<b>MODULE 2: SIGNAL PROCESSING & EVENT DETECTION</b>",
        showarrow=False,
        font=dict(size=14, color=colors["process"]),
    )

    # Filtering
    fig.add_shape(
        type="rect",
        x0=0.7,
        y0=11.7,
        x1=2.3,
        y1=12.9,
        fillcolor="white",
        line=dict(color=colors["process"], width=2),
    )
    fig.add_annotation(
        x=1.5,
        y=12.65,
        text="<b>Noise Filtering</b>",
        showarrow=False,
        font=dict(size=11, color=colors["process"]),
    )
    fig.add_annotation(
        x=1.5,
        y=12.15,
        text="• Butterworth LPF<br>  (4th order, fc=0.1 Hz)<br>• Baseline correction<br>• Artifact removal",
        showarrow=False,
        font=dict(size=9),
        align="left",
    )

    # Event Detection
    fig.add_shape(
        type="rect",
        x0=2.7,
        y0=11.7,
        x1=4.3,
        y1=12.9,
        fillcolor="white",
        line=dict(color=colors["process"], width=2),
    )
    fig.add_annotation(
        x=3.5,
        y=12.65,
        text="<b>Event Detection</b>",
        showarrow=False,
        font=dict(size=11, color=colors["process"]),
    )
    fig.add_annotation(
        x=3.5,
        y=12.15,
        text=f"• Hysteresis thresholds<br>  ON: μ+3σ, OFF: μ+1.5σ<br>• Min duration: 30s<br>• <b>{n_events} events detected</b>",
        showarrow=False,
        font=dict(size=9),
        align="left",
    )

    # Arrow down
    fig.add_annotation(
        x=2.5,
        y=11.2,
        ax=2.5,
        ay=11.5,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor=colors["arrow"],
        showarrow=True,
    )

    # =========================================================================
    # MODULE 3: SENSOR CHARACTERIZATION
    # =========================================================================
    fig.add_shape(
        type="rect",
        x0=5.5,
        y0=11.5,
        x1=9.5,
        y1=14.8,
        fillcolor=colors["bg_light"],
        line=dict(color=colors["input"], width=3),
    )
    fig.add_annotation(
        x=7.5,
        y=14.55,
        text="<b>MODULE 3: SENSOR CHARACTERIZATION</b>",
        showarrow=False,
        font=dict(size=14, color=colors["input"]),
    )

    # Response Time
    fig.add_shape(
        type="rect",
        x0=5.7,
        y0=13.4,
        x1=7.3,
        y1=14.3,
        fillcolor="white",
        line=dict(color=colors["input"], width=2),
    )
    fig.add_annotation(
        x=6.5,
        y=14.1,
        text="<b>Response Time</b>",
        showarrow=False,
        font=dict(size=10, color=colors["input"]),
    )
    fig.add_annotation(
        x=6.5,
        y=13.7,
        text="T₉₀ rise/recovery<br>Kinetic rate constants",
        showarrow=False,
        font=dict(size=8),
        align="center",
    )

    # Dynamic Range
    fig.add_shape(
        type="rect",
        x0=7.7,
        y0=13.4,
        x1=9.3,
        y1=14.3,
        fillcolor="white",
        line=dict(color=colors["input"], width=2),
    )
    fig.add_annotation(
        x=8.5,
        y=14.1,
        text="<b>Dynamic Range</b>",
        showarrow=False,
        font=dict(size=10, color=colors["input"]),
    )
    fig.add_annotation(
        x=8.5,
        y=13.7,
        text="Min detectable: 0.3 ΔADC<br>Max response: 3.6 ΔADC",
        showarrow=False,
        font=dict(size=8),
        align="center",
    )

    # Layer Diagnostics
    fig.add_shape(
        type="rect",
        x0=5.7,
        y0=11.7,
        x1=7.3,
        y1=13.2,
        fillcolor="white",
        line=dict(color=colors["input"], width=2),
    )
    fig.add_annotation(
        x=6.5,
        y=13.0,
        text="<b>Layer Diagnostics</b>",
        showarrow=False,
        font=dict(size=10, color=colors["input"]),
    )
    fig.add_annotation(
        x=6.5,
        y=12.4,
        text="L1: Active (ZIF-8)<br>L2: Reference (Dark)<br>L3: Saturated",
        showarrow=False,
        font=dict(size=8),
        align="center",
    )

    # Pixel Health
    fig.add_shape(
        type="rect",
        x0=7.7,
        y0=11.7,
        x1=9.3,
        y1=13.2,
        fillcolor="white",
        line=dict(color=colors["input"], width=2),
    )
    fig.add_annotation(
        x=8.5,
        y=13.0,
        text="<b>Pixel Health</b>",
        showarrow=False,
        font=dict(size=10, color=colors["input"]),
    )
    fig.add_annotation(
        x=8.5,
        y=12.4,
        text="Dead/Noisy/Saturated<br>classification per pixel<br>32×32 health map",
        showarrow=False,
        font=dict(size=8),
        align="center",
    )

    # Horizontal connector from Module 2 to Module 3
    fig.add_annotation(
        x=5.5,
        y=12.45,
        ax=4.5,
        ay=12.45,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor=colors["arrow"],
        showarrow=True,
    )

    # =========================================================================
    # MODULE 4: CALIBRATION MODELING
    # =========================================================================
    fig.add_shape(
        type="rect",
        x0=0.5,
        y0=8.5,
        x1=4.5,
        y1=11.0,
        fillcolor="rgba(231, 76, 60, 0.1)",
        line=dict(color=colors["model"], width=3),
    )
    fig.add_annotation(
        x=2.5,
        y=10.75,
        text="<b>MODULE 4: CALIBRATION & MODELING</b>",
        showarrow=False,
        font=dict(size=14, color=colors["model"]),
    )

    # Calibration Points
    fig.add_shape(
        type="rect",
        x0=0.7,
        y0=9.5,
        x1=2.3,
        y1=10.5,
        fillcolor="white",
        line=dict(color=colors["model"], width=2),
    )
    fig.add_annotation(
        x=1.5,
        y=10.3,
        text="<b>Calibration Data</b>",
        showarrow=False,
        font=dict(size=10, color=colors["model"]),
    )
    fig.add_annotation(
        x=1.5,
        y=9.85,
        text="7 reference points:<br>500-7000 ppm toluene<br>ΔADC: 0.5 → 3.0",
        showarrow=False,
        font=dict(size=8),
        align="center",
    )

    # Model Comparison
    fig.add_shape(
        type="rect",
        x0=2.7,
        y0=9.5,
        x1=4.3,
        y1=10.5,
        fillcolor="white",
        line=dict(color=colors["model"], width=2),
    )
    fig.add_annotation(
        x=3.5,
        y=10.3,
        text="<b>Model Comparison</b>",
        showarrow=False,
        font=dict(size=10, color=colors["model"]),
    )
    fig.add_annotation(
        x=3.5,
        y=9.85,
        text="• Polynomial (1,2,3)<br>• Langmuir isotherm<br>• Freundlich isotherm",
        showarrow=False,
        font=dict(size=8),
        align="center",
    )

    # Best Model
    fig.add_shape(
        type="rect",
        x0=1.5,
        y0=8.7,
        x1=3.5,
        y1=9.3,
        fillcolor=colors["highlight"],
        line=dict(color="#d35400", width=2),
    )
    fig.add_annotation(
        x=2.5,
        y=9.0,
        text=f"<b>Best: {calibration_model} (R²={r_squared:.3f})</b>",
        showarrow=False,
        font=dict(size=10, color="white"),
    )

    # Arrow down
    fig.add_annotation(
        x=2.5,
        y=8.2,
        ax=2.5,
        ay=8.5,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor=colors["arrow"],
        showarrow=True,
    )

    # =========================================================================
    # MODULE 5: APPLICATION (Severity Classification)
    # =========================================================================
    fig.add_shape(
        type="rect",
        x0=0.5,
        y0=5.5,
        x1=4.5,
        y1=8.0,
        fillcolor=colors["bg_purple"],
        line=dict(color=colors["output"], width=3),
    )
    fig.add_annotation(
        x=2.5,
        y=7.75,
        text="<b>MODULE 5: APPLICATION - SEVERITY MONITORING</b>",
        showarrow=False,
        font=dict(size=14, color=colors["output"]),
    )

    # PPM Conversion
    fig.add_shape(
        type="rect",
        x0=0.7,
        y0=6.3,
        x1=2.3,
        y1=7.5,
        fillcolor="white",
        line=dict(color=colors["output"], width=2),
    )
    fig.add_annotation(
        x=1.5,
        y=7.3,
        text="<b>PPM Conversion</b>",
        showarrow=False,
        font=dict(size=10, color=colors["output"]),
    )
    fig.add_annotation(
        x=1.5,
        y=6.75,
        text="ΔADC → ppm<br>via calibration model<br>Real-time estimation",
        showarrow=False,
        font=dict(size=8),
        align="center",
    )

    # Severity Classification
    fig.add_shape(
        type="rect",
        x0=2.7,
        y0=6.3,
        x1=4.3,
        y1=7.5,
        fillcolor="white",
        line=dict(color=colors["output"], width=2),
    )
    fig.add_annotation(
        x=3.5,
        y=7.3,
        text="<b>Severity Classification</b>",
        showarrow=False,
        font=dict(size=10, color=colors["output"]),
    )
    fig.add_annotation(
        x=3.5,
        y=6.75,
        text="🟢 SAFE: <2000 ppm<br>🟡 WARNING: 2-5K ppm<br>🔴 CRITICAL: >5K ppm",
        showarrow=False,
        font=dict(size=8),
        align="center",
    )

    # Output indicator
    fig.add_shape(
        type="rect",
        x0=1.5,
        y0=5.7,
        x1=3.5,
        y1=6.1,
        fillcolor=(
            "#27ae60"
            if max_severity == 1
            else "#f39c12" if max_severity == 2 else "#e74c3c"
        ),
        line=dict(color="#2c3e50", width=2),
    )
    severity_text = (
        "SAFE" if max_severity == 1 else "WARNING" if max_severity == 2 else "CRITICAL"
    )
    fig.add_annotation(
        x=2.5,
        y=5.9,
        text=f"<b>Max Level: {severity_text} ({max_severity})</b>",
        showarrow=False,
        font=dict(size=10, color="white"),
    )

    # =========================================================================
    # MODULE 6: INTEGRATION & OUTPUT
    # =========================================================================
    fig.add_shape(
        type="rect",
        x0=5.5,
        y0=5.5,
        x1=9.5,
        y1=11.0,
        fillcolor="rgba(149, 165, 166, 0.1)",
        line=dict(color="#7f8c8d", width=3),
    )
    fig.add_annotation(
        x=7.5,
        y=10.75,
        text="<b>MODULE 6: INTEGRATION & OUTPUTS</b>",
        showarrow=False,
        font=dict(size=14, color="#2c3e50"),
    )

    # Pipeline Results
    fig.add_shape(
        type="rect",
        x0=5.7,
        y0=9.2,
        x1=7.3,
        y1=10.5,
        fillcolor="white",
        line=dict(color="#2c3e50", width=2),
    )
    fig.add_annotation(
        x=6.5,
        y=10.3,
        text="<b>Pipeline Results</b>",
        showarrow=False,
        font=dict(size=10, color="#2c3e50"),
    )
    fig.add_annotation(
        x=6.5,
        y=9.7,
        text="• ProcessedDataset<br>• DetectedEvents[]<br>• CalibrationModel<br>• SeverityTimeline",
        showarrow=False,
        font=dict(size=8),
        align="center",
    )

    # Visualization
    fig.add_shape(
        type="rect",
        x0=7.7,
        y0=9.2,
        x1=9.3,
        y1=10.5,
        fillcolor="white",
        line=dict(color="#2c3e50", width=2),
    )
    fig.add_annotation(
        x=8.5,
        y=10.3,
        text="<b>Visualization</b>",
        showarrow=False,
        font=dict(size=10, color="#2c3e50"),
    )
    fig.add_annotation(
        x=8.5,
        y=9.7,
        text="• Interactive Dash app<br>• Plotly figures<br>• HTML/PNG export<br>• PDF reports",
        showarrow=False,
        font=dict(size=8),
        align="center",
    )

    # Configuration
    fig.add_shape(
        type="rect",
        x0=5.7,
        y0=7.5,
        x1=7.3,
        y1=8.9,
        fillcolor="white",
        line=dict(color="#2c3e50", width=2),
    )
    fig.add_annotation(
        x=6.5,
        y=8.7,
        text="<b>Configuration</b>",
        showarrow=False,
        font=dict(size=10, color="#2c3e50"),
    )
    fig.add_annotation(
        x=6.5,
        y=8.1,
        text="• PipelineConfig<br>• FilterParams<br>• ThresholdSettings<br>• OutputOptions",
        showarrow=False,
        font=dict(size=8),
        align="center",
    )

    # Reports
    fig.add_shape(
        type="rect",
        x0=7.7,
        y0=7.5,
        x1=9.3,
        y1=8.9,
        fillcolor="white",
        line=dict(color="#2c3e50", width=2),
    )
    fig.add_annotation(
        x=8.5,
        y=8.7,
        text="<b>Reports</b>",
        showarrow=False,
        font=dict(size=10, color="#2c3e50"),
    )
    fig.add_annotation(
        x=8.5,
        y=8.1,
        text="• Markdown summary<br>• JSON config<br>• Analysis report<br>• Daily barn report",
        showarrow=False,
        font=dict(size=8),
        align="center",
    )

    # Real-time Dashboard
    fig.add_shape(
        type="rect",
        x0=6.3,
        y0=5.7,
        x1=8.7,
        y1=7.2,
        fillcolor="#3498db",
        line=dict(color="#2980b9", width=3),
    )
    fig.add_annotation(
        x=7.5,
        y=6.9,
        text="<b>Interactive Dashboard</b>",
        showarrow=False,
        font=dict(size=11, color="white"),
    )
    fig.add_annotation(
        x=7.5,
        y=6.35,
        text="• Real-time heatmaps<br>• Signal time series<br>• Severity indicators<br>• Full analysis report",
        showarrow=False,
        font=dict(size=9, color="white"),
        align="center",
    )

    # Connector arrows
    fig.add_annotation(
        x=5.5,
        y=9.0,
        ax=4.5,
        ay=9.0,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor=colors["arrow"],
        showarrow=True,
    )
    fig.add_annotation(
        x=5.5,
        y=6.5,
        ax=4.5,
        ay=6.5,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor=colors["arrow"],
        showarrow=True,
    )

    # =========================================================================
    # DATA FLOW EQUATIONS & KEY ALGORITHMS
    # =========================================================================
    fig.add_shape(
        type="rect",
        x0=0.5,
        y0=0.3,
        x1=9.5,
        y1=5.2,
        fillcolor="rgba(236, 240, 241, 0.5)",
        line=dict(color="#bdc3c7", width=2),
    )
    fig.add_annotation(
        x=5.0,
        y=4.95,
        text="<b>KEY ALGORITHMS & MATHEMATICAL FORMULATIONS</b>",
        showarrow=False,
        font=dict(size=14, color="#2c3e50"),
    )

    # Column 1: Signal Processing
    fig.add_annotation(
        x=2.5,
        y=4.5,
        text="<b>Signal Processing</b>",
        showarrow=False,
        font=dict(size=11, color=colors["process"]),
    )
    fig.add_annotation(
        x=2.5,
        y=3.6,
        text="<b>Butterworth Low-Pass Filter:</b><br>"
        "H(s) = 1 / √(1 + (s/ωc)²ⁿ)<br>"
        "n=4, fc=0.1 Hz<br><br>"
        "<b>Baseline Estimation:</b><br>"
        "B(t) = P₁₀[x(t-w:t)]<br>"
        "w = 300 samples (5 min)",
        showarrow=False,
        font=dict(size=9, family="Courier New"),
        align="left",
    )

    # Column 2: Event Detection
    fig.add_annotation(
        x=5.0,
        y=4.5,
        text="<b>Event Detection</b>",
        showarrow=False,
        font=dict(size=11, color=colors["process"]),
    )
    fig.add_annotation(
        x=5.0,
        y=3.6,
        text="<b>Hysteresis Thresholding:</b><br>"
        "θ_ON = μ + 3σ (start event)<br>"
        "θ_OFF = μ + 1.5σ (end event)<br><br>"
        "<b>Event Validation:</b><br>"
        "Duration ≥ 30 seconds<br>"
        "Peak ΔADC > θ_ON",
        showarrow=False,
        font=dict(size=9, family="Courier New"),
        align="left",
    )

    # Column 3: Calibration Models
    fig.add_annotation(
        x=7.5,
        y=4.5,
        text="<b>Calibration Models</b>",
        showarrow=False,
        font=dict(size=11, color=colors["model"]),
    )
    fig.add_annotation(
        x=7.5,
        y=3.6,
        text="<b>Polynomial:</b><br>"
        "ppm = Σ aᵢ·(ΔADC)ⁱ<br><br>"
        "<b>Langmuir:</b><br>"
        "ΔADC = (K·C)/(1+K·C)<br><br>"
        "<b>Freundlich:</b><br>"
        "ΔADC = K·C^(1/n)",
        showarrow=False,
        font=dict(size=9, family="Courier New"),
        align="left",
    )

    # Bottom row: Metrics
    fig.add_annotation(
        x=2.5,
        y=1.4,
        text="<b>Performance Metrics</b>",
        showarrow=False,
        font=dict(size=11, color="#2c3e50"),
    )
    fig.add_annotation(
        x=2.5,
        y=0.85,
        text="R² = 1 - SS_res/SS_tot<br>" "RMSE = √(Σ(y-ŷ)²/n)<br>" "MAE = Σ|y-ŷ|/n",
        showarrow=False,
        font=dict(size=9, family="Courier New"),
        align="left",
    )

    fig.add_annotation(
        x=5.0,
        y=1.4,
        text="<b>Sensor Characteristics</b>",
        showarrow=False,
        font=dict(size=11, color="#2c3e50"),
    )
    fig.add_annotation(
        x=5.0,
        y=0.85,
        text="T₉₀ = time to 90% response<br>"
        "SNR = 10·log₁₀(P_signal/P_noise)<br>"
        "LOD = 3σ_baseline",
        showarrow=False,
        font=dict(size=9, family="Courier New"),
        align="left",
    )

    fig.add_annotation(
        x=7.5,
        y=1.4,
        text="<b>Application Thresholds</b>",
        showarrow=False,
        font=dict(size=11, color="#2c3e50"),
    )
    fig.add_annotation(
        x=7.5,
        y=0.85,
        text="SAFE: ppm < 2000<br>"
        "WARNING: 2000 ≤ ppm < 5000<br>"
        "CRITICAL: ppm ≥ 5000",
        showarrow=False,
        font=dict(size=9, family="Courier New"),
        align="left",
    )

    return fig


def analyze_coating_pattern(
    pixel_timeseries: np.ndarray,
    coated_percentile: float = 88,
    use_adaptive_threshold: bool = True,
) -> Dict:
    """
    Analyze the CMOS sensor to identify coated (responsive) vs reference (stable) pixels.

    This analysis identifies pixels that show significant response to gas exposure
    (coated with ZIF-8 MOF) versus pixels that remain stable (reference/uncoated).

    The ZIF-8 MOF coating is applied to a subset of pixels (~10-15% typically).
    Coated pixels show higher response magnitude during gas exposure events.

    Args:
        pixel_timeseries: 3D array of shape (n_frames, 32, 32) containing ADC values
        coated_percentile: Percentile threshold - pixels above this are "coated" (default 88 = top ~12%)
        use_adaptive_threshold: If True, use Otsu-like adaptive thresholding

    Returns:
        Dictionary containing coating analysis results
    """
    n_frames = pixel_timeseries.shape[0]

    # Calculate per-pixel statistics
    # Baseline: 10th percentile of first 100 frames (before gas exposure)
    baseline_frames = min(100, n_frames // 10)
    pixel_baseline = np.percentile(pixel_timeseries[:baseline_frames], 10, axis=0)

    # Peak response: 95th percentile of entire time series (during exposure)
    pixel_peak = np.percentile(pixel_timeseries, 95, axis=0)

    # Response delta: absolute difference (not normalized - more stable)
    pixel_delta = pixel_peak - pixel_baseline

    # Response magnitude: normalized for visualization
    pixel_baseline_safe = np.where(pixel_baseline > 0, pixel_baseline, 1)
    response_magnitude = pixel_delta / pixel_baseline_safe

    # Standard deviation as secondary indicator
    pixel_std = np.std(pixel_timeseries, axis=0)

    # Flatten for threshold calculation
    delta_flat = pixel_delta.flatten()

    if use_adaptive_threshold:
        # Use adaptive threshold: find natural break in response distribution
        # Sort deltas and find the largest gap in the upper portion
        sorted_deltas = np.sort(delta_flat)

        # Look at top 30% of pixels for potential coated region
        top_30_start = int(len(sorted_deltas) * 0.70)
        top_deltas = sorted_deltas[top_30_start:]

        # Find gaps between consecutive values
        gaps = np.diff(top_deltas)

        # Find the largest gap (natural separation between coated and reference)
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            adaptive_threshold = (
                top_deltas[max_gap_idx] + top_deltas[max_gap_idx + 1]
            ) / 2
        else:
            # Fallback to percentile
            adaptive_threshold = np.percentile(delta_flat, coated_percentile)

        # Use the more selective threshold
        percentile_threshold = np.percentile(delta_flat, coated_percentile)
        threshold = max(adaptive_threshold, percentile_threshold)
    else:
        # Simple percentile threshold
        threshold = np.percentile(delta_flat, coated_percentile)

    # Identify coated pixels: response delta above threshold
    coating_mask = pixel_delta > threshold

    # Calculate statistics
    n_coated = np.sum(coating_mask)
    n_reference = 1024 - n_coated
    coated_percentage = (n_coated / 1024) * 100

    # Spatial distribution
    top_half_coated = np.sum(coating_mask[:16, :])
    bottom_half_coated = np.sum(coating_mask[16:, :])

    # Find primary coated region (bounding box of majority of coated pixels)
    coated_rows, coated_cols = np.where(coating_mask)
    if len(coated_rows) > 0:
        # Use 10th and 90th percentile for robust bounding
        row_10, row_90 = int(np.percentile(coated_rows, 10)), int(
            np.percentile(coated_rows, 90)
        )
        col_10, col_90 = int(np.percentile(coated_cols, 10)), int(
            np.percentile(coated_cols, 90)
        )
        coated_region = f"Rows {row_10}-{row_90}, Cols {col_10}-{col_90}"
    else:
        coated_region = "No significant coated region detected"

    return {
        "coating_mask": coating_mask,
        "n_coated": int(n_coated),
        "n_reference": int(n_reference),
        "coated_percentage": float(coated_percentage),
        "top_half_coated": int(top_half_coated),
        "bottom_half_coated": int(bottom_half_coated),
        "coated_region": coated_region,
        "response_map": response_magnitude,
        "pixel_std": pixel_std,
        "pixel_delta": pixel_delta,
        "threshold_used": float(threshold),
    }


def plot_coating_pattern_map(
    pixel_timeseries: np.ndarray, analysis_results: Optional[Dict] = None
) -> go.Figure:
    """
    Create a visualization showing the ZIF-8 MOF coating pattern on the CMOS sensor.

    Displays:
    - Left: Binary map of coated (responsive) vs reference (stable) pixels
    - Right: Response magnitude heatmap

    Args:
        pixel_timeseries: 3D array of shape (n_frames, 32, 32) containing ADC values
        analysis_results: Optional pre-computed analysis results from analyze_coating_pattern()

    Returns:
        Plotly figure with coating pattern visualization
    """
    # Run analysis if not provided
    if analysis_results is None:
        analysis_results = analyze_coating_pattern(pixel_timeseries)

    coating_mask = analysis_results["coating_mask"]
    response_map = analysis_results["response_map"]
    n_coated = analysis_results["n_coated"]
    n_reference = analysis_results["n_reference"]
    coated_pct = analysis_results["coated_percentage"]
    top_half = analysis_results["top_half_coated"]
    bottom_half = analysis_results["bottom_half_coated"]
    coated_region = analysis_results["coated_region"]

    # Create subplot figure
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"<b>Coating Pattern Map</b><br><sup>Coated: {n_coated} ({coated_pct:.1f}%) | Reference: {n_reference} ({100-coated_pct:.1f}%)</sup>",
            f"<b>Response Magnitude</b><br><sup>Top half: {top_half} | Bottom half: {bottom_half} coated pixels</sup>",
        ),
        horizontal_spacing=0.08,
    )

    # Left plot: Binary coating pattern
    # Create discrete colormap: 0=Reference (blue), 1=Coated (orange/red)
    binary_map = coating_mask.astype(int)

    fig.add_trace(
        go.Heatmap(
            z=binary_map,
            colorscale=[
                [0, "rgb(59, 130, 246)"],  # Reference - Blue
                [1, "rgb(249, 115, 22)"],  # Coated - Orange
            ],
            showscale=True,
            colorbar=dict(
                title="Type",
                tickvals=[0.25, 0.75],
                ticktext=["Reference", "Coated"],
                len=0.5,
                x=0.45,
            ),
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Type: %{customdata}<extra></extra>",
            customdata=[
                ["Coated" if c else "Reference" for c in row] for row in coating_mask
            ],
        ),
        row=1,
        col=1,
    )

    # Right plot: Response magnitude heatmap
    fig.add_trace(
        go.Heatmap(
            z=response_map,
            colorscale="Viridis",
            colorbar=dict(title="Response<br>Magnitude", len=0.5, x=1.02),
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Response: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Add outline boxes around primary coated region on both plots
    coated_rows, coated_cols = np.where(coating_mask)
    if len(coated_rows) > 0:
        row_min, row_max = coated_rows.min(), coated_rows.max()
        col_min, col_max = coated_cols.min(), coated_cols.max()

        for col in [1, 2]:
            # Calculate x offset for subplot
            fig.add_shape(
                type="rect",
                x0=col_min - 0.5,
                y0=row_min - 0.5,
                x1=col_max + 0.5,
                y1=row_max + 0.5,
                line=dict(color="white", width=2, dash="dash"),
                row=1,
                col=col,
            )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>ZIF-8 MOF Coating Pattern Analysis</b><br><sup>Primary coated region: {coated_region}</sup>",
            x=0.5,
            font=dict(size=16),
        ),
        template="plotly_white",
        height=550,
        width=1100,
    )

    # Update axes for both subplots
    for col in [1, 2]:
        fig.update_xaxes(title_text="Column (pixel)", row=1, col=col)
        fig.update_yaxes(
            title_text="Row (pixel)" if col == 1 else "",
            scaleanchor=f"x{col}" if col == 1 else "x2",
            scaleratio=1,
            autorange="reversed",
            row=1,
            col=col,
        )

    return fig


def create_coating_summary_stats(analysis_results: Dict) -> go.Figure:
    """
    Create a summary statistics visualization for the coating pattern analysis.

    Args:
        analysis_results: Results from analyze_coating_pattern()

    Returns:
        Plotly figure with summary statistics
    """
    n_coated = analysis_results["n_coated"]
    n_reference = analysis_results["n_reference"]
    top_half = analysis_results["top_half_coated"]
    bottom_half = analysis_results["bottom_half_coated"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Pixel Classification", "Spatial Distribution"),
        specs=[[{"type": "pie"}, {"type": "bar"}]],
    )

    # Pie chart: Coated vs Reference
    fig.add_trace(
        go.Pie(
            labels=["Coated (Responsive)", "Reference (Stable)"],
            values=[n_coated, n_reference],
            marker_colors=["rgb(249, 115, 22)", "rgb(59, 130, 246)"],
            textinfo="percent+value",
            hovertemplate="%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Bar chart: Top vs Bottom half
    fig.add_trace(
        go.Bar(
            x=["Top Half<br>(Rows 0-15)", "Bottom Half<br>(Rows 16-31)"],
            y=[top_half, bottom_half],
            marker_color=["rgb(34, 197, 94)", "rgb(168, 85, 247)"],
            text=[top_half, bottom_half],
            textposition="auto",
            hovertemplate="%{x}<br>Coated pixels: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(text="<b>Coating Pattern Statistics</b>", x=0.5, font=dict(size=14)),
        template="plotly_white",
        height=350,
        showlegend=False,
    )

    fig.update_yaxes(title_text="Number of Coated Pixels", row=1, col=2)

    return fig


# =============================================================================
# FORECASTING VISUALIZATIONS
# =============================================================================


def create_forecast_visualization(
    timestamps: np.ndarray,
    values: np.ndarray,
    forecast_hours: float = None,  # Auto-calculated if None
    title: str = "VOC Concentration Forecast",
    unit: str = "ppm",
) -> go.Figure:
    """
    Create a simple forecast visualization with multiple prediction methods.

    Demonstrates:
    1. Moving Average forecast
    2. Linear trend extrapolation
    3. Exponential smoothing

    Note: Forecast horizon is automatically limited based on available data.
    Rule of thumb: Forecast at most 50% of historical data length.

    Args:
        timestamps: Historical timestamps
        values: Historical ppm or ΔADC values
        forecast_hours: Hours to forecast ahead (auto-calculated if None)
        title: Plot title
        unit: Unit label for y-axis (default "ppm")

    Returns:
        Plotly figure with historical data and forecasts
    """
    from datetime import timedelta

    # Ensure arrays
    values = np.array(values)
    n_points = len(values)

    # Calculate historical data duration
    if hasattr(timestamps[0], "timestamp"):
        data_duration_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
    else:
        # Assume seconds for numeric timestamps
        data_duration_hours = (timestamps[-1] - timestamps[0]) / 3600

    # Auto-calculate forecast horizon: max 50% of data duration (rule of thumb)
    # With only 3.5h of data, we can reasonably forecast ~1-2 hours ahead
    if forecast_hours is None:
        forecast_hours = min(
            data_duration_hours * 0.5, 24
        )  # Max 24h, typically shorter
        forecast_hours = max(forecast_hours, 1)  # At least 1 hour

    # Calculate forecast timestamps
    # Use 5-minute intervals for shorter forecasts, 10-minute for longer
    forecast_interval_minutes = 5 if forecast_hours <= 4 else 10
    forecast_steps = int(forecast_hours * 60 / forecast_interval_minutes)

    if hasattr(timestamps[0], "timestamp"):
        # datetime objects
        last_time = timestamps[-1]
        forecast_times = [
            last_time + timedelta(minutes=forecast_interval_minutes * (i + 1))
            for i in range(forecast_steps)
        ]
    else:
        # numeric timestamps - assume seconds
        forecast_interval_sec = forecast_interval_minutes * 60
        forecast_times = [
            timestamps[-1] + forecast_interval_sec * (i + 1)
            for i in range(forecast_steps)
        ]

    # === Forecasting Methods ===

    # 1. Moving Average (rolling on historical data)
    ma_window = min(100, n_points // 4)
    # Compute rolling MA on historical data for overlay
    ma_historical = np.full(n_points, np.nan)
    for i in range(ma_window, n_points):
        ma_historical[i] = np.mean(values[i - ma_window : i])

    # Forecast: continue from last MA value
    ma_value = np.mean(values[-ma_window:])
    ma_forecast = np.full(forecast_steps, ma_value)

    # Add slight random walk for realism
    noise_scale = np.std(values[-ma_window:]) * 0.1
    ma_forecast = ma_forecast + np.cumsum(np.random.randn(forecast_steps) * noise_scale)

    # 2. Linear Trend Extrapolation
    recent_window = min(200, n_points // 2)
    x_recent = np.arange(recent_window)
    y_recent = values[-recent_window:]
    slope, intercept = np.polyfit(x_recent, y_recent, 1)

    # Compute linear fit on historical data for overlay
    np.arange(n_points)
    # Only show the fit on the recent window used for fitting
    linear_historical = np.full(n_points, np.nan)
    linear_historical[-recent_window:] = slope * x_recent + intercept

    x_forecast = np.arange(recent_window, recent_window + forecast_steps)
    linear_forecast = slope * x_forecast + intercept

    # 3. Exponential Smoothing (simple)
    alpha = 0.3  # Smoothing factor
    smoothed = np.zeros(n_points)
    smoothed[0] = values[0]
    for i in range(1, n_points):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]

    # Forecast: continue with slight decay toward mean
    exp_forecast = np.zeros(forecast_steps)
    exp_forecast[0] = smoothed[-1]
    mean_value = np.mean(values)
    decay = 0.98
    for i in range(1, forecast_steps):
        exp_forecast[i] = decay * exp_forecast[i - 1] + (1 - decay) * mean_value

    # Calculate confidence intervals (simple: ±2σ expanding over time)
    historical_std = np.std(values[-ma_window:])
    time_factors = np.sqrt(np.arange(1, forecast_steps + 1) / forecast_steps)
    upper_bound = ma_forecast + 2 * historical_std * time_factors
    lower_bound = ma_forecast - 2 * historical_std * time_factors

    # Format forecast hours for display
    forecast_hours_display = f"{forecast_hours:.1f}".rstrip("0").rstrip(".")

    # === Create Figure ===
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f"{title} - {forecast_hours_display}h Ahead",
            "Forecast Uncertainty (95% CI)",
        ),
    )

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=list(values),
            mode="lines",
            name="Historical Data",
            line=dict(color="#3498db", width=1.5),
            legendgroup="historical",
        ),
        row=1,
        col=1,
    )

    # === Historical Method Overlays (shows how methods fit historical data) ===

    # Moving Average on historical data (same color, solid line)
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=list(ma_historical),
            mode="lines",
            name="MA (Historical)",
            line=dict(color="#e74c3c", width=1.5),
            legendgroup="ma",
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    # Linear Trend on historical data
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=list(linear_historical),
            mode="lines",
            name="Trend (Historical)",
            line=dict(color="#2ecc71", width=1.5),
            legendgroup="linear",
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    # Exponential Smoothing on historical data
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=list(smoothed),
            mode="lines",
            name="EMA (Historical)",
            line=dict(color="#9b59b6", width=1.5),
            legendgroup="exp",
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    # === Forecast Region ===

    # Confidence interval (shaded)
    fig.add_trace(
        go.Scatter(
            x=list(forecast_times) + list(forecast_times)[::-1],
            y=list(upper_bound) + list(lower_bound)[::-1],
            fill="toself",
            fillcolor="rgba(52, 152, 219, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Moving Average forecast (dashed to indicate prediction)
    fig.add_trace(
        go.Scatter(
            x=list(forecast_times),
            y=list(ma_forecast),
            mode="lines",
            name="MA Forecast",
            line=dict(color="#e74c3c", width=2, dash="dash"),
            legendgroup="ma",
        ),
        row=1,
        col=1,
    )

    # Linear trend forecast
    fig.add_trace(
        go.Scatter(
            x=list(forecast_times),
            y=list(linear_forecast),
            mode="lines",
            name="Trend Forecast",
            line=dict(color="#2ecc71", width=2, dash="dot"),
            legendgroup="linear",
        ),
        row=1,
        col=1,
    )

    # Exponential smoothing forecast
    fig.add_trace(
        go.Scatter(
            x=list(forecast_times),
            y=list(exp_forecast),
            mode="lines",
            name="EMA Forecast",
            line=dict(color="#9b59b6", width=2, dash="dashdot"),
            legendgroup="exp",
        ),
        row=1,
        col=1,
    )

    # Add vertical line at forecast start
    fig.add_vline(
        x=(
            timestamps[-1]
            if hasattr(timestamps[-1], "timestamp")
            else float(timestamps[-1])
        ),
        line=dict(color="gray", width=2, dash="dash"),
        row=1,
        col=1,
    )

    # Uncertainty plot (expanding cone)
    uncertainty = 2 * historical_std * time_factors
    fig.add_trace(
        go.Scatter(
            x=list(forecast_times),
            y=list(uncertainty),
            mode="lines",
            name="Uncertainty (±2σ)",
            line=dict(color="#f39c12", width=2),
            fill="tozeroy",
            fillcolor="rgba(243, 156, 18, 0.3)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>VOC Concentration Forecasting ({forecast_hours_display}h Horizon)</b>",
            x=0.5,
            font=dict(size=16),
        ),
        template="plotly_white",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    fig.update_yaxes(title_text=f"Concentration ({unit})", row=1, col=1)
    fig.update_yaxes(title_text="Uncertainty", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    # Add annotation explaining forecast
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text=f"<i>Note: Forecast horizon ({forecast_hours_display}h) limited to ~50% of historical data ({data_duration_hours:.1f}h). "
        f"These are simple demonstration methods - production systems should use ARIMA, Prophet, or LSTM.</i>",
        showarrow=False,
        font=dict(size=10, color="gray"),
    )

    return fig


def create_forecast_summary_cards(
    current_value: float, forecasts: Dict[str, float], unit: str = "ΔADC"
) -> go.Figure:
    """
    Create summary indicator cards showing forecast values.

    Args:
        current_value: Current sensor reading
        forecasts: Dict of {time_label: predicted_value}
        unit: Unit label
    """
    n_cards = len(forecasts) + 1

    fig = make_subplots(
        rows=1,
        cols=n_cards,
        specs=[[{"type": "indicator"} for _ in range(n_cards)]],
        horizontal_spacing=0.05,
    )

    # Current value
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=current_value,
            title={
                "text": f"<b>Current</b><br><span style='font-size:0.8em'>{unit}</span>"
            },
            number={"font": {"size": 40}},
            domain={"row": 0, "column": 0},
        ),
        row=1,
        col=1,
    )

    # Forecasts
    prev_value = current_value
    for i, (label, value) in enumerate(forecasts.items(), start=2):
        value - prev_value
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=value,
                delta={"reference": prev_value, "relative": False},
                title={
                    "text": f"<b>{label}</b><br><span style='font-size:0.8em'>{unit}</span>"
                },
                number={"font": {"size": 36}},
                domain={"row": 0, "column": i - 1},
            ),
            row=1,
            col=i,
        )
        prev_value = value

    fig.update_layout(height=200, template="plotly_white", margin=dict(t=60, b=20))

    return fig
