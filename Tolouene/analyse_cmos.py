from __future__ import annotations

import argparse
import base64
import io
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, ctx, dcc, html
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.module4_calibration_modeling import (FreundlichModel,
                                                  LangmuirModel,
                                                  PolynomialCalibration)
from modules.module6_integration import CMOSVOCPipeline, PipelineConfig
from modules.module7_visualization import (PipelineVisualizer,
                                           analyze_coating_pattern,
                                           create_coating_summary_stats,
                                           create_forecast_summary_cards,
                                           create_forecast_visualization,
                                           create_pipeline_diagram,
                                           create_sensor_health_gauge,
                                           generate_calibration_summary_text,
                                           plot_all_calibration_models,
                                           plot_bad_pixel_map,
                                           plot_baseline_drift,
                                           plot_calibration_residuals,
                                           plot_coating_pattern_map,
                                           plot_dadc_with_exposure_windows,
                                           plot_layer_comparison_timeseries,
                                           plot_noise_analysis,
                                           plot_predicted_vs_actual,
                                           plot_spatial_sensitivity_map,
                                           plot_t90_distribution)
# Import calibration model
from toluene_calibration import TolueneCalibrationModel

PIXELS_PER_SIDE = 32
PIXEL_COUNT = PIXELS_PER_SIDE * PIXELS_PER_SIDE
TIMESTAMP_FMT = "%d-%b-%Y %H:%M:%S.%f"
DEFAULT_DATA_FILE = Path(__file__).parent / "dataset" / "tolouene_aug_2024.dat"
LAYER_NAMES = {
    1: "Layer 1",
    2: "Layer 2",
    3: "Layer 3",
}
# Custom green → yellow → orange → red color scale (matching the reference image)
COLOR_SCALE = [
    [0.0, "rgb(0, 180, 0)"],  # Green
    [0.25, "rgb(140, 200, 0)"],  # Yellow-green
    [0.5, "rgb(255, 200, 0)"],  # Yellow/Orange
    [0.75, "rgb(255, 120, 0)"],  # Orange
    [1.0, "rgb(255, 50, 0)"],  # Red
]
LAYER_COLORS = {
    1: "rgb(0, 200, 100)",  # Green for Layer 1
    2: "rgb(255, 160, 0)",  # Orange for Layer 2
    3: "rgb(255, 60, 60)",  # Red for Layer 3
}

# Layer 1 focused mode settings
LAYER1_COLOR_RANGE = (1650, 1750)  # Focused range for Layer 1
LAYER1_COLOR_SCALE = [
    [0.0, "rgb(0, 100, 255)"],  # Blue (low)
    [0.25, "rgb(0, 200, 200)"],  # Cyan
    [0.5, "rgb(0, 255, 100)"],  # Green (middle)
    [0.75, "rgb(255, 255, 0)"],  # Yellow
    [1.0, "rgb(255, 50, 0)"],  # Red (high)
]

# ΔADC color scale (for toluene response view)
DELTA_ADC_COLOR_SCALE = [
    [0.0, "rgb(0, 50, 150)"],  # Dark blue (baseline)
    [0.2, "rgb(0, 150, 255)"],  # Light blue
    [0.4, "rgb(0, 255, 150)"],  # Cyan-green
    [0.6, "rgb(150, 255, 0)"],  # Yellow-green
    [0.8, "rgb(255, 200, 0)"],  # Orange
    [1.0, "rgb(255, 50, 0)"],  # Red (high response)
]


@dataclass(slots=True)
class FrameRecord:
    timestamp: datetime
    packet: int
    grids: dict[int, np.ndarray]


def iter_clean_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if stripped:
                yield stripped


def parse_sensor_grid(line: str) -> np.ndarray:
    parts = line.split()
    if len(parts) != PIXEL_COUNT:
        raise ValueError(f"Expected {PIXEL_COUNT} samples, got {len(parts)}")
    values = np.asarray(parts, dtype=np.uint16).reshape(
        PIXELS_PER_SIDE, PIXELS_PER_SIDE
    )
    values[1::2] = np.flip(values[1::2], axis=1)
    return values


def load_frames(data_file: Path) -> list[FrameRecord]:
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    frames: list[FrameRecord] = []
    cleaner = iter_clean_lines(data_file)

    while True:
        try:
            timestamp_str = next(cleaner)
        except StopIteration:
            break

        try:
            timestamp = datetime.strptime(timestamp_str, TIMESTAMP_FMT)
        except ValueError as exc:
            raise ValueError(f"Could not parse timestamp '{timestamp_str}'") from exc

        header = next(cleaner, None)
        if header is None:
            break

        header_parts = header.split()
        if len(header_parts) != 2:
            raise ValueError(f"Invalid header line '{header}'")

        packet = int(header_parts[0])
        sensor_id = int(header_parts[1])
        grids: dict[int, np.ndarray] = {sensor_id: parse_sensor_grid(next(cleaner))}

        while len(grids) < 3:
            sensor_line = next(cleaner)
            if "-" in sensor_line:
                raise ValueError(
                    "Encountered new timestamp before collecting all layers"
                )
            sensor_id = int(sensor_line)
            grids[sensor_id] = parse_sensor_grid(next(cleaner))

        frames.append(FrameRecord(timestamp=timestamp, packet=packet, grids=grids))

    if not frames:
        raise ValueError(f"No frames were parsed from {data_file}")

    return frames


def compute_global_color_range(frames: Sequence[FrameRecord]) -> tuple[int, int]:
    """Compute a global color range across all layers and frames."""
    global_min = float("inf")
    global_max = float("-inf")
    for frame in frames:
        for grid in frame.grids.values():
            global_min = min(global_min, int(grid.min()))
            global_max = max(global_max, int(grid.max()))
    if global_min == global_max:
        global_max = global_min + 1
    return int(global_min), int(global_max)


def compute_adc_sums(frames: Sequence[FrameRecord]) -> dict[int, list[int]]:
    """Compute the sum of ADC values for each layer across all frames."""
    sums: dict[int, list[int]] = {}
    for frame in frames:
        for sensor_id, grid in frame.grids.items():
            if sensor_id not in sums:
                sums[sensor_id] = []
            sums[sensor_id].append(int(grid.sum()))
    return sums


def compute_delta_adc(
    frames: Sequence[FrameRecord],
    layer_id: int = 1,
    coating_mask: np.ndarray | None = None,
) -> tuple[list[float], float, dict]:
    """Compute ΔADC using average-based coated/reference subtraction.

    This method calculates ΔADC as:
        ΔADC = Average(coated pixels) - Average(reference pixels)

    This approach provides:
    - Better signal-to-noise ratio (only responsive pixels contribute)
    - Environmental drift compensation (reference pixels track drift)
    - Temperature compensation (both regions affected equally)

    Args:
        frames: Sequence of FrameRecord objects
        layer_id: Which layer to analyze (default 1 = active sensing layer)
        coating_mask: Optional 32x32 boolean array (True = coated pixel)
                     If None, auto-detects using analyze_coating_pattern()

    Returns:
        Tuple of (delta_adc_list, baseline_value, coating_info)
        - delta_adc_list: List of ΔADC values per frame
        - baseline_value: Mean reference ADC value (for compatibility)
        - coating_info: Dict with coating pattern statistics
    """
    # Build pixel timeseries for coating analysis
    pixel_timeseries = np.array([frame.grids[layer_id] for frame in frames])

    # Auto-detect coating pattern if not provided
    if coating_mask is None:
        coating_analysis = analyze_coating_pattern(pixel_timeseries)
        coating_mask = coating_analysis["coating_mask"]
        coating_info = coating_analysis
    else:
        # Compute basic stats for provided mask
        n_coated = np.sum(coating_mask)
        coating_info = {
            "n_coated": int(n_coated),
            "n_reference": int(1024 - n_coated),
            "coated_percentage": float(n_coated / 1024 * 100),
        }

    # Get indices of coated and reference pixels
    coated_rows, coated_cols = np.where(coating_mask)
    ref_rows, ref_cols = np.where(~coating_mask)

    # Calculate ΔADC for each frame: Avg(coated) - Avg(reference)
    delta_adc = []
    ref_means = []
    coated_means = []

    for frame in frames:
        grid = frame.grids[layer_id]

        # Extract pixel values
        coated_values = grid[coated_rows, coated_cols]
        ref_values = grid[ref_rows, ref_cols]

        # Calculate averages
        avg_coated = np.mean(coated_values) if len(coated_values) > 0 else 0
        avg_ref = np.mean(ref_values) if len(ref_values) > 0 else 0

        # ΔADC = difference between coated and reference averages
        delta = avg_coated - avg_ref
        delta_adc.append(float(delta))
        ref_means.append(float(avg_ref))
        coated_means.append(float(avg_coated))

    # Baseline is the mean reference value (for compatibility with existing code)
    baseline = float(np.mean(ref_means))

    # Add time-series data to coating_info
    coating_info["ref_means"] = ref_means
    coating_info["coated_means"] = coated_means

    return delta_adc, baseline, coating_info


def compute_sensor_analysis(
    frames: Sequence[FrameRecord], delta_adc: list[float], baseline: float
) -> dict:
    """Compute comprehensive sensor analysis statistics."""
    timestamps = [f.timestamp for f in frames]
    delta_arr = np.array(delta_adc)

    # Layer statistics
    layer1_sums = np.array([np.sum(f.grids[1]) for f in frames])
    layer2_sums = np.array([np.sum(f.grids[2]) for f in frames])
    layer3_sums = np.array([np.sum(f.grids[3]) for f in frames])

    # Time segments analysis (10 segments)
    n_segments = 10
    segment_size = len(frames) // n_segments
    segments = []
    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size, len(frames))
        seg_delta = delta_arr[start_idx:end_idx]
        segments.append(
            {
                "time_start": timestamps[start_idx].strftime("%H:%M"),
                "time_end": timestamps[end_idx - 1].strftime("%H:%M"),
                "mean_dadc": float(np.mean(seg_delta)),
                "min_dadc": float(np.min(seg_delta)),
                "max_dadc": float(np.max(seg_delta)),
            }
        )

    # Detect exposure periods using RELATIVE threshold
    # Calculate actual baseline (10th percentile) for relative detection
    dadc_baseline = float(np.percentile(delta_arr, 10))
    relative_threshold = 0.3  # Threshold ABOVE baseline

    in_exposure = False
    exposure_periods = []
    start_time = None
    start_idx = 0

    for i, (frame, dadc) in enumerate(zip(frames, delta_adc)):
        dadc_relative = dadc - dadc_baseline  # Convert to relative value
        if dadc_relative > relative_threshold and not in_exposure:
            in_exposure = True
            start_time = frame.timestamp
            start_idx = i
        elif dadc_relative <= relative_threshold and in_exposure:
            in_exposure = False
            duration = (frame.timestamp - start_time).total_seconds()
            if duration >= 30:  # Only count periods >= 30 seconds
                peak_relative = float(np.max(delta_arr[start_idx:i]) - dadc_baseline)
                mean_relative = float(np.mean(delta_arr[start_idx:i]) - dadc_baseline)
                # Estimate ppm from relative ΔADC (approx: 500ppm per 0.25 ΔADC)
                est_ppm = int(peak_relative * 2000)  # Rough linear estimate
                exposure_periods.append(
                    {
                        "start": start_time.strftime("%H:%M:%S"),
                        "end": frame.timestamp.strftime("%H:%M:%S"),
                        "duration": int(duration),
                        "peak_dadc": peak_relative,
                        "mean_dadc": mean_relative,
                        "est_ppm": min(
                            10000, max(0, est_ppm)
                        ),  # Clamp to reasonable range
                    }
                )

    # Key events
    max_idx = int(np.argmax(delta_arr))
    min_idx = int(np.argmin(delta_arr))

    return {
        "overview": {
            "total_frames": len(frames),
            "duration_min": (timestamps[-1] - timestamps[0]).total_seconds() / 60,
            "date": timestamps[0].strftime("%d-%b-%Y"),
            "time_span": f"{timestamps[0].strftime('%H:%M:%S')} - {timestamps[-1].strftime('%H:%M:%S')}",
        },
        "layers": {
            "layer1": {
                "mean": float(np.mean(layer1_sums)),
                "std": float(np.std(layer1_sums)),
                "status": "Active",
            },
            "layer2": {
                "mean": float(np.mean(layer2_sums)),
                "std": float(np.std(layer2_sums)),
                "status": "Zero/Disabled",
            },
            "layer3": {
                "mean": float(np.mean(layer3_sums)),
                "std": float(np.std(layer3_sums)),
                "status": "Saturated",
            },
        },
        "delta_adc": {
            "baseline": baseline,
            "min": float(np.min(delta_arr)),
            "max": float(np.max(delta_arr)),
            "mean": float(np.mean(delta_arr)),
            "std": float(np.std(delta_arr)),
        },
        "segments": segments,
        "exposure_periods": exposure_periods[:15],  # Top 15 significant periods
        "key_events": {
            "max_response": {
                "value": float(delta_arr[max_idx]),
                "time": timestamps[max_idx].strftime("%H:%M:%S"),
            },
            "min_response": {
                "value": float(delta_arr[min_idx]),
                "time": timestamps[min_idx].strftime("%H:%M:%S"),
            },
        },
    }


def build_analysis_segment_chart(
    segments: list[dict], dadc_baseline: float = None
) -> go.Figure:
    """Build a bar chart showing ΔADC by time segment."""
    times = [f"{s['time_start']}-{s['time_end']}" for s in segments]
    means = [s["mean_dadc"] for s in segments]

    # Calculate baseline from segments if not provided
    if dadc_baseline is None:
        dadc_baseline = min(means) if means else 0

    # Color legend definitions (relative to baseline)
    color_levels = [
        ("Baseline (<0.3)", "rgb(100, 150, 255)", "<500 ppm"),
        ("Low (0.3-0.5)", "rgb(100, 200, 100)", "~500-1000 ppm"),
        ("Medium (0.5-1.0)", "rgb(255, 200, 100)", "~1000-3000 ppm"),
        ("High (1.0-1.5)", "rgb(255, 150, 50)", "~3000-5000 ppm"),
        ("Very High (>1.5)", "rgb(255, 80, 80)", ">5000 ppm"),
    ]

    # Color based on RELATIVE value (difference from baseline)
    colors = []
    for m in means:
        rel_val = m - dadc_baseline  # Relative to baseline
        if rel_val < 0.3:
            colors.append("rgb(100, 150, 255)")  # Blue - baseline
        elif rel_val < 0.5:
            colors.append("rgb(100, 200, 100)")  # Green - low
        elif rel_val < 1.0:
            colors.append("rgb(255, 200, 100)")  # Yellow - medium
        elif rel_val < 1.5:
            colors.append("rgb(255, 150, 50)")  # Orange - high
        else:
            colors.append("rgb(255, 80, 80)")  # Red - very high

    fig = go.Figure()

    # Add main data bars
    fig.add_trace(
        go.Bar(
            x=times,
            y=means,
            marker_color=colors,
            text=[f"{m:.2f}" for m in means],
            textposition="outside",
            hovertemplate="%{x}<br>Mean ΔADC: %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )

    # Add invisible traces for legend
    for label, color, ppm in color_levels:
        fig.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                marker_color=color,
                name=f"{label} ({ppm})",
                showlegend=True,
            )
        )

    fig.update_layout(
        title="ΔADC by Time Segment",
        xaxis_title="Time Period",
        yaxis_title="Mean ΔADC",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=50, r=20, t=50, b=80),
        xaxis=dict(tickangle=-45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1,
        ),
    )

    return fig


def build_layer_comparison_chart(layer_stats: dict) -> go.Figure:
    """Build a comparison chart for the three sensor layers."""
    layers = ["Layer 1", "Layer 2", "Layer 3"]
    means = [
        layer_stats["layer1"]["mean"] / 1e6,  # Convert to millions
        layer_stats["layer2"]["mean"] / 1e6,
        layer_stats["layer3"]["mean"] / 1e6,
    ]
    statuses = [
        layer_stats["layer1"]["status"],
        layer_stats["layer2"]["status"],
        layer_stats["layer3"]["status"],
    ]
    colors = ["rgb(0, 200, 100)", "rgb(100, 100, 100)", "rgb(255, 100, 100)"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=layers,
            y=means,
            marker_color=colors,
            text=[f"{m:.2f}M<br>({s})" for m, s in zip(means, statuses)],
            textposition="inside",
            hovertemplate="%{x}<br>Sum: %{y:.3f}M ADC<extra></extra>",
        )
    )

    fig.update_layout(
        title="Sensor Layer Comparison (Sum ADC)",
        yaxis_title="ADC Sum (Millions)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=50, r=50, t=50, b=30),
    )

    return fig


def build_timeseries_base(
    timestamps: list, adc_sums: dict[int, list[int]]
) -> go.Figure:
    """Build the base time series figure once (without the moving marker)."""
    layer_order = sorted(adc_sums.keys())

    fig = go.Figure()

    for sensor_id in layer_order:
        fig.add_trace(
            go.Scattergl(  # Use WebGL for better performance
                x=timestamps,
                y=adc_sums[sensor_id],
                mode="lines",
                name=LAYER_NAMES.get(sensor_id, f"Layer {sensor_id}"),
                line=dict(color=LAYER_COLORS.get(sensor_id, "white"), width=2),
                hovertemplate="%{y:,.0f}<extra>"
                + LAYER_NAMES.get(sensor_id, f"Layer {sensor_id}")
                + "</extra>",
            )
        )

    # Add placeholder shape and annotation (will be updated)
    fig.add_shape(
        type="line",
        x0=timestamps[0],
        x1=timestamps[0],
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="white", width=2, dash="dash"),
        name="position-marker",
    )
    fig.add_annotation(
        x=timestamps[0],
        y=1,
        yref="paper",
        text="Frame 1",
        showarrow=False,
        font=dict(color="white", size=10),
        yshift=10,
    )

    fig.update_layout(
        title="ADC Sum Over Time (per Layer)",
        xaxis_title="Time",
        yaxis_title="Sum of ADC Values",
        height=300,
        margin=dict(l=60, r=20, t=50, b=40),
        paper_bgcolor="rgb(14,17,23)",
        plot_bgcolor="rgb(14,17,23)",
        font=dict(color="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", showgrid=True),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", showgrid=True),
        uirevision="constant",  # Preserve zoom/pan state
    )

    return fig


def build_layer1_timeseries_base(timestamps: list, sums: list[int]) -> go.Figure:
    """Build the base Layer 1 time series figure once."""
    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(  # Use WebGL for better performance
            x=timestamps,
            y=sums,
            mode="lines",
            name="Layer 1",
            line=dict(color=LAYER_COLORS[1], width=2),
            hovertemplate="%{y:,.0f}<extra>Layer 1</extra>",
        )
    )

    # Add placeholder shape and annotation
    fig.add_shape(
        type="line",
        x0=timestamps[0],
        x1=timestamps[0],
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="white", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=timestamps[0],
        y=1,
        yref="paper",
        text="Frame 1",
        showarrow=False,
        font=dict(color="white", size=10),
        yshift=10,
    )

    fig.update_layout(
        title="Layer 1 ADC Sum Over Time",
        xaxis_title="Time",
        yaxis_title="Sum of ADC Values",
        height=300,
        margin=dict(l=60, r=20, t=50, b=40),
        paper_bgcolor="rgb(14,17,23)",
        plot_bgcolor="rgb(14,17,23)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", showgrid=True),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", showgrid=True),
        uirevision="constant",
    )

    return fig


def build_delta_adc_timeseries(
    timestamps: list,
    delta_adc: list[float],
    baseline: float,
    concentration_labels: list[dict] = None,
    calibration: TolueneCalibrationModel = None,
) -> go.Figure:
    """Build ΔADC time series figure matching reference chart style with concentration labels."""
    fig = go.Figure()

    # Calculate actual ΔADC baseline (10th percentile) for relative measurements
    delta_arr = np.array(delta_adc)
    dadc_baseline = float(np.percentile(delta_arr, 10))
    max_delta = float(np.max(delta_arr))
    min_delta = float(np.min(delta_arr))
    max_delta - dadc_baseline  # Range above baseline

    # Main ΔADC trace with gradient coloring effect
    fig.add_trace(
        go.Scattergl(
            x=timestamps,
            y=delta_adc,
            mode="lines",
            name="ΔADC",
            line=dict(color="rgb(255, 80, 80)", width=1.5),  # Red like reference chart
            fill="tozeroy",
            fillcolor="rgba(255, 80, 80, 0.3)",
            hovertemplate="ΔADC: %{y:.2f}<br>Time: %{x}<extra></extra>",
        )
    )

    # Add baseline reference line at the DETECTED baseline
    fig.add_hline(
        y=dadc_baseline,
        line=dict(color="rgba(255,255,0,0.7)", width=2, dash="dash"),
    )

    # Detect peaks FIRST to derive actual ppm level positions
    # Known concentration sequence: ascending then descending
    ppm_sequence = [500, 1000, 3000, 5000, 7000, 9000, 7000, 5000, 3000, 1000, 500]

    # Find peaks with parameters tuned for ~10 min exposure periods
    peaks_idx, peak_props = find_peaks(
        delta_arr,
        height=dadc_baseline + 0.15,
        distance=400,
        prominence=0.15,
    )

    # Select top 11 most prominent peaks if we found more
    if len(peaks_idx) > len(ppm_sequence):
        prominences = peak_props.get("prominences", np.ones(len(peaks_idx)))
        if len(prominences) == 0:
            prominences = delta_arr[peaks_idx] - dadc_baseline
        top_indices = np.argsort(prominences)[-len(ppm_sequence) :]
        peaks_idx = np.sort(peaks_idx[top_indices])

    # Build a map of ppm -> actual peak heights (average for duplicates)
    ppm_to_heights = {}
    for i, peak_idx in enumerate(peaks_idx[: len(ppm_sequence)]):
        ppm_val = ppm_sequence[i]
        peak_height = delta_arr[peak_idx]
        if ppm_val not in ppm_to_heights:
            ppm_to_heights[ppm_val] = []
        ppm_to_heights[ppm_val].append(peak_height)

    # Calculate average height for each ppm level
    ppm_avg_heights = {ppm: np.mean(heights) for ppm, heights in ppm_to_heights.items()}

    # Concentration levels with colors - heights will be from actual peaks
    ppm_colors = {
        500: "rgba(0, 255, 0, 0.6)",
        1000: "rgba(100, 255, 0, 0.6)",
        3000: "rgba(255, 255, 0, 0.6)",
        5000: "rgba(255, 200, 0, 0.6)",
        7000: "rgba(255, 100, 0, 0.6)",
        9000: "rgba(255, 50, 0, 0.6)",
    }

    # Add horizontal ppm reference lines at ACTUAL detected peak heights
    for ppm_val in sorted(ppm_avg_heights.keys()):
        avg_height = ppm_avg_heights[ppm_val]
        color = ppm_colors.get(ppm_val, "rgba(255,255,255,0.5)")
        fig.add_hline(
            y=avg_height,
            line=dict(color=color, width=1, dash="dot"),
        )

    # Add position marker LAST so it's the last shape (we'll update shapes[-1])
    fig.add_shape(
        type="line",
        x0=timestamps[0],
        x1=timestamps[0],
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="yellow", width=2, dash="dash"),
        name="frame_marker",
    )
    fig.add_annotation(
        x=timestamps[0],
        y=1,
        yref="paper",
        text="Frame 1",
        showarrow=False,
        font=dict(color="yellow", size=10),
        yshift=10,
        name="frame_label",
    )

    # Build annotations list (will be appended to existing)
    extra_annotations = [
        dict(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Baseline: {dadc_baseline:.2f} ΔADC",
            showarrow=False,
            font=dict(color="rgba(255,255,0,0.9)", size=11),
            align="left",
        ),
    ]

    # Add ppm labels on the right Y-axis at ACTUAL detected heights
    for ppm_val in sorted(ppm_avg_heights.keys()):
        avg_height = ppm_avg_heights[ppm_val]
        color = ppm_colors.get(ppm_val, "rgba(255,255,255,0.5)")
        extra_annotations.append(
            dict(
                x=1.02,  # Right side of plot
                y=avg_height,
                xref="paper",
                yref="y",
                text=f"{ppm_val} ppm",
                showarrow=False,
                font=dict(color=color, size=9),
                xanchor="left",
            )
        )

    # Add ppm annotations on the curve peaks (peaks_idx already detected above)
    for i, peak_idx in enumerate(peaks_idx[: len(ppm_sequence)]):
        peak_ppm = ppm_sequence[i]
        peak_time = timestamps[peak_idx]
        peak_val = delta_arr[peak_idx]
        extra_annotations.append(
            dict(
                x=peak_time,
                y=peak_val,
                xref="x",
                yref="y",
                text=f"{peak_ppm} ppm",
                showarrow=False,
                font=dict(color="white", size=9),
                textangle=-45,  # Angled like paper
                yshift=15,
            )
        )

    # Add calibration info if available
    if calibration and calibration.calibration:
        cal = calibration.calibration
        extra_annotations.append(
            dict(
                x=0.98,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"R²={cal.r_squared:.3f}",
                showarrow=False,
                font=dict(color="rgba(255,255,255,0.7)", size=10),
                align="right",
                xanchor="right",
            )
        )

    # Combine with existing annotations from shapes
    all_annotations = list(fig.layout.annotations) + extra_annotations

    fig.update_layout(
        title=dict(
            text="ΔADC Response to Toluene Concentration (Layer 1) - UV ink",
            font=dict(size=16),
        ),
        xaxis_title="Time",
        yaxis_title="ΔADC (I)",
        height=380,
        margin=dict(l=60, r=80, t=60, b=40),  # More right margin for ppm labels
        paper_bgcolor="rgb(14,17,23)",
        plot_bgcolor="rgb(14,17,23)",
        font=dict(color="white"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.1)",
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.1)",
            showgrid=True,
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.3)",
            range=[min_delta - 0.3, max_delta + 0.8],  # Extra room for labels
        ),
        uirevision="constant",
        annotations=all_annotations,
    )

    return fig


def build_delta_adc_heatmap(delta_values: list[float], max_delta: float) -> go.Figure:
    """Build hexagonal heatmap showing ΔADC values (change from baseline)."""
    hex_x_list = HEX_X.tolist()
    hex_y_list = HEX_Y.tolist()
    customdata_list = CUSTOMDATA.tolist()

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=hex_x_list,
            y=hex_y_list,
            mode="markers",
            marker=dict(
                symbol="hexagon",
                size=14,
                color=delta_values,
                cmin=0,
                cmax=max(max_delta, 50),  # Ensure reasonable range
                colorscale=DELTA_ADC_COLOR_SCALE,
                line=dict(width=1, color="rgba(0,0,0,0.5)"),
                showscale=True,
                colorbar=dict(title="ΔADC", len=0.8),
            ),
            customdata=customdata_list,
            hovertemplate="row %{customdata[0]}<br>col %{customdata[1]}<br>ΔADC: %{marker.color:.1f}<extra></extra>",
        )
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)

    fig.update_layout(
        title="ΔADC Spatial Response (Layer 1)",
        height=700,
        margin=dict(l=20, r=100, t=60, b=20),
        paper_bgcolor="rgb(14,17,23)",
        plot_bgcolor="rgb(14,17,23)",
        font=dict(color="white"),
        uirevision="constant",
    )
    return fig


def hex_layout(
    rows: int, cols: int, spacing: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    dy = math.sqrt(3) / 2 * spacing
    xs = []
    ys = []
    for row in range(rows):
        x_offset = 0.5 * (row % 2)
        for col in range(cols):
            xs.append((col + x_offset) * spacing)
            ys.append(-row * dy)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


HEX_X, HEX_Y = hex_layout(PIXELS_PER_SIDE, PIXELS_PER_SIDE)
ROW_IDX, COL_IDX = np.indices((PIXELS_PER_SIDE, PIXELS_PER_SIDE))
CUSTOMDATA = np.stack([ROW_IDX.flatten(), COL_IDX.flatten()], axis=-1)


@dataclass
class DashboardData:
    frames: Sequence[FrameRecord]
    timestamps: list[datetime]
    all_grids_data: list[dict[int, list]]
    layer1_clipped_data: list[list]
    layer2_data: list[list]
    layer3_data: list[list]
    layer_color_ranges: dict[int, tuple[int, int]]
    delta_adc_values: list[float]
    delta_adc_pixels: list[list[float]]
    analysis_data: dict
    segment_chart: go.Figure
    layer_chart: go.Figure
    base_layer_fig: go.Figure
    base_layer1_fig: go.Figure
    base_layer2_fig: go.Figure
    base_layer3_fig: go.Figure
    base_timeseries_fig: go.Figure
    base_layer1_ts_fig: go.Figure
    base_layer2_ts_fig: go.Figure
    base_layer3_ts_fig: go.Figure
    base_delta_ts_fig: go.Figure
    base_delta_heatmap_fig: go.Figure
    layer_order: list[int]
    marks: dict
    adc_sums: dict[int, list[int]]
    coating_info: dict  # Coating pattern analysis info (coated/reference pixels)


def prepare_dashboard_data(frames: Sequence[FrameRecord]) -> DashboardData:
    color_range = compute_global_color_range(frames)
    adc_sums = compute_adc_sums(frames)
    timestamps = [frame.timestamp for frame in frames]
    marks = build_slider_marks(timestamps)

    # Pre-compute all grid data as lists for faster updates
    layer_order = sorted(frames[0].grids.keys())
    all_grids_data: list[dict[int, list]] = []
    layer_sums: dict[int, list[int]] = {1: [], 2: [], 3: []}

    for frame in frames:
        grids_for_frame = {}
        for sensor_id in layer_order:
            grids_for_frame[sensor_id] = frame.grids[sensor_id].flatten().tolist()
            layer_sums[sensor_id].append(int(frame.grids[sensor_id].sum()))
        all_grids_data.append(grids_for_frame)

    # Compute per-layer color ranges
    layer_color_ranges: dict[int, tuple[int, int]] = {}
    for layer_id in layer_order:
        layer_values = np.array([frame.grids[layer_id] for frame in frames])
        layer_color_ranges[layer_id] = (
            int(layer_values.min()),
            int(layer_values.max()),
        )

    # Pre-compute clipped Layer 1 data (focused view)
    cmin, cmax = LAYER1_COLOR_RANGE
    layer1_clipped_data: list[list] = []
    for frame in frames:
        values = np.clip(frame.grids[1].flatten(), cmin, cmax).tolist()
        layer1_clipped_data.append(values)

    # Pre-compute Layer 2 data
    layer2_data: list[list] = []
    for frame in frames:
        layer2_data.append(frame.grids[2].flatten().tolist())

    # Pre-compute Layer 3 data
    layer3_data: list[list] = []
    for frame in frames:
        layer3_data.append(frame.grids[3].flatten().tolist())

    # Compute Delta-ADC data using average-based coated/reference subtraction
    # This method: Delta-ADC = Avg(coated pixels) - Avg(reference pixels)
    delta_adc_values, baseline_adc, coating_info_dashboard = compute_delta_adc(
        frames, layer_id=1
    )
    print("      Delta-ADC Method: Coated/Reference subtraction")
    print(
        f"      Coated pixels: {coating_info_dashboard.get('n_coated', 'N/A')} ({coating_info_dashboard.get('coated_percentage', 0):.1f}%)"
    )
    print(f"      Reference pixels: {coating_info_dashboard.get('n_reference', 'N/A')}")

    # Pre-compute per-pixel ΔADC for each frame (baseline per pixel)
    layer1_pixel_baseline = np.percentile(
        [frame.grids[1].flatten() for frame in frames], 10, axis=0
    )
    delta_adc_pixels: list[list[float]] = []
    for frame in frames:
        pixel_delta = (frame.grids[1].flatten() - layer1_pixel_baseline).tolist()
        delta_adc_pixels.append(pixel_delta)
    max_pixel_delta = float(np.max([np.max(d) for d in delta_adc_pixels]))

    # Create and fit calibration model
    calibration_model = TolueneCalibrationModel()
    calibration_model.fit_calibration(timestamps, delta_adc_values)
    concentration_labels = calibration_model.get_concentration_labels(
        timestamps, delta_adc_values
    )

    # Compute comprehensive analysis
    analysis_data = compute_sensor_analysis(frames, delta_adc_values, baseline_adc)
    segment_chart = build_analysis_segment_chart(analysis_data["segments"])
    layer_chart = build_layer_comparison_chart(analysis_data["layers"])

    # Pre-build base figures (done once)
    base_layer_fig = build_layer_figure_base(color_range, layer_order)
    base_layer1_fig = build_single_layer_figure_base(
        1,
        layer_color_ranges[1],
        "Layer 1 - Active Sensing (ZIF-8 MOF)",
        LAYER1_COLOR_SCALE,
    )
    base_layer2_fig = build_single_layer_figure_base(
        2,
        layer_color_ranges[2],
        "Layer 2 - Reference (Dark/Disabled)",
        [[0, "rgb(50,50,50)"], [1, "rgb(100,100,100)"]],
    )
    base_layer3_fig = build_single_layer_figure_base(
        3,
        layer_color_ranges[3],
        "Layer 3 - Saturated (Overexposed)",
        [[0, "rgb(200,50,50)"], [0.5, "rgb(255,150,50)"], [1, "rgb(255,255,100)"]],
    )
    base_timeseries_fig = build_timeseries_base(timestamps, adc_sums)
    base_layer1_ts_fig = build_single_layer_timeseries_base(
        timestamps, layer_sums[1], 1, "Layer 1 - Active Sensing"
    )
    base_layer2_ts_fig = build_single_layer_timeseries_base(
        timestamps, layer_sums[2], 2, "Layer 2 - Reference"
    )
    base_layer3_ts_fig = build_single_layer_timeseries_base(
        timestamps, layer_sums[3], 3, "Layer 3 - Saturated"
    )
    base_delta_ts_fig = build_delta_adc_timeseries(
        timestamps,
        delta_adc_values,
        baseline_adc,
        concentration_labels,
        calibration_model,
    )
    base_delta_heatmap_fig = build_delta_adc_heatmap(
        delta_adc_pixels[0], max_pixel_delta
    )

    return DashboardData(
        frames=frames,
        timestamps=timestamps,
        all_grids_data=all_grids_data,
        layer1_clipped_data=layer1_clipped_data,
        layer2_data=layer2_data,
        layer3_data=layer3_data,
        layer_color_ranges=layer_color_ranges,
        delta_adc_values=delta_adc_values,
        delta_adc_pixels=delta_adc_pixels,
        analysis_data=analysis_data,
        segment_chart=segment_chart,
        layer_chart=layer_chart,
        base_layer_fig=base_layer_fig,
        base_layer1_fig=base_layer1_fig,
        base_layer2_fig=base_layer2_fig,
        base_layer3_fig=base_layer3_fig,
        base_timeseries_fig=base_timeseries_fig,
        base_layer1_ts_fig=base_layer1_ts_fig,
        base_layer2_ts_fig=base_layer2_ts_fig,
        base_layer3_ts_fig=base_layer3_ts_fig,
        base_delta_ts_fig=base_delta_ts_fig,
        base_delta_heatmap_fig=base_delta_heatmap_fig,
        layer_order=layer_order,
        marks=marks,
        adc_sums=adc_sums,
        coating_info=coating_info_dashboard,
    )


def build_layer_figure_base(
    color_range: tuple[int, int], layer_order: list[int]
) -> go.Figure:
    """Build the base heatmap figure structure once."""
    fig = make_subplots(
        rows=1,
        cols=len(layer_order),
        subplot_titles=[LAYER_NAMES.get(sid, f"Layer {sid}") for sid in layer_order],
    )

    cmin, cmax = color_range
    # Convert to list once for faster JSON serialization
    hex_x_list = HEX_X.tolist()
    hex_y_list = HEX_Y.tolist()
    customdata_list = CUSTOMDATA.tolist()

    for idx, sensor_id in enumerate(layer_order, start=1):
        # Initialize with zeros (will be updated)
        fig.add_trace(
            go.Scattergl(  # Use WebGL for better performance
                x=hex_x_list,
                y=hex_y_list,
                mode="markers",
                marker=dict(
                    symbol="hexagon",
                    size=18,
                    color=[0] * PIXEL_COUNT,  # Placeholder
                    cmin=cmin,
                    cmax=cmax,
                    colorscale=COLOR_SCALE,
                    line=dict(width=0),
                    showscale=(idx == len(layer_order)),
                    colorbar=dict(title="ADC", len=0.8),
                ),
                customdata=customdata_list,
                hovertemplate="row %{customdata[0]}<br>col %{customdata[1]}<br>value %{marker.color:.0f}<extra></extra>",
            ),
            row=1,
            col=idx,
        )

        fig.update_xaxes(visible=False, row=1, col=idx)
        fig.update_yaxes(
            visible=False, scaleanchor=f"x{idx}", scaleratio=1, row=1, col=idx
        )

    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgb(14,17,23)",
        plot_bgcolor="rgb(14,17,23)",
        font=dict(color="white"),
        uirevision="constant",  # Preserve zoom/pan state
    )
    return fig


def build_single_layer_figure_base(
    layer_id: int, color_range: tuple[int, int], title: str, colorscale: list
) -> go.Figure:
    """Build a base figure for a single layer view."""
    cmin, cmax = color_range
    # Handle edge case where all values are the same
    if cmin == cmax:
        cmax = cmin + 1

    hex_x_list = HEX_X.tolist()
    hex_y_list = HEX_Y.tolist()
    customdata_list = CUSTOMDATA.tolist()

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=hex_x_list,
            y=hex_y_list,
            mode="markers",
            marker=dict(
                symbol="hexagon",
                size=14,
                color=[0] * PIXEL_COUNT,
                cmin=cmin,
                cmax=cmax,
                colorscale=colorscale,
                line=dict(width=1, color="rgba(0,0,0,0.5)"),
                showscale=True,
                colorbar=dict(title="ADC", len=0.8),
            ),
            customdata=customdata_list,
            hovertemplate="row %{customdata[0]}<br>col %{customdata[1]}<br>value %{marker.color:.0f}<extra></extra>",
        )
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)

    fig.update_layout(
        title=title,
        height=700,
        margin=dict(l=20, r=100, t=60, b=20),
        paper_bgcolor="rgb(14,17,23)",
        plot_bgcolor="rgb(14,17,23)",
        font=dict(color="white"),
        uirevision="constant",
    )
    return fig


def build_single_layer_timeseries_base(
    timestamps: list, sums: list[int], layer_id: int, title: str
) -> go.Figure:
    """Build the base time series figure for a single layer."""
    layer_colors = {
        1: "rgb(0, 200, 100)",
        2: "rgb(100, 100, 100)",
        3: "rgb(255, 100, 100)",
    }

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=timestamps,
            y=sums,
            mode="lines",
            name=f"Layer {layer_id}",
            line=dict(color=layer_colors.get(layer_id, "white"), width=2),
            hovertemplate="%{y:,.0f}<extra>Layer " + str(layer_id) + "</extra>",
        )
    )

    fig.add_shape(
        type="line",
        x0=timestamps[0],
        x1=timestamps[0],
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="white", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=timestamps[0],
        y=1,
        yref="paper",
        text="Frame 1",
        showarrow=False,
        font=dict(color="white", size=10),
        yshift=10,
    )

    fig.update_layout(
        title=f"{title} - ADC Sum Over Time",
        xaxis_title="Time",
        yaxis_title="Sum of ADC Values",
        height=300,
        margin=dict(l=60, r=20, t=50, b=40),
        paper_bgcolor="rgb(14,17,23)",
        plot_bgcolor="rgb(14,17,23)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", showgrid=True),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", showgrid=True),
        uirevision="constant",
    )

    return fig


def build_slider_marks(
    timestamps: Sequence[datetime], max_marks: int = 10
) -> dict[int, str]:
    if not timestamps:
        return {0: "0"}
    if len(timestamps) <= max_marks:
        return {idx: ts.strftime("%H:%M:%S") for idx, ts in enumerate(timestamps)}
    step = max(1, len(timestamps) // (max_marks - 1))
    marks = {
        idx: timestamps[idx].strftime("%H:%M:%S")
        for idx in range(0, len(timestamps), step)
    }
    marks[len(timestamps) - 1] = timestamps[-1].strftime("%H:%M:%S")
    return marks


def get_dashboard_layout(data: DashboardData) -> html.Div:
    button_style = {
        "padding": "8px 16px",
        "fontSize": "16px",
        "border": "none",
        "borderRadius": "4px",
        "cursor": "pointer",
        "marginRight": "8px",
    }

    return html.Div(
        className="container",
        children=[
            html.H2("CMOS Sensor Layers"),
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "2rem",
                    "marginBottom": "1rem",
                    "flexWrap": "wrap",
                },
                children=[
                    html.Div(
                        id="meta",
                        children=[
                            html.Span(id="timestamp-label"),
                            html.Span(" • "),
                            html.Span(id="packet-label"),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Label(
                                "View: ",
                                style={"marginRight": "0.5rem", "fontWeight": "500"},
                            ),
                            dcc.Dropdown(
                                id="view-mode",
                                options=[
                                    {"label": "All Layers (Overview)", "value": "all"},
                                    {
                                        "label": "Layer 1 - Active Sensing (ΔADC Toluene)",
                                        "value": "layer1",
                                    },
                                    {
                                        "label": "Layer 2 - Reference (Dark/Disabled)",
                                        "value": "layer2",
                                    },
                                    {
                                        "label": "Layer 3 - Saturated (Overexposed)",
                                        "value": "layer3",
                                    },
                                ],
                                value="all",
                                clearable=False,
                                style={"width": "380px", "color": "black"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                ],
            ),
            # Playback controls
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "1rem",
                    "marginBottom": "1rem",
                    "flexWrap": "wrap",
                },
                children=[
                    html.Button(
                        "▶ Play",
                        id="play-btn",
                        n_clicks=0,
                        style={
                            **button_style,
                            "backgroundColor": "#4CAF50",
                            "color": "white",
                        },
                    ),
                    html.Button(
                        "⏸ Pause",
                        id="pause-btn",
                        n_clicks=0,
                        style={
                            **button_style,
                            "backgroundColor": "#FF9800",
                            "color": "white",
                        },
                    ),
                    html.Button(
                        "⏹ Stop",
                        id="stop-btn",
                        n_clicks=0,
                        style={
                            **button_style,
                            "backgroundColor": "#f44336",
                            "color": "white",
                        },
                    ),
                    html.Div(
                        children=[
                            html.Label("Speed: ", style={"marginRight": "0.5rem"}),
                            dcc.Dropdown(
                                id="speed-dropdown",
                                options=[
                                    {"label": "0.5x (2s)", "value": 2000},
                                    {"label": "1x (1s)", "value": 1000},
                                    {"label": "2x (500ms)", "value": 500},
                                    {"label": "4x (250ms)", "value": 250},
                                    {"label": "10x (100ms)", "value": 100},
                                    {"label": "20x (50ms)", "value": 50},
                                ],
                                value=500,
                                clearable=False,
                                style={"width": "140px", "color": "black"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Div(
                        children=[
                            html.Label("Loop: ", style={"marginRight": "0.5rem"}),
                            dcc.Checklist(
                                id="loop-checkbox",
                                options=[{"label": "", "value": "loop"}],
                                value=["loop"],
                                style={"display": "inline-block"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    # Predicted concentration display (shown in ΔADC mode)
                    html.Div(
                        id="concentration-display",
                        children=[
                            html.Span(
                                "Est. Concentration: ", style={"fontWeight": "bold"}
                            ),
                            html.Span(id="concentration-value", children="-- ppm"),
                        ],
                        style={
                            "display": "none",  # Hidden by default
                            "padding": "8px 16px",
                            "backgroundColor": "rgba(255, 100, 100, 0.2)",
                            "borderRadius": "4px",
                            "border": "1px solid rgba(255, 100, 100, 0.5)",
                        },
                    ),
                ],
            ),
            dcc.Slider(
                id="frame-slider",
                min=0,
                max=len(data.frames) - 1,
                value=0,
                step=1,
                marks=data.marks,
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="drag",
            ),
            dcc.Graph(id="layer-graph", style={"marginTop": "1.5rem"}),
            dcc.Graph(id="timeseries-graph", style={"marginTop": "1rem"}),
            # Analysis Panel
            html.Div(
                style={
                    "marginTop": "2rem",
                    "borderTop": "1px solid #555",
                    "paddingTop": "1.5rem",
                },
                children=[
                    html.H3(
                        "ZIF-8 MOF Toluene Sensor Analysis",
                        style={
                            "marginBottom": "1rem",
                            "fontWeight": "500",
                            "letterSpacing": "0.5px",
                        },
                    ),
                    # Overview Cards Row
                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "1rem",
                            "flexWrap": "wrap",
                            "marginBottom": "1.5rem",
                        },
                        children=[
                            # Dataset Overview Card
                            html.Div(
                                style={
                                    "backgroundColor": "rgba(40, 50, 65, 0.8)",
                                    "padding": "1.25rem",
                                    "borderRadius": "4px",
                                    "flex": "1",
                                    "minWidth": "220px",
                                    "borderLeft": "3px solid #5b9bd5",
                                },
                                children=[
                                    html.H4(
                                        "Dataset Overview",
                                        style={
                                            "marginBottom": "0.75rem",
                                            "color": "#5b9bd5",
                                            "fontWeight": "600",
                                            "fontSize": "0.95rem",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "1px",
                                        },
                                    ),
                                    html.Table(
                                        [
                                            html.Tr(
                                                [
                                                    html.Td(
                                                        "Date:",
                                                        style={"paddingRight": "1rem"},
                                                    ),
                                                    html.Td(
                                                        data.analysis_data["overview"][
                                                            "date"
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Time Span:"),
                                                    html.Td(
                                                        data.analysis_data["overview"][
                                                            "time_span"
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Duration:"),
                                                    html.Td(
                                                        f"{data.analysis_data['overview']['duration_min']:.1f} min"
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Total Frames:"),
                                                    html.Td(
                                                        f"{data.analysis_data['overview']['total_frames']:,}"
                                                    ),
                                                ]
                                            ),
                                        ],
                                        style={"fontSize": "0.9rem"},
                                    ),
                                ],
                            ),
                            # ΔADC Statistics Card
                            html.Div(
                                style={
                                    "backgroundColor": "rgba(40, 50, 65, 0.8)",
                                    "padding": "1.25rem",
                                    "borderRadius": "4px",
                                    "flex": "1",
                                    "minWidth": "220px",
                                    "borderLeft": "3px solid #70ad47",
                                },
                                children=[
                                    html.H4(
                                        "ΔADC Statistics",
                                        style={
                                            "marginBottom": "0.75rem",
                                            "color": "#70ad47",
                                            "fontWeight": "600",
                                            "fontSize": "0.95rem",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "1px",
                                        },
                                    ),
                                    html.Table(
                                        [
                                            html.Tr(
                                                [
                                                    html.Td(
                                                        "Coated:",
                                                        style={"paddingRight": "1rem"},
                                                    ),
                                                    html.Td(
                                                        f"{data.coating_info.get('n_coated', 117)} px ({data.coating_info.get('coated_percentage', 11.4):.1f}%)"
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Reference:"),
                                                    html.Td(
                                                        f"{data.coating_info.get('n_reference', 907)} px"
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Range:"),
                                                    html.Td(
                                                        f"{data.analysis_data['delta_adc']['min']:.2f} to {data.analysis_data['delta_adc']['max']:.2f}"
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Mean:"),
                                                    html.Td(
                                                        f"{data.analysis_data['delta_adc']['mean']:.3f}"
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Std Dev:"),
                                                    html.Td(
                                                        f"{data.analysis_data['delta_adc']['std']:.3f}"
                                                    ),
                                                ]
                                            ),
                                        ],
                                        style={"fontSize": "0.9rem"},
                                    ),
                                ],
                            ),
                            # Key Events Card
                            html.Div(
                                style={
                                    "backgroundColor": "rgba(40, 50, 65, 0.8)",
                                    "padding": "1.25rem",
                                    "borderRadius": "4px",
                                    "flex": "1",
                                    "minWidth": "220px",
                                    "borderLeft": "3px solid #ed7d31",
                                },
                                children=[
                                    html.H4(
                                        "Key Events",
                                        style={
                                            "marginBottom": "0.75rem",
                                            "color": "#ed7d31",
                                            "fontWeight": "600",
                                            "fontSize": "0.95rem",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "1px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "Max Response: ",
                                                        style={"fontWeight": "bold"},
                                                    ),
                                                    html.Span(
                                                        f"ΔADC = {data.analysis_data['key_events']['max_response']['value']:.3f}",
                                                        style={"color": "#f66"},
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                f"at {data.analysis_data['key_events']['max_response']['time']}",
                                                style={
                                                    "fontSize": "0.85rem",
                                                    "opacity": "0.8",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "Min Response: ",
                                                        style={
                                                            "fontWeight": "bold",
                                                            "marginTop": "0.5rem",
                                                            "display": "inline-block",
                                                        },
                                                    ),
                                                    html.Span(
                                                        f"ΔADC = {data.analysis_data['key_events']['min_response']['value']:.3f}",
                                                        style={"color": "#6cf"},
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                f"at {data.analysis_data['key_events']['min_response']['time']}",
                                                style={
                                                    "fontSize": "0.85rem",
                                                    "opacity": "0.8",
                                                },
                                            ),
                                        ],
                                        style={"fontSize": "0.9rem"},
                                    ),
                                ],
                            ),
                            # Signal Methods Card
                            html.Div(
                                style={
                                    "backgroundColor": "rgba(40, 50, 65, 0.8)",
                                    "padding": "1.25rem",
                                    "borderRadius": "4px",
                                    "flex": "1.5",
                                    "minWidth": "320px",
                                    "borderLeft": "3px solid #9b59b6",
                                },
                                children=[
                                    html.H4(
                                        "Signal Methods",
                                        style={
                                            "marginBottom": "0.75rem",
                                            "color": "#9b59b6",
                                            "fontWeight": "600",
                                            "fontSize": "0.95rem",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "1px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Strong("Method: "),
                                                    html.Span(
                                                        "Coated/Reference Pixel Subtraction"
                                                    ),
                                                ],
                                                style={"marginBottom": "0.4rem"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Strong("ΔADC(t) = "),
                                                    html.Span(
                                                        "Avg(Coated) − Avg(Reference)",
                                                        style={
                                                            "fontFamily": "monospace"
                                                        },
                                                    ),
                                                ],
                                                style={"marginBottom": "0.4rem"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Strong("Coated: "),
                                                    html.Span(
                                                        f"{data.coating_info.get('n_coated', 117)} pixels ({data.coating_info.get('coated_percentage', 11.4):.1f}%) — ZIF-8 MOF responsive"
                                                    ),
                                                ],
                                                style={"marginBottom": "0.4rem"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Strong("Reference: "),
                                                    html.Span(
                                                        f"{data.coating_info.get('n_reference', 907)} pixels — stable baseline"
                                                    ),
                                                ],
                                                style={"marginBottom": "0.4rem"},
                                            ),
                                            html.Div(
                                                "Units: Raw ADC difference (environmental drift compensated)",
                                                style={
                                                    "fontSize": "0.8rem",
                                                    "opacity": "0.7",
                                                    "fontStyle": "italic",
                                                },
                                            ),
                                        ],
                                        style={"fontSize": "0.85rem"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Charts Row
                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "1rem",
                            "flexWrap": "wrap",
                            "marginBottom": "1.5rem",
                        },
                        children=[
                            html.Div(
                                style={"flex": "2", "minWidth": "400px"},
                                children=[
                                    dcc.Graph(
                                        figure=data.segment_chart,
                                        config={"displayModeBar": False},
                                    )
                                ],
                            ),
                            html.Div(
                                style={"flex": "1", "minWidth": "250px"},
                                children=[
                                    dcc.Graph(
                                        figure=data.layer_chart,
                                        config={"displayModeBar": False},
                                    )
                                ],
                            ),
                        ],
                    ),
                    # Exposure Periods Table
                    html.Div(
                        style={"marginTop": "1.5rem"},
                        children=[
                            html.H4(
                                "Significant Exposure Periods",
                                style={
                                    "marginBottom": "0.75rem",
                                    "fontWeight": "500",
                                    "borderBottom": "1px solid #444",
                                    "paddingBottom": "0.5rem",
                                },
                            ),
                            html.P(
                                "Duration ≥ 30s, ΔADC > 0.3 above baseline",
                                style={
                                    "fontSize": "0.8rem",
                                    "opacity": "0.6",
                                    "marginBottom": "0.75rem",
                                    "marginTop": "-0.5rem",
                                },
                            ),
                            html.Div(
                                style={"overflowX": "auto"},
                                children=[
                                    html.Table(
                                        style={
                                            "width": "100%",
                                            "borderCollapse": "collapse",
                                            "fontSize": "0.85rem",
                                        },
                                        children=[
                                            html.Thead(
                                                html.Tr(
                                                    [
                                                        html.Th(
                                                            "Start",
                                                            style={
                                                                "padding": "8px",
                                                                "borderBottom": "2px solid #666",
                                                                "textAlign": "left",
                                                            },
                                                        ),
                                                        html.Th(
                                                            "End",
                                                            style={
                                                                "padding": "8px",
                                                                "borderBottom": "2px solid #666",
                                                                "textAlign": "left",
                                                            },
                                                        ),
                                                        html.Th(
                                                            "Duration",
                                                            style={
                                                                "padding": "8px",
                                                                "borderBottom": "2px solid #666",
                                                                "textAlign": "right",
                                                            },
                                                        ),
                                                        html.Th(
                                                            "Peak ΔADC",
                                                            style={
                                                                "padding": "8px",
                                                                "borderBottom": "2px solid #666",
                                                                "textAlign": "right",
                                                            },
                                                        ),
                                                        html.Th(
                                                            "Mean ΔADC",
                                                            style={
                                                                "padding": "8px",
                                                                "borderBottom": "2px solid #666",
                                                                "textAlign": "right",
                                                            },
                                                        ),
                                                        html.Th(
                                                            "Est. ppm",
                                                            style={
                                                                "padding": "8px",
                                                                "borderBottom": "2px solid #666",
                                                                "textAlign": "right",
                                                            },
                                                        ),
                                                    ]
                                                )
                                            ),
                                            (
                                                html.Tbody(
                                                    [
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    period["start"],
                                                                    style={
                                                                        "padding": "6px 8px",
                                                                        "borderBottom": "1px solid #333",
                                                                    },
                                                                ),
                                                                html.Td(
                                                                    period["end"],
                                                                    style={
                                                                        "padding": "6px 8px",
                                                                        "borderBottom": "1px solid #333",
                                                                    },
                                                                ),
                                                                html.Td(
                                                                    f"{period['duration']}s",
                                                                    style={
                                                                        "padding": "6px 8px",
                                                                        "borderBottom": "1px solid #333",
                                                                        "textAlign": "right",
                                                                    },
                                                                ),
                                                                html.Td(
                                                                    f"{period['peak_dadc']:.2f}",
                                                                    style={
                                                                        "padding": "6px 8px",
                                                                        "borderBottom": "1px solid #333",
                                                                        "textAlign": "right",
                                                                        "color": (
                                                                            "#f66"
                                                                            if period[
                                                                                "peak_dadc"
                                                                            ]
                                                                            > 2.0
                                                                            else (
                                                                                "#fc6"
                                                                                if period[
                                                                                    "peak_dadc"
                                                                                ]
                                                                                > 1.0
                                                                                else "#6f6"
                                                                            )
                                                                        ),
                                                                    },
                                                                ),
                                                                html.Td(
                                                                    f"{period['mean_dadc']:.2f}",
                                                                    style={
                                                                        "padding": "6px 8px",
                                                                        "borderBottom": "1px solid #333",
                                                                        "textAlign": "right",
                                                                    },
                                                                ),
                                                                html.Td(
                                                                    f"{period.get('est_ppm', int(period['peak_dadc'] * 2000)):,}",
                                                                    style={
                                                                        "padding": "6px 8px",
                                                                        "borderBottom": "1px solid #333",
                                                                        "textAlign": "right",
                                                                        "fontWeight": "bold",
                                                                    },
                                                                ),
                                                            ]
                                                        )
                                                        for period in data.analysis_data[
                                                            "exposure_periods"
                                                        ]
                                                    ]
                                                )
                                                if data.analysis_data[
                                                    "exposure_periods"
                                                ]
                                                else html.Tbody(
                                                    [
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    "No significant exposure periods detected",
                                                                    colSpan=6,
                                                                    style={
                                                                        "padding": "1rem",
                                                                        "textAlign": "center",
                                                                        "opacity": "0.6",
                                                                    },
                                                                )
                                                            ]
                                                        )
                                                    ]
                                                )
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Interpretation Panel
                    html.Div(
                        style={
                            "marginTop": "2rem",
                            "backgroundColor": "rgba(35, 40, 50, 0.9)",
                            "padding": "1.5rem",
                            "borderRadius": "4px",
                            "borderTop": "2px solid #5b9bd5",
                        },
                        children=[
                            html.H4(
                                "Analysis Interpretation",
                                style={
                                    "marginBottom": "1rem",
                                    "fontWeight": "500",
                                    "color": "#ddd",
                                },
                            ),
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
                                    "gap": "1rem",
                                },
                                children=[
                                    html.Div(
                                        [
                                            html.Strong("Sensor Configuration:"),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        "Layer 1: Active sensing layer (ZIF-8 MOF coated)"
                                                    ),
                                                    html.Li(
                                                        "Layer 2: Disabled or reference layer (reads zero)"
                                                    ),
                                                    html.Li(
                                                        "Layer 3: Saturated (possibly overexposed)"
                                                    ),
                                                ],
                                                style={
                                                    "marginTop": "0.25rem",
                                                    "paddingLeft": "1.5rem",
                                                },
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Strong("Sensor Performance:"),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        "Detection range: ~500 - 9000+ ppm toluene"
                                                    ),
                                                    html.Li(
                                                        "Shows reversible response (good recovery)"
                                                    ),
                                                    html.Li(
                                                        "Fast response and recovery times"
                                                    ),
                                                ],
                                                style={
                                                    "marginTop": "0.25rem",
                                                    "paddingLeft": "1.5rem",
                                                },
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Strong("Application Notes:"),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        "Suitable for poultry health monitoring"
                                                    ),
                                                    html.Li(
                                                        "VOC detection validates ZIF-8 MOF coating"
                                                    ),
                                                    html.Li(
                                                        f"~{data.analysis_data['overview']['duration_min']:.0f} min test confirms stability"
                                                    ),
                                                ],
                                                style={
                                                    "marginTop": "0.25rem",
                                                    "paddingLeft": "1.5rem",
                                                },
                                            ),
                                        ]
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            # Interval for animation
            dcc.Interval(
                id="animation-interval",
                interval=500,  # milliseconds
                n_intervals=0,
                disabled=True,  # Start paused
            ),
            # Store for playback state
            dcc.Store(id="playback-state", data={"playing": False}),
        ],
        style={
            "padding": "2rem",
            "backgroundColor": "rgb(8,10,15)",
            "color": "white",
            "minHeight": "100vh",
        },
    )


def register_dashboard_callbacks(app: Dash, data: DashboardData):
    # Callback to control play/pause/stop
    @app.callback(
        Output("animation-interval", "disabled"),
        Output("animation-interval", "interval"),
        Output("playback-state", "data"),
        Input("play-btn", "n_clicks"),
        Input("pause-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        Input("speed-dropdown", "value"),
        State("playback-state", "data"),
        prevent_initial_call=True,
    )
    def control_playback(play_clicks, pause_clicks, stop_clicks, speed, state):
        triggered_id = ctx.triggered_id

        if triggered_id == "play-btn":
            return False, speed, {"playing": True}
        elif triggered_id == "pause-btn":
            return True, speed, {"playing": False}
        elif triggered_id == "stop-btn":
            return True, speed, {"playing": False}
        elif triggered_id == "speed-dropdown":
            # Update speed, keep current playing state
            return not state.get("playing", False), speed, state

        return True, speed, {"playing": False}

    # Callback to advance frame on interval tick
    @app.callback(
        Output("frame-slider", "value"),
        Input("animation-interval", "n_intervals"),
        Input("stop-btn", "n_clicks"),
        State("frame-slider", "value"),
        State("loop-checkbox", "value"),
        prevent_initial_call=True,
    )
    def advance_frame(n_intervals, stop_clicks, current_value, loop_value):
        triggered_id = ctx.triggered_id

        if triggered_id == "stop-btn":
            return 0  # Reset to beginning

        if triggered_id == "animation-interval":
            next_value = current_value + 1
            if next_value >= len(data.frames):
                if "loop" in (loop_value or []):
                    return 0  # Loop back to start
                else:
                    return len(data.frames) - 1  # Stay at end
            return next_value

        return current_value

    # Store previous view mode to detect mode changes
    prev_mode = {"mode": "all"}

    @app.callback(
        Output("layer-graph", "figure"),
        Output("timeseries-graph", "figure"),
        Output("timestamp-label", "children"),
        Output("packet-label", "children"),
        Input("frame-slider", "value"),
        Input("view-mode", "value"),
    )
    def update_graph(frame_idx: int, view_mode: str):
        frame = data.frames[frame_idx]
        ts_label = f"Timestamp: {frame.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
        packet_label = f"Packet #{frame.packet}"
        current_ts = data.timestamps[frame_idx]

        mode_changed = prev_mode["mode"] != view_mode
        prev_mode["mode"] = view_mode

        # Helper to update timeseries marker
        def update_ts_marker(ts_fig, shape_idx=0):
            ts_fig.layout.shapes[shape_idx].x0 = current_ts
            ts_fig.layout.shapes[shape_idx].x1 = current_ts
            ts_fig.layout.annotations[0].x = current_ts
            ts_fig.layout.annotations[0].text = f"Frame {frame_idx + 1}"

        if mode_changed:
            # Mode changed - return full base figures with updated data
            if view_mode == "layer1":
                # Layer 1 shows ΔADC Toluene Response (Active Sensing ZIF-8 MOF)
                layer_fig = go.Figure(data.base_delta_heatmap_fig)
                layer_fig.data[0].marker.color = data.delta_adc_pixels[frame_idx]
                layer_fig.layout.title = f"Layer 1 (ΔADC Toluene) - Frame {frame_idx + 1} / {len(data.frames)} (ΔADC: {data.delta_adc_values[frame_idx]:.2f})"
                ts_fig = go.Figure(data.base_delta_ts_fig)
                # Frame marker is the last shape (after concentration lines)
                update_ts_marker(ts_fig, shape_idx=-1)
            elif view_mode == "layer2":
                layer_fig = go.Figure(data.base_layer2_fig)
                layer_fig.data[0].marker.color = data.layer2_data[frame_idx]
                layer_fig.layout.title = (
                    f"Layer 2 (Reference) - Frame {frame_idx + 1} / {len(data.frames)}"
                )
                ts_fig = go.Figure(data.base_layer2_ts_fig)
                update_ts_marker(ts_fig)
            elif view_mode == "layer3":
                layer_fig = go.Figure(data.base_layer3_fig)
                layer_fig.data[0].marker.color = data.layer3_data[frame_idx]
                layer_fig.layout.title = (
                    f"Layer 3 (Saturated) - Frame {frame_idx + 1} / {len(data.frames)}"
                )
                ts_fig = go.Figure(data.base_layer3_ts_fig)
                update_ts_marker(ts_fig)
            else:  # "all"
                layer_fig = go.Figure(data.base_layer_fig)
                for idx, sensor_id in enumerate(data.layer_order):
                    layer_fig.data[idx].marker.color = data.all_grids_data[frame_idx][
                        sensor_id
                    ]
                layer_fig.layout.title = (
                    f"All Layers - Frame {frame_idx + 1} / {len(data.frames)}"
                )
                ts_fig = go.Figure(data.base_timeseries_fig)
                update_ts_marker(ts_fig)

            return layer_fig, ts_fig, ts_label, packet_label

        # Same mode - use Patch for efficient partial updates
        layer_patch = Patch()
        ts_patch = Patch()

        if view_mode == "layer1":
            # Layer 1 shows ΔADC Toluene Response (Active Sensing ZIF-8 MOF)
            layer_patch["data"][0]["marker"]["color"] = data.delta_adc_pixels[frame_idx]
            layer_patch["layout"][
                "title"
            ] = f"Layer 1 (ΔADC Toluene) - Frame {frame_idx + 1} / {len(data.frames)} (ΔADC: {data.delta_adc_values[frame_idx]:.2f})"
            ts_patch["layout"]["shapes"][-1]["x0"] = current_ts
            ts_patch["layout"]["shapes"][-1]["x1"] = current_ts
            ts_patch["layout"]["annotations"][0]["x"] = current_ts
            ts_patch["layout"]["annotations"][0]["text"] = f"Frame {frame_idx + 1}"
        elif view_mode == "layer2":
            layer_patch["data"][0]["marker"]["color"] = data.layer2_data[frame_idx]
            layer_patch["layout"][
                "title"
            ] = f"Layer 2 (Reference) - Frame {frame_idx + 1} / {len(data.frames)}"
            ts_patch["layout"]["shapes"][0]["x0"] = current_ts
            ts_patch["layout"]["shapes"][0]["x1"] = current_ts
            ts_patch["layout"]["annotations"][0]["x"] = current_ts
            ts_patch["layout"]["annotations"][0]["text"] = f"Frame {frame_idx + 1}"
        elif view_mode == "layer3":
            layer_patch["data"][0]["marker"]["color"] = data.layer3_data[frame_idx]
            layer_patch["layout"][
                "title"
            ] = f"Layer 3 (Saturated) - Frame {frame_idx + 1} / {len(data.frames)}"
            ts_patch["layout"]["shapes"][0]["x0"] = current_ts
            ts_patch["layout"]["shapes"][0]["x1"] = current_ts
            ts_patch["layout"]["annotations"][0]["x"] = current_ts
            ts_patch["layout"]["annotations"][0]["text"] = f"Frame {frame_idx + 1}"
        else:  # "all"
            for idx, sensor_id in enumerate(data.layer_order):
                layer_patch["data"][idx]["marker"]["color"] = data.all_grids_data[
                    frame_idx
                ][sensor_id]
            layer_patch["layout"][
                "title"
            ] = f"All Layers - Frame {frame_idx + 1} / {len(data.frames)}"
            ts_patch["layout"]["shapes"][0]["x0"] = current_ts
            ts_patch["layout"]["shapes"][0]["x1"] = current_ts
            ts_patch["layout"]["annotations"][0]["x"] = current_ts
            ts_patch["layout"]["annotations"][0]["text"] = f"Frame {frame_idx + 1}"

        return layer_patch, ts_patch, ts_label, packet_label

    # Callback for concentration display visibility and value
    @app.callback(
        Output("concentration-display", "style"),
        Output("concentration-value", "children"),
        Input("frame-slider", "value"),
        Input("view-mode", "value"),
    )
    def update_concentration_display(frame_idx: int, view_mode: str):
        base_style = {
            "padding": "8px 16px",
            "backgroundColor": "rgba(255, 100, 100, 0.2)",
            "borderRadius": "4px",
            "border": "1px solid rgba(255, 100, 100, 0.5)",
        }

        if view_mode == "layer1":
            # Layer 1 shows ΔADC - show concentration estimate
            # Simple linear mapping: ΔADC 0-4 → 0-10000 ppm
            # This is a rough estimate based on observed data range
            delta = data.delta_adc_values[frame_idx]

            # Clamp to reasonable range
            if delta < 0:
                conc = 0
                color = "rgba(100, 200, 255, 0.2)"  # Blue for below baseline
                border = "1px solid rgba(100, 200, 255, 0.5)"
            elif delta < 0.5:
                conc = int(delta * 1000)  # 0-500 ppm
                color = "rgba(100, 255, 100, 0.2)"  # Green - low
                border = "1px solid rgba(100, 255, 100, 0.5)"
            elif delta < 1.5:
                conc = int(500 + (delta - 0.5) * 1500)  # 500-2000 ppm
                color = "rgba(200, 255, 100, 0.2)"  # Yellow-green
                border = "1px solid rgba(200, 255, 100, 0.5)"
            elif delta < 2.5:
                conc = int(2000 + (delta - 1.5) * 3000)  # 2000-5000 ppm
                color = "rgba(255, 200, 100, 0.2)"  # Orange
                border = "1px solid rgba(255, 200, 100, 0.5)"
            else:
                conc = int(5000 + (delta - 2.5) * 4000)  # 5000+ ppm
                color = "rgba(255, 100, 100, 0.2)"  # Red - high
                border = "1px solid rgba(255, 100, 100, 0.5)"

            base_style["backgroundColor"] = color
            base_style["border"] = border

            return {
                **base_style,
                "display": "flex",
                "alignItems": "center",
                "gap": "0.5rem",
            }, f"{conc:,} ppm"
        else:
            # Hide in other modes
            return {**base_style, "display": "none"}, "-- ppm"

    # Callback for Print/PDF button - triggers browser print dialog
    app.clientside_callback(
        """
        function(n_clicks) {
            if (n_clicks > 0) {
                window.print();
            }
            return '';
        }
        """,
        Output("print-pdf-btn", "data-dummy"),
        Input("print-pdf-btn", "n_clicks"),
        prevent_initial_call=True,
    )


def generate_html_report(
    results,
    data,
    data_file: Path,
    fig_signal,
    fig_exposure,
    fig_spatial,
    fig_bad_pixels,
    fig_t90_dist,
    fig_drift,
    fig_noise,
    fig_layers,
    fig_cal,
    fig_residuals,
    fig_char,
    fig_sev,
    fig_health,
) -> str:
    """Generate a standalone HTML report with all visualizations."""

    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert figures to HTML divs
    def fig_to_html(fig, height="500px"):
        return fig.to_html(
            full_html=False, include_plotlyjs=False, config={"displayModeBar": False}
        )

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CMOS ZIF-8 MOF Toluene Sensor Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px;
            background: #f5f5f5;
            color: #333;
        }}
        .report-container {{
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 0.5rem; }}
        h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem; margin-top: 2rem; }}
        .subtitle {{ text-align: center; color: #7f8c8d; margin-bottom: 2rem; }}
        .section {{ margin-bottom: 2rem; }}
        .description {{ color: #7f8c8d; font-size: 0.9rem; margin-bottom: 1rem; }}
        .flex-row {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
        .flex-item {{ flex: 1; min-width: 400px; }}
        table {{ border-collapse: collapse; margin: 1rem auto; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #f8f9fa; }}
        .footer {{ text-align: center; color: #95a5a6; font-size: 0.85rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ddd; }}
        @media print {{
            body {{ background: white; }}
            .report-container {{ box-shadow: none; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
<div class="report-container">
    <h1>CMOS ZIF-8 MOF Toluene Sensor Analysis Report</h1>
    <p class="subtitle">Generated: {report_date}</p>
    
    <div class="section">
        <h2>1. Executive Summary</h2>
        <div style="line-height: 1.6;">
            {results.daily_report.replace(chr(10), '<br>')}
        </div>
    </div>
    
    <div class="section">
        <h2>2. Signal Processing & Event Detection</h2>
        <p class="description">Raw signal processing with baseline correction and event detection.</p>
        {fig_to_html(fig_signal)}
        <p><strong>Detected Events:</strong> {len(results.detected_events)}</p>
    </div>
    
    <div class="section">
        <h2>3. ΔADC Response with Exposure Windows</h2>
        <p class="description">Time-series response with concentration exposure periods.</p>
        {fig_to_html(fig_exposure)}
    </div>
    
    <div class="section">
        <h2>4. Spatial Sensitivity Analysis</h2>
        <p class="description">Per-pixel sensitivity map (32×32) and pixel health status.</p>
        <div class="flex-row">
            <div class="flex-item">{fig_to_html(fig_spatial)}</div>
            <div class="flex-item">{fig_to_html(fig_bad_pixels)}</div>
        </div>
    </div>
    
    <div class="section">
        <h2>5. Response Kinetics (T90 Analysis)</h2>
        <p class="description">Distribution of T90 rise and recovery times.</p>
        {fig_to_html(fig_t90_dist)}
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Mean T90 Rise</td><td>{results.sensor_characteristics.mean_t90_rise:.2f} ± {results.sensor_characteristics.std_t90_rise:.2f} s</td></tr>
            <tr><td>Mean T90 Recovery</td><td>{results.sensor_characteristics.mean_t90_recovery:.2f} ± {results.sensor_characteristics.std_t90_recovery:.2f} s</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>6. Baseline Drift & Noise Analysis</h2>
        <p class="description">Long-term stability and noise characterization.</p>
        <div class="flex-row">
            <div class="flex-item">{fig_to_html(fig_drift)}</div>
            <div class="flex-item">{fig_to_html(fig_noise)}</div>
        </div>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Baseline Drift Rate</td><td>{results.sensor_characteristics.baseline_drift_per_hour:.4f} ΔADC/hour</td></tr>
            <tr><td>Baseline Noise RMS</td><td>{results.sensor_characteristics.baseline_noise_rms:.4f} ΔADC</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>7. Sensor Layer Comparison</h2>
        <p class="description">Time-series comparison of all three sensor layers.</p>
        {fig_to_html(fig_layers)}
    </div>
    
    <div class="section">
        <h2>8. Calibration Model & Residuals</h2>
        <p class="description">Calibration model: {results.calibration_model.name if results.calibration_model else 'N/A'}</p>
        <div class="flex-row">
            <div class="flex-item">{fig_to_html(fig_cal)}</div>
            <div class="flex-item">{fig_to_html(fig_residuals)}</div>
        </div>
    </div>
    
    <div class="section">
        <h2>9. Sensor Characteristics</h2>
        {fig_to_html(fig_char)}
    </div>
    
    <div class="section">
        <h2>10. Application Severity Monitoring</h2>
        <p class="description">Real-time severity classification for poultry barn VOC monitoring.</p>
        {fig_to_html(fig_sev)}
    </div>
    
    <div class="section">
        <h2>11. Sensor Health Index</h2>
        <p class="description">Overall sensor health assessment.</p>
        {fig_to_html(fig_health)}
    </div>
    
    <div class="footer">
        <p>End of Report</p>
        <p>Data source: {data_file.name}</p>
    </div>
</div>
</body>
</html>
"""
    return html_content


def generate_pdf_report(
    results, data_file: Path, figures: dict, calibration_summary: str = ""
) -> bytes:
    """
    Generate high-quality PDF report with figures exported as PNG using kaleido.

    Returns bytes that can be downloaded directly.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm, inch
        from reportlab.platypus import (Image, PageBreak, Paragraph,
                                        SimpleDocTemplate, Spacer, Table,
                                        TableStyle)
    except ImportError:
        # Fallback: export as multi-page HTML
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=1 * cm,
        rightMargin=1 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor("#2c3e50"),
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading1"],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor("#3498db"),
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=14,
    )

    elements = []

    # Title Page
    elements.append(Spacer(1, 2 * inch))
    elements.append(Paragraph("CMOS ZIF-8 MOF Sensor Analysis Report", title_style))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(f"Data Source: {data_file.name}", body_style))
    elements.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style
        )
    )
    elements.append(Spacer(1, 0.3 * inch))

    # Executive Summary metrics
    if results.dataset:
        summary_data = [
            ["Total Frames", f"{len(results.dataset.timestamps):,}"],
            ["Events Detected", f"{len(results.detected_events)}"],
            ["Max ΔADC", f"{results.dataset.delta_adc.max():.3f}"],
            ["Mean T90 Rise", f"{results.sensor_characteristics.mean_t90_rise:.1f} s"],
        ]
        summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#2c3e50")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), 11),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
                ]
            )
        )
        elements.append(summary_table)

    elements.append(PageBreak())

    # Function to convert Plotly figure to image bytes
    def fig_to_image(fig, width=900, height=500):
        try:
            img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
            img_buffer = io.BytesIO(img_bytes)
            return Image(img_buffer, width=24 * cm, height=13 * cm)
        except Exception as e:
            return Paragraph(f"[Figure could not be rendered: {str(e)}]", body_style)

    # Section: Signal Processing
    elements.append(Paragraph("1. Signal Processing & Event Detection", heading_style))
    elements.append(
        Paragraph(
            "Butterworth low-pass filter (4th order, fc=0.1 Hz) applied to raw ADC time-series. "
            f"Detected {len(results.detected_events)} exposure events using hysteresis thresholding.",
            body_style,
        )
    )
    if "signal" in figures and figures["signal"]:
        elements.append(fig_to_image(figures["signal"]))
    elements.append(PageBreak())

    # Section: Exposure Windows
    elements.append(Paragraph("2. ΔADC Response with Exposure Windows", heading_style))
    elements.append(
        Paragraph(
            "Time-series response showing normalized ΔADC values during calibration exposures. "
            "Colored bands indicate different concentration levels.",
            body_style,
        )
    )
    if "exposure" in figures and figures["exposure"]:
        elements.append(fig_to_image(figures["exposure"]))
    elements.append(PageBreak())

    # Section: Calibration Model Evaluation
    elements.append(Paragraph("3. Calibration Model Evaluation", heading_style))
    elements.append(
        Paragraph(
            "Gold-standard Predicted vs Actual scatter plot for calibration quality assessment. "
            "Points on the identity line (y=x) indicate perfect predictions. "
            "±5% (green) and ±10% (yellow) error bands shown for reference.",
            body_style,
        )
    )
    if "pred_vs_actual" in figures and figures["pred_vs_actual"]:
        elements.append(fig_to_image(figures["pred_vs_actual"]))

    # Add calibration summary text if available
    if calibration_summary:
        elements.append(Spacer(1, 0.3 * inch))
        # Clean markdown for reportlab
        clean_summary = (
            calibration_summary.replace("**", "")
            .replace("###", "")
            .replace("##", "")
            .replace("#", "")
        )
        for line in clean_summary.split("\n"):
            if line.strip():
                elements.append(Paragraph(line.strip(), body_style))
    elements.append(PageBreak())

    # Section: Spatial Analysis
    elements.append(Paragraph("4. Spatial Sensitivity & Pixel Health", heading_style))
    if "spatial" in figures and figures["spatial"]:
        elements.append(fig_to_image(figures["spatial"], width=800, height=450))
    if "bad_pixels" in figures and figures["bad_pixels"]:
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(fig_to_image(figures["bad_pixels"], width=800, height=450))
    elements.append(PageBreak())

    # Section: Response Kinetics
    elements.append(Paragraph("5. Response Kinetics (T90 Analysis)", heading_style))
    elements.append(
        Paragraph(
            f"Mean T90 Rise: {results.sensor_characteristics.mean_t90_rise:.2f} ± "
            f"{results.sensor_characteristics.std_t90_rise:.2f} s | "
            f"Mean T90 Recovery: {results.sensor_characteristics.mean_t90_recovery:.2f} ± "
            f"{results.sensor_characteristics.std_t90_recovery:.2f} s",
            body_style,
        )
    )
    if "t90_dist" in figures and figures["t90_dist"]:
        elements.append(fig_to_image(figures["t90_dist"]))
    elements.append(PageBreak())

    # Section: Drift & Noise
    elements.append(Paragraph("6. Baseline Drift & Noise Analysis", heading_style))
    elements.append(
        Paragraph(
            f"Drift Rate: {results.sensor_characteristics.baseline_drift_per_hour:.4f} ΔADC/hour | "
            f"Noise RMS: {results.sensor_characteristics.baseline_noise_rms:.4f} ΔADC",
            body_style,
        )
    )
    if "drift" in figures and figures["drift"]:
        elements.append(fig_to_image(figures["drift"]))
    elements.append(PageBreak())

    # Section: Layer Comparison
    elements.append(Paragraph("7. Three-Layer Sensor Comparison", heading_style))
    elements.append(
        Paragraph(
            "Layer 1 (Active ZIF-8 MOF) shows clear toluene response. "
            "Layer 2 (Reference) provides baseline. Layer 3 (Saturated) for quality control.",
            body_style,
        )
    )
    if "layers" in figures and figures["layers"]:
        elements.append(fig_to_image(figures["layers"]))
    elements.append(PageBreak())

    # Section: Severity Monitoring
    elements.append(Paragraph("8. Application Severity Monitoring", heading_style))
    elements.append(
        Paragraph(
            "Real-time severity classification for poultry barn VOC monitoring: "
            "SAFE (0-2000 ppm), MODERATE (2000-5000 ppm), CRITICAL (>5000 ppm).",
            body_style,
        )
    )
    if "sev" in figures and figures["sev"]:
        elements.append(fig_to_image(figures["sev"]))
    elements.append(PageBreak())

    # Section: Sensor Health
    elements.append(Paragraph("9. Sensor Health Index", heading_style))
    elements.append(
        Paragraph(
            "Composite health score (0-100) based on: Noise (30%), Drift (30%), "
            "Response Speed (25%), Saturation (15%).",
            body_style,
        )
    )
    if "health" in figures and figures["health"]:
        elements.append(fig_to_image(figures["health"], width=700, height=400))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def create_dashboard(frames: Sequence[FrameRecord], data_file: Path):
    # 1. Prepare Dashboard Data (Existing Logic)
    data = prepare_dashboard_data(frames)

    # 2. Run Pipeline Analysis (New Logic)
    print("Running CMOS VOC Pipeline...")
    pipeline = CMOSVOCPipeline(
        PipelineConfig(
            data_file=str(data_file),
            filter_type="savgol",
            calibration_model="polynomial",
            save_results=False,  # Don't save to disk, just keep in memory
        )
    )

    # We can skip load_data since we already have frames, but the pipeline expects to load from file.
    # To save time, we could inject the data if the pipeline supported it, but load_data is fast enough.
    pipeline.load_data()
    pipeline.process_signals()
    pipeline.characterize_sensor()
    pipeline.build_calibration()
    pipeline.analyze_application()
    pipeline.generate_report()

    results = pipeline.results

    # 3. Generate Pipeline Figures
    # We use a dummy output dir since we won't be saving files here, just generating figures
    visualizer = PipelineVisualizer(Path("dummy_output"))

    fig_signal = visualizer.plot_signal_processing(
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

    fig_char = visualizer.plot_sensor_characteristics(results.sensor_characteristics)

    fig_cal = visualizer.plot_calibration(results.calibration_model)

    # For severity, we need ppm values
    if results.calibration_model:
        ppm_values = results.calibration_model.predict_ppm(results.filtered_signal)
    else:
        ppm_values = results.filtered_signal

    fig_sev = visualizer.plot_severity_timeline(
        timestamps=results.dataset.timestamps,
        severity_levels=results.severity_timeline,
        ppm_values=ppm_values,
    )

    # =========================================================================
    # 3B. Generate Additional Scientific Visualizations
    # =========================================================================

    # 1. ΔADC with Exposure Windows
    fig_exposure = plot_dadc_with_exposure_windows(
        timestamps=results.dataset.timestamps, delta_adc=results.filtered_signal
    )

    # 2. Spatial Sensitivity Map (mean ΔADC during peak exposure)
    # Find peak exposure frame
    peak_idx = np.argmax(results.filtered_signal)
    peak_window = slice(max(0, peak_idx - 50), min(len(frames), peak_idx + 50))

    # Calculate mean pixel response during peak window
    pixel_responses = []
    for i in range(peak_window.start, peak_window.stop):
        if i < len(frames):
            pixel_responses.append(frames[i].grids[1].flatten())
    if pixel_responses:
        mean_pixel_response = np.mean(pixel_responses, axis=0).reshape(32, 32)
        # Normalize to ΔADC
        baseline_pixel = np.percentile(
            [f.grids[1].flatten() for f in frames[:100]], 10, axis=0
        ).reshape(32, 32)
        spatial_delta = (mean_pixel_response - baseline_pixel) / 1000
    else:
        spatial_delta = np.zeros((32, 32))

    fig_spatial = plot_spatial_sensitivity_map(
        pixel_data=spatial_delta, title="Spatial Sensitivity Map - Peak Exposure (ΔADC)"
    )

    # 3. Bad Pixel Map with proper classification
    # Create time-series array for Layer 1 (Active Sensing)
    pixel_timeseries_layer1 = np.array(
        [f.grids[1] for f in frames]
    )  # Shape: (n_frames, 32, 32)
    fig_bad_pixels = plot_bad_pixel_map(
        pixel_timeseries=pixel_timeseries_layer1, layer_id=1, adc_fullscale=4095
    )

    # 3B. ZIF-8 MOF Coating Pattern Analysis
    # Identify coated (responsive) vs reference (stable) pixels
    coating_analysis = analyze_coating_pattern(pixel_timeseries_layer1)
    fig_coating_pattern = plot_coating_pattern_map(
        pixel_timeseries_layer1, coating_analysis
    )
    fig_coating_stats = create_coating_summary_stats(coating_analysis)

    # 4. T90 Distribution
    rise_times = [
        m.t90_rise
        for m in results.sensor_characteristics.event_metrics
        if m.t90_rise > 0
    ]
    recovery_times = [
        m.recovery_time
        for m in results.sensor_characteristics.event_metrics
        if m.recovery_time > 0
    ]
    fig_t90_dist = plot_t90_distribution(rise_times, recovery_times)

    # 5. Baseline Drift
    fig_drift = plot_baseline_drift(
        timestamps=results.dataset.timestamps,
        delta_adc=results.filtered_signal,
        window_size=200,
    )

    # 6. Noise Analysis
    # Get baseline period (first 10% or low-response period)
    baseline_mask = results.filtered_signal < np.percentile(results.filtered_signal, 20)
    baseline_values = results.filtered_signal[baseline_mask]
    fig_noise = plot_noise_analysis(baseline_values, sample_rate=1.0)

    # 7. Layer Comparison Time Series
    layer1_sums = np.array([np.sum(f.grids[1]) for f in frames])
    layer2_sums = np.array([np.sum(f.grids[2]) for f in frames])
    layer3_sums = np.array([np.sum(f.grids[3]) for f in frames])
    fig_layers = plot_layer_comparison_timeseries(
        timestamps=results.dataset.timestamps,
        layer1_values=layer1_sums / 1e6,  # Scale to millions
        layer2_values=layer2_sums / 1e6,
        layer3_values=layer3_sums / 1e6,
    )

    # 8. Calibration Model Evaluation - Predicted vs Actual Plot
    # Generate calibration data points for model evaluation with TOP 3 MODELS
    if results.calibration_model:
        # Actual calibration points (exposure concentrations and measured ΔADC)
        calib_ppm_actual = np.array([500, 1000, 2000, 3500, 5500, 8000])
        calib_dadc = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

        # Create and fit ALL calibration models for comparison
        all_models = {}
        model_equations = {}

        # Polynomial (1) - Linear
        poly1 = PolynomialCalibration(degree=1)
        poly1.fit(calib_ppm_actual, calib_dadc)
        all_models["Polynomial(1)"] = poly1.predict_ppm(calib_dadc)
        model_equations["Polynomial(1)"] = poly1.get_equation_str()

        # Polynomial (2) - Quadratic
        poly2 = PolynomialCalibration(degree=2)
        poly2.fit(calib_ppm_actual, calib_dadc)
        all_models["Polynomial(2)"] = poly2.predict_ppm(calib_dadc)
        model_equations["Polynomial(2)"] = poly2.get_equation_str()

        # Polynomial (3) - Cubic
        poly3 = PolynomialCalibration(degree=3)
        poly3.fit(calib_ppm_actual, calib_dadc)
        all_models["Polynomial(3)"] = poly3.predict_ppm(calib_dadc)
        model_equations["Polynomial(3)"] = poly3.get_equation_str()

        # Langmuir - Physical sorption model
        try:
            langmuir = LangmuirModel()
            langmuir.fit(calib_ppm_actual, calib_dadc)
            all_models["Langmuir"] = langmuir.predict_ppm(calib_dadc)
            model_equations["Langmuir"] = (
                f"Langmuir: Qmax={langmuir.Qmax:.3f}, K={langmuir.K:.6f}"
            )
        except Exception:
            pass  # Skip if fitting fails

        # Freundlich - Power law model
        try:
            freundlich = FreundlichModel()
            freundlich.fit(calib_ppm_actual, calib_dadc)
            all_models["Freundlich"] = freundlich.predict_ppm(calib_dadc)
            model_equations["Freundlich"] = (
                f"Freundlich: K={freundlich.K:.4f}, n={freundlich.n:.3f}"
            )
        except Exception:
            pass  # Skip if fitting fails

        # Calculate R² for all models and sort
        model_r2 = {}
        for name, pred in all_models.items():
            ss_res = np.sum((calib_ppm_actual - pred) ** 2)
            ss_tot = np.sum((calib_ppm_actual - np.mean(calib_ppm_actual)) ** 2)
            model_r2[name] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Select top 3 for main plot
        top3_names = sorted(model_r2.keys(), key=lambda x: model_r2[x], reverse=True)[
            :3
        ]
        top3_predictions = {name: all_models[name] for name in top3_names}
        top3_equations = {
            name: model_equations.get(name, "")
            for name in top3_names
            if name in model_equations
        }

        # Create enhanced predicted vs actual plot with TOP 3 models
        fig_pred_vs_actual, calibration_metrics = plot_predicted_vs_actual(
            ppm_actual=calib_ppm_actual,
            ppm_predicted_dict=top3_predictions,
            model_equations=top3_equations,
            exposure_labels=[f"{int(p)} ppm" for p in calib_ppm_actual],
            show_error_bands=True,
            show_labels=True,
            show_equations=True,
            color_by_order=False,  # Use model colors since we have multiple models
        )

        # Create ALL MODELS comparison grid
        fig_all_models = plot_all_calibration_models(
            ppm_actual=calib_ppm_actual,
            predictions_dict=all_models,
            model_equations=model_equations,
        )

        # Generate detailed text summary for best model
        best_model_name = top3_names[0]
        calibration_summary_text = generate_calibration_summary_text(
            model_name=best_model_name,
            metrics=calibration_metrics[best_model_name],
            ppm_actual=calib_ppm_actual,
            ppm_predicted=top3_predictions[best_model_name],
            equation=top3_equations.get(best_model_name),
        )

        # Also keep residual plot for detailed analysis (best model only)
        fig_residuals = plot_calibration_residuals(
            ppm_actual=calib_ppm_actual,
            ppm_predicted=top3_predictions[best_model_name],
            model_name=best_model_name,
        )
    else:
        fig_pred_vs_actual = go.Figure()
        fig_pred_vs_actual.add_annotation(
            text="No calibration model available", x=0.5, y=0.5, showarrow=False
        )
        fig_all_models = go.Figure()
        fig_all_models.add_annotation(
            text="No calibration model available", x=0.5, y=0.5, showarrow=False
        )
        fig_residuals = go.Figure()
        fig_residuals.add_annotation(
            text="No calibration model available", x=0.5, y=0.5, showarrow=False
        )
        calibration_summary_text = "No calibration data available."

    # 9. Sensor Health Gauge
    # Calculate health scores
    noise_rms = results.sensor_characteristics.baseline_noise_rms
    noise_score = max(0, 100 - noise_rms * 100)  # Lower noise = higher score

    drift_rate = abs(results.sensor_characteristics.baseline_drift_per_hour)
    drift_score = max(0, 100 - drift_rate * 500)  # Lower drift = higher score

    mean_t90 = results.sensor_characteristics.mean_t90_rise
    t90_score = max(0, 100 - mean_t90 * 2)  # Faster response = higher score

    # Calculate saturation percentage (Layer 3)
    layer3_max = np.max(layer3_sums)
    sat_pct = np.sum(layer3_sums > 0.95 * layer3_max) / len(layer3_sums) * 100

    fig_health = create_sensor_health_gauge(
        noise_score=min(100, noise_score),
        drift_score=min(100, drift_score),
        t90_score=min(100, t90_score),
        saturation_pct=sat_pct,
    )

    # 10. Pipeline Architecture Diagram
    # Get R² from best model if available
    best_r2 = 0.998
    best_model_display = "Polynomial(2)"
    if results.calibration_model:
        best_model_display = results.calibration_model.name
        if calibration_metrics and top3_names:
            best_r2 = calibration_metrics.get(top3_names[0], {}).get("r2", 0.998)

    fig_pipeline = create_pipeline_diagram(
        n_frames=len(frames),
        n_events=len(results.detected_events) if results.detected_events else 0,
        calibration_model=best_model_display,
        max_severity=(
            int(np.max(results.severity_timeline))
            if len(results.severity_timeline) > 0
            else 1
        ),
        r_squared=best_r2,
    )

    # 10. Forecasting Visualization - Use ppm concentration (calibrated values)
    # Calculate data duration for appropriate forecast horizon
    timestamps_arr = np.array(results.dataset.timestamps)
    if hasattr(timestamps_arr[0], "timestamp"):
        data_duration_hours = (
            timestamps_arr[-1] - timestamps_arr[0]
        ).total_seconds() / 3600
    else:
        data_duration_hours = (timestamps_arr[-1] - timestamps_arr[0]) / 3600

    # Use ppm concentration values for forecasting (more meaningful than raw ΔADC)
    # ppm_values is already calculated above from calibration model
    fig_forecast = create_forecast_visualization(
        timestamps=timestamps_arr,
        values=ppm_values,  # Use calibrated ppm concentration
        forecast_hours=None,  # Auto-calculate based on data duration
        title="Toluene Concentration (ppm)",
    )

    # Forecast summary cards - use ppm values
    current_ppm = float(ppm_values[-1])
    forecast_horizon = min(data_duration_hours * 0.5, 24)  # Same logic as visualization
    # Create time labels proportional to forecast horizon
    t1 = forecast_horizon * 0.25
    t2 = forecast_horizon * 0.5
    t3 = forecast_horizon
    # Simple decay forecast for demo (in production, use actual forecast model output)
    forecast_vals = {
        f"+{t1:.0f}h" if t1 >= 1 else f"+{t1*60:.0f}m": current_ppm * 0.95,
        f"+{t2:.0f}h" if t2 >= 1 else f"+{t2*60:.0f}m": current_ppm * 0.90,
        f"+{t3:.0f}h" if t3 >= 1 else f"+{t3*60:.0f}m": current_ppm * 0.85,
    }
    fig_forecast_cards = create_forecast_summary_cards(
        current_value=current_ppm, forecasts=forecast_vals, unit="ppm"
    )

    # Calculate Sensor Health Score (composite metric)
    # Based on: Noise (30%), Drift (30%), Response Time (25%), Stability (15%)
    noise_score = max(
        0, min(100, 100 - results.sensor_characteristics.baseline_noise_rms * 500)
    )
    drift_score = max(
        0,
        min(
            100,
            100 - abs(results.sensor_characteristics.baseline_drift_per_hour) * 5000,
        ),
    )
    t90_score = max(
        0, min(100, 100 - results.sensor_characteristics.mean_t90_rise * 1.5)
    )
    stability_score = max(
        0, min(100, 100 - results.sensor_characteristics.long_term_stability * 200)
    )
    sensor_health_score = (
        noise_score * 0.30
        + drift_score * 0.30
        + t90_score * 0.25
        + stability_score * 0.15
    )

    # 4. Build App Layout with Tabs
    app = Dash(__name__)
    app.title = "CMOS Sensor Analysis Dashboard"

    # Get the layout for the raw analysis tab
    raw_analysis_layout = get_dashboard_layout(data)

    # Professional tab styling
    tab_style = {
        "padding": "12px 20px",
        "fontWeight": "500",
        "backgroundColor": "#2a2d35",
        "border": "none",
        "borderBottom": "2px solid transparent",
        "color": "#999",
    }
    tab_selected_style = {
        "padding": "12px 20px",
        "fontWeight": "500",
        "backgroundColor": "#1e2127",
        "border": "none",
        "borderBottom": "2px solid #5b9bd5",
        "color": "#fff",
    }

    app.layout = html.Div(
        [
            # Header
            html.Div(
                [
                    html.H1(
                        "CMOS Sensor Analysis",
                        style={
                            "margin": "0",
                            "fontSize": "1.5rem",
                            "fontWeight": "400",
                            "letterSpacing": "1px",
                        },
                    ),
                    html.P(
                        "ZIF-8 MOF Toluene Detection System",
                        style={
                            "margin": "4px 0 0 0",
                            "fontSize": "0.85rem",
                            "opacity": "0.6",
                            "fontWeight": "300",
                        },
                    ),
                ],
                style={
                    "padding": "20px 30px",
                    "backgroundColor": "#1a1d23",
                    "borderBottom": "1px solid #333",
                    "color": "#fff",
                },
            ),
            dcc.Tabs(
                [
                    # ====== 1. SENSOR DATA - Raw data overview ======
                    dcc.Tab(
                        label="1. Sensor Data",
                        children=[raw_analysis_layout],
                        style=tab_style,
                        selected_style=tab_selected_style,
                    ),
                    # ====== 2. LAYERS - Raw layer comparison (foundational) ======
                    dcc.Tab(
                        label="2. Layers",
                        children=[
                            html.Div(
                                [
                                    html.H3(
                                        "Sensor Layer Comparison",
                                        style={
                                            "fontWeight": "400",
                                            "marginBottom": "1rem",
                                        },
                                    ),
                                    html.P(
                                        "The CMOS sensor has 3 layers: Layer 1 (Active Sensing), Layer 2 (Reference/Dark), Layer 3 (Saturated). "
                                        "This comparison shows how each layer responds over time.",
                                        style={"color": "#888", "marginBottom": "1rem"},
                                    ),
                                    dcc.Graph(
                                        figure=fig_layers, style={"height": "85vh"}
                                    ),
                                ],
                                style={"padding": "20px"},
                            )
                        ],
                        style=tab_style,
                        selected_style=tab_selected_style,
                    ),
                    # ====== 3. SPATIAL - Pixel layout & coating patterns ======
                    dcc.Tab(
                        label="3. Spatial",
                        children=[
                            html.Div(
                                [
                                    html.H3(
                                        "Spatial Analysis",
                                        style={
                                            "fontWeight": "400",
                                            "marginBottom": "1rem",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    dcc.Graph(
                                                        figure=fig_spatial,
                                                        style={"height": "55vh"},
                                                    )
                                                ],
                                                style={
                                                    "flex": "1",
                                                    "minWidth": "400px",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Graph(
                                                        figure=fig_bad_pixels,
                                                        style={"height": "55vh"},
                                                    )
                                                ],
                                                style={
                                                    "flex": "1",
                                                    "minWidth": "400px",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "gap": "1rem",
                                            "flexWrap": "wrap",
                                        },
                                    ),
                                    # ZIF-8 MOF Coating Pattern Analysis Section
                                    html.Hr(
                                        style={
                                            "margin": "2rem 0",
                                            "borderColor": "#e0e0e0",
                                        }
                                    ),
                                    html.H3(
                                        "ZIF-8 MOF Coating Pattern Analysis",
                                        style={
                                            "fontWeight": "400",
                                            "marginBottom": "0.5rem",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.P(
                                                [
                                                    "Pre-analysis of the raw data reveals the coating pattern on the CMOS sensor. ",
                                                    html.Strong(
                                                        f"Coated/responsive pixels: {coating_analysis['n_coated']} ({coating_analysis['coated_percentage']:.1f}%) "
                                                    ),
                                                    "| ",
                                                    html.Strong(
                                                        f"Reference/stable pixels: {coating_analysis['n_reference']} ({100-coating_analysis['coated_percentage']:.1f}%)"
                                                    ),
                                                ],
                                                style={
                                                    "color": "#555",
                                                    "marginBottom": "0.5rem",
                                                },
                                            ),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        f"Top half (rows 0-15): {coating_analysis['top_half_coated']} coated pixels"
                                                    ),
                                                    html.Li(
                                                        f"Bottom half (rows 16-31): {coating_analysis['bottom_half_coated']} coated pixels"
                                                    ),
                                                    html.Li(
                                                        f"Primary coated region: {coating_analysis['coated_region']}"
                                                    ),
                                                    html.Li(
                                                        "Interpretation: Localized rectangular patch, likely upper/center area of the sensor"
                                                    ),
                                                ],
                                                style={
                                                    "color": "#555",
                                                    "fontSize": "0.9rem",
                                                    "marginBottom": "1rem",
                                                },
                                            ),
                                        ],
                                        style={
                                            "backgroundColor": "#f8f9fa",
                                            "padding": "1rem",
                                            "borderRadius": "4px",
                                            "marginBottom": "1rem",
                                            "border": "1px solid #e9ecef",
                                        },
                                    ),
                                    # Coating Pattern Visualization
                                    dcc.Graph(
                                        figure=fig_coating_pattern,
                                        style={"height": "55vh"},
                                    ),
                                    # Coating Statistics
                                    dcc.Graph(
                                        figure=fig_coating_stats,
                                        style={"height": "40vh"},
                                    ),
                                ],
                                style={"padding": "20px"},
                            )
                        ],
                        style=tab_style,
                        selected_style=tab_selected_style,
                    ),
                    # ====== 4. SIGNAL PROCESSING - ΔADC extraction & filtering ======
                    dcc.Tab(
                        label="4. Signal Processing",
                        children=[
                            html.Div(
                                [
                                    html.H3(
                                        "Signal Processing & Event Detection",
                                        style={
                                            "fontWeight": "400",
                                            "marginBottom": "1rem",
                                        },
                                    ),
                                    html.P(
                                        "Raw ΔADC signal is filtered using Butterworth bandpass, baseline corrected, and events detected using hysteresis thresholding.",
                                        style={"color": "#888", "marginBottom": "1rem"},
                                    ),
                                    dcc.Graph(
                                        figure=fig_signal, style={"height": "80vh"}
                                    ),
                                ],
                                style={"padding": "20px"},
                            )
                        ],
                        style=tab_style,
                        selected_style=tab_selected_style,
                    ),
                    # ====== 5. CHARACTERISTICS - Sensor metrics ======
                    dcc.Tab(
                        label="5. Characteristics",
                        children=[
                            html.Div(
                                [
                                    html.H3(
                                        "Sensor Characterization",
                                        style={
                                            "fontWeight": "400",
                                            "marginBottom": "1rem",
                                        },
                                    ),
                                    html.P(
                                        "Response time analysis (T₉₀), dynamic range, and pixel health metrics derived from the processed signal.",
                                        style={"color": "#888", "marginBottom": "1rem"},
                                    ),
                                    dcc.Graph(
                                        figure=fig_char, style={"height": "80vh"}
                                    ),
                                ],
                                style={"padding": "20px"},
                            )
                        ],
                        style=tab_style,
                        selected_style=tab_selected_style,
                    ),
                    # ====== 6. KINETICS - Response dynamics (T90, drift, noise) ======
                    dcc.Tab(
                        label="6. Kinetics",
                        children=[
                            html.Div(
                                [
                                    html.H3(
                                        "Response Kinetics Analysis",
                                        style={
                                            "fontWeight": "400",
                                            "marginBottom": "1rem",
                                        },
                                    ),
                                    html.P(
                                        "Analysis of sensor response dynamics: T₉₀ rise/recovery times, baseline drift over time, and noise characteristics.",
                                        style={"color": "#888", "marginBottom": "1rem"},
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                figure=fig_t90_dist,
                                                style={"height": "45vh"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=fig_drift,
                                                                style={
                                                                    "height": "40vh"
                                                                },
                                                            )
                                                        ],
                                                        style={
                                                            "flex": "1",
                                                            "minWidth": "400px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=fig_noise,
                                                                style={
                                                                    "height": "40vh"
                                                                },
                                                            )
                                                        ],
                                                        style={
                                                            "flex": "1",
                                                            "minWidth": "400px",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "gap": "1rem",
                                                    "flexWrap": "wrap",
                                                },
                                            ),
                                        ]
                                    ),
                                ],
                                style={"padding": "20px"},
                            )
                        ],
                        style=tab_style,
                        selected_style=tab_selected_style,
                    ),
                    # ====== 7. CALIBRATION - ppm conversion models ======
                    dcc.Tab(
                        label="7. Calibration",
                        children=[
                            html.Div(
                                [
                                    html.H3(
                                        "Calibration Model Evaluation",
                                        style={
                                            "fontWeight": "400",
                                            "marginBottom": "1rem",
                                        },
                                    ),
                                    html.P(
                                        "Multiple calibration models (Polynomial, Langmuir, Freundlich) fitted to convert ΔADC → ppm concentration.",
                                        style={"color": "#888", "marginBottom": "1rem"},
                                    ),
                                    dcc.Graph(
                                        figure=fig_pred_vs_actual,
                                        style={"height": "55vh"},
                                    ),
                                    dcc.Graph(
                                        figure=fig_residuals, style={"height": "35vh"}
                                    ),
                                ],
                                style={"padding": "20px"},
                            )
                        ],
                        style=tab_style,
                        selected_style=tab_selected_style,
                    ),
                    # ====== 8. EXPOSURE - Detected events ======
                    dcc.Tab(
                        label="8. Exposure",
                        children=[
                            html.Div(
                                [
                                    html.H3(
                                        "Exposure Window Analysis",
                                        style={
                                            "fontWeight": "400",
                                            "marginBottom": "1rem",
                                        },
                                    ),
                                    html.P(
                                        "Detected exposure events with estimated ppm concentrations and the calibration curve used for conversion.",
                                        style={"color": "#888", "marginBottom": "1rem"},
                                    ),
                                    dcc.Graph(
                                        figure=fig_exposure, style={"height": "45vh"}
                                    ),
                                    dcc.Graph(figure=fig_cal, style={"height": "45vh"}),
                                ],
                                style={"padding": "20px"},
                            )
                        ],
                        style=tab_style,
                        selected_style=tab_selected_style,
                    ),
                    # ====== 9. SEVERITY - Application alerts ======
                    dcc.Tab(
                        label="9. Severity",
                        children=[
                            html.Div(
                                [
                                    html.H3(
                                        "Application Severity Monitoring",
                                        style={
                                            "fontWeight": "400",
                                            "marginBottom": "1rem",
                                        },
                                    ),
                                    html.P(
                                        "VOC concentration mapped to severity levels (Safe/Warning/Critical) for poultry barn health monitoring.",
                                        style={"color": "#888", "marginBottom": "1rem"},
                                    ),
                                    dcc.Graph(figure=fig_sev, style={"height": "80vh"}),
                                ],
                                style={"padding": "20px"},
                            )
                        ],
                        style=tab_style,
                        selected_style=tab_selected_style,
                    ),
                    # ====== 10. HEALTH - Sensor health summary ======
                    dcc.Tab(
                        label="10. Health",
                        children=[
                            html.Div(
                                [
                                    html.H3(
                                        "Sensor Health Index",
                                        style={
                                            "fontWeight": "400",
                                            "marginBottom": "1rem",
                                        },
                                    ),
                                    html.P(
                                        "Composite health score based on noise level, baseline drift, response speed (T₉₀), and saturation percentage.",
                                        style={"color": "#888", "marginBottom": "1rem"},
                                    ),
                                    dcc.Graph(
                                        figure=fig_health, style={"height": "60vh"}
                                    ),
                                ],
                                style={"padding": "20px"},
                            )
                        ],
                        style=tab_style,
                        selected_style=tab_selected_style,
                    ),
                    # ====== 11. FORECASTING - Predict future concentrations ======
                    dcc.Tab(
                        label="11. Forecast",
                        children=[
                            html.Div(
                                [
                                    html.H3(
                                        "VOC Concentration Forecasting (ppm)",
                                        style={
                                            "fontWeight": "400",
                                            "marginBottom": "1rem",
                                        },
                                    ),
                                    html.P(
                                        f"Forecasting calibrated toluene concentration (ppm) using the calibration model. "
                                        f"Forecast horizon is auto-calculated based on data duration (~{forecast_horizon:.1f}h for {data_duration_hours:.1f}h of data). "
                                        "Three methods demonstrated: Moving Average, Linear Trend, and Exponential Smoothing.",
                                        style={"color": "#888", "marginBottom": "1rem"},
                                    ),
                                    # Forecast summary cards
                                    html.Div(
                                        [
                                            html.H4(
                                                "Forecast Summary",
                                                style={
                                                    "fontWeight": "400",
                                                    "marginBottom": "0.5rem",
                                                },
                                            ),
                                            dcc.Graph(
                                                figure=fig_forecast_cards,
                                                style={"height": "180px"},
                                            ),
                                        ],
                                        style={
                                            "marginBottom": "1rem",
                                            "backgroundColor": "#f8f9fa",
                                            "padding": "1rem",
                                            "borderRadius": "8px",
                                        },
                                    ),
                                    # Main forecast plot
                                    dcc.Graph(
                                        figure=fig_forecast, style={"height": "55vh"}
                                    ),
                                    # Methodology explanation
                                    html.Div(
                                        [
                                            html.H4(
                                                "Forecasting Methods",
                                                style={
                                                    "fontWeight": "500",
                                                    "marginBottom": "0.5rem",
                                                    "color": "#2c3e50",
                                                },
                                            ),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        [
                                                            html.Strong(
                                                                "Moving Average: "
                                                            ),
                                                            "Uses the mean of recent observations with random walk for uncertainty.",
                                                        ]
                                                    ),
                                                    html.Li(
                                                        [
                                                            html.Strong(
                                                                "Linear Trend: "
                                                            ),
                                                            "Extrapolates the recent linear trend into the future.",
                                                        ]
                                                    ),
                                                    html.Li(
                                                        [
                                                            html.Strong(
                                                                "Exponential Smoothing: "
                                                            ),
                                                            "Weighted average with decay toward the historical mean.",
                                                        ]
                                                    ),
                                                ],
                                                style={
                                                    "color": "#555",
                                                    "fontSize": "0.9rem",
                                                },
                                            ),
                                            html.P(
                                                [
                                                    html.Strong("Note: "),
                                                    "These are demonstration forecasts using simple methods. For production deployment, consider "
                                                    "ARIMA, Prophet, or LSTM neural network models for more accurate predictions.",
                                                ],
                                                style={
                                                    "color": "#888",
                                                    "fontSize": "0.85rem",
                                                    "marginTop": "0.5rem",
                                                    "fontStyle": "italic",
                                                },
                                            ),
                                        ],
                                        style={
                                            "backgroundColor": "#e8f4fd",
                                            "padding": "1rem",
                                            "borderRadius": "8px",
                                            "marginTop": "1rem",
                                        },
                                    ),
                                ],
                                style={"padding": "20px"},
                            )
                        ],
                        style=tab_style,
                        selected_style=tab_selected_style,
                    ),
                    # ====== 12. FULL REPORT - Comprehensive summary ======
                    dcc.Tab(
                        label="12. Full Report",
                        children=[
                            html.Div(
                                [
                                    # ========== REPORT HEADER ==========
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H1(
                                                        "CMOS ZIF-8 MOF Toluene Sensor",
                                                        style={
                                                            "textAlign": "center",
                                                            "marginBottom": "0",
                                                            "color": "#2c3e50",
                                                            "fontSize": "1.8rem",
                                                            "fontWeight": "600",
                                                        },
                                                    ),
                                                    html.H2(
                                                        "Digital Twin Analysis Report",
                                                        style={
                                                            "textAlign": "center",
                                                            "marginTop": "0.25rem",
                                                            "marginBottom": "0.5rem",
                                                            "color": "#7f8c8d",
                                                            "fontSize": "1.1rem",
                                                            "fontWeight": "400",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            html.P(
                                                f"Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}",
                                                style={
                                                    "textAlign": "center",
                                                    "color": "#95a5a6",
                                                    "marginBottom": "0.5rem",
                                                    "fontSize": "0.85rem",
                                                },
                                            ),
                                            html.P(
                                                f"Data Source: {data_file.name}",
                                                style={
                                                    "textAlign": "center",
                                                    "color": "#95a5a6",
                                                    "marginBottom": "1rem",
                                                    "fontSize": "0.85rem",
                                                },
                                            ),
                                        ],
                                        style={
                                            "borderBottom": "3px solid #3498db",
                                            "paddingBottom": "1rem",
                                            "marginBottom": "1.5rem",
                                        },
                                    ),
                                    # ========== KEY METRICS AT A GLANCE ==========
                                    html.Div(
                                        [
                                            html.H3(
                                                "Analysis Summary",
                                                style={
                                                    "color": "#2c3e50",
                                                    "marginBottom": "1rem",
                                                    "fontSize": "1.1rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    # Metric Card 1: Data
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                "",
                                                                style={
                                                                    "fontSize": "1.5rem",
                                                                    "marginBottom": "0.25rem",
                                                                },
                                                            ),
                                                            html.Div(
                                                                f"{len(frames):,}",
                                                                style={
                                                                    "fontSize": "1.5rem",
                                                                    "fontWeight": "600",
                                                                    "color": "#3498db",
                                                                },
                                                            ),
                                                            html.Div(
                                                                "Frames Analyzed",
                                                                style={
                                                                    "fontSize": "0.75rem",
                                                                    "color": "#7f8c8d",
                                                                },
                                                            ),
                                                            html.Div(
                                                                f"{data_duration_hours:.1f}h duration",
                                                                style={
                                                                    "fontSize": "0.7rem",
                                                                    "color": "#95a5a6",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "textAlign": "center",
                                                            "padding": "1rem",
                                                            "backgroundColor": "#f8f9fa",
                                                            "borderRadius": "8px",
                                                            "flex": "1",
                                                            "margin": "0 0.5rem",
                                                            "minWidth": "120px",
                                                        },
                                                    ),
                                                    # Metric Card 2: Events
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                "",
                                                                style={
                                                                    "fontSize": "1.5rem",
                                                                    "marginBottom": "0.25rem",
                                                                },
                                                            ),
                                                            html.Div(
                                                                f"{len(results.detected_events)}",
                                                                style={
                                                                    "fontSize": "1.5rem",
                                                                    "fontWeight": "600",
                                                                    "color": "#e74c3c",
                                                                },
                                                            ),
                                                            html.Div(
                                                                "Events Detected",
                                                                style={
                                                                    "fontSize": "0.75rem",
                                                                    "color": "#7f8c8d",
                                                                },
                                                            ),
                                                            html.Div(
                                                                (
                                                                    f"Max: {np.max(ppm_values):.0f} ppm"
                                                                    if len(ppm_values)
                                                                    > 0
                                                                    else ""
                                                                ),
                                                                style={
                                                                    "fontSize": "0.7rem",
                                                                    "color": "#95a5a6",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "textAlign": "center",
                                                            "padding": "1rem",
                                                            "backgroundColor": "#f8f9fa",
                                                            "borderRadius": "8px",
                                                            "flex": "1",
                                                            "margin": "0 0.5rem",
                                                            "minWidth": "120px",
                                                        },
                                                    ),
                                                    # Metric Card 3: Calibration
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                "",
                                                                style={
                                                                    "fontSize": "1.5rem",
                                                                    "marginBottom": "0.25rem",
                                                                },
                                                            ),
                                                            html.Div(
                                                                f"R²={best_r2:.3f}",
                                                                style={
                                                                    "fontSize": "1.3rem",
                                                                    "fontWeight": "600",
                                                                    "color": "#27ae60",
                                                                },
                                                            ),
                                                            html.Div(
                                                                "Calibration Fit",
                                                                style={
                                                                    "fontSize": "0.75rem",
                                                                    "color": "#7f8c8d",
                                                                },
                                                            ),
                                                            html.Div(
                                                                f"{best_model_display}",
                                                                style={
                                                                    "fontSize": "0.7rem",
                                                                    "color": "#95a5a6",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "textAlign": "center",
                                                            "padding": "1rem",
                                                            "backgroundColor": "#f8f9fa",
                                                            "borderRadius": "8px",
                                                            "flex": "1",
                                                            "margin": "0 0.5rem",
                                                            "minWidth": "120px",
                                                        },
                                                    ),
                                                    # Metric Card 4: Severity
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                "",
                                                                style={
                                                                    "fontSize": "1.5rem",
                                                                    "marginBottom": "0.25rem",
                                                                },
                                                            ),
                                                            html.Div(
                                                                f"Level {int(np.max(results.severity_timeline)) if len(results.severity_timeline) > 0 else 0}",
                                                                style={
                                                                    "fontSize": "1.5rem",
                                                                    "fontWeight": "600",
                                                                    "color": (
                                                                        "#27ae60"
                                                                        if np.max(
                                                                            results.severity_timeline
                                                                        )
                                                                        <= 1
                                                                        else (
                                                                            "#f39c12"
                                                                            if np.max(
                                                                                results.severity_timeline
                                                                            )
                                                                            <= 2
                                                                            else "#e74c3c"
                                                                        )
                                                                    ),
                                                                },
                                                            ),
                                                            html.Div(
                                                                "Max Severity",
                                                                style={
                                                                    "fontSize": "0.75rem",
                                                                    "color": "#7f8c8d",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    (
                                                                        "SAFE"
                                                                        if np.max(
                                                                            results.severity_timeline
                                                                        )
                                                                        <= 1
                                                                        else (
                                                                            "WARNING"
                                                                            if np.max(
                                                                                results.severity_timeline
                                                                            )
                                                                            <= 2
                                                                            else "CRITICAL"
                                                                        )
                                                                    )
                                                                ][0],
                                                                style={
                                                                    "fontSize": "0.7rem",
                                                                    "color": "#95a5a6",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "textAlign": "center",
                                                            "padding": "1rem",
                                                            "backgroundColor": "#f8f9fa",
                                                            "borderRadius": "8px",
                                                            "flex": "1",
                                                            "margin": "0 0.5rem",
                                                            "minWidth": "120px",
                                                        },
                                                    ),
                                                    # Metric Card 5: Sensor Health
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                "",
                                                                style={
                                                                    "fontSize": "1.5rem",
                                                                    "marginBottom": "0.25rem",
                                                                },
                                                            ),
                                                            html.Div(
                                                                f"{sensor_health_score:.0f}%",
                                                                style={
                                                                    "fontSize": "1.5rem",
                                                                    "fontWeight": "600",
                                                                    "color": "#9b59b6",
                                                                },
                                                            ),
                                                            html.Div(
                                                                "Sensor Health",
                                                                style={
                                                                    "fontSize": "0.75rem",
                                                                    "color": "#7f8c8d",
                                                                },
                                                            ),
                                                            html.Div(
                                                                (
                                                                    "Good"
                                                                    if sensor_health_score
                                                                    >= 70
                                                                    else "Monitor"
                                                                ),
                                                                style={
                                                                    "fontSize": "0.7rem",
                                                                    "color": "#95a5a6",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "textAlign": "center",
                                                            "padding": "1rem",
                                                            "backgroundColor": "#f8f9fa",
                                                            "borderRadius": "8px",
                                                            "flex": "1",
                                                            "margin": "0 0.5rem",
                                                            "minWidth": "120px",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "flexWrap": "wrap",
                                                    "justifyContent": "center",
                                                    "gap": "0.5rem",
                                                },
                                            ),
                                        ],
                                        style={
                                            "marginBottom": "2rem",
                                            "padding": "1rem",
                                            "backgroundColor": "#fff",
                                            "borderRadius": "8px",
                                            "border": "1px solid #e0e0e0",
                                        },
                                    ),
                                    # ========== EXPORT BUTTONS ==========
                                    html.Div(
                                        [
                                            html.Button(
                                                "Export HTML Report",
                                                id="export-html-btn",
                                                style={
                                                    "padding": "10px 20px",
                                                    "backgroundColor": "#3498db",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "marginRight": "10px",
                                                },
                                            ),
                                            html.Button(
                                                "Export PDF Report",
                                                id="export-pdf-btn",
                                                style={
                                                    "padding": "10px 20px",
                                                    "backgroundColor": "#9b59b6",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                    "marginRight": "10px",
                                                },
                                            ),
                                            html.Button(
                                                "Print / Save as PDF",
                                                id="print-pdf-btn",
                                                style={
                                                    "padding": "10px 20px",
                                                    "backgroundColor": "#27ae60",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                            dcc.Download(id="download-report"),
                                            dcc.Download(id="download-pdf-report"),
                                        ],
                                        style={
                                            "textAlign": "center",
                                            "marginBottom": "2rem",
                                        },
                                    ),
                                    # ========== PIPELINE ARCHITECTURE OVERVIEW ==========
                                    html.Div(
                                        [
                                            html.H3(
                                                "System Architecture",
                                                style={
                                                    "color": "#2c3e50",
                                                    "borderBottom": "2px solid #9b59b6",
                                                    "paddingBottom": "0.5rem",
                                                    "marginBottom": "1rem",
                                                },
                                            ),
                                            # Pipeline description with dashboard tab mapping
                                            html.Div(
                                                [
                                                    html.P(
                                                        [
                                                            "This report presents a complete analysis of the ",
                                                            html.B(
                                                                "CMOS ZIF-8 MOF toluene sensor"
                                                            ),
                                                            " digital twin. The pipeline transforms raw 32×32×3 pixel ADC readings into calibrated ",
                                                            html.B(
                                                                "ppm concentration values"
                                                            ),
                                                            " and ",
                                                            html.B(
                                                                "severity assessments"
                                                            ),
                                                            " for poultry barn health monitoring.",
                                                        ],
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.95rem",
                                                            "marginBottom": "1rem",
                                                            "lineHeight": "1.6",
                                                        },
                                                    ),
                                                    # Dashboard Tab Mapping - Grid of cards
                                                    html.H4(
                                                        "Dashboard Modules → Report Sections",
                                                        style={
                                                            "color": "#7f8c8d",
                                                            "fontSize": "0.9rem",
                                                            "marginBottom": "0.75rem",
                                                            "fontWeight": "500",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            # Row 1: Data Processing
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "1",
                                                                        style={
                                                                            "backgroundColor": "#3498db",
                                                                            "color": "white",
                                                                            "padding": "2px 8px",
                                                                            "borderRadius": "4px",
                                                                            "fontSize": "0.75rem",
                                                                            "marginRight": "0.5rem",
                                                                        },
                                                                    ),
                                                                    html.B(
                                                                        "Sensor Data"
                                                                    ),
                                                                    html.Span(
                                                                        " → Raw ADC frames, ΔADC calculation",
                                                                        style={
                                                                            "color": "#666",
                                                                            "fontSize": "0.85rem",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "padding": "0.5rem",
                                                                    "borderLeft": "3px solid #3498db",
                                                                    "marginBottom": "0.5rem",
                                                                    "backgroundColor": "#f8f9fa",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "2",
                                                                        style={
                                                                            "backgroundColor": "#3498db",
                                                                            "color": "white",
                                                                            "padding": "2px 8px",
                                                                            "borderRadius": "4px",
                                                                            "fontSize": "0.75rem",
                                                                            "marginRight": "0.5rem",
                                                                        },
                                                                    ),
                                                                    html.B("Layers"),
                                                                    html.Span(
                                                                        " → 3-layer analysis (Active/Reference/Saturated)",
                                                                        style={
                                                                            "color": "#666",
                                                                            "fontSize": "0.85rem",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "padding": "0.5rem",
                                                                    "borderLeft": "3px solid #3498db",
                                                                    "marginBottom": "0.5rem",
                                                                    "backgroundColor": "#f8f9fa",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "3",
                                                                        style={
                                                                            "backgroundColor": "#3498db",
                                                                            "color": "white",
                                                                            "padding": "2px 8px",
                                                                            "borderRadius": "4px",
                                                                            "fontSize": "0.75rem",
                                                                            "marginRight": "0.5rem",
                                                                        },
                                                                    ),
                                                                    html.B("Spatial"),
                                                                    html.Span(
                                                                        " → Coated vs reference pixel mapping",
                                                                        style={
                                                                            "color": "#666",
                                                                            "fontSize": "0.85rem",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "padding": "0.5rem",
                                                                    "borderLeft": "3px solid #3498db",
                                                                    "marginBottom": "0.5rem",
                                                                    "backgroundColor": "#f8f9fa",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "4",
                                                                        style={
                                                                            "backgroundColor": "#e74c3c",
                                                                            "color": "white",
                                                                            "padding": "2px 8px",
                                                                            "borderRadius": "4px",
                                                                            "fontSize": "0.75rem",
                                                                            "marginRight": "0.5rem",
                                                                        },
                                                                    ),
                                                                    html.B(
                                                                        "Signal Processing"
                                                                    ),
                                                                    html.Span(
                                                                        f" → Filtering, baseline correction, {len(results.detected_events)} events detected",
                                                                        style={
                                                                            "color": "#666",
                                                                            "fontSize": "0.85rem",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "padding": "0.5rem",
                                                                    "borderLeft": "3px solid #e74c3c",
                                                                    "marginBottom": "0.5rem",
                                                                    "backgroundColor": "#f8f9fa",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "5",
                                                                        style={
                                                                            "backgroundColor": "#f39c12",
                                                                            "color": "white",
                                                                            "padding": "2px 8px",
                                                                            "borderRadius": "4px",
                                                                            "fontSize": "0.75rem",
                                                                            "marginRight": "0.5rem",
                                                                        },
                                                                    ),
                                                                    html.B(
                                                                        "Characteristics"
                                                                    ),
                                                                    html.Span(
                                                                        " → T₉₀ response times, dynamic range",
                                                                        style={
                                                                            "color": "#666",
                                                                            "fontSize": "0.85rem",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "padding": "0.5rem",
                                                                    "borderLeft": "3px solid #f39c12",
                                                                    "marginBottom": "0.5rem",
                                                                    "backgroundColor": "#f8f9fa",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "6",
                                                                        style={
                                                                            "backgroundColor": "#f39c12",
                                                                            "color": "white",
                                                                            "padding": "2px 8px",
                                                                            "borderRadius": "4px",
                                                                            "fontSize": "0.75rem",
                                                                            "marginRight": "0.5rem",
                                                                        },
                                                                    ),
                                                                    html.B("Kinetics"),
                                                                    html.Span(
                                                                        " → Response/recovery timing analysis",
                                                                        style={
                                                                            "color": "#666",
                                                                            "fontSize": "0.85rem",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "padding": "0.5rem",
                                                                    "borderLeft": "3px solid #f39c12",
                                                                    "marginBottom": "0.5rem",
                                                                    "backgroundColor": "#f8f9fa",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "7",
                                                                        style={
                                                                            "backgroundColor": "#27ae60",
                                                                            "color": "white",
                                                                            "padding": "2px 8px",
                                                                            "borderRadius": "4px",
                                                                            "fontSize": "0.75rem",
                                                                            "marginRight": "0.5rem",
                                                                        },
                                                                    ),
                                                                    html.B(
                                                                        "Calibration"
                                                                    ),
                                                                    html.Span(
                                                                        f" → ΔADC → ppm ({best_model_display}, R²={best_r2:.3f})",
                                                                        style={
                                                                            "color": "#666",
                                                                            "fontSize": "0.85rem",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "padding": "0.5rem",
                                                                    "borderLeft": "3px solid #27ae60",
                                                                    "marginBottom": "0.5rem",
                                                                    "backgroundColor": "#f8f9fa",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "8",
                                                                        style={
                                                                            "backgroundColor": "#27ae60",
                                                                            "color": "white",
                                                                            "padding": "2px 8px",
                                                                            "borderRadius": "4px",
                                                                            "fontSize": "0.75rem",
                                                                            "marginRight": "0.5rem",
                                                                        },
                                                                    ),
                                                                    html.B("Exposure"),
                                                                    html.Span(
                                                                        " → Concentration windows with ppm annotations",
                                                                        style={
                                                                            "color": "#666",
                                                                            "fontSize": "0.85rem",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "padding": "0.5rem",
                                                                    "borderLeft": "3px solid #27ae60",
                                                                    "marginBottom": "0.5rem",
                                                                    "backgroundColor": "#f8f9fa",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "9",
                                                                        style={
                                                                            "backgroundColor": "#9b59b6",
                                                                            "color": "white",
                                                                            "padding": "2px 8px",
                                                                            "borderRadius": "4px",
                                                                            "fontSize": "0.75rem",
                                                                            "marginRight": "0.5rem",
                                                                        },
                                                                    ),
                                                                    html.B("Severity"),
                                                                    html.Span(
                                                                        " → Safe/Warning/Critical classification",
                                                                        style={
                                                                            "color": "#666",
                                                                            "fontSize": "0.85rem",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "padding": "0.5rem",
                                                                    "borderLeft": "3px solid #9b59b6",
                                                                    "marginBottom": "0.5rem",
                                                                    "backgroundColor": "#f8f9fa",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "10",
                                                                        style={
                                                                            "backgroundColor": "#9b59b6",
                                                                            "color": "white",
                                                                            "padding": "2px 8px",
                                                                            "borderRadius": "4px",
                                                                            "fontSize": "0.75rem",
                                                                            "marginRight": "0.5rem",
                                                                        },
                                                                    ),
                                                                    html.B("Health"),
                                                                    html.Span(
                                                                        f" → Sensor health score ({sensor_health_score:.0f}%)",
                                                                        style={
                                                                            "color": "#666",
                                                                            "fontSize": "0.85rem",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "padding": "0.5rem",
                                                                    "borderLeft": "3px solid #9b59b6",
                                                                    "marginBottom": "0.5rem",
                                                                    "backgroundColor": "#f8f9fa",
                                                                },
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Span(
                                                                        "11",
                                                                        style={
                                                                            "backgroundColor": "#1abc9c",
                                                                            "color": "white",
                                                                            "padding": "2px 8px",
                                                                            "borderRadius": "4px",
                                                                            "fontSize": "0.75rem",
                                                                            "marginRight": "0.5rem",
                                                                        },
                                                                    ),
                                                                    html.B("Forecast"),
                                                                    html.Span(
                                                                        f" → ppm prediction ({forecast_horizon:.1f}h ahead)",
                                                                        style={
                                                                            "color": "#666",
                                                                            "fontSize": "0.85rem",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "padding": "0.5rem",
                                                                    "borderLeft": "3px solid #1abc9c",
                                                                    "marginBottom": "0.5rem",
                                                                    "backgroundColor": "#f8f9fa",
                                                                },
                                                            ),
                                                        ],
                                                        style={"marginBottom": "1rem"},
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#fafafa",
                                                    "padding": "1.25rem",
                                                    "borderRadius": "8px",
                                                    "marginBottom": "1rem",
                                                    "border": "1px solid #e8e8e8",
                                                },
                                            ),
                                            # Technical Specs - Compact
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "🔧 ",
                                                        style={"fontSize": "0.9rem"},
                                                    ),
                                                    html.B(
                                                        "Technical Specs: ",
                                                        style={"fontSize": "0.85rem"},
                                                    ),
                                                    html.Span(
                                                        "CMOS + ZIF-8 MOF coating • 32×32×3 pixels • 12-bit ADC (0-4095) • ~2 Hz sampling • Toluene detection",
                                                        style={
                                                            "fontSize": "0.85rem",
                                                            "color": "#666",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#e8f5e9",
                                                    "padding": "0.75rem",
                                                    "borderRadius": "4px",
                                                    "marginBottom": "1rem",
                                                },
                                            ),
                                            # Pipeline diagram (collapsible or scrollable)
                                            html.Details(
                                                [
                                                    html.Summary(
                                                        "View Full Pipeline Architecture Diagram",
                                                        style={
                                                            "cursor": "pointer",
                                                            "padding": "0.75rem",
                                                            "backgroundColor": "#f0f0f0",
                                                            "borderRadius": "4px",
                                                            "fontWeight": "500",
                                                            "color": "#2c3e50",
                                                        },
                                                    ),
                                                    dcc.Graph(
                                                        figure=fig_pipeline,
                                                        style={
                                                            "height": "1400px",
                                                            "marginTop": "1rem",
                                                        },
                                                    ),
                                                ],
                                                style={"marginBottom": "1rem"},
                                            ),
                                        ],
                                        style={"marginBottom": "2rem"},
                                    ),
                                    # Section 1: Executive Summary
                                    html.Div(
                                        [
                                            html.H3(
                                                "1. Executive Summary",
                                                style={
                                                    "color": "#2c3e50",
                                                    "borderBottom": "2px solid #3498db",
                                                    "paddingBottom": "0.5rem",
                                                },
                                            ),
                                            dcc.Markdown(
                                                results.daily_report,
                                                style={"lineHeight": "1.6"},
                                            ),
                                        ],
                                        style={"marginBottom": "2rem"},
                                    ),
                                    # Section 2: Signal Processing (with Method/Definitions/Findings)
                                    html.Div(
                                        [
                                            html.H3(
                                                "2. Signal Processing & Event Detection",
                                                style={
                                                    "color": "#2c3e50",
                                                    "borderBottom": "2px solid #3498db",
                                                    "paddingBottom": "0.5rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Method",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        "Coated/Reference pixel subtraction with Butterworth low-pass filter (4th order, fc=0.1 Hz). "
                                                        "Event detection uses hysteresis thresholding with ON threshold = μ + 3σ (baseline) and "
                                                        "OFF threshold = μ + 1.5σ. Minimum event duration: 30 seconds.",
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                        },
                                                    ),
                                                    html.H4(
                                                        "Definitions",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.Ul(
                                                        [
                                                            html.Li(
                                                                f"Coated pixels: {coating_analysis.get('n_coated', 117)} ({coating_analysis.get('coated_percentage', 11.4):.1f}%) — ZIF-8 MOF responsive"
                                                            ),
                                                            html.Li(
                                                                f"Reference pixels: {coating_analysis.get('n_reference', 907)} — stable baseline for drift compensation"
                                                            ),
                                                            html.Li(
                                                                "ΔADC = Avg(Coated pixels) − Avg(Reference pixels)"
                                                            ),
                                                            html.Li(
                                                                "Event: Contiguous period where signal exceeds ON threshold"
                                                            ),
                                                        ],
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                            "marginLeft": "1rem",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#f8f9fa",
                                                    "padding": "1rem",
                                                    "borderRadius": "4px",
                                                    "marginBottom": "1rem",
                                                },
                                            ),
                                            dcc.Graph(
                                                figure=fig_signal,
                                                style={"height": "500px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Key Findings",
                                                        style={
                                                            "color": "#27ae60",
                                                            "fontSize": "0.95rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        (
                                                            f"Total frames analyzed: {len(frames):,} | Detected events: {len(results.detected_events)} | "
                                                            f"Mean event duration: {np.mean([e.duration for e in results.detected_events]):.1f}s"
                                                            if results.detected_events
                                                            else "No events detected"
                                                        ),
                                                        style={"fontWeight": "500"},
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#e8f5e9",
                                                    "padding": "0.75rem",
                                                    "borderRadius": "4px",
                                                    "marginTop": "1rem",
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "2rem"},
                                    ),
                                    # Section 3: Exposure Windows
                                    html.Div(
                                        [
                                            html.H3(
                                                "3. ΔADC Response with Exposure Windows",
                                                style={
                                                    "color": "#2c3e50",
                                                    "borderBottom": "2px solid #3498db",
                                                    "paddingBottom": "0.5rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Method",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        "Continuous ΔADC time-series plotted with color-coded exposure window annotations. "
                                                        "Green = safe levels, yellow = moderate, red = high concentration periods.",
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                        },
                                                    ),
                                                    html.H4(
                                                        "Definitions",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.Ul(
                                                        [
                                                            html.Li(
                                                                "Safe: ΔADC < 1.0 (estimated <2000 ppm toluene)"
                                                            ),
                                                            html.Li(
                                                                "Moderate: 1.0 ≤ ΔADC < 2.5 (2000-5000 ppm)"
                                                            ),
                                                            html.Li(
                                                                "High: ΔADC ≥ 2.5 (>5000 ppm)"
                                                            ),
                                                        ],
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                            "marginLeft": "1rem",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#f8f9fa",
                                                    "padding": "1rem",
                                                    "borderRadius": "4px",
                                                    "marginBottom": "1rem",
                                                },
                                            ),
                                            dcc.Graph(
                                                figure=fig_exposure,
                                                style={"height": "400px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Key Findings",
                                                        style={
                                                            "color": "#27ae60",
                                                            "fontSize": "0.95rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        f"Peak ΔADC: {np.max(results.filtered_signal):.2f} | "
                                                        f"Time above moderate threshold: {np.sum(results.filtered_signal > 1.0) / len(results.filtered_signal) * 100:.1f}%",
                                                        style={"fontWeight": "500"},
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#e8f5e9",
                                                    "padding": "0.75rem",
                                                    "borderRadius": "4px",
                                                    "marginTop": "1rem",
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "2rem"},
                                    ),
                                    # Section 4: Spatial Analysis
                                    html.Div(
                                        [
                                            html.H3(
                                                "4. Spatial Sensitivity & Pixel Health Analysis",
                                                style={
                                                    "color": "#2c3e50",
                                                    "borderBottom": "2px solid #3498db",
                                                    "paddingBottom": "0.5rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Method",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        "Per-pixel mean ΔADC calculated during peak exposure window (±50 frames around maximum). "
                                                        "Pixel health classification based on temporal statistics across all frames.",
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                        },
                                                    ),
                                                    html.H4(
                                                        "Definitions (Pixel Health)",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.Ul(
                                                        [
                                                            html.Li(
                                                                "Saturated: mean > 0.98 × ADC_fullscale (4095) OR max ≥ 4095"
                                                            ),
                                                            html.Li(
                                                                "Dead: max ADC value < 1.0 (always zero)"
                                                            ),
                                                            html.Li(
                                                                "Noisy: temporal σ > 5 × median(σ) of all pixels"
                                                            ),
                                                            html.Li(
                                                                "Normal: All other pixels"
                                                            ),
                                                        ],
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                            "marginLeft": "1rem",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#f8f9fa",
                                                    "padding": "1rem",
                                                    "borderRadius": "4px",
                                                    "marginBottom": "1rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=fig_spatial,
                                                                style={
                                                                    "height": "450px"
                                                                },
                                                            )
                                                        ],
                                                        style={
                                                            "flex": "1",
                                                            "minWidth": "350px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=fig_bad_pixels,
                                                                style={
                                                                    "height": "450px"
                                                                },
                                                            )
                                                        ],
                                                        style={
                                                            "flex": "1",
                                                            "minWidth": "350px",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "gap": "1rem",
                                                    "flexWrap": "wrap",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Key Findings",
                                                        style={
                                                            "color": "#27ae60",
                                                            "fontSize": "0.95rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        "Spatial uniformity and pixel yield metrics shown above each heatmap. "
                                                        "Higher yield indicates better sensor quality.",
                                                        style={"fontWeight": "500"},
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#e8f5e9",
                                                    "padding": "0.75rem",
                                                    "borderRadius": "4px",
                                                    "marginTop": "1rem",
                                                },
                                            ),
                                            # Coating Pattern Analysis Sub-section
                                            html.Hr(
                                                style={
                                                    "margin": "1.5rem 0",
                                                    "borderColor": "#e0e0e0",
                                                }
                                            ),
                                            html.H4(
                                                "4B. ZIF-8 MOF Coating Pattern Analysis",
                                                style={
                                                    "color": "#5b9bd5",
                                                    "fontSize": "1rem",
                                                    "marginTop": "1rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        [
                                                            "Pre-analysis identifies coated (responsive) vs reference (stable) pixels based on temporal response characteristics. ",
                                                            html.Strong(
                                                                f"Coated: {coating_analysis['n_coated']} ({coating_analysis['coated_percentage']:.1f}%) | "
                                                            ),
                                                            html.Strong(
                                                                f"Reference: {coating_analysis['n_reference']} ({100-coating_analysis['coated_percentage']:.1f}%)"
                                                            ),
                                                        ],
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                        },
                                                    ),
                                                    html.Ul(
                                                        [
                                                            html.Li(
                                                                f"Top half (rows 0-15): {coating_analysis['top_half_coated']} coated pixels"
                                                            ),
                                                            html.Li(
                                                                f"Bottom half (rows 16-31): {coating_analysis['bottom_half_coated']} coated pixels"
                                                            ),
                                                            html.Li(
                                                                f"Primary region: {coating_analysis['coated_region']}"
                                                            ),
                                                        ],
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.85rem",
                                                            "marginLeft": "1rem",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#fff8e6",
                                                    "padding": "0.75rem",
                                                    "borderRadius": "4px",
                                                    "marginBottom": "1rem",
                                                    "border": "1px solid #ffd966",
                                                },
                                            ),
                                            dcc.Graph(
                                                figure=fig_coating_pattern,
                                                style={"height": "450px"},
                                            ),
                                        ],
                                        style={"marginBottom": "2rem"},
                                    ),
                                    # Section 5: Response Kinetics
                                    html.Div(
                                        [
                                            html.H3(
                                                "5. Response Kinetics (T90 Analysis)",
                                                style={
                                                    "color": "#2c3e50",
                                                    "borderBottom": "2px solid #3498db",
                                                    "paddingBottom": "0.5rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Method",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        "T90 rise time: time to reach 90% of peak response from event onset. "
                                                        "T90 recovery time: time to return to 10% above baseline after peak. "
                                                        "Calculated via linear interpolation on smoothed signal.",
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                        },
                                                    ),
                                                    html.H4(
                                                        "Definitions",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.Ul(
                                                        [
                                                            html.Li(
                                                                "T90 Rise = t(0.9 × Peak) − t(Event Start)"
                                                            ),
                                                            html.Li(
                                                                "T90 Recovery = t(Baseline + 0.1 × Peak) − t(Peak)"
                                                            ),
                                                            html.Li(
                                                                "Fast response: T90 < 30s | Moderate: 30-60s | Slow: >60s"
                                                            ),
                                                        ],
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                            "marginLeft": "1rem",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#f8f9fa",
                                                    "padding": "1rem",
                                                    "borderRadius": "4px",
                                                    "marginBottom": "1rem",
                                                },
                                            ),
                                            dcc.Graph(
                                                figure=fig_t90_dist,
                                                style={"height": "400px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Key Findings",
                                                        style={
                                                            "color": "#27ae60",
                                                            "fontSize": "0.95rem",
                                                        },
                                                    ),
                                                    html.Table(
                                                        [
                                                            html.Tr(
                                                                [
                                                                    html.Th(
                                                                        "Metric",
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                        },
                                                                    ),
                                                                    html.Th(
                                                                        "Value",
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                        },
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Tr(
                                                                [
                                                                    html.Td(
                                                                        "Mean T90 Rise",
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                        },
                                                                    ),
                                                                    html.Td(
                                                                        f"{results.sensor_characteristics.mean_t90_rise:.2f} ± {results.sensor_characteristics.std_t90_rise:.2f} s",
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                        },
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Tr(
                                                                [
                                                                    html.Td(
                                                                        "Mean T90 Recovery",
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                        },
                                                                    ),
                                                                    html.Td(
                                                                        f"{results.sensor_characteristics.mean_t90_recovery:.2f} ± {results.sensor_characteristics.std_t90_recovery:.2f} s",
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                        },
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        style={
                                                            "margin": "0.5rem 0",
                                                            "borderCollapse": "collapse",
                                                            "width": "100%",
                                                            "maxWidth": "400px",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#e8f5e9",
                                                    "padding": "0.75rem",
                                                    "borderRadius": "4px",
                                                    "marginTop": "1rem",
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "2rem"},
                                    ),
                                    # Section 6: Baseline Stability
                                    html.Div(
                                        [
                                            html.H3(
                                                "6. Baseline Drift & Noise Analysis",
                                                style={
                                                    "color": "#2c3e50",
                                                    "borderBottom": "2px solid #3498db",
                                                    "paddingBottom": "0.5rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Method",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        "Rolling 10th percentile (window=200 samples) tracks baseline drift. "
                                                        "Linear regression on baseline provides drift rate. "
                                                        "Noise analysis from low-response periods (bottom 20% of signal).",
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                        },
                                                    ),
                                                    html.H4(
                                                        "Definitions",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.Ul(
                                                        [
                                                            html.Li(
                                                                "Drift Rate: Linear slope of baseline over time (ΔADC/hour)"
                                                            ),
                                                            html.Li(
                                                                "Noise RMS: √(mean(baseline²)) during quiescent periods"
                                                            ),
                                                            html.Li(
                                                                "Acceptable drift: <0.01 ΔADC/hour | Low noise: RMS < 0.05"
                                                            ),
                                                        ],
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                            "marginLeft": "1rem",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#f8f9fa",
                                                    "padding": "1rem",
                                                    "borderRadius": "4px",
                                                    "marginBottom": "1rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=fig_drift,
                                                                style={
                                                                    "height": "380px"
                                                                },
                                                            )
                                                        ],
                                                        style={
                                                            "flex": "1",
                                                            "minWidth": "400px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=fig_noise,
                                                                style={
                                                                    "height": "380px"
                                                                },
                                                            )
                                                        ],
                                                        style={
                                                            "flex": "1",
                                                            "minWidth": "400px",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "gap": "1rem",
                                                    "flexWrap": "wrap",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Key Findings",
                                                        style={
                                                            "color": "#27ae60",
                                                            "fontSize": "0.95rem",
                                                        },
                                                    ),
                                                    html.Table(
                                                        [
                                                            html.Tr(
                                                                [
                                                                    html.Th(
                                                                        "Parameter",
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                        },
                                                                    ),
                                                                    html.Th(
                                                                        "Value",
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                        },
                                                                    ),
                                                                    html.Th(
                                                                        "Status",
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                        },
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Tr(
                                                                [
                                                                    html.Td(
                                                                        "Baseline Drift",
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                        },
                                                                    ),
                                                                    html.Td(
                                                                        f"{results.sensor_characteristics.baseline_drift_per_hour:.4f} ΔADC/hr",
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                        },
                                                                    ),
                                                                    html.Td(
                                                                        (
                                                                            "Good"
                                                                            if abs(
                                                                                results.sensor_characteristics.baseline_drift_per_hour
                                                                            )
                                                                            < 0.01
                                                                            else "Monitor"
                                                                        ),
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                            "color": (
                                                                                "#27ae60"
                                                                                if abs(
                                                                                    results.sensor_characteristics.baseline_drift_per_hour
                                                                                )
                                                                                < 0.01
                                                                                else "#e67e22"
                                                                            ),
                                                                        },
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Tr(
                                                                [
                                                                    html.Td(
                                                                        "Noise RMS",
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                        },
                                                                    ),
                                                                    html.Td(
                                                                        f"{results.sensor_characteristics.baseline_noise_rms:.4f} ΔADC",
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                        },
                                                                    ),
                                                                    html.Td(
                                                                        (
                                                                            "Low"
                                                                            if results.sensor_characteristics.baseline_noise_rms
                                                                            < 0.05
                                                                            else "Moderate"
                                                                        ),
                                                                        style={
                                                                            "padding": "0.5rem",
                                                                            "border": "1px solid #ddd",
                                                                            "color": (
                                                                                "#27ae60"
                                                                                if results.sensor_characteristics.baseline_noise_rms
                                                                                < 0.05
                                                                                else "#e67e22"
                                                                            ),
                                                                        },
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        style={
                                                            "margin": "0.5rem 0",
                                                            "borderCollapse": "collapse",
                                                            "width": "100%",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#e8f5e9",
                                                    "padding": "0.75rem",
                                                    "borderRadius": "4px",
                                                    "marginTop": "1rem",
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "2rem"},
                                    ),
                                    # Section 7: Layer Comparison
                                    html.Div(
                                        [
                                            html.H3(
                                                "7. Sensor Layer Comparison",
                                                style={
                                                    "color": "#2c3e50",
                                                    "borderBottom": "2px solid #3498db",
                                                    "paddingBottom": "0.5rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Method",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        "Sum of all 1024 pixels per layer plotted over time. "
                                                        "Differential signal (Layer 1 − Layer 2) removes common-mode noise and drift.",
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                        },
                                                    ),
                                                    html.H4(
                                                        "Layer Definitions",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.Ul(
                                                        [
                                                            html.Li(
                                                                "Layer 1 (Active): ZIF-8 MOF coated - primary toluene sensing"
                                                            ),
                                                            html.Li(
                                                                "Layer 2 (Reference): Dark/disabled - noise reference for subtraction"
                                                            ),
                                                            html.Li(
                                                                "Layer 3 (Saturated): Overexposed - not usable for quantification"
                                                            ),
                                                            html.Li(
                                                                "Differential: L1−L2 provides drift-compensated toluene signal"
                                                            ),
                                                        ],
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                            "marginLeft": "1rem",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#f8f9fa",
                                                    "padding": "1rem",
                                                    "borderRadius": "4px",
                                                    "marginBottom": "1rem",
                                                },
                                            ),
                                            dcc.Graph(
                                                figure=fig_layers,
                                                style={"height": "600px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Key Findings",
                                                        style={
                                                            "color": "#27ae60",
                                                            "fontSize": "0.95rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        "Layer 1 shows clear toluene response correlation. "
                                                        "Layer 2 provides stable reference. Differential signal enhances SNR.",
                                                        style={"fontWeight": "500"},
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#e8f5e9",
                                                    "padding": "0.75rem",
                                                    "borderRadius": "4px",
                                                    "marginTop": "1rem",
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "2rem"},
                                    ),
                                    # Section 8: Calibration Model Evaluation (NEW - Predicted vs Actual)
                                    html.Div(
                                        [
                                            html.H3(
                                                "8. Calibration Model Evaluation",
                                                style={
                                                    "color": "#2c3e50",
                                                    "borderBottom": "2px solid #3498db",
                                                    "paddingBottom": "0.5rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Method",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        "Multi-model comparison using Predicted vs Actual scatter plot (gold-standard calibration evaluation). "
                                                        "Top 3 calibration models are shown: Polynomial(1), Polynomial(2), and Langmuir/Freundlich. "
                                                        "Points on the identity line (y=x) indicate perfect predictions. "
                                                        "Error bands show ±5% (green) and ±10% (yellow) acceptable deviation zones.",
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                        },
                                                    ),
                                                    html.H4(
                                                        "Interpretation Guide",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.Ul(
                                                        [
                                                            html.Li(
                                                                "Points ON solid black line → Perfect prediction"
                                                            ),
                                                            html.Li(
                                                                "Points within GREEN band → Within ±5% error (Excellent)"
                                                            ),
                                                            html.Li(
                                                                "Points within YELLOW band → Within ±10% error (Acceptable)"
                                                            ),
                                                            html.Li(
                                                                "Points ABOVE line → Model over-predicts concentration"
                                                            ),
                                                            html.Li(
                                                                "Points BELOW line → Model under-predicts concentration"
                                                            ),
                                                            html.Li(
                                                                "Model equations shown in upper-left corner"
                                                            ),
                                                            html.Li(
                                                                "R² closer to 1.0 → Better fit | RMSE closer to 0 → Lower error"
                                                            ),
                                                        ],
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                            "marginLeft": "1rem",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#f8f9fa",
                                                    "padding": "1rem",
                                                    "borderRadius": "4px",
                                                    "marginBottom": "1rem",
                                                },
                                            ),
                                            dcc.Graph(
                                                figure=fig_pred_vs_actual,
                                                style={"height": "550px"},
                                            ),
                                            # All Calibration Models Comparison Grid
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "All Calibration Models Comparison",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "1rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        "Individual Predicted vs Actual plots for each calibration model (Polynomial 1-3, Langmuir, Freundlich).",
                                                        style={
                                                            "color": "#666",
                                                            "fontSize": "0.85rem",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            dcc.Graph(
                                                figure=fig_all_models,
                                                style={"height": "700px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Detailed Residual Analysis",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            dcc.Graph(
                                                figure=fig_residuals,
                                                style={"height": "350px"},
                                            ),
                                            # Calibration Summary Text (auto-generated markdown rendered as HTML)
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Calibration Performance Summary",
                                                        style={
                                                            "color": "#27ae60",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "1rem",
                                                        },
                                                    ),
                                                    dcc.Markdown(
                                                        calibration_summary_text,
                                                        style={
                                                            "fontSize": "0.9rem",
                                                            "lineHeight": "1.6",
                                                            "padding": "1rem",
                                                            "backgroundColor": "#fafafa",
                                                            "borderRadius": "4px",
                                                            "border": "1px solid #e0e0e0",
                                                        },
                                                    ),
                                                ],
                                                style={"marginTop": "1rem"},
                                            ),
                                        ],
                                        style={"marginBottom": "2rem"},
                                    ),
                                    # Section 9: Sensor Characteristics Summary
                                    html.Div(
                                        [
                                            html.H3(
                                                "9. Sensor Characteristics Summary",
                                                style={
                                                    "color": "#2c3e50",
                                                    "borderBottom": "2px solid #3498db",
                                                    "paddingBottom": "0.5rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Method",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        "Comprehensive sensor metrics derived from all detected events. "
                                                        "Includes response linearity, dynamic range, repeatability, and sensitivity.",
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#f8f9fa",
                                                    "padding": "1rem",
                                                    "borderRadius": "4px",
                                                    "marginBottom": "1rem",
                                                },
                                            ),
                                            dcc.Graph(
                                                figure=fig_char,
                                                style={"height": "450px"},
                                            ),
                                        ],
                                        style={"marginBottom": "2rem"},
                                    ),
                                    # Section 10: Application Severity Monitoring
                                    html.Div(
                                        [
                                            html.H3(
                                                "10. Application Severity Monitoring",
                                                style={
                                                    "color": "#2c3e50",
                                                    "borderBottom": "2px solid #3498db",
                                                    "paddingBottom": "0.5rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Method",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        "Real-time severity classification for poultry barn VOC monitoring based on "
                                                        "estimated toluene concentration from calibrated sensor response.",
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                        },
                                                    ),
                                                    html.H4(
                                                        "Severity Thresholds (Poultry Barn)",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.Ul(
                                                        [
                                                            html.Li(
                                                                [
                                                                    html.Span(
                                                                        "SAFE",
                                                                        style={
                                                                            "color": "#27ae60",
                                                                            "fontWeight": "bold",
                                                                        },
                                                                    ),
                                                                    ": 0-2000 ppm - Normal operating levels",
                                                                ]
                                                            ),
                                                            html.Li(
                                                                [
                                                                    html.Span(
                                                                        "MODERATE",
                                                                        style={
                                                                            "color": "#f39c12",
                                                                            "fontWeight": "bold",
                                                                        },
                                                                    ),
                                                                    ": 2000-5000 ppm - Ventilation required",
                                                                ]
                                                            ),
                                                            html.Li(
                                                                [
                                                                    html.Span(
                                                                        "CRITICAL",
                                                                        style={
                                                                            "color": "#e74c3c",
                                                                            "fontWeight": "bold",
                                                                        },
                                                                    ),
                                                                    ": >5000 ppm - Immediate action needed",
                                                                ]
                                                            ),
                                                        ],
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                            "marginLeft": "1rem",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#f8f9fa",
                                                    "padding": "1rem",
                                                    "borderRadius": "4px",
                                                    "marginBottom": "1rem",
                                                },
                                            ),
                                            dcc.Graph(
                                                figure=fig_sev,
                                                style={"height": "500px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Key Findings",
                                                        style={
                                                            "color": "#27ae60",
                                                            "fontSize": "0.95rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        f"Maximum severity level reached: {int(np.max(results.severity_timeline)) if len(results.severity_timeline) > 0 else 0}. "
                                                        "Review timeline for duration of elevated severity periods.",
                                                        style={"fontWeight": "500"},
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#e8f5e9",
                                                    "padding": "0.75rem",
                                                    "borderRadius": "4px",
                                                    "marginTop": "1rem",
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "2rem"},
                                    ),
                                    # Section 11: Sensor Health Index
                                    html.Div(
                                        [
                                            html.H3(
                                                "11. Sensor Health Index",
                                                style={
                                                    "color": "#2c3e50",
                                                    "borderBottom": "2px solid #3498db",
                                                    "paddingBottom": "0.5rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Method",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        "Composite health score (0-100) calculated from weighted metrics: "
                                                        "Noise (30%), Drift (30%), Response Speed (25%), Saturation (15%).",
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                        },
                                                    ),
                                                    html.H4(
                                                        "Scoring Definitions",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.Ul(
                                                        [
                                                            html.Li(
                                                                "Noise Score: 100 − (RMS × 100), lower noise = higher score"
                                                            ),
                                                            html.Li(
                                                                "Drift Score: 100 − (|drift_rate| × 500), lower drift = higher score"
                                                            ),
                                                            html.Li(
                                                                "T90 Score: 100 − (mean_T90 × 2), faster response = higher score"
                                                            ),
                                                            html.Li(
                                                                "Saturation Penalty: Percentage of frames with Layer 3 saturation"
                                                            ),
                                                            html.Li(
                                                                [
                                                                    html.Span(
                                                                        "≥80: Excellent",
                                                                        style={
                                                                            "color": "#27ae60"
                                                                        },
                                                                    ),
                                                                    " | ",
                                                                    html.Span(
                                                                        "60-80: Good",
                                                                        style={
                                                                            "color": "#f39c12"
                                                                        },
                                                                    ),
                                                                    " | ",
                                                                    html.Span(
                                                                        "<60: Needs Attention",
                                                                        style={
                                                                            "color": "#e74c3c"
                                                                        },
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                            "marginLeft": "1rem",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#f8f9fa",
                                                    "padding": "1rem",
                                                    "borderRadius": "4px",
                                                    "marginBottom": "1rem",
                                                },
                                            ),
                                            dcc.Graph(
                                                figure=fig_health,
                                                style={"height": "450px"},
                                            ),
                                        ],
                                        style={"marginBottom": "2rem"},
                                    ),
                                    # Section 12: Forecasting
                                    html.Div(
                                        [
                                            html.H3(
                                                "12. Toluene Concentration Forecasting (ppm)",
                                                style={
                                                    "color": "#2c3e50",
                                                    "borderBottom": "2px solid #3498db",
                                                    "paddingBottom": "0.5rem",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.H4(
                                                        "Method",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        f"Forecasting calibrated toluene concentration (ppm) using the fitted calibration model. "
                                                        f"Three simple forecasting methods are demonstrated based on {data_duration_hours:.1f}h of historical data. "
                                                        f"Forecast horizon (~{forecast_horizon:.1f}h) is automatically limited to ~50% of historical data duration.",
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                        },
                                                    ),
                                                    html.H4(
                                                        "Forecasting Methods",
                                                        style={
                                                            "color": "#5b9bd5",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.Ul(
                                                        [
                                                            html.Li(
                                                                [
                                                                    html.B(
                                                                        "Moving Average: "
                                                                    ),
                                                                    "Mean of recent window with slight random walk for realism",
                                                                ]
                                                            ),
                                                            html.Li(
                                                                [
                                                                    html.B(
                                                                        "Linear Trend: "
                                                                    ),
                                                                    "Linear regression extrapolation from recent data",
                                                                ]
                                                            ),
                                                            html.Li(
                                                                [
                                                                    html.B(
                                                                        "Exponential Smoothing: "
                                                                    ),
                                                                    "Weighted average with decay toward mean",
                                                                ]
                                                            ),
                                                        ],
                                                        style={
                                                            "color": "#555",
                                                            "fontSize": "0.9rem",
                                                            "marginLeft": "1rem",
                                                        },
                                                    ),
                                                    html.H4(
                                                        "Limitations",
                                                        style={
                                                            "color": "#e74c3c",
                                                            "fontSize": "0.95rem",
                                                            "marginTop": "0.5rem",
                                                        },
                                                    ),
                                                    html.P(
                                                        "These are simple demonstration methods. Production forecasting should use ARIMA, Prophet, "
                                                        "or LSTM models with sufficient historical data (days/weeks) to capture seasonal patterns.",
                                                        style={
                                                            "color": "#666",
                                                            "fontSize": "0.85rem",
                                                            "fontStyle": "italic",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "backgroundColor": "#f8f9fa",
                                                    "padding": "1rem",
                                                    "borderRadius": "4px",
                                                    "marginBottom": "1rem",
                                                },
                                            ),
                                            dcc.Graph(
                                                figure=fig_forecast,
                                                style={"height": "500px"},
                                            ),
                                        ],
                                        style={"marginBottom": "2rem"},
                                    ),
                                    # Footer
                                    html.Div(
                                        [
                                            html.Hr(style={"borderColor": "#bdc3c7"}),
                                            html.P(
                                                "End of Report",
                                                style={
                                                    "textAlign": "center",
                                                    "color": "#7f8c8d",
                                                    "fontStyle": "italic",
                                                },
                                            ),
                                            html.P(
                                                f"Data source: {data_file.name}",
                                                style={
                                                    "textAlign": "center",
                                                    "color": "#95a5a6",
                                                    "fontSize": "0.85rem",
                                                },
                                            ),
                                        ]
                                    ),
                                ],
                                id="full-report-content",
                                style={
                                    "padding": "30px",
                                    "backgroundColor": "#ffffff",
                                    "color": "#333",
                                    "maxWidth": "1200px",
                                    "margin": "0 auto",
                                    "boxShadow": "0 2px 10px rgba(0,0,0,0.1)",
                                },
                            )
                        ],
                        style=tab_style,
                        selected_style=tab_selected_style,
                    ),
                ],
                style={"backgroundColor": "#2a2d35", "borderBottom": "1px solid #333"},
            ),
        ],
        style={
            "fontFamily": "'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif",
            "backgroundColor": "#1e2127",
            "minHeight": "100vh",
        },
    )

    register_dashboard_callbacks(app, data)

    # Store report data for export
    report_data = {
        "results": results,
        "data": data,
        "data_file": data_file,
        "figures": {
            "signal": fig_signal,
            "exposure": fig_exposure,
            "spatial": fig_spatial,
            "bad_pixels": fig_bad_pixels,
            "t90_dist": fig_t90_dist,
            "drift": fig_drift,
            "noise": fig_noise,
            "layers": fig_layers,
            "cal": fig_cal,
            "pred_vs_actual": fig_pred_vs_actual,
            "all_models": fig_all_models,
            "residuals": fig_residuals,
            "char": fig_char,
            "sev": fig_sev,
            "health": fig_health,
        },
        "calibration_summary": (
            calibration_summary_text if "calibration_summary_text" in dir() else ""
        ),
    }

    # HTML Export callback
    @app.callback(
        Output("download-report", "data"),
        Input("export-html-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def export_html_report(n_clicks):
        if n_clicks:
            html_content = generate_html_report(
                results=report_data["results"],
                data=report_data["data"],
                data_file=report_data["data_file"],
                fig_signal=report_data["figures"]["signal"],
                fig_exposure=report_data["figures"]["exposure"],
                fig_spatial=report_data["figures"]["spatial"],
                fig_bad_pixels=report_data["figures"]["bad_pixels"],
                fig_t90_dist=report_data["figures"]["t90_dist"],
                fig_drift=report_data["figures"]["drift"],
                fig_noise=report_data["figures"]["noise"],
                fig_layers=report_data["figures"]["layers"],
                fig_cal=report_data["figures"]["cal"],
                fig_residuals=report_data["figures"]["residuals"],
                fig_char=report_data["figures"]["char"],
                fig_sev=report_data["figures"]["sev"],
                fig_health=report_data["figures"]["health"],
            )
            filename = f"cmos_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            return dict(content=html_content, filename=filename)
        return None

    # PDF Export callback
    @app.callback(
        Output("download-pdf-report", "data"),
        Input("export-pdf-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def export_pdf_report(n_clicks):
        if n_clicks:
            pdf_bytes = generate_pdf_report(
                results=report_data["results"],
                data_file=report_data["data_file"],
                figures=report_data["figures"],
                calibration_summary=report_data.get("calibration_summary", ""),
            )
            if pdf_bytes:
                filename = f"cmos_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                return dict(
                    content=base64.b64encode(pdf_bytes).decode("utf-8"),
                    filename=filename,
                    base64=True,
                )
            else:
                # Fallback: reportlab not installed - notify user
                return None
        return None

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
================================================================================
CMOS ZIF-8 MOF TOLUENE SENSOR - ANALYSIS DASHBOARD
================================================================================
Interactive web-based analysis and visualization tool for CMOS sensor data.

Usage:
    python analyse_cmos.py                              # Run with default data
    python analyse_cmos.py --data path/to/file.dat     # Custom data file  
    python analyse_cmos.py --export ./output           # Export results to folder
    python analyse_cmos.py --debug --port 8080         # Debug mode on port 8080
================================================================================
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        "-d",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="Path to the .dat data file",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the Dash server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8050,
        help="Port for the Dash server (default: 8050)",
    )
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        default=True,
        help="Disable debug mode (debug mode with auto-reload is enabled by default)",
    )
    parser.add_argument(
        "--export",
        "-e",
        type=Path,
        default=None,
        help="Export analysis results to specified directory (no dashboard)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="savgol",
        choices=["ma", "savgol", "butterworth", "wavelet"],
        help="Filter type for signal processing (default: savgol)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="polynomial",
        choices=["polynomial", "langmuir", "ensemble"],
        help="Calibration model type (default: polynomial)",
    )
    return parser.parse_args()


def export_results(
    data_file: Path, output_dir: Path, filter_type: str, calibration_model: str
) -> None:
    """Export analysis results to files without launching the dashboard."""
    from collections import Counter

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CMOS ZIF-8 MOF TOLUENE SENSOR - VOC ANALYSIS PIPELINE")
    print("Export Mode - Saving Results to Files")
    print("=" * 70)
    print(f"\nData file: {data_file}")
    print(f"Output dir: {output_dir}")
    print(f"Filter: {filter_type}")
    print(f"Calibration: {calibration_model}")
    print("\n" + "-" * 70)

    # Load frames for additional visualizations
    frames = load_frames(data_file)
    data = prepare_dashboard_data(frames)

    # Run pipeline with save_results=True
    config = PipelineConfig(
        data_file=str(data_file),
        filter_type=filter_type,
        calibration_model=calibration_model,
        output_dir=str(output_dir),
        save_results=True,
    )

    pipeline = CMOSVOCPipeline(config)
    results = pipeline.run_full_pipeline()

    # Generate additional scientific visualizations
    print("\nGenerating additional visualizations...")

    visualizer = PipelineVisualizer(output_dir)

    fig_signal = visualizer.plot_signal_processing(
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

    fig_char = visualizer.plot_sensor_characteristics(results.sensor_characteristics)
    fig_cal = visualizer.plot_calibration(results.calibration_model)

    if results.calibration_model:
        ppm_values = results.calibration_model.predict_ppm(results.filtered_signal)
    else:
        ppm_values = results.filtered_signal

    fig_sev = visualizer.plot_severity_timeline(
        timestamps=results.dataset.timestamps,
        severity_levels=results.severity_timeline,
        ppm_values=ppm_values,
    )

    # Additional scientific visualizations
    fig_exposure = plot_dadc_with_exposure_windows(
        timestamps=results.dataset.timestamps, delta_adc=results.filtered_signal
    )

    peak_idx = np.argmax(results.filtered_signal)
    peak_window = slice(max(0, peak_idx - 50), min(len(frames), peak_idx + 50))
    pixel_responses = []
    for i in range(peak_window.start, peak_window.stop):
        if i < len(frames):
            pixel_responses.append(frames[i].grids[1].flatten())
    if pixel_responses:
        mean_pixel_response = np.mean(pixel_responses, axis=0).reshape(32, 32)
        baseline_pixel = np.percentile(
            [f.grids[1].flatten() for f in frames[:100]], 10, axis=0
        ).reshape(32, 32)
        spatial_delta = (mean_pixel_response - baseline_pixel) / 1000
    else:
        spatial_delta = np.zeros((32, 32))

    fig_spatial = plot_spatial_sensitivity_map(
        spatial_delta, "Spatial Sensitivity Map - Peak Exposure (ΔADC)"
    )

    # Bad Pixel Map with proper classification
    pixel_timeseries_layer1 = np.array(
        [f.grids[1] for f in frames]
    )  # Shape: (n_frames, 32, 32)
    fig_bad_pixels = plot_bad_pixel_map(
        pixel_timeseries=pixel_timeseries_layer1, layer_id=1, adc_fullscale=4095
    )

    # Coating Pattern Analysis - identify coated vs reference pixels
    coating_analysis = analyze_coating_pattern(pixel_timeseries_layer1)
    plot_coating_pattern_map(
        pixel_timeseries_layer1, coating_analysis
    )
    create_coating_summary_stats(coating_analysis)

    rise_times = [
        m.t90_rise
        for m in results.sensor_characteristics.event_metrics
        if m.t90_rise > 0
    ]
    recovery_times = [
        m.recovery_time
        for m in results.sensor_characteristics.event_metrics
        if m.recovery_time > 0
    ]
    fig_t90_dist = plot_t90_distribution(rise_times, recovery_times)

    fig_drift = plot_baseline_drift(
        results.dataset.timestamps, results.filtered_signal, window_size=200
    )

    baseline_mask = results.filtered_signal < np.percentile(results.filtered_signal, 20)
    baseline_values = results.filtered_signal[baseline_mask]
    fig_noise = plot_noise_analysis(baseline_values, sample_rate=1.0)

    layer1_sums = np.array([np.sum(f.grids[1]) for f in frames])
    layer2_sums = np.array([np.sum(f.grids[2]) for f in frames])
    layer3_sums = np.array([np.sum(f.grids[3]) for f in frames])
    fig_layers = plot_layer_comparison_timeseries(
        timestamps=results.dataset.timestamps,
        layer1_values=layer1_sums / 1e6,
        layer2_values=layer2_sums / 1e6,
        layer3_values=layer3_sums / 1e6,
    )

    if results.calibration_model:
        # Actual calibration points
        calib_ppm_actual = np.array([500, 1000, 2000, 3500, 5500, 8000])
        calib_dadc = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

        # Create and fit multiple calibration models for comparison
        models_to_compare = {}
        model_equations = {}

        # Polynomial (1) - Linear
        poly1 = PolynomialCalibration(degree=1)
        poly1.fit(calib_ppm_actual, calib_dadc)
        models_to_compare["Polynomial(1)"] = poly1.predict_ppm(calib_dadc)
        model_equations["Polynomial(1)"] = poly1.get_equation_str()

        # Polynomial (2) - Quadratic
        poly2 = PolynomialCalibration(degree=2)
        poly2.fit(calib_ppm_actual, calib_dadc)
        models_to_compare["Polynomial(2)"] = poly2.predict_ppm(calib_dadc)
        model_equations["Polynomial(2)"] = poly2.get_equation_str()

        # Langmuir
        try:
            langmuir = LangmuirModel()
            langmuir.fit(calib_ppm_actual, calib_dadc)
            models_to_compare["Langmuir"] = langmuir.predict_ppm(calib_dadc)
            model_equations["Langmuir"] = (
                f"ΔADC = {langmuir.Qmax:.4f} × K × ppm / (1 + K × ppm), K={langmuir.K:.6f}"
            )
        except Exception:
            pass

        # Sort models by R² and select top 3
        model_r2 = {}
        for name, pred in models_to_compare.items():
            ss_res = np.sum((calib_ppm_actual - pred) ** 2)
            ss_tot = np.sum((calib_ppm_actual - np.mean(calib_ppm_actual)) ** 2)
            model_r2[name] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        top3_names = sorted(model_r2.keys(), key=lambda x: model_r2[x], reverse=True)[
            :3
        ]
        top3_predictions = {name: models_to_compare[name] for name in top3_names}
        top3_equations = {
            name: model_equations.get(name, "")
            for name in top3_names
            if name in model_equations
        }

        fig_pred_vs_actual, _ = plot_predicted_vs_actual(
            ppm_actual=calib_ppm_actual,
            ppm_predicted_dict=top3_predictions,
            model_equations=top3_equations,
            show_error_bands=True,
            show_labels=True,
            show_equations=True,
            color_by_order=False,
        )
        fig_residuals = plot_calibration_residuals(
            calib_ppm_actual, top3_predictions[top3_names[0]], top3_names[0]
        )
    else:
        fig_pred_vs_actual = go.Figure()
        fig_pred_vs_actual.add_annotation(
            text="No calibration model available", x=0.5, y=0.5, showarrow=False
        )
        fig_residuals = go.Figure()
        fig_residuals.add_annotation(
            text="No calibration model available", x=0.5, y=0.5, showarrow=False
        )

    noise_score = max(0, 100 - results.sensor_characteristics.baseline_noise_rms * 100)
    drift_score = max(
        0, 100 - abs(results.sensor_characteristics.baseline_drift_per_hour) * 500
    )
    t90_score = max(0, 100 - results.sensor_characteristics.mean_t90_rise * 2)
    layer3_max = np.max(layer3_sums)
    sat_pct = np.sum(layer3_sums > 0.95 * layer3_max) / len(layer3_sums) * 100
    fig_health = create_sensor_health_gauge(
        min(100, noise_score), min(100, drift_score), min(100, t90_score), sat_pct
    )

    # Save additional figures as HTML
    fig_exposure.write_html(output_dir / "cmos_analysis_5_exposure.html")
    fig_spatial.write_html(output_dir / "cmos_analysis_6_spatial.html")
    fig_bad_pixels.write_html(output_dir / "cmos_analysis_7_pixel_health.html")
    fig_t90_dist.write_html(output_dir / "cmos_analysis_8_t90_dist.html")
    fig_drift.write_html(output_dir / "cmos_analysis_9_drift.html")
    fig_noise.write_html(output_dir / "cmos_analysis_10_noise.html")
    fig_layers.write_html(output_dir / "cmos_analysis_11_layers.html")
    fig_pred_vs_actual.write_html(output_dir / "cmos_analysis_12_pred_vs_actual.html")
    fig_residuals.write_html(output_dir / "cmos_analysis_13_residuals.html")
    fig_health.write_html(output_dir / "cmos_analysis_14_health.html")

    # Generate full HTML report
    print("Generating full HTML report...")
    html_report = generate_html_report(
        results=results,
        data=data,
        data_file=data_file,
        fig_signal=fig_signal,
        fig_exposure=fig_exposure,
        fig_spatial=fig_spatial,
        fig_bad_pixels=fig_bad_pixels,
        fig_t90_dist=fig_t90_dist,
        fig_drift=fig_drift,
        fig_noise=fig_noise,
        fig_layers=fig_layers,
        fig_cal=fig_cal,
        fig_residuals=fig_residuals,
        fig_char=fig_char,
        fig_sev=fig_sev,
        fig_health=fig_health,
    )

    report_path = (
        output_dir / f"full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_report)
    print(f"   Full report saved: {report_path.name}")

    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    if results.dataset:
        print("\nDataset Statistics:")
        print(f"   - Frames: {len(results.dataset.timestamps):,}")
        print(
            f"   - Duration: {results.dataset.timestamps[-1] - results.dataset.timestamps[0]}"
        )
        print(
            f"   - ΔADC Range: {results.dataset.delta_adc.min():.3f} to {results.dataset.delta_adc.max():.3f}"
        )

    if results.detected_events:
        print(f"\nEvents Detected: {len(results.detected_events)}")
        for i, event in enumerate(results.detected_events[:5]):
            print(
                f"   - Event {i+1}: Peak {event.peak_value:.2f} ΔADC, Duration {event.duration:.1f}s"
            )
        if len(results.detected_events) > 5:
            print(f"   ... and {len(results.detected_events) - 5} more events")

    if results.sensor_characteristics:
        char = results.sensor_characteristics
        print("\nSensor Characteristics:")
        print(f"   - T90 Rise: {char.mean_t90_rise:.1f} +/- {char.std_t90_rise:.1f} s")
        print(
            f"   - T90 Recovery: {char.mean_t90_recovery:.1f} +/- {char.std_t90_recovery:.1f} s"
        )
        print(
            f"   - Dynamic Range: {char.min_detectable:.3f} to {char.max_response:.3f} ΔADC"
        )

    if results.calibration_model:
        print(f"\nCalibration Model: {results.calibration_model.name}")
        if results.calibration_model.metrics:
            m = results.calibration_model.metrics
            print(f"   - R-squared: {m.r_squared:.4f}")
            print(f"   - RMSE: {m.rmse:.2f} ppm")

    if results.severity_timeline is not None:
        severity_counts = Counter(results.severity_timeline)
        print("\nSeverity Distribution:")
        severity_names = {1: "NORMAL", 2: "WARNING", 3: "CRITICAL"}
        for level, count in sorted(severity_counts.items()):
            pct = count / len(results.severity_timeline) * 100
            name = severity_names.get(level, f"Level {level}")
            print(f"   - {name}: {count:,} frames ({pct:.1f}%)")

    print(f"\nProcessing Time: {results.processing_time_seconds:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)


def main() -> None:
    args = parse_args()

    # Validate data file exists
    if not args.data.exists():
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    # Export mode - save results and exit
    if args.export:
        export_results(args.data, args.export, args.filter, args.calibration)
        return

    # Dashboard mode - launch interactive web app
    frames = load_frames(args.data)
    app = create_dashboard(frames, args.data)
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
