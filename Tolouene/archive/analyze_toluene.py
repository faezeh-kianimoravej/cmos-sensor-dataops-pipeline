"""
CMOS Toluene Sensor Data Analysis Script

This script helps explore the ZIF-8 MOF toluene detection data:
- Calculates ΔADC (change from baseline) like the reference chart
- Compares all 3 sensor matrices/layers
- Identifies concentration peaks
"""

import sys
from pathlib import Path

# Add parent directory to path to import from analyse_cmos
sys.path.insert(0, str(Path(__file__).parent))


import numpy as np
from analyse_cmos import load_frames


def analyze_toluene_response(data_file: Path):
    """Analyze the toluene concentration response from CMOS sensor data."""

    print(f"Loading data from: {data_file}")
    frames = load_frames(data_file)
    print(f"Loaded {len(frames)} frames")

    # Time range
    start_time = frames[0].timestamp
    end_time = frames[-1].timestamp
    duration = end_time - start_time
    print(f"\nTime range: {start_time} to {end_time}")
    print(f"Duration: {duration}")
    print(
        f"Average frame rate: {len(frames) / duration.total_seconds():.2f} frames/sec"
    )

    # Analyze each layer
    print("\n" + "=" * 60)
    print("LAYER ANALYSIS (All 3 Matrices)")
    print("=" * 60)

    layer_ids = sorted(frames[0].grids.keys())

    for layer_id in layer_ids:
        print(f"\n--- Layer {layer_id} ---")

        # Collect all ADC sums for this layer
        adc_sums = [int(frame.grids[layer_id].sum()) for frame in frames]
        adc_means = [float(frame.grids[layer_id].mean()) for frame in frames]

        # Statistics
        baseline_adc = np.percentile(
            adc_sums, 10
        )  # Approximate baseline (10th percentile)
        max_adc = max(adc_sums)
        min_adc = min(adc_sums)

        # Calculate ΔADC (change from baseline)
        delta_adc = [(s - baseline_adc) / 1000 for s in adc_sums]  # Normalize
        max_delta = max(delta_adc)

        print(f"  ADC Sum Range: {min_adc:,} to {max_adc:,}")
        print(f"  Baseline (10th %ile): {baseline_adc:,.0f}")
        print(f"  Max ΔADC: {max_delta:.2f} (normalized)")
        print(f"  Mean pixel ADC: {np.mean(adc_means):.1f}")
        print(
            f"  Pixel ADC range: {min([f.grids[layer_id].min() for f in frames])} - {max([f.grids[layer_id].max() for f in frames])}"
        )

    # Find peaks (potential toluene concentration changes)
    print("\n" + "=" * 60)
    print("PEAK DETECTION (Potential Concentration Changes)")
    print("=" * 60)

    # Use Layer 1 for peak detection
    layer1_sums = [int(frame.grids[1].sum()) for frame in frames]
    baseline = np.percentile(layer1_sums, 10)

    # Simple peak detection: find local maxima above threshold
    threshold = baseline + (max(layer1_sums) - baseline) * 0.3

    in_peak = False
    peaks = []
    peak_start = None

    for i, (adc_sum, frame) in enumerate(zip(layer1_sums, frames)):
        if adc_sum > threshold and not in_peak:
            in_peak = True
            peak_start = i
        elif adc_sum < threshold and in_peak:
            in_peak = False
            peak_max_idx = peak_start + np.argmax(layer1_sums[peak_start:i])
            peaks.append(
                {
                    "idx": peak_max_idx,
                    "time": frames[peak_max_idx].timestamp,
                    "adc": layer1_sums[peak_max_idx],
                    "delta": (layer1_sums[peak_max_idx] - baseline) / 1000,
                }
            )

    print(f"\nFound {len(peaks)} potential concentration peaks:")
    print("(Based on 10-minute gas switching cycles)\n")

    for i, peak in enumerate(peaks[:15], 1):  # Show first 15 peaks
        print(
            f"  Peak {i:2d}: {peak['time'].strftime('%H:%M:%S')} | "
            f"ADC Sum: {peak['adc']:,} | ΔADC: {peak['delta']:.2f}"
        )

    if len(peaks) > 15:
        print(f"  ... and {len(peaks) - 15} more peaks")

    # Estimate concentration pattern
    print("\n" + "=" * 60)
    print("CONCENTRATION PATTERN ESTIMATE")
    print("=" * 60)
    print("""
Based on the reference chart, the experiment follows this pattern:
  - 500 ppm → 1000 ppm → 3000 ppm → 5000 ppm → 7000 ppm → 9000 ppm
  - Then decreasing: 7000 → 5000 → 3000 → 1000 → 500 ppm
  - Each concentration held for ~10 minutes
  - Nitrogen baseline between each toluene exposure

Higher ΔADC values correspond to higher toluene concentrations.
The ZIF-8 MOF coating on the CMOS sensor adsorbs toluene molecules,
changing the electrical properties detected by each pixel.
    """)

    return frames, layer1_sums, peaks


if __name__ == "__main__":
    data_file = Path(__file__).parent / "dataset" / "tolouene_aug_2024.dat"

    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        sys.exit(1)

    frames, adc_data, peaks = analyze_toluene_response(data_file)

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
To further explore the data:

1. Run the dashboard with 'All Layers' view to compare all 3 matrices:
   python Tolouene/analyse_cmos.py

2. Use the slider to navigate to peak times identified above

3. Compare Layer 1, 2, and 3 responses - they may have different
   sensitivities or detect different aspects of the gas response

4. The spatial pattern (hexagonal heatmap) shows which pixels
   respond most strongly - useful for identifying sensor hotspots
   or calibration issues
    """)
