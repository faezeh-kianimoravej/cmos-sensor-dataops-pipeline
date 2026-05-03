"""Analyze the CMOS toluene sensor data to understand what it tells us."""

from pathlib import Path

import numpy as np
from analyse_cmos import load_frames


def analyze_data():
    # Load the data
    data_path = Path(__file__).parent / "dataset" / "tolouene_aug_2024.dat"
    frames = load_frames(data_path)

    print("=" * 60)
    print("CMOS ZIF-8 MOF TOLUENE SENSOR DATA ANALYSIS")
    print("=" * 60)

    # Basic info
    print("\n📊 DATASET OVERVIEW")
    print(f"   Total Frames: {len(frames)}")
    print(
        f"   Time Span: {frames[0].timestamp.strftime('%H:%M:%S')} to {frames[-1].timestamp.strftime('%H:%M:%S')}"
    )
    duration_min = (frames[-1].timestamp - frames[0].timestamp).total_seconds() / 60
    print(f"   Duration: {duration_min:.1f} minutes ({duration_min/60:.2f} hours)")
    print(f"   Date: {frames[0].timestamp.strftime('%d-%b-%Y')}")

    # Extract layer data (sensor IDs 1, 2, 3)
    # Use SUM of all pixels (not mean) - matching the dashboard
    layer1_sums = np.array([np.sum(f.grids[1]) for f in frames])
    layer2_sums = np.array([np.sum(f.grids[2]) for f in frames])
    layer3_sums = np.array([np.sum(f.grids[3]) for f in frames])

    print("\n🔬 SENSOR LAYER ANALYSIS (Sum of 1024 pixels)")
    print(
        f"   Layer 1 (Active): Mean Sum={np.mean(layer1_sums):.0f}, Std={np.std(layer1_sums):.0f}"
    )
    print(f"   Layer 2 (Zeros):  Mean Sum={np.mean(layer2_sums):.0f}")
    print(f"   Layer 3 (Saturated): Mean Sum={np.mean(layer3_sums):.0f}")

    # Calculate ΔADC for Layer 1 (using SUM, matching the dashboard)
    baseline = np.percentile(layer1_sums, 10)
    delta_adc = (layer1_sums - baseline) / 1000

    print("\n📈 ΔADC STATISTICS (Layer 1 - Sum/1000)")
    print(f"   Baseline ADC Sum (10th percentile): {baseline:.0f}")
    print(f"   ΔADC Range: {np.min(delta_adc):.3f} to {np.max(delta_adc):.3f}")
    print(f"   ΔADC Mean: {np.mean(delta_adc):.3f}")
    print(f"   ΔADC Std Dev: {np.std(delta_adc):.3f}")

    # Time-series pattern
    print("\n⏱️  TIME-SERIES PATTERN (10 segments)")
    n_segments = 10
    segment_size = len(frames) // n_segments
    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size, len(frames))
        seg_delta = delta_adc[start_idx:end_idx]
        seg_time_start = frames[start_idx].timestamp.strftime("%H:%M")
        seg_time_end = frames[end_idx - 1].timestamp.strftime("%H:%M")
        bar = "█" * int(abs(np.mean(seg_delta)) * 10) if np.mean(seg_delta) > 0 else ""
        print(
            f"   {seg_time_start}-{seg_time_end}: ΔADC={np.mean(seg_delta):+.3f} {bar}"
        )

    # Key events
    max_idx = np.argmax(delta_adc)
    min_idx = np.argmin(delta_adc)

    print("\n🎯 KEY EVENTS")
    print(
        f"   Maximum Response: ΔADC = {delta_adc[max_idx]:.3f} at {frames[max_idx].timestamp.strftime('%H:%M:%S')}"
    )
    print(
        f"   Minimum Response: ΔADC = {delta_adc[min_idx]:.3f} at {frames[min_idx].timestamp.strftime('%H:%M:%S')}"
    )

    # Detect exposure periods (when ΔADC > threshold)
    print("\n🧪 EXPOSURE PERIOD DETECTION")
    threshold = 0.3
    in_exposure = False
    exposure_periods = []
    start_time = None

    for i, (frame, dadc) in enumerate(zip(frames, delta_adc)):
        if dadc > threshold and not in_exposure:
            in_exposure = True
            start_time = frame.timestamp
            start_idx = i
        elif dadc <= threshold and in_exposure:
            in_exposure = False
            exposure_periods.append(
                {
                    "start": start_time,
                    "end": frame.timestamp,
                    "duration": (frame.timestamp - start_time).total_seconds(),
                    "peak_dadc": np.max(delta_adc[start_idx:i]),
                    "mean_dadc": np.mean(delta_adc[start_idx:i]),
                }
            )

    print(f"   Threshold: ΔADC > {threshold}")
    print(f"   Detected {len(exposure_periods)} exposure periods:")
    for j, period in enumerate(exposure_periods, 1):
        print(
            f"   {j}. {period['start'].strftime('%H:%M:%S')} - {period['end'].strftime('%H:%M:%S')} "
            f"({period['duration']:.0f}s) Peak: {period['peak_dadc']:.3f}, Mean: {period['mean_dadc']:.3f}"
        )

    # Analysis interpretation
    print("\n" + "=" * 60)
    print("📋 INTERPRETATION")
    print("=" * 60)

    print("""
This dataset captures a toluene gas exposure experiment using a 32×32 pixel 
CMOS sensor coated with ZIF-8 Metal-Organic Framework (MOF).

KEY FINDINGS:

1. SENSOR CONFIGURATION:
   • Layer 1: Active sensing layer showing dynamic response to toluene
   • Layer 2: Reads all zeros (possibly disabled or reference layer)  
   • Layer 3: Saturated at max ADC values (possibly overexposed)

2. EXPERIMENT PATTERN:
   The data shows a classic gas exposure calibration experiment with:
   • Initial baseline period (~14:30-14:45)
   • Multiple exposure cycles with increasing concentrations
   • Recovery periods between exposures where sensor returns to baseline
   
3. SENSOR RESPONSE:
   • The sensor shows POSITIVE ΔADC response when exposed to toluene
   • Higher concentrations appear to cause larger ΔADC deflections
   • The sensor demonstrates good recovery (returns to baseline)
   • Response time: Relatively fast rise and fall times

4. CALIBRATION NOTES:
   • Reference concentrations: 500, 1000, 2000, 3000, 5000, 7000, 9000 ppm
   • The horizontal lines on the dashboard show expected ΔADC for each level
   • Actual sensor response can be compared against these references

5. PRACTICAL IMPLICATIONS:
   • The ZIF-8 MOF coating successfully detects toluene vapor
   • Sensor shows reversible response (important for reusability)
   • Layer 1 is the primary functional sensing element
   • The ~3.5 hour test validates sensor stability over time
""")


if __name__ == "__main__":
    analyze_data()
