# Region Selection Visual Guide

## Matrix Visualization with Rotation

```
BEFORE ROTATION:
  (Original Layout)
  Col 0  Col 1  Col 2  ...
Row 0  ●     ●     ●
Row 1    ●     ●     ●
Row 2  ●     ●     ●
...

AFTER ROTATION (-90°):
  Using transformation: (x, y) → (y, -x)
  
Row 0 becomes left edge
Col 0 becomes top edge
Coordinates are rotated to match physical sensor orientation
```

---

## Interactive Selection Flow

```
START
  ↓
┌─────────────────────────────────────────┐
│  USER SEES PIXEL MATRIX (32×32 hexagons) │
│  Matrix is rotated -90 degrees          │
└─────────────────────────────────────────┘
  ↓
  ├─→ Click "Start Selecting Reference" 
  │     ↓
  │   [Selection Mode = "reference"]
  │     ↓
  │   User clicks pixels → Toggle in/out
  │     ↓
  │   Pixels tracked in reference-region-store
  │     ↓
  │   Status shows: "Reference: N pixels" (red if <10, green if ≥10)
  │     ↓
  │   Click button again to exit
  │
  ├─→ Click "Start Selecting Coated"
  │     ↓
  │   [Selection Mode = "coated"]
  │     ↓
  │   User clicks pixels → Toggle in/out
  │     ↓
  │   Pixels tracked in coated-region-store
  │     ↓
  │   Status shows: "Coated: N pixels" (red if <10, green if ≥10)
  │     ↓
  │   Click button again to exit
  │
  └─→ Click "✓ Apply Regions & Compute Cleaned Signal"
        ↓
      ┌─────────────────────────────────┐
      │ VALIDATION                      │
      │ ✓ Reference: N ≥ 10?            │
      │ ✓ Coated: N ≥ 10?               │
      └─────────────────────────────────┘
        ↓
        ├─→ INVALID (RED X) → Show error message
        │                     User selects more pixels
        │
        └─→ VALID (GREEN ✓) → Compute signals
              ↓
              1. reference_sums = sum(ref_pixels) per frame
              2. coated_sums = sum(coated_pixels) per frame
              3. cleaned_signal = coated_sums - reference_sums
              ↓
              Store in DashboardData
              ↓
              Show success message: "Regions applied! Ref: N pixels | Coated: M pixels"
              ↓
              Ready for ΔADC calculations using cleaned_signal
```

---

## Callback Dependency Graph

```
┌──────────────────────┐
│  ref-select-btn      │
│  (Click)             │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ toggle_ref_selection()           │
│ Updates: selection-mode-store    │
└──────────────────────────────────┘
       │
       ├─────────────────────────┐
       │                         │
       ▼                         ▼
┌──────────────────────┐  ┌──────────────────────┐
│ layer-graph          │  │ selection-mode-store │
│ (Click Points)       │  │ = "reference"        │
└──────┬───────────────┘  └──────────────────────┘
       │                           │
       └─────────────┬─────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │ handle_pixel_click()           │
        │ - Reads click (row, col)       │
        │ - If mode="reference": Add/rm  │
        │ Updates: reference-region-store│
        └────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │ update_region_status()         │
        │ - Validates region             │
        │ - Updates ref-status           │
        │ - Color: 🟢 or 🔴             │
        └────────────────────────────────┘


┌──────────────────────┐
│ apply-regions-btn    │
│ (Click)              │
└──────┬───────────────┘
       │
       ├─────────────────────────────────┐
       │                                 │
       ▼                                 ▼
┌──────────────────────┐  ┌──────────────────────┐
│ reference-region-store
│ = [list of (r,c)]   │  │ coated-region-store  │
└──────────────────────┘  │ = [list of (r,c)]   │
       │                  └──────────────────────┘
       │                           │
       └─────────────┬─────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │ apply_regions()                │
        │ - Validate both regions        │
        │ - compute_cleaned_signal()     │
        │ - Update DashboardData         │
        │ - Update region-error-msg      │
        └────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │ DashboardData Updated          │
        │ - reference_region             │
        │ - coated_region                │
        │ - reference_sums[]             │
        │ - coated_sums[]                │
        │ - cleaned_signal[]             │
        └────────────────────────────────┘
```

---

## Data Flow Example

### INPUT
```
Frame 0: Layer 1 grid
  ┌────────────────────┐
  │ . . . . . . . . .  │
  │ . . . . . . . . .  │  Reference Region Pixels: {(5,8), (5,9), (6,8), (6,9), ...}
  │ . . . . . . . . .  │  Coated Region Pixels: {(20,20), (20,21), (21,20), (21,21), ...}
  │ . . . . . . . . .  │
  │ . . R R . . . . .  │  (R = Reference, C = Coated)
  │ . . R R . . . . .  │
  │ . . . . . . . . .  │
  │ . . . . . . C C .  │
  │ . . . . . . C C .  │
  └────────────────────┘
```

### COMPUTATION
```
For each frame i:
  reference_sum[i] = sum of ADC values at (5,8), (5,9), (6,8), (6,9), ...
                   = 1500

  coated_sum[i]    = sum of ADC values at (20,20), (20,21), (21,20), (21,21), ...
                   = 3200

  cleaned_signal[i] = coated_sum[i] - reference_sum[i]
                    = 3200 - 1500
                    = 1700
```

### OUTPUT
```
Time Series Data Available for Analysis:

Frame:  0      1      2      3      4    ...
Ref:    1500   1480   1520   1510   1490  ... (baseline varies with temperature)
Coated: 3200   3220   3180   3250   3300  ... (includes both ref signal + toluene response)
Clean:  1700   1740   1660   1740   1810  ... (toluene response only, ref noise removed!)
          ▲
          │
          └─ This is the signal used for ΔADC calculation
```

---

## Validation Rules

```
┌─────────────────────────────────────────────────┐
│         REGION VALIDATION                       │
├─────────────────────────────────────────────────┤
│ Minimum Pixels Required: 10 (configurable)      │
│ Recommended: 15-20 pixels per region            │
│                                                 │
│ ✗ INVALID                                       │
│   - 0 pixels selected                           │
│   - 1-9 pixels selected                         │
│   → Shows red error message                     │
│   → "Apply" button is disabled                  │
│                                                 │
│ ✓ VALID                                         │
│   - ≥10 pixels selected                         │
│   → Shows green success message                 │
│   → "Apply" button is enabled                   │
└─────────────────────────────────────────────────┘
```

---

## Key Formulas

### Cleaned Signal Calculation
```
cleaned_signal[i] = coated_sum[i] - reference_sum[i]

Where:
  coated_sum[i] = Σ(Layer1[r,c]) for (r,c) ∈ coated_region
  reference_sum[i] = Σ(Layer1[r,c]) for (r,c) ∈ reference_region
```

### Next Step: ΔADC from Cleaned Signal
```
Baseline = P₁₀(cleaned_signal)  // 10th percentile

ΔADC[i] = (cleaned_signal[i] - Baseline) / 1000
          
This removes the reference noise and gives pure toluene response!
```

---

## Status Indicators

```
Reference Region Status:
┌─────────────────────────────────────────┐
│ Status: Reference: 0 pixels             │  ← 🔴 RED (invalid)
│         (minimum 10 advised)            │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Status: Reference: 5 pixels             │  ← 🔴 RED (below minimum)
│         (minimum 10 advised)            │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Status: Reference: 18 pixels ✓          │  ← 🟢 GREEN (valid)
└─────────────────────────────────────────┘

Error Message (when applying):
┌─────────────────────────────────────────┐
│ ❌ Reference: 8 pixels (minimum 10       │
│    advised)  |  ❌ Coated: 5 pixels      │
│    (minimum 10 advised)                 │
└─────────────────────────────────────────┘

Success Message:
┌─────────────────────────────────────────┐
│ ✓ Regions applied successfully!         │
│   Reference: 15 pixels | Coated: 22     │
│   pixels                                │
└─────────────────────────────────────────┘
```
