# Region Selection & Matrix Rotation Implementation Summary

## Overview
Updated `analyse_cmos.py` to support interactive region selection for signal cleaning (Reference vs Coated pixels) and rotated the sensor matrix display by -90 degrees.

---

## Changes Implemented

### 1. **Matrix Rotation (-90 degrees)**

**File:** `analyse_cmos.py`, lines ~800-820

**Function:** `hex_layout(rows, cols, spacing)`

**What Changed:**
- Original transformation: `x = col_offset * spacing`, `y = -row * dy`
- New transformation: Apply -90 degree rotation `(x, y) → (y, -x)`
  ```python
  rotated_x = orig_y
  rotated_y = -orig_x
  ```
- This rotates the hexagonal grid layout to match the physical sensor orientation

**Impact:** All visualizations now display with the matrix rotated -90 degrees

---

### 2. **Region Selection Constants & Validation**

**File:** `analyse_cmos.py`, lines ~45-51

**New Constants:**
```python
MIN_PIXELS_PER_REGION = 10  # Minimum advised pixels per region
DEFAULT_REFERENCE_REGION = set()  # User-selectable, no default
DEFAULT_COATED_REGION = set()  # User-selectable, no default
```

**New Helper Functions:**

#### `validate_region_selection(region_indices, region_name)`
- Validates that selected region has at least `MIN_PIXELS_PER_REGION` pixels
- Returns: `(is_valid: bool, message: str)`
- Example: `validate_region_selection(region_set, "Reference")` → `(True, "Reference: 25 pixels ✓")`

#### `compute_region_sum(frame, region_indices, layer_id=1)`
- Computes sum of ADC values in a selected pixel region
- Returns: Sum of ADC values (int)

#### `compute_cleaned_signal(frames, coated_indices, reference_indices, layer_id=1)`
- Computes three aligned signals from user-selected regions:
  - `reference_sums`: Per-frame sum of reference region pixels
  - `coated_sums`: Per-frame sum of coated region pixels
  - `cleaned_signal`: `coated_sums - reference_sums` (cleaned signal for downstream analysis)
- Returns: `(reference_sums, coated_sums, cleaned_signal)`

---

### 3. **DashboardData Extension**

**File:** `analyse_cmos.py`, lines ~823-862

**New Fields Added:**
```python
@dataclass
class DashboardData:
    # ... existing fields ...
    
    # Region selection fields
    reference_region: set  # Set of (row, col) tuples for reference pixels
    coated_region: set     # Set of (row, col) tuples for coated pixels
    reference_sums: list[int]   # Per-frame sum of reference region
    coated_sums: list[int]      # Per-frame sum of coated region
    cleaned_signal: list[int]   # Cleaned signal: coated - reference
```

**Initialization:** Set to empty in `prepare_dashboard_data()` function (lines ~943-947)

---

### 4. **Interactive Dashboard UI for Region Selection**

**File:** `analyse_cmos.py`, lines ~1260-1360

**New UI Section:** "Region Selection for Signal Cleaning"

**Components:**

#### Reference Region Panel (Red Theme)
- Button: "🔍 Start Selecting Reference" - Toggle selection mode
- Button: "🗑️ Clear Reference" - Clear all selected reference pixels
- Status Display: Shows number of selected pixels with validation status

#### Coated Region Panel (Green Theme)
- Button: "🔍 Start Selecting Coated" - Toggle selection mode
- Button: "🗑️ Clear Coated" - Clear all selected coated pixels
- Status Display: Shows number of selected pixels with validation status

#### Apply Button
- **"✓ Apply Regions & Compute Cleaned Signal"**
- Validates both regions (minimum 10 pixels each)
- Computes cleaned signal from the selected regions
- Shows success/error messages

#### Hidden Data Stores (for state management)
```python
dcc.Store(id="reference-region-store", data={"indices": []})
dcc.Store(id="coated-region-store", data={"indices": []})
dcc.Store(id="selection-mode-store", data={"mode": None})  # "reference", "coated", or None
```

---

### 5. **Interactive Dash Callbacks**

**File:** `analyse_cmos.py`, lines ~1668-1810

**Callbacks Implemented:**

#### `toggle_ref_selection()`
- Triggered by: "Start Selecting Reference" button
- Action: Toggle selection mode on/off for reference region
- Updates: `selection-mode-store`

#### `toggle_coat_selection()`
- Triggered by: "Start Selecting Coated" button
- Action: Toggle selection mode on/off for coated region
- Updates: `selection-mode-store`

#### `clear_reference()`
- Clears all reference pixels from store
- Updates: `reference-region-store`

#### `clear_coated()`
- Clears all coated pixels from store
- Updates: `coated-region-store`

#### `update_region_status()`
- Triggered by: Region store changes
- Action: Updates status displays with pixel counts and validation status
- Color coding:
  - 🟢 Green: Valid (≥10 pixels)
  - 🔴 Red: Invalid (<10 pixels or empty)

#### `apply_regions()`
- Triggered by: "Apply Regions & Compute Cleaned Signal" button
- Actions:
  1. Validates both regions (minimum 10 pixels each)
  2. Stores selected regions in `DashboardData` object
  3. Computes cleaned signals using `compute_cleaned_signal()`
  4. Stores `reference_sums`, `coated_sums`, and `cleaned_signal`
- Output: Success message or error messages showing which regions are invalid

#### `handle_pixel_click()`
- Triggered by: Click on pixel matrix graph (`layer-graph`)
- Action: 
  - When selection mode is "reference": Toggle reference pixel in/out
  - When selection mode is "coated": Toggle coated pixel in/out
  - When selection mode is None: Do nothing (no action during normal viewing)
- Reads: customdata from click event (row, col)
- Updates: `reference-region-store` or `coated-region-store` accordingly

---

## User Workflow

### Step 1: Start Reference Selection
1. Click **"🔍 Start Selecting Reference"** button
2. Button highlights (selection mode active)
3. User clicks on pixels in the matrix to select them
4. Selected pixels are stored and count appears in status

### Step 2: Apply Reference Region
1. Continue clicking to add/remove reference pixels (minimum 10 advised)
2. Click **"🔍 Start Selecting Reference"** again to exit selection mode
3. Status shows: "Reference: N pixels ✓" if valid

### Step 3: Select Coated Region
1. Click **"🔍 Start Selecting Coated"** button
2. Click pixels to select coated region (minimum 10 advised)
3. Exit selection mode when done

### Step 4: Apply & Compute
1. Click **"✓ Apply Regions & Compute Cleaned Signal"**
2. System validates both regions
3. If valid:
   - ✓ Success message with pixel counts
   - `cleaned_signal = coated_sums - reference_sums` computed for all frames
4. If invalid:
   - ❌ Error message showing which regions need more pixels

### Step 5: Use Cleaned Signal
- The computed `cleaned_signal` is now available in `DashboardData`
- Can be used for downstream ΔADC calculations and visualizations

---

## Data Structure Example

After region selection and applying:

```python
data.reference_region = {(5, 10), (5, 11), (6, 10), ...}  # 25 pixels total
data.coated_region = {(15, 15), (15, 16), (16, 15), ...}  # 30 pixels total

data.reference_sums = [1500, 1520, 1495, ...]  # Per-frame sum of reference pixels
data.coated_sums = [3200, 3250, 3180, ...]     # Per-frame sum of coated pixels
data.cleaned_signal = [1700, 1730, 1685, ...]  # cleaned = coated - reference
```

---

## Next Steps

### Phase 2: Use Cleaned Signal for ΔADC
1. Modify `compute_delta_adc()` to work with `cleaned_signal` instead of Layer 1 total
2. Update visualization functions to display cleaned signal ΔADC
3. Add visualization mode toggle: "Absolute ADC" | "Cleaned Signal" | "ΔADC from Cleaned"

### Phase 3: Visualization Updates (Optional)
1. Highlight selected pixels in the matrix visualization (different color)
2. Show reference/coated regions in the heatmap (overlay or separate panel)
3. Add comparison chart: Reference vs Coated vs Cleaned signals over time

---

## Technical Notes

- **Rotation Math**: -90° rotation matrix: `(x, y) → (y, -x)` transforms coordinates correctly
- **Pixel Indexing**: Uses (row, col) tuples for consistency with numpy array indexing
- **Validation**: Minimum 10 pixels per region is configurable via `MIN_PIXELS_PER_REGION` constant
- **Stateless Selection**: Selection mode is tracked in stores, not global state
- **Allow Duplicate Outputs**: Pixel click callback uses `allow_duplicate=True` because both store outputs are modified in the same callback

---

## Files Modified

- `c:\Users\yywema\Documents\WP4_OBSeRVeD\OBSERVED_WP4_CMOS_DigitalTwinModel\Tolouene\analyse_cmos.py`

Total changes:
- ✅ 4 new helper functions (validate, compute_region_sum, compute_cleaned_signal, hex_layout update)
- ✅ 5 new DashboardData fields
- ✅ 1 UI panel with interactive controls
- ✅ 7 new Dash callbacks for region management
- ✅ 1 matrix rotation transformation
