"""
================================================================================
CMOS ZIF-8 MOF TOLUENE SENSOR ANALYSIS MODULES
================================================================================

A comprehensive modular system for analyzing CMOS sensor data
for poultry barn VOC monitoring applications.

Modules:
--------
✓ Module 1: Data Understanding & Preprocessing
✓ Module 2: Signal Processing & Noise Reduction
✓ Module 3: Sensor Characterization
✓ Module 4: Calibration & Modeling
✓ Module 5: Application Design (Poultry Barn VOC Monitoring)
✓ Module 6: Final Integration & Roadmap
✓ Module 7: Visualization Library

Quick Start:
------------
```python
from Tolouene.modules import quick_analysis

# Run complete analysis
results = quick_analysis("path/to/data.dat")

# Or use individual modules
from Tolouene.modules.module1_data_preprocessing import load_frames
from Tolouene.modules.module2_signal_processing import SignalProcessor
from Tolouene.modules.module3_sensor_characterization import full_sensor_characterization
from Tolouene.modules.module4_calibration_modeling import quick_calibration
from Tolouene.modules.module5_application_design import RealTimeMonitor
from Tolouene.modules.module6_integration import CMOSVOCPipeline
from Tolouene.modules.module7_visualization import plot_predicted_vs_actual
```
"""

__version__ = "1.0.0"
__author__ = "CMOS Sensor Analysis Team"

__all__ = [
    # Module 1
    "SensorConfig",
    "FrameRecord",
    "ProcessedDataset",
    "load_frames",
    "process_dataset",
    # Module 2
    "SignalProcessor",
    "FilterPipeline",
    "EventDetector",
    "DetectedEvent",
    # Module 3
    "SensorCharacteristics",
    "full_sensor_characterization",
    "analyze_layers",
    # Module 4
    "CalibrationModel",
    "PolynomialCalibration",
    "LangmuirModel",
    "quick_calibration",
    "compare_models",
    # Module 5
    "SeverityLevel",
    "classify_severity",
    "RealTimeMonitor",
    "AlarmManager",
    "BarnSimulator",
    # Module 6
    "CMOSVOCPipeline",
    "PipelineConfig",
    "quick_analysis",
    # Module 7
    "plot_predicted_vs_actual",
    "generate_calibration_summary_text",
    "plot_all_calibration_models",
    "create_pipeline_diagram",
]

# Convenient imports
try:
    from .module1_data_preprocessing import (FrameRecord, ProcessedDataset,
                                             SensorConfig, load_frames,
                                             process_dataset)
    from .module2_signal_processing import (DetectedEvent, EventDetector,
                                            FilterPipeline, SignalProcessor)
    from .module3_sensor_characterization import (SensorCharacteristics,
                                                  analyze_layers,
                                                  full_sensor_characterization)
    from .module4_calibration_modeling import (CalibrationModel, LangmuirModel,
                                               PolynomialCalibration,
                                               compare_models,
                                               quick_calibration)
    from .module5_application_design import (AlarmManager, BarnSimulator,
                                             RealTimeMonitor, SeverityLevel,
                                             classify_severity)
    from .module6_integration import (CMOSVOCPipeline, PipelineConfig,
                                      quick_analysis)
    from .module7_visualization import (create_pipeline_diagram,
                                        generate_calibration_summary_text,
                                        plot_all_calibration_models,
                                        plot_predicted_vs_actual)
except ImportError as e:
    # Allow partial imports during development
    import warnings

    warnings.warn(f"Some modules could not be imported: {e}")
