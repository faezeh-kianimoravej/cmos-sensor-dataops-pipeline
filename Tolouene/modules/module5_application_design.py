"""
================================================================================
MODULE 5: APPLICATION DESIGN - POULTRY BARN VOC MONITORING
================================================================================
Status: IN PROGRESS

Objective:
----------
Design a complete VOC monitoring application for poultry barns:
- Health severity classification
- Alarm systems
- Deployment simulation
- Real-time dashboard design

Context:
--------
Toluene and other VOCs are biomarkers for poultry health issues:
- Respiratory diseases (aspergillosis, infectious bronchitis)
- Metabolic disorders
- Stress responses
- Environmental contamination

Dependencies:
-------------
numpy, pandas, datetime
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# 5.1 VOC SEVERITY LEVELS FOR POULTRY HEALTH
# =============================================================================


class SeverityLevel(Enum):
    """VOC severity classification for poultry health."""

    NORMAL = auto()  # Baseline - healthy conditions
    WARNING = auto()  # Action required - ventilation/check
    CRITICAL = auto()  # Immediate danger - evacuate/isolate


@dataclass
class SeverityThreshold:
    """Threshold definition for a severity level."""

    level: SeverityLevel
    min_ppm: float
    max_ppm: float
    min_dadc: float
    max_dadc: float
    color: str
    description: str
    action: str


# Poultry-specific VOC thresholds based on research
POULTRY_VOC_THRESHOLDS = [
    SeverityThreshold(
        level=SeverityLevel.NORMAL,
        min_ppm=0,
        max_ppm=2000,
        min_dadc=0,
        max_dadc=1.5,
        color="#22c55e",  # Green
        description="Normal barn conditions",
        action="Continue normal monitoring",
    ),
    SeverityThreshold(
        level=SeverityLevel.WARNING,
        min_ppm=2000,
        max_ppm=5000,
        min_dadc=1.5,
        max_dadc=2.5,
        color="#f97316",  # Orange
        description="Warning - Poor air quality",
        action="Increase ventilation, inspect flock",
    ),
    SeverityThreshold(
        level=SeverityLevel.CRITICAL,
        min_ppm=5000,
        max_ppm=float("inf"),
        min_dadc=2.5,
        max_dadc=float("inf"),
        color="#ef4444",  # Red
        description="Critical - Dangerous VOC levels",
        action="Immediate veterinary attention, evacuate if necessary",
    ),
]


def classify_severity(delta_adc: float) -> SeverityThreshold:
    """Classify a ΔADC reading into severity level."""
    # Handle negative values (baseline drift/noise) as Normal
    if delta_adc < 0:
        return POULTRY_VOC_THRESHOLDS[0]  # Normal

    for threshold in POULTRY_VOC_THRESHOLDS:
        if threshold.min_dadc <= delta_adc < threshold.max_dadc:
            return threshold
    return POULTRY_VOC_THRESHOLDS[-1]  # Emergency for anything above


def classify_severity_ppm(ppm: float) -> SeverityThreshold:
    """Classify a ppm reading into severity level."""
    for threshold in POULTRY_VOC_THRESHOLDS:
        if threshold.min_ppm <= ppm < threshold.max_ppm:
            return threshold
    return POULTRY_VOC_THRESHOLDS[-1]


# =============================================================================
# 5.2 EVENT SCORING SYSTEM
# =============================================================================


@dataclass
class VOCEvent:
    """Represents a VOC exposure event."""

    event_id: str
    start_time: datetime
    end_time: Optional[datetime]
    peak_dadc: float
    peak_ppm: float
    duration_minutes: float
    severity: SeverityLevel
    score: float
    is_active: bool = False


@dataclass
class EventScore:
    """Scoring breakdown for an event."""

    intensity_score: float  # Based on peak concentration
    duration_score: float  # Based on exposure duration
    rate_score: float  # Based on rate of rise
    total_score: float
    risk_category: str


class EventScorer:
    """
    Score VOC events based on poultry health impact.

    Scoring factors:
    1. Intensity (peak concentration)
    2. Duration (exposure time)
    3. Rate of change (sudden vs gradual)
    4. Cumulative exposure (AUC)
    """

    # Scoring weights
    INTENSITY_WEIGHT = 0.4
    DURATION_WEIGHT = 0.35
    RATE_WEIGHT = 0.15
    CUMULATIVE_WEIGHT = 0.10

    def __init__(self):
        self.event_history: List[VOCEvent] = []

    def score_event(
        self,
        peak_dadc: float,
        duration_minutes: float,
        rate_of_rise: float,  # ΔADC per minute
        cumulative_exposure: Optional[float] = None,
    ) -> EventScore:
        """Calculate event score (0-100)."""

        # Intensity score (0-100)
        # Normalize to 0-4 ΔADC range
        intensity_score = min(100, (peak_dadc / 4.0) * 100)

        # Duration score (0-100)
        # Critical after 60 minutes, max at 180 minutes
        duration_score = min(100, (duration_minutes / 180) * 100)

        # Rate score (0-100)
        # Rapid rises are more concerning
        # Normalize to 0.5 ΔADC/min = 100
        rate_score = min(100, (rate_of_rise / 0.5) * 100)

        # Total weighted score
        total = (
            intensity_score * self.INTENSITY_WEIGHT
            + duration_score * self.DURATION_WEIGHT
            + rate_score * self.RATE_WEIGHT
        )

        # Add cumulative if available
        if cumulative_exposure is not None:
            # Normalize: 100 ppm-hours = max concern
            cumulative_score = min(100, cumulative_exposure / 100 * 100)
            total = total * 0.9 + cumulative_score * 0.1

        # Risk category
        if total < 25:
            risk = "Low"
        elif total < 50:
            risk = "Moderate"
        elif total < 75:
            risk = "High"
        else:
            risk = "Critical"

        return EventScore(
            intensity_score=intensity_score,
            duration_score=duration_score,
            rate_score=rate_score,
            total_score=total,
            risk_category=risk,
        )

    def calculate_cumulative_exposure(
        self,
        dadc_values: np.ndarray,
        timestamps: np.ndarray,  # In hours
        ppm_model: Optional[Callable] = None,
    ) -> float:
        """
        Calculate cumulative exposure (ppm-hours).

        Integral of concentration over time.
        """
        if ppm_model is None:
            # Simple linear approximation
            ppm_values = dadc_values * 2000  # Rough conversion
        else:
            ppm_values = ppm_model(dadc_values)

        # Trapezoidal integration
        dt = np.diff(timestamps)
        avg_ppm = (ppm_values[:-1] + ppm_values[1:]) / 2
        cumulative = np.sum(avg_ppm * dt)

        return float(cumulative)


# =============================================================================
# 5.3 ALARM SYSTEM
# =============================================================================


class AlarmPriority(Enum):
    """Alarm priority levels."""

    INFO = 1
    WARNING = 2
    ALERT = 3
    ALARM = 4
    CRITICAL = 5


@dataclass
class Alarm:
    """Individual alarm instance."""

    alarm_id: str
    timestamp: datetime
    priority: AlarmPriority
    severity: SeverityLevel
    message: str
    dadc_value: float
    ppm_value: float
    acknowledged: bool = False
    cleared: bool = False
    cleared_time: Optional[datetime] = None


class AlarmManager:
    """
    Manages alarm generation, acknowledgment, and history.

    Features:
    - Hysteresis to prevent alarm chatter
    - Alarm suppression (dead time)
    - Acknowledgment tracking
    - History logging
    """

    def __init__(
        self,
        hysteresis_margin: float = 0.1,
        dead_time_seconds: float = 60,
        max_history: int = 1000,
    ):
        self.hysteresis_margin = hysteresis_margin
        self.dead_time_seconds = dead_time_seconds
        self.max_history = max_history

        self.active_alarms: Dict[str, Alarm] = {}
        self.alarm_history: deque = deque(maxlen=max_history)
        self.last_alarm_time: Dict[SeverityLevel, datetime] = {}
        self.current_level: SeverityLevel = SeverityLevel.NORMAL

        self._alarm_counter = 0

    def _generate_alarm_id(self) -> str:
        """Generate unique alarm ID."""
        self._alarm_counter += 1
        return f"ALM-{datetime.now().strftime('%Y%m%d')}-{self._alarm_counter:04d}"

    def check_and_generate_alarms(
        self, delta_adc: float, ppm_value: float, timestamp: datetime
    ) -> List[Alarm]:
        """
        Check if alarms should be generated.

        Returns list of new alarms generated.
        """
        new_alarms = []
        threshold = classify_severity(delta_adc)

        # Check for level increase (with hysteresis)
        if threshold.level.value > self.current_level.value:
            # Only trigger if above hysteresis margin
            margin_dadc = threshold.min_dadc + self.hysteresis_margin
            if delta_adc >= margin_dadc:
                # Check dead time
                last_time = self.last_alarm_time.get(threshold.level)
                if (
                    last_time is None
                    or (timestamp - last_time).total_seconds() >= self.dead_time_seconds
                ):
                    alarm = self._create_alarm(
                        threshold, delta_adc, ppm_value, timestamp
                    )
                    new_alarms.append(alarm)
                    self.active_alarms[alarm.alarm_id] = alarm
                    self.last_alarm_time[threshold.level] = timestamp

        # Update current level (with hysteresis on decrease)
        if threshold.level.value >= self.current_level.value:
            self.current_level = threshold.level
        elif delta_adc < (
            classify_severity(delta_adc).min_dadc - self.hysteresis_margin
        ):
            self.current_level = threshold.level
            # Clear alarms for higher levels
            self._clear_higher_alarms(threshold.level, timestamp)

        return new_alarms

    def _create_alarm(
        self,
        threshold: SeverityThreshold,
        delta_adc: float,
        ppm_value: float,
        timestamp: datetime,
    ) -> Alarm:
        """Create a new alarm."""
        priority = AlarmPriority(min(threshold.level.value, 5))

        message = f"{threshold.level.name}: {threshold.description}. {threshold.action}"

        return Alarm(
            alarm_id=self._generate_alarm_id(),
            timestamp=timestamp,
            priority=priority,
            severity=threshold.level,
            message=message,
            dadc_value=delta_adc,
            ppm_value=ppm_value,
        )

    def _clear_higher_alarms(self, current_level: SeverityLevel, timestamp: datetime):
        """Clear alarms for levels above current."""
        to_clear = []
        for alarm_id, alarm in self.active_alarms.items():
            if alarm.severity.value > current_level.value and not alarm.cleared:
                alarm.cleared = True
                alarm.cleared_time = timestamp
                self.alarm_history.append(alarm)
                to_clear.append(alarm_id)

        for alarm_id in to_clear:
            del self.active_alarms[alarm_id]

    def acknowledge_alarm(self, alarm_id: str) -> bool:
        """Acknowledge an alarm."""
        if alarm_id in self.active_alarms:
            self.active_alarms[alarm_id].acknowledged = True
            return True
        return False

    def get_active_alarms(self) -> List[Alarm]:
        """Get all active (uncleared) alarms."""
        return list(self.active_alarms.values())

    def get_alarm_summary(self) -> Dict:
        """Get summary of alarm status."""
        active = self.get_active_alarms()
        return {
            "total_active": len(active),
            "unacknowledged": sum(1 for a in active if not a.acknowledged),
            "by_priority": {
                p.name: sum(1 for a in active if a.priority == p) for p in AlarmPriority
            },
            "current_level": self.current_level.name,
        }


# =============================================================================
# 5.4 REAL-TIME MONITORING SYSTEM
# =============================================================================


@dataclass
class SensorReading:
    """Single sensor reading."""

    timestamp: datetime
    delta_adc: float
    ppm: float
    layer1_sum: float
    temperature: Optional[float] = None
    humidity: Optional[float] = None


@dataclass
class BarnStatus:
    """Current barn monitoring status."""

    timestamp: datetime
    current_severity: SeverityLevel
    current_dadc: float
    current_ppm: float
    trend: str  # 'rising', 'falling', 'stable'
    active_alarms: int
    event_score: float
    recommendation: str


class RealTimeMonitor:
    """
    Real-time VOC monitoring system for poultry barn.

    Features:
    - Moving average smoothing
    - Trend detection
    - Alarm integration
    - Status reporting
    """

    def __init__(
        self,
        window_size: int = 10,
        ppm_model: Optional[Callable] = None,
        alarm_manager: Optional[AlarmManager] = None,
    ):
        self.window_size = window_size
        self.ppm_model = ppm_model or self._default_ppm_model
        self.alarm_manager = alarm_manager or AlarmManager()
        self.event_scorer = EventScorer()

        self.reading_buffer: deque = deque(maxlen=window_size)
        self.trend_buffer: deque = deque(maxlen=60)  # 1 minute for trend

        self.current_event: Optional[VOCEvent] = None
        self.event_start_dadc: float = 0

    def _default_ppm_model(self, dadc: float) -> float:
        """Default ΔADC to ppm conversion."""
        # Linear approximation: 0.5 ΔADC = 500 ppm, 3.5 ΔADC = 9000 ppm
        return max(0, 500 + (dadc - 0.5) * (8500 / 3.0))

    def process_reading(self, reading: SensorReading) -> BarnStatus:
        """Process new sensor reading and return status."""
        self.reading_buffer.append(reading)
        self.trend_buffer.append(reading.delta_adc)

        # Smoothed current value
        smoothed_dadc = np.mean([r.delta_adc for r in self.reading_buffer])
        current_ppm = self.ppm_model(smoothed_dadc)

        # Check severity
        severity = classify_severity(smoothed_dadc)

        # Check alarms
        self.alarm_manager.check_and_generate_alarms(
            smoothed_dadc, current_ppm, reading.timestamp
        )

        # Detect trend
        trend = self._detect_trend()

        # Update event tracking
        self._update_event_tracking(smoothed_dadc, reading.timestamp)

        # Calculate current event score
        event_score = 0.0
        if self.current_event is not None:
            duration = (
                reading.timestamp - self.current_event.start_time
            ).total_seconds() / 60
            rate = (smoothed_dadc - self.event_start_dadc) / max(duration, 0.1)
            score = self.event_scorer.score_event(
                peak_dadc=self.current_event.peak_dadc,
                duration_minutes=duration,
                rate_of_rise=rate,
            )
            event_score = score.total_score

        # Generate recommendation
        recommendation = self._generate_recommendation(severity, trend, event_score)

        return BarnStatus(
            timestamp=reading.timestamp,
            current_severity=severity.level,
            current_dadc=smoothed_dadc,
            current_ppm=current_ppm,
            trend=trend,
            active_alarms=len(self.alarm_manager.get_active_alarms()),
            event_score=event_score,
            recommendation=recommendation,
        )

    def _detect_trend(self) -> str:
        """Detect trend from recent readings."""
        if len(self.trend_buffer) < 5:
            return "stable"

        values = np.array(self.trend_buffer)
        # Linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        # Threshold for trend detection
        if slope > 0.01:  # Rising
            return "rising"
        elif slope < -0.01:  # Falling
            return "falling"
        else:
            return "stable"

    def _update_event_tracking(self, dadc: float, timestamp: datetime):
        """Track VOC events."""
        threshold = classify_severity(dadc)

        if threshold.level.value >= SeverityLevel.ELEVATED.value:
            if self.current_event is None:
                # Start new event
                self.current_event = VOCEvent(
                    event_id=f"EVT-{timestamp.strftime('%Y%m%d%H%M%S')}",
                    start_time=timestamp,
                    end_time=None,
                    peak_dadc=dadc,
                    peak_ppm=self.ppm_model(dadc),
                    duration_minutes=0,
                    severity=threshold.level,
                    score=0,
                    is_active=True,
                )
                self.event_start_dadc = dadc
            else:
                # Update ongoing event
                if dadc > self.current_event.peak_dadc:
                    self.current_event.peak_dadc = dadc
                    self.current_event.peak_ppm = self.ppm_model(dadc)
                if threshold.level.value > self.current_event.severity.value:
                    self.current_event.severity = threshold.level
                self.current_event.duration_minutes = (
                    timestamp - self.current_event.start_time
                ).total_seconds() / 60
        else:
            if self.current_event is not None:
                # End event
                self.current_event.end_time = timestamp
                self.current_event.is_active = False
                self.current_event.duration_minutes = (
                    timestamp - self.current_event.start_time
                ).total_seconds() / 60
                # Store in history
                self.event_scorer.event_history.append(self.current_event)
                self.current_event = None

    def _generate_recommendation(
        self, severity: SeverityThreshold, trend: str, event_score: float
    ) -> str:
        """Generate actionable recommendation."""
        recommendations = [severity.action]

        if trend == "rising" and severity.level.value >= SeverityLevel.ELEVATED.value:
            recommendations.append("VOC levels rising - prepare for escalation")

        if event_score > 50:
            recommendations.append("Extended exposure - consider flock inspection")

        if severity.level == SeverityLevel.CRITICAL:
            recommendations.append("URGENT: Veterinary consultation recommended")

        return " | ".join(recommendations)


# =============================================================================
# 5.5 BARN DEPLOYMENT SIMULATION
# =============================================================================


@dataclass
class BarnConfig:
    """Configuration for poultry barn monitoring."""

    barn_id: str
    barn_name: str
    capacity: int  # Number of birds
    area_sqm: float
    num_sensors: int
    sensor_positions: List[Tuple[float, float]]  # (x, y) coordinates
    ventilation_rate: float  # m³/hour
    baseline_ppm: float = 200  # Normal background VOC


class BarnSimulator:
    """
    Simulate VOC levels in a poultry barn.

    Used for testing and training purposes.
    """

    def __init__(self, config: BarnConfig, seed: int = 42):
        self.config = config
        np.random.seed(seed)

        self.current_time = datetime.now()
        self.base_level = config.baseline_ppm
        self.event_active = False
        self.event_peak = 0
        self.event_start = None

    def simulate_reading(self) -> SensorReading:
        """Generate simulated sensor reading."""
        self.current_time += timedelta(seconds=1)

        # Base level with noise
        noise = np.random.normal(0, 20)  # ±20 ppm noise

        # Diurnal variation (higher in afternoon)
        hour = self.current_time.hour
        diurnal = 50 * np.sin(2 * np.pi * (hour - 6) / 24)

        # Random event injection (5% chance per minute)
        if not self.event_active and np.random.random() < 0.05 / 60:
            self.event_active = True
            self.event_peak = np.random.uniform(1000, 6000)  # Random severity
            self.event_start = self.current_time

        # Event contribution
        event_level = 0
        if self.event_active:
            elapsed = (self.current_time - self.event_start).total_seconds()
            # Rise phase (30 seconds to peak)
            if elapsed < 30:
                event_level = self.event_peak * (elapsed / 30)
            # Decay phase (exponential)
            else:
                event_level = self.event_peak * np.exp(-(elapsed - 30) / 60)

            # End event when below threshold
            if event_level < 100:
                self.event_active = False
                self.event_start = None

        # Total PPM
        total_ppm = max(0, self.base_level + noise + diurnal + event_level)

        # Convert to ΔADC
        delta_adc = (total_ppm - 500) * (3.0 / 8500) + 0.5
        delta_adc = max(0, delta_adc)

        # Simulated Layer 1 sum
        layer1_sum = 1736172 + delta_adc * 1000 + np.random.normal(0, 50)

        return SensorReading(
            timestamp=self.current_time,
            delta_adc=delta_adc,
            ppm=total_ppm,
            layer1_sum=layer1_sum,
            temperature=25 + np.random.normal(0, 1),
            humidity=60 + np.random.normal(0, 5),
        )

    def simulate_period(self, duration_minutes: int) -> List[SensorReading]:
        """Simulate readings for a period."""
        readings = []
        for _ in range(duration_minutes * 60):  # 1 reading per second
            readings.append(self.simulate_reading())
        return readings


# =============================================================================
# 5.6 DASHBOARD COMPONENTS
# =============================================================================


def generate_status_card_html(status: BarnStatus) -> str:
    """Generate HTML for status card."""
    severity_colors = {
        SeverityLevel.NORMAL: "#22c55e",
        SeverityLevel.ELEVATED: "#eab308",
        SeverityLevel.HIGH: "#f97316",
        SeverityLevel.CRITICAL: "#ef4444",
        SeverityLevel.EMERGENCY: "#7f1d1d",
    }

    color = severity_colors.get(status.current_severity, "#gray")

    return f"""
    <div style="background: linear-gradient(135deg, {color}20, {color}10); 
                border-left: 4px solid {color}; padding: 16px; border-radius: 8px;">
        <h3 style="color: {color}; margin: 0;">{status.current_severity.name}</h3>
        <p style="font-size: 24px; font-weight: bold; margin: 8px 0;">
            {status.current_ppm:.0f} ppm ({status.current_dadc:.2f} ΔADC)
        </p>
        <p style="color: #666;">Trend: {status.trend.upper()} | 
           Alarms: {status.active_alarms} | 
           Event Score: {status.event_score:.1f}</p>
        <p style="font-size: 12px; color: #888;">{status.recommendation}</p>
    </div>
    """


def generate_alarm_table_html(alarms: List[Alarm]) -> str:
    """Generate HTML table for active alarms."""
    if not alarms:
        return "<p>No active alarms</p>"

    rows = []
    for alarm in sorted(alarms, key=lambda a: a.priority.value, reverse=True):
        ack_btn = "✓" if alarm.acknowledged else "○"
        rows.append(f"""
            <tr>
                <td>{alarm.timestamp.strftime('%H:%M:%S')}</td>
                <td><span style="color: {'red' if alarm.priority.value >= 4 else 'orange'};">
                    {alarm.priority.name}</span></td>
                <td>{alarm.message[:50]}...</td>
                <td>{alarm.ppm_value:.0f}</td>
                <td>{ack_btn}</td>
            </tr>
        """)

    return f"""
    <table style="width: 100%; border-collapse: collapse;">
        <thead>
            <tr>
                <th>Time</th>
                <th>Priority</th>
                <th>Message</th>
                <th>PPM</th>
                <th>Ack</th>
            </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """


# =============================================================================
# 5.7 REPORTING
# =============================================================================


@dataclass
class DailyReport:
    """Daily monitoring report."""

    date: str
    total_events: int
    max_severity_reached: SeverityLevel
    max_ppm: float
    max_dadc: float
    total_elevated_minutes: float
    total_high_minutes: float
    total_critical_minutes: float
    alarms_generated: int
    recommendations: List[str]


def generate_daily_report(
    readings: List[SensorReading], events: List[VOCEvent], alarms: List[Alarm]
) -> DailyReport:
    """Generate daily summary report."""

    if not readings:
        return DailyReport(
            date=datetime.now().strftime("%Y-%m-%d"),
            total_events=0,
            max_severity_reached=SeverityLevel.NORMAL,
            max_ppm=0,
            max_dadc=0,
            total_elevated_minutes=0,
            total_high_minutes=0,
            total_critical_minutes=0,
            alarms_generated=0,
            recommendations=["No data available"],
        )

    date_str = readings[0].timestamp.strftime("%Y-%m-%d")

    # Calculate statistics
    dadc_values = np.array([r.delta_adc for r in readings])
    ppm_values = np.array([r.ppm for r in readings])

    max_dadc = float(np.max(dadc_values))
    max_ppm = float(np.max(ppm_values))

    # Time in each severity zone (assuming 1 reading/second)
    elevated_count = np.sum((dadc_values >= 0.5) & (dadc_values < 1.5))
    high_count = np.sum((dadc_values >= 1.5) & (dadc_values < 2.5))
    critical_count = np.sum(dadc_values >= 2.5)

    # Max severity
    if critical_count > 0:
        max_severity = SeverityLevel.CRITICAL
    elif high_count > 0:
        max_severity = SeverityLevel.HIGH
    elif elevated_count > 0:
        max_severity = SeverityLevel.ELEVATED
    else:
        max_severity = SeverityLevel.NORMAL

    # Generate recommendations
    recommendations = []
    if high_count > 60:  # More than 1 minute high
        recommendations.append("Consider veterinary inspection")
    if critical_count > 0:
        recommendations.append("Review ventilation system")
    if len(events) > 5:
        recommendations.append("Multiple exposure events - investigate sources")
    if not recommendations:
        recommendations.append("Normal operations - continue standard monitoring")

    return DailyReport(
        date=date_str,
        total_events=len(events),
        max_severity_reached=max_severity,
        max_ppm=max_ppm,
        max_dadc=max_dadc,
        total_elevated_minutes=elevated_count / 60,
        total_high_minutes=high_count / 60,
        total_critical_minutes=critical_count / 60,
        alarms_generated=len(alarms),
        recommendations=recommendations,
    )


def format_report_markdown(report: DailyReport) -> str:
    """Format daily report as Markdown."""
    return f"""
# Daily VOC Monitoring Report
## {report.date}

### Summary
- **Total Events:** {report.total_events}
- **Max Severity:** {report.max_severity_reached.name}
- **Peak Concentration:** {report.max_ppm:.0f} ppm ({report.max_dadc:.2f} ΔADC)

### Exposure Times
| Severity | Duration |
|----------|----------|
| Elevated | {report.total_elevated_minutes:.1f} min |
| High | {report.total_high_minutes:.1f} min |
| Critical | {report.total_critical_minutes:.1f} min |

### Alarms Generated
Total: {report.alarms_generated}

### Recommendations
{"".join(f'- {r}' + chr(10) for r in report.recommendations)}
"""


# =============================================================================
# MODULE 5 SUMMARY
# =============================================================================

MODULE_5_SUMMARY = """
================================================================================
MODULE 5 SUMMARY: APPLICATION DESIGN - POULTRY BARN VOC MONITORING
================================================================================

COMPLETED COMPONENTS:
---------------------
✓ 5.1 VOC Severity Levels
    - 5 levels: NORMAL → ELEVATED → HIGH → CRITICAL → EMERGENCY
    - PPM and ΔADC thresholds
    - Color coding and descriptions
    - Action recommendations per level

✓ 5.2 Event Scoring System
    - Multi-factor scoring (intensity, duration, rate)
    - Cumulative exposure calculation (ppm-hours)
    - Risk categorization (Low → Critical)
    - Event history tracking

✓ 5.3 Alarm System
    - Priority-based alarms (INFO → CRITICAL)
    - Hysteresis for chatter prevention
    - Dead time suppression
    - Acknowledgment workflow
    - Alarm history logging

✓ 5.4 Real-Time Monitoring
    - Moving average smoothing
    - Trend detection (rising/falling/stable)
    - Event tracking and scoring
    - Automatic recommendations
    - BarnStatus updates

✓ 5.5 Barn Deployment Simulation
    - Configurable barn parameters
    - Realistic VOC simulation
    - Diurnal variations
    - Random event injection
    - Testing and training data

✓ 5.6 Dashboard Components
    - Status card HTML generation
    - Alarm table generation
    - Visual severity indicators

✓ 5.7 Reporting
    - Daily report generation
    - Exposure time summaries
    - Markdown formatting
    - Automated recommendations

POULTRY-SPECIFIC THRESHOLDS:
----------------------------
| Level     | PPM Range    | ΔADC Range  | Action                    |
|-----------|-------------|-------------|---------------------------|
| NORMAL    | 0-500       | 0-0.5       | Continue monitoring       |
| ELEVATED  | 500-2000    | 0.5-1.5     | Increase ventilation      |
| HIGH      | 2000-5000   | 1.5-2.5     | Veterinary consultation   |
| CRITICAL  | 5000-7000   | 2.5-3.0     | Immediate attention       |
| EMERGENCY | 7000+       | 3.0+        | Evacuate, call authorities|

NEXT STEPS → MODULE 6:
----------------------
- End-to-end pipeline integration
- Complete code organization
- Future work roadmap
- Deployment documentation
================================================================================
"""


if __name__ == "__main__":
    print(MODULE_5_SUMMARY)

    # Quick demo
    print("\n--- Simulation Demo ---")
    config = BarnConfig(
        barn_id="BARN-001",
        barn_name="North Poultry House",
        capacity=10000,
        area_sqm=500,
        num_sensors=4,
        sensor_positions=[(0, 0), (10, 0), (0, 10), (10, 10)],
        ventilation_rate=5000,
    )

    simulator = BarnSimulator(config)
    monitor = RealTimeMonitor()

    print("Simulating 60 seconds of data...")
    for _ in range(60):
        reading = simulator.simulate_reading()
        status = monitor.process_reading(reading)

    print(
        f"Final status: {status.current_severity.name} - {status.current_ppm:.0f} ppm"
    )
    print(f"Active alarms: {status.active_alarms}")
