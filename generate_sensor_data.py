"""
generate_sensor_data.py
-----------------------
Generates synthetic WITSML-like drilling sensor data for an offshore
directional well. Produces a time-series CSV that the signal_collector.py
(Eyes) will read and feed to the domain_agent.py (Brain).

Well profile: Offshore directional well
  - 0    to 1500m : Vertical section (conductor + surface casing)
  - 1500 to 2800m : Build section (0 to 45 degrees inclination)
  - 2800 to 4500m : Tangent section (45 degrees — highest NPT risk zone)

Parameters generated:
  depth, timestamp, operation_state,
  wob, rpm, torque, torque_variance,
  hookload, delta_hookload,
  spp, rop, ecd,
  flow_rate, spm, pit_volume

Four injected NPT anomaly windows (trend-based, not threshold spikes):
  1. Mechanical Stuck Pipe  — rows 180-230  (tangent section, connection)
  2. Washout               — rows 320-370  (tangent section, drilling)
  3. Bit Balling           — rows 450-490  (tangent section, drilling)
  4. Twist-off Risk        — rows 560-620  (tangent section, drilling)

Design philosophy:
  Anomalies are TRENDS, not spikes. Each dysfunction builds gradually
  over 20-30 rows before becoming obvious — because the agent should
  catch the early warning phase, not the crisis phase.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ---------------------------------------------------------------------------
# Seed for reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------------------
# Well profile constants
# ---------------------------------------------------------------------------
TOTAL_ROWS       = 700
START_DEPTH      = 1500.0   # Start in build section (m)
END_DEPTH        = 4500.0   # End of tangent section (m)
DEPTH_INCREMENT  = (END_DEPTH - START_DEPTH) / TOTAL_ROWS
START_TIME       = datetime(2026, 4, 1, 6, 0, 0)
TIME_INCREMENT   = timedelta(minutes=2)  # 2-minute survey intervals

# ---------------------------------------------------------------------------
# Baseline parameter ranges — offshore directional well, tangent section
# ---------------------------------------------------------------------------

# String weight increases with depth — dynamic hookload baseline
def baseline_hookload(depth):
    """Hookload increases with depth as string weight grows."""
    return 180 + (depth - 1500) * 0.04 + np.random.normal(0, 2)

def baseline_wob(depth):
    """WOB varies with formation hardness — harder at depth."""
    return 15 + np.random.normal(0, 1.5)

def baseline_rpm():
    return 120 + np.random.normal(0, 3)

def baseline_torque(wob, rpm):
    """Torque correlates with WOB and RPM."""
    return (wob * 18 + rpm * 0.8) + np.random.normal(0, 15)

def baseline_spp(depth):
    """SPP increases with depth due to hydrostatic head."""
    return 220 + (depth - 1500) * 0.01 + np.random.normal(0, 3)

def baseline_rop():
    return 18 + np.random.normal(0, 2)

def baseline_ecd(depth):
    """ECD increases slightly with depth."""
    return 1.45 + (depth - 1500) * 0.00005 + np.random.normal(0, 0.005)

def baseline_flow_rate():
    return 850 + np.random.normal(0, 10)

def baseline_spm():
    return 65 + np.random.normal(0, 1)

def baseline_pit_volume():
    return 320 + np.random.normal(0, 2)


# ---------------------------------------------------------------------------
# Operation state sequencer
# ---------------------------------------------------------------------------

def get_operation_state(row_idx):
    """
    Realistic state machine for offshore directional drilling.
    Connections every ~30 rows (60 min at 2-min intervals).
    Occasional reaming passes.
    """
    cycle = row_idx % 35
    if cycle in [0, 1, 2]:           # Connection every 35 rows
        return "CONNECTION"
    elif cycle in [3, 4]:            # Short reaming after connection
        return "REAMING"
    else:
        return "DRILLING"


# ---------------------------------------------------------------------------
# NPT anomaly injection functions
# ---------------------------------------------------------------------------

def inject_stuck_pipe(row, base_row, hookload_base, torque_base):
    """
    Mechanical Stuck Pipe — trend-based signature.
    Builds over 50 rows. Peak danger at row 30+.

    Signature:
    - Delta hookload trending up at connections (overpull)
    - Torque average elevated and climbing
    - WOB dropping (cannot feed string forward)
    - SPP slight increase (pack-off beginning)
    """
    progress = (row - base_row) / 50.0  # 0.0 to 1.0

    delta_hookload_extra = progress * 25   # Overpull building to 25 tons
    torque_factor        = 1 + progress * 0.35  # Torque up 35%
    wob_factor           = 1 - progress * 0.25  # WOB down 25%
    spp_factor           = 1 + progress * 0.08  # SPP up 8%

    return {
        "delta_hookload_extra": delta_hookload_extra,
        "torque_factor":        torque_factor,
        "wob_factor":           wob_factor,
        "spp_factor":           spp_factor,
        "rop_factor":           1 - progress * 0.30,
        "rpm_factor":           1.0,
        "flow_factor":          1.0,
        "spm_factor":           1.0,
        "pit_factor":           1.0,
        "npt_label":            "STUCK_PIPE_RISK",
        "npt_phase":            "EARLY" if progress < 0.4 else "DEVELOPING",
    }


def inject_washout(row, base_row):
    """
    Drillstring Washout — trend-based signature.
    Builds over 50 rows. Gradual, not sudden.

    Signature:
    - SPP steady decrease (fluid escaping through washout)
    - SPM increasing (pump compensating for pressure loss)
    - Pit volume increasing (fluid returning to surface)
    - Torque and hookload STABLE (distinguishes from stuck pipe)
    - ROP slightly decreasing (less hydraulic energy at bit)
    """
    progress = (row - base_row) / 50.0

    return {
        "delta_hookload_extra": 0,
        "torque_factor":        1.0,                     # Stable — key discriminator
        "wob_factor":           1.0,
        "spp_factor":           1 - progress * 0.12,     # SPP drops 12%
        "rop_factor":           1 - progress * 0.10,     # Slight ROP decrease
        "rpm_factor":           1.0,
        "flow_factor":          1 + progress * 0.08,     # Flow up slightly
        "spm_factor":           1 + progress * 0.10,     # SPM up (compensating)
        "pit_factor":           1 + progress * 0.04,     # Pit volume increasing
        "npt_label":            "WASHOUT_CANDIDATE",
        "npt_phase":            "EARLY" if progress < 0.4 else "DEVELOPING",
    }


def inject_bit_balling(row, base_row):
    """
    Bit Balling — trend-based signature.
    Rapid onset over 40 rows.

    Signature:
    - ROP sudden drop >30% (clay packing around bit)
    - WOB increasing (driller pushing harder)
    - Torque erratic/decreasing (bit not cutting, just spinning)
    - SPP slight increase (plugging)
    - RPM stable initially
    """
    progress = (row - base_row) / 40.0

    # Torque oscillation — erratic, not steady
    torque_oscillation = 1 + np.sin(progress * 12) * 0.15 * progress

    return {
        "delta_hookload_extra": 0,
        "torque_factor":        torque_oscillation * (1 - progress * 0.20),
        "wob_factor":           1 + progress * 0.35,     # WOB up 35%
        "spp_factor":           1 + progress * 0.06,     # SPP up 6%
        "rop_factor":           1 - progress * 0.45,     # ROP drops 45%
        "rpm_factor":           1.0,
        "flow_factor":          1.0,
        "spm_factor":           1.0,
        "pit_factor":           1.0,
        "npt_label":            "BIT_BALLING_RISK",
        "npt_phase":            "EARLY" if progress < 0.35 else "DEVELOPING",
    }


def inject_twist_off_risk(row, base_row):
    """
    Twist-off Risk — cyclic fatigue buildup (10-30 min before event).
    This is the EARLY WARNING phase — agent should catch this
    BEFORE the string fails.

    Signature:
    - Torque oscillation amplitude INCREASING (cyclic fatigue)
    - Torque variance increasing over time
    - RPM instability building
    - Average torque stays roughly same (distinguishes from stuck pipe)
    """
    progress = (row - base_row) / 60.0

    # Oscillation amplitude increases with progress
    amplitude    = 0.05 + progress * 0.35
    oscillation  = np.sin(progress * 20) * amplitude
    rpm_wobble   = 1 + np.sin(progress * 15) * progress * 0.12

    # Late stage — actual twist-off
    if progress > 0.85:
        return {
            "delta_hookload_extra": -30,        # Hookload drops (lighter string)
            "torque_factor":        0.05,        # Torque near zero
            "wob_factor":           0,
            "spp_factor":           1.0,
            "rop_factor":           3.0,         # ROP spikes (freefalling)
            "rpm_factor":           2.5,         # RPM spikes (free spinning)
            "flow_factor":          1.0,
            "spm_factor":           1.0,
            "pit_factor":           1.0,
            "npt_label":            "TWIST_OFF_OCCURRED",
            "npt_phase":            "CRITICAL",
        }

    return {
        "delta_hookload_extra": 0,
        "torque_factor":        1 + oscillation,
        "wob_factor":           1.0,
        "spp_factor":           1.0,
        "rop_factor":           1.0,
        "rpm_factor":           rpm_wobble,
        "flow_factor":          1.0,
        "spm_factor":           1.0,
        "pit_factor":           1.0,
        "npt_label":            "TWIST_OFF_RISK",
        "npt_phase":            "EARLY" if progress < 0.5 else "DEVELOPING",
    }


# ---------------------------------------------------------------------------
# Main data generation
# ---------------------------------------------------------------------------

def generate(output_path: str = "data/sensor_data.csv"):
    """Generate synthetic sensor data and write to CSV."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []

    # Anomaly windows — (start_row, end_row, type)
    ANOMALY_WINDOWS = {
        range(180, 230): "stuck_pipe",
        range(320, 370): "washout",
        range(450, 490): "bit_balling",
        range(560, 620): "twist_off",
    }

    print("Generating synthetic drilling sensor data...")
    print(f"  Well type  : Offshore directional (45° tangent section)")
    print(f"  Depth range: {START_DEPTH}m – {END_DEPTH}m")
    print(f"  Total rows : {TOTAL_ROWS}")
    print(f"  NPT windows: Stuck pipe (180-230), Washout (320-370), "
          f"Bit balling (450-490), Twist-off (560-620)\n")

    prev_torques = []  # Rolling window for variance calculation

    for i in range(TOTAL_ROWS):
        depth     = START_DEPTH + i * DEPTH_INCREMENT
        timestamp = START_TIME + i * TIME_INCREMENT
        op_state  = get_operation_state(i)

        # Baseline values
        hkld_base  = baseline_hookload(depth)
        wob_base   = baseline_wob(depth)
        rpm_base   = baseline_rpm()
        torq_base  = baseline_torque(wob_base, rpm_base)
        spp_base   = baseline_spp(depth)
        rop_base   = baseline_rop()
        ecd_base   = baseline_ecd(depth)
        flow_base  = baseline_flow_rate()
        spm_base   = baseline_spm()
        pit_base   = baseline_pit_volume()

        # During connection — adjust baseline
        if op_state == "CONNECTION":
            wob_base  = 0
            rop_base  = 0
            rpm_base  = 0
            torq_base = np.random.normal(5, 2)   # Near zero on connection

        # Check for anomaly injection
        anomaly_type = None
        for window, atype in ANOMALY_WINDOWS.items():
            if i in window:
                anomaly_type = atype
                base_row     = window.start
                break

        # Apply anomaly modifiers
        mod = {
            "delta_hookload_extra": 0,
            "torque_factor":        1.0,
            "wob_factor":           1.0,
            "spp_factor":           1.0,
            "rop_factor":           1.0,
            "rpm_factor":           1.0,
            "flow_factor":          1.0,
            "spm_factor":           1.0,
            "pit_factor":           1.0,
            "npt_label":            "NORMAL",
            "npt_phase":            "NORMAL",
        }

        if anomaly_type == "stuck_pipe":
            mod = inject_stuck_pipe(i, base_row, hkld_base, torq_base)
        elif anomaly_type == "washout":
            mod = inject_washout(i, base_row)
        elif anomaly_type == "bit_balling":
            mod = inject_bit_balling(i, base_row)
        elif anomaly_type == "twist_off":
            mod = inject_twist_off_risk(i, base_row)

        # Compute final values
        wob       = max(0, wob_base   * mod["wob_factor"])
        rpm       = max(0, rpm_base   * mod["rpm_factor"])
        torque    = max(0, torq_base  * mod["torque_factor"])
        spp       = max(0, spp_base   * mod["spp_factor"])
        rop       = max(0, rop_base   * mod["rop_factor"])
        flow_rate = max(0, flow_base  * mod["flow_factor"])
        spm       = max(0, spm_base   * mod["spm_factor"])
        pit_vol   = max(0, pit_base   * mod["pit_factor"])
        hookload  = max(0, hkld_base  + mod["delta_hookload_extra"])

        # Dynamic overpull threshold — 10% of hookload
        expected_hookload  = hkld_base
        delta_hookload     = hookload - expected_hookload
        overpull_threshold = expected_hookload * 0.10

        # Torque variance — rolling 10-row window
        prev_torques.append(torque)
        if len(prev_torques) > 10:
            prev_torques.pop(0)
        torque_variance = round(np.var(prev_torques), 2)

        records.append({
            "row_idx":             i,
            "timestamp":           timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "depth_m":             round(depth, 1),
            "operation_state":     op_state,
            "wob_tons":            round(wob, 2),
            "rpm":                 round(rpm, 1),
            "torque_knm":          round(torque, 2),
            "torque_variance":     torque_variance,
            "hookload_tons":       round(hookload, 2),
            "expected_hookload":   round(expected_hookload, 2),
            "delta_hookload_tons": round(delta_hookload, 2),
            "overpull_threshold":  round(overpull_threshold, 2),
            "spp_bar":             round(spp, 2),
            "rop_m_hr":            round(rop, 2),
            "ecd_sg":              round(ecd_base, 4),
            "flow_rate_lpm":       round(flow_rate, 1),
            "spm":                 round(spm, 1),
            "pit_volume_m3":       round(pit_vol, 2),
            "npt_label":           mod["npt_label"],
            "npt_phase":           mod["npt_phase"],
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    # Summary
    normal_rows   = len(df[df["npt_label"] == "NORMAL"])
    anomaly_rows  = len(df[df["npt_label"] != "NORMAL"])
    print(f"✅ Generated {len(df)} rows → {output_path}")
    print(f"   Normal rows  : {normal_rows}")
    print(f"   Anomaly rows : {anomaly_rows}")
    print(f"\n   NPT label distribution:")
    for label, count in df["npt_label"].value_counts().items():
        print(f"   {label:<25} : {count} rows")


if __name__ == "__main__":
    generate()
