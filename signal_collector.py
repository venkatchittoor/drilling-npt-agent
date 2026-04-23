from dotenv import load_dotenv
load_dotenv()

"""
signal_collector.py
-------------------
The EYES of the drilling-npt-agent.

Reads the synthetic WITSML-like sensor CSV and prepares
rolling windows of sensor readings for the domain agent.

Key responsibilities:
  1. Read sensor stream row by row
  2. Compute rolling deltas — rate of change per parameter
  3. Detect operation state transitions
  4. Package a 10-row rolling window for the Brain

Why rolling windows?
  A single sensor reading tells you nothing. A 10-row window
  (20 minutes of drilling data) reveals TRENDS — which is what
  the domain agent reasons about. Torque at 800 kNm means nothing.
  Torque increasing 8% per reading for 10 consecutive readings
  means stuck pipe risk.

The Eyes never interpret. They observe, compute, and package.
Interpretation is the domain agent's job.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SENSOR_DATA_PATH = "data/sensor_data.csv"
WINDOW_SIZE      = 10   # Rolling window — 20 minutes at 2-min intervals
STRIDE           = 5    # Step between windows — overlap for continuity


# ---------------------------------------------------------------------------
# Delta computation — rate of change per parameter
# ---------------------------------------------------------------------------

def compute_deltas(window_df: pd.DataFrame) -> dict:
    """
    Compute rate of change for key parameters across the window.
    Returns delta as percentage change from first to last reading.
    """
    def pct_change(col):
        first = window_df[col].iloc[0]
        last  = window_df[col].iloc[-1]
        if first == 0 or pd.isna(first):
            return 0.0
        return round((last - first) / abs(first) * 100, 2)

    def trend_direction(col):
        """Is the parameter consistently increasing, decreasing, or oscillating?"""
        values = window_df[col].values
        diffs  = np.diff(values)
        pos    = np.sum(diffs > 0)
        neg    = np.sum(diffs < 0)
        if pos >= len(diffs) * 0.7:
            return "INCREASING"
        elif neg >= len(diffs) * 0.7:
            return "DECREASING"
        else:
            return "OSCILLATING"

    return {
        "delta_torque_pct":       pct_change("torque_knm"),
        "delta_wob_pct":          pct_change("wob_tons"),
        "delta_spp_pct":          pct_change("spp_bar"),
        "delta_rop_pct":          pct_change("rop_m_hr"),
        "delta_hookload_pct":     pct_change("hookload_tons"),
        "delta_rpm_pct":          pct_change("rpm"),
        "delta_spm_pct":          pct_change("spm"),
        "delta_pit_volume_pct":   pct_change("pit_volume_m3"),
        "torque_trend":           trend_direction("torque_knm"),
        "hookload_trend":         trend_direction("hookload_tons"),
        "spp_trend":              trend_direction("spp_bar"),
        "rop_trend":              trend_direction("rop_m_hr"),
        "rpm_trend":              trend_direction("rpm"),
    }


# ---------------------------------------------------------------------------
# Window packager
# ---------------------------------------------------------------------------

def package_window(window_df: pd.DataFrame) -> dict:
    """
    Package a rolling window of sensor readings into a structured
    dict ready for the domain agent.

    Includes:
    - Current readings (last row)
    - Window statistics (min, max, mean, variance)
    - Delta trends (rate of change)
    - Operation state context
    - Overpull assessment at connections
    """
    current      = window_df.iloc[-1]
    op_states    = window_df["operation_state"].tolist()
    current_state = current["operation_state"]

    # Overpull assessment — only meaningful at CONNECTION
    overpull_tons      = current["delta_hookload_tons"]
    overpull_threshold = current["overpull_threshold"]
    overpull_flag      = (
        current_state == "CONNECTION" and
        overpull_tons > overpull_threshold
    )

    # Torque character — variance tells us oscillation vs drift
    torque_variance_mean = window_df["torque_variance"].mean()
    torque_variance_trend = (
        "INCREASING" if window_df["torque_variance"].iloc[-1] >
                        window_df["torque_variance"].iloc[0] * 1.2
        else "STABLE"
    )

    deltas = compute_deltas(window_df)

    return {
        "window_start_row":    int(window_df.iloc[0]["row_idx"]),
        "window_end_row":      int(current["row_idx"]),
        "timestamp":           current["timestamp"],
        "depth_m":             round(current["depth_m"], 1),
        "operation_state":     current_state,
        "state_transitions":   list(dict.fromkeys(op_states)),  # Unique ordered states

        # Current readings
        "current": {
            "wob_tons":            current["wob_tons"],
            "rpm":                 current["rpm"],
            "torque_knm":          current["torque_knm"],
            "hookload_tons":       current["hookload_tons"],
            "expected_hookload":   current["expected_hookload"],
            "delta_hookload_tons": round(overpull_tons, 2),
            "overpull_threshold":  round(overpull_threshold, 2),
            "spp_bar":             current["spp_bar"],
            "rop_m_hr":            current["rop_m_hr"],
            "ecd_sg":              current["ecd_sg"],
            "flow_rate_lpm":       current["flow_rate_lpm"],
            "spm":                 current["spm"],
            "pit_volume_m3":       current["pit_volume_m3"],
        },

        # Window statistics
        "window_stats": {
            "torque_mean":         round(window_df["torque_knm"].mean(), 2),
            "torque_min":          round(window_df["torque_knm"].min(), 2),
            "torque_max":          round(window_df["torque_knm"].max(), 2),
            "torque_variance_mean":round(torque_variance_mean, 2),
            "torque_variance_trend": torque_variance_trend,
            "spp_mean":            round(window_df["spp_bar"].mean(), 2),
            "spp_min":             round(window_df["spp_bar"].min(), 2),
            "rop_mean":            round(window_df["rop_m_hr"].mean(), 2),
            "hookload_mean":       round(window_df["hookload_tons"].mean(), 2),
            "pit_volume_mean":     round(window_df["pit_volume_m3"].mean(), 2),
            "pit_volume_max":      round(window_df["pit_volume_m3"].max(), 2),
        },

        # Trend deltas
        "deltas": deltas,

        # Flags
        "flags": {
            "overpull_flag":       overpull_flag,
            "overpull_tons":       round(overpull_tons, 2),
            "overpull_threshold":  round(overpull_threshold, 2),
            "connection_in_window": "CONNECTION" in op_states,
        },

        # Ground truth label (NOT shown to agent — for validation only)
        "_ground_truth": {
            "npt_label": current["npt_label"],
            "npt_phase": current["npt_phase"],
        }
    }


# ---------------------------------------------------------------------------
# Stream generator — yields windows from the CSV
# ---------------------------------------------------------------------------

def stream_windows(
    path: str = SENSOR_DATA_PATH,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    start_row: int = 0,
    end_row: int = None,
):
    """
    Generator that yields rolling windows from the sensor CSV.
    Each yield is a packaged window dict ready for the domain agent.

    Usage:
        for window in stream_windows():
            alert = domain_agent.analyze(window)
    """
    df = pd.read_csv(path)

    if end_row is None:
        end_row = len(df)

    total_windows = 0
    for start in range(start_row, end_row - window_size, stride):
        end        = start + window_size
        window_df  = df.iloc[start:end].copy()
        total_windows += 1
        yield package_window(window_df)

    print(f"   [Signal Collector] Streamed {total_windows} windows "
          f"(window={window_size}, stride={stride})")


# ---------------------------------------------------------------------------
# Batch loader — loads all windows at once (for testing)
# ---------------------------------------------------------------------------

def load_all_windows(
    path: str = SENSOR_DATA_PATH,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> list:
    """Load all windows into a list — useful for testing and inspection."""
    return list(stream_windows(path, window_size, stride))


# ---------------------------------------------------------------------------
# Anomaly window filter — returns only windows containing NPT events
# For focused agent testing without running all 700 rows
# ---------------------------------------------------------------------------

def load_anomaly_windows(path: str = SENSOR_DATA_PATH) -> list:
    """
    Return one representative window per NPT event type.
    Useful for focused testing of the domain agent's reasoning.
    Selects windows from the EARLY phase — agent should catch these.
    """
    anomaly_centers = {
        "STUCK_PIPE_RISK":   195,   # Mid early-warning phase
        "WASHOUT_CANDIDATE": 335,   # Mid early-warning phase
        "BIT_BALLING_RISK":  462,   # Mid early-warning phase
        "TWIST_OFF_RISK":    575,   # Mid early-warning phase
    }

    df      = pd.read_csv(path)
    windows = []

    for label, center_row in anomaly_centers.items():
        start     = max(0, center_row - WINDOW_SIZE // 2)
        end       = start + WINDOW_SIZE
        window_df = df.iloc[start:end].copy()
        window    = package_window(window_df)
        window["_test_label"] = label
        windows.append(window)
        print(f"   [Signal Collector] Loaded {label} window "
              f"(rows {start}-{end}, depth {window['depth_m']}m)")

    return windows


if __name__ == "__main__":
    import json
    print("Testing Signal Collector — loading anomaly windows...\n")
    windows = load_anomaly_windows()
    print(f"\n✅ Loaded {len(windows)} anomaly windows")
    print("\n--- SAMPLE WINDOW (STUCK PIPE) ---")
    stuck_window = next(w for w in windows if "STUCK" in w["_test_label"])
    print(json.dumps(stuck_window, indent=2, default=str))
