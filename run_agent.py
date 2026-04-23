from dotenv import load_dotenv
load_dotenv()

"""
run_agent.py
------------
The single entrypoint for the drilling-npt-agent.

Orchestrates the full pipeline:
  Eyes   → signal_collector.py   (read sensor stream, package windows)
  Brain  → domain_agent.py       (domain-expert 3-turn reasoning)
  Hands  → alert_writer.py       (persist alerts to file + Delta)

Usage:
  python run_agent.py                    # full run — all windows
  python run_agent.py --anomaly-only     # focused run — 4 NPT windows only
  python run_agent.py --dry-run          # Eyes only — no agent calls
  python run_agent.py --well "WELL-001"  # tag alerts with well name
"""

import argparse
import json
import sys
from datetime import datetime, timezone

from signal_collector import stream_windows, load_anomaly_windows
from domain_agent import analyze
from alert_writer import write_alerts

WELL_NAME_DEFAULT = "SYNTHETIC-OFFSHORE-001"


def print_banner(well_name: str, mode: str):
    print()
    print("=" * 60)
    print("  DRILLING NPT AGENT")
    print(f"  Well     : {well_name}")
    print(f"  Mode     : {mode}")
    print(f"  Started  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    print()


def print_alert_summary(alert: dict):
    """Print a single alert in a clean driller-friendly format."""
    severity  = alert.get("alert_severity", "MONITOR")
    npt_type  = alert.get("npt_type", "UNKNOWN")
    depth     = alert.get("depth_m", 0)
    confidence = alert.get("confidence", "")
    window    = alert.get("intervention_window", "")

    severity_emoji = {
        "CRITICAL": "🔴",
        "HIGH":     "🟠",
        "MEDIUM":   "🟡",
        "LOW":      "🟢",
        "MONITOR":  "⚪",
    }.get(severity, "⚪")

    print(f"\n  {severity_emoji} [{severity}] {npt_type} at {depth}m "
          f"[{confidence} confidence | Window: {window}]")
    print(f"     {alert.get('primary_diagnosis', '')[:80]}")


def print_final_summary(alerts: list, duration_sec: float, run_id: str):
    """Print end-of-run summary."""
    critical = sum(1 for a in alerts if a.get("alert_severity") == "CRITICAL")
    high     = sum(1 for a in alerts if a.get("alert_severity") == "HIGH")
    medium   = sum(1 for a in alerts if a.get("alert_severity") == "MEDIUM")
    normal   = sum(1 for a in alerts if a.get("alert_severity") == "MONITOR")

    npt_types = set(a.get("npt_type") for a in alerts
                    if a.get("npt_type") not in ("NORMAL", "UNCERTAIN", None))

    print()
    print("=" * 60)
    print(f"  RUN COMPLETE")
    print(f"  Run ID      : {run_id}")
    print(f"  Duration    : {duration_sec}s")
    print(f"  Windows     : {len(alerts)} analyzed")
    print(f"  🔴 CRITICAL : {critical}")
    print(f"  🟠 HIGH     : {high}")
    print(f"  🟡 MEDIUM   : {medium}")
    print(f"  ⚪ MONITOR  : {normal}")
    if npt_types:
        print(f"  NPT types   : {', '.join(sorted(npt_types))}")
    print("=" * 60)


def run_full(well_name: str, dry_run: bool = False) -> list:
    """
    Full production run — streams all windows from sensor CSV.
    Only calls domain agent on windows with detected anomalies.
    """
    print(f"  Streaming all windows from sensor data...")
    alerts     = []
    window_count = 0

    for window in stream_windows():
        window_count += 1

        if dry_run:
            # Eyes only — print window metadata, skip agent
            depth = window.get("depth_m")
            state = window.get("operation_state")
            gt    = window.get("_ground_truth", {})
            label = gt.get("npt_label", "NORMAL")
            if label != "NORMAL":
                print(f"  [DRY RUN] Window {window_count} — "
                      f"{depth}m | {state} | GT: {label}")
            continue

        alert = analyze(window)
        alerts.append(alert)

        # Print non-monitor alerts immediately
        if alert.get("alert_severity") != "MONITOR":
            print_alert_summary(alert)

    if dry_run:
        print(f"\n  [DRY RUN] {window_count} windows scanned. No agent calls made.")
        return []

    return alerts


def run_anomaly_focused(well_name: str) -> list:
    """
    Focused run — only the 4 NPT anomaly windows.
    Faster for testing and portfolio demos.
    """
    print(f"  Loading 4 NPT anomaly windows (focused mode)...")
    windows = load_anomaly_windows()
    alerts  = []

    for window in windows:
        alert = analyze(window)
        alerts.append(alert)
        print_alert_summary(alert)

    return alerts


def main():
    parser = argparse.ArgumentParser(
        description="Drilling NPT Agent — domain-expert AI monitoring for offshore drilling"
    )
    parser.add_argument(
        "--anomaly-only", "-a",
        action="store_true",
        help="Run focused mode — analyze 4 NPT anomaly windows only (faster)"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Eyes only — stream windows without calling domain agent"
    )
    parser.add_argument(
        "--well", "-w",
        type=str,
        default=WELL_NAME_DEFAULT,
        help=f"Well name tag for alerts (default: {WELL_NAME_DEFAULT})"
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing alerts to file and Delta table"
    )
    args = parser.parse_args()

    mode = ("DRY RUN" if args.dry_run
            else "ANOMALY FOCUSED" if args.anomaly_only
            else "FULL STREAM")

    print_banner(args.well, mode)

    started_at = datetime.now(timezone.utc).isoformat()
    run_id     = f"npt-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    # Run
    if args.dry_run:
        run_full(args.well, dry_run=True)
        sys.exit(0)
    elif args.anomaly_only:
        alerts = run_anomaly_focused(args.well)
    else:
        alerts = run_full(args.well)

    completed_at = datetime.now(timezone.utc).isoformat()
    duration_sec = round(
        (datetime.fromisoformat(completed_at) -
         datetime.fromisoformat(started_at)).total_seconds(), 1
    )

    print_final_summary(alerts, duration_sec, run_id)

    # Write
    if not args.no_write and alerts:
        write_alerts(
            alerts,
            run_id=run_id,
            well_name=args.well,
            started_at=started_at,
            completed_at=completed_at,
            duration_sec=duration_sec,
        )

    # Exit code — CRITICAL alerts trigger non-zero exit
    # Useful for wiring into pipeline alerting systems
    critical_count = sum(1 for a in alerts if a.get("alert_severity") == "CRITICAL")
    if critical_count > 0:
        print(f"\n⚠️  {critical_count} CRITICAL alert(s) — exit code 2")
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
