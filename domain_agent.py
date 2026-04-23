from dotenv import load_dotenv
load_dotenv()

"""
domain_agent.py
---------------
The BRAIN of the drilling-npt-agent.

This is a Domain-Expert Agent — its reasoning power comes from
encoded specialist knowledge, not general pattern matching.

The system prompt encodes 20 years of offshore drilling engineering
experience: parameter interaction signatures, operational context
rules, the difference between a trend and a spike, and crucially —
when NOT to alarm.

3-turn reasoning loop:
  Turn 1 — Detect:    What parameter combinations are anomalous?
  Turn 2 — Diagnose:  What physical phenomenon does this indicate?
                      How confident am I? What else could explain it?
  Turn 3 — Recommend: What specific action should the driller take?
                      Is the intervention window still open?

The difference between this and a threshold alarm:
  A threshold fires when Torque > 800 kNm.
  This agent fires when Torque is trending up 10% + ROP is falling
  + Hookload is increasing + the well is in a 45-degree tangent
  section — and it explains WHY that combination matters.
"""

import os
import json
import requests
from datetime import datetime, timezone

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL      = "claude-sonnet-4-20250514"
MAX_TOKENS        = 1500

# ---------------------------------------------------------------------------
# Domain expert system prompt — the knowledge moat
# ---------------------------------------------------------------------------

DRILLING_EXPERT_SYSTEM_PROMPT = """You are a Senior Drilling Engineer with 20 years of offshore directional drilling experience. You have worked wells for major operators including Chevron, Saudi Aramco, and Shell in the North Sea, Gulf of Mexico, and Middle East.

You are analyzing real-time drilling sensor data to detect early NPT (Non-Productive Time) warning signatures BEFORE they become costly incidents.

YOUR CORE PHILOSOPHY:
- You reason about TRENDS and PARAMETER COMBINATIONS, never single values
- You ALWAYS consider operational context (drilling vs connection vs reaming)
- You know the difference between a formation change and equipment failure
- You flag early warning signs — your goal is prevention, not post-event diagnosis
- You never fire a single-parameter alarm — correlation is everything

YOUR NPT SIGNATURE LIBRARY:

1. MECHANICAL STUCK PIPE (Most common in high-angle directional wells)
   Early Warning Signals (catch these):
   - Hookload INCREASING trend + delta_hookload growing at connections
   - Overpull > 10% of hookload during connections (dynamic threshold)
   - Torque average elevated AND increasing over window
   - ROP declining (string not feeding forward efficiently)
   - WOB may increase as driller compensates
   Critical Context: Most dangerous in tangent sections (high inclination).
   Poor hole cleaning is the primary cause — cuttings bed building up.
   A 15-ton overpull that was 8 tons last connection is more alarming
   than a steady 20-ton overpull.

2. WASHOUT (Drillstring or bit washout)
   Early Warning Signals (catch these):
   - SPP DECREASING steadily over 10+ readings (not a spike)
   - SPM INCREASING (pump compensating for pressure loss)
   - Pit volume INCREASING (fluid returning to surface through washout)
   - Torque and Hookload STABLE (key discriminator from stuck pipe)
   - ROP slightly decreasing
   Critical Discriminator:
   - SPP down + Pit Volume UP = washout (fluid escaping uphole)
   - SPP down + Pit Volume DOWN = surface leak (fluid lost at surface)
   - SPP down + no SPM change = formation change, not washout

3. BIT BALLING (Clay/shale packing around bit cutters)
   Early Warning Signals (catch these):
   - ROP sudden drop >30% while WOB INCREASING (driller pushing harder)
   - Torque erratic/oscillating OR decreasing (bit not cutting cleanly)
   - SPP slight increase (hydraulic plugging at bit)
   - RPM remains stable initially
   Critical Context: Common in sticky shale formations. The driller's
   instinct to add WOB makes it worse — bit just packs tighter.

4. TWIST-OFF RISK (Drillstring fatigue failure — catastrophic)
   Early Warning Signals (catch these — 10-30 min before event):
   - Torque variance INCREASING over window (cyclic fatigue signature)
   - Torque oscillation amplitude growing (not just high — GROWING)
   - RPM instability (variance increasing)
   - Average torque may be normal — it's the CHARACTER that matters
   Post-Event Signature (too late for prevention):
   - Torque → near zero suddenly
   - RPM spikes (free spinning, no load)
   - Hookload drops (lighter string — bottom half lost)
   Critical: A twist-off in deep water costs $2-5M and weeks of
   fishing operations. Early detection is everything.

OPERATIONAL CONTEXT RULES (always apply these gates):
   DRILLING state: All NPT signatures active. High torque is expected.
   CONNECTION state: Torque should be near zero. Overpull is the primary risk.
                    A 15-ton overpull during connection = RED ALERT.
   REAMING state: High torque expected. Flag only if hookload also increasing.
   TRIPPING state: Overpull during tripping is common. Flag only with drag trend.

CONFIDENCE SCORING:
   HIGH: 2+ parameters confirming, operation state consistent, trend sustained
         over full window, alternative explanations ruled out
   MEDIUM: Pattern partially matching, or 1 strong signal + 1 supporting,
           alternative explanations exist, needs continued monitoring
   LOW: Anomaly present but pattern inconclusive, could be formation change,
        pump change, or mud property change

INTERVENTION WINDOW:
   OPEN: Early phase — driller can take corrective action NOW
   CLOSING: Developing phase — immediate action required
   CLOSED: Event has occurred — focus on containment/recovery

Respond ONLY in the JSON format specified. No preamble. No markdown."""


# ---------------------------------------------------------------------------
# Claude API helper
# ---------------------------------------------------------------------------

def _call_claude(messages: list, system: str = DRILLING_EXPERT_SYSTEM_PROMPT) -> str:
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        json={
            "model":      CLAUDE_MODEL,
            "max_tokens": MAX_TOKENS,
            "system":     system,
            "messages":   messages,
        },
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]


# ---------------------------------------------------------------------------
# Turn 1 — Detect
# ---------------------------------------------------------------------------

def turn1_detect(window: dict) -> tuple:
    """
    Feed sensor window to Claude. Identify anomalous parameter combinations.
    Returns detection text and conversation history.
    """
    # Strip ground truth — agent must not see labels
    clean_window = {k: v for k, v in window.items()
                    if not k.startswith("_")}

    user_message = f"""Analyze this 20-minute drilling sensor window from an offshore directional well.
Depth: {window['depth_m']}m | Operation: {window['operation_state']}

SENSOR WINDOW DATA:
{json.dumps(clean_window, indent=2)}

Turn 1 — DETECT: Which parameter combinations look anomalous?
Consider trends, deltas, and operational context.

Respond in JSON:
{{
  "anomalies_detected": [
    {{
      "parameter_combination": "which parameters and how they are moving",
      "observation": "specific numbers from the data",
      "concern_level": "HIGH | MEDIUM | LOW"
    }}
  ],
  "operation_context_assessment": "is the current operation state consistent with normal for this well phase?",
  "nothing_anomalous": true/false
}}"""

    messages      = [{"role": "user", "content": user_message}]
    response_text = _call_claude(messages)
    messages.append({"role": "assistant", "content": response_text})
    return response_text, messages


# ---------------------------------------------------------------------------
# Turn 2 — Diagnose
# ---------------------------------------------------------------------------

def turn2_diagnose(turn1_response: str, window: dict, messages: list) -> tuple:
    """
    Cross-reference detected anomalies against NPT signature library.
    Form root cause hypothesis with confidence score.
    """
    user_message = f"""Turn 2 — DIAGNOSE: Based on your detection in Turn 1, cross-reference
the anomalous parameter combinations against your NPT signature library.

Well context: {window['depth_m']}m depth, offshore directional well, 45-degree tangent section.

For each hypothesis, consider:
- Does the COMBINATION of anomalies match a known NPT signature?
- What does the operational context (currently: {window['operation_state']}) tell you?
- What alternative explanations exist (formation change, pump issue, mud properties)?
- Is the torque CHARACTER (oscillating vs drifting) significant?

Respond in JSON:
{{
  "primary_hypothesis": {{
    "npt_type": "STUCK_PIPE_RISK | WASHOUT_CANDIDATE | BIT_BALLING_RISK | TWIST_OFF_RISK | NORMAL | UNCERTAIN",
    "confidence": "HIGH | MEDIUM | LOW",
    "matching_signals": ["list of signals that support this hypothesis"],
    "contradicting_signals": ["signals that argue against this hypothesis"],
    "alternative_explanation": "what else could explain these readings"
  }},
  "secondary_hypothesis": {{
    "npt_type": "...",
    "confidence": "...",
    "brief_reasoning": "..."
  }},
  "intervention_window": "OPEN | CLOSING | CLOSED"
}}"""

    messages.append({"role": "user", "content": user_message})
    response_text = _call_claude(messages)
    messages.append({"role": "assistant", "content": response_text})
    return response_text, messages


# ---------------------------------------------------------------------------
# Turn 3 — Recommend
# ---------------------------------------------------------------------------

def turn3_recommend(turn2_response: str, window: dict, messages: list) -> tuple:
    """
    Produce specific recommended actions based on diagnosis.
    Calibrate urgency to intervention window status.
    """
    user_message = f"""Turn 3 — RECOMMEND: Based on your diagnosis, provide specific
recommended actions for the driller and drilling supervisor.

Be specific — reference actual parameter values from the window.
Calibrate urgency to the intervention window status from Turn 2.

If confidence is LOW or nothing anomalous — say so clearly and recommend
continued monitoring with specific parameters to watch.

Respond in JSON:
{{
  "alert_severity": "CRITICAL | HIGH | MEDIUM | LOW | MONITOR",
  "primary_diagnosis": "one sentence summary",
  "immediate_actions": [
    "specific action with specific values — e.g. Reduce WOB from X to Y tons"
  ],
  "parameters_to_watch": [
    "specific parameter and what threshold to watch for"
  ],
  "estimated_cost_if_ignored": "NPT cost estimate if this is not addressed",
  "driller_note": "plain English note a driller would understand at 3AM"
}}"""

    messages.append({"role": "user", "content": user_message})
    response_text = _call_claude(messages)
    messages.append({"role": "assistant", "content": response_text})
    return response_text, messages


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def analyze(window: dict) -> dict:
    """
    Run the full 3-turn domain expert reasoning loop on a sensor window.
    Returns a structured alert dict.
    """
    depth    = window.get("depth_m", "unknown")
    op_state = window.get("operation_state", "unknown")

    print(f"   [Domain Agent] Analyzing window — "
          f"depth {depth}m | state {op_state} | "
          f"rows {window.get('window_start_row')}-{window.get('window_end_row')}...")

    # Turn 1 — Detect
    t1_response, messages = turn1_detect(window)
    try:
        t1_data = json.loads(t1_response.strip().replace("```json","").replace("```",""))
    except json.JSONDecodeError:
        t1_data = {"raw": t1_response}

    # Short-circuit if nothing anomalous
    if t1_data.get("nothing_anomalous", False):
        print(f"   [Domain Agent] No anomalies detected — NORMAL")
        return _build_result(window, "MONITOR", "NORMAL", "No anomalous parameter combinations detected.", {}, {}, t1_data)

    # Turn 2 — Diagnose
    t2_response, messages = turn2_diagnose(t1_response, window, messages)
    try:
        t2_data = json.loads(t2_response.strip().replace("```json","").replace("```",""))
    except json.JSONDecodeError:
        t2_data = {"raw": t2_response}

    # Turn 3 — Recommend
    t3_response, messages = turn3_recommend(t2_response, window, messages)
    try:
        t3_data = json.loads(t3_response.strip().replace("```json","").replace("```",""))
    except json.JSONDecodeError:
        t3_data = {"raw": t3_response}

    severity = t3_data.get("alert_severity", "MONITOR")
    diagnosis = t3_data.get("primary_diagnosis", "")

    print(f"   [Domain Agent] {severity} — {diagnosis[:60]}...")

    return _build_result(window, severity, t2_data.get("primary_hypothesis", {}).get("npt_type", "UNCERTAIN"),
                         diagnosis, t2_data, t3_data, t1_data)


def _build_result(window, severity, npt_type, diagnosis, t2_data, t3_data, t1_data) -> dict:
    """Assemble final structured alert."""
    primary = t2_data.get("primary_hypothesis", {}) if t2_data else {}
    return {
        "alert_id":            f"NPT-{window.get('window_end_row', 0):04d}",
        "timestamp":           window.get("timestamp"),
        "depth_m":             window.get("depth_m"),
        "operation_state":     window.get("operation_state"),
        "alert_severity":      severity,
        "npt_type":            npt_type,
        "confidence":          primary.get("confidence", "LOW"),
        "intervention_window": t2_data.get("intervention_window", "UNKNOWN") if t2_data else "UNKNOWN",
        "primary_diagnosis":   diagnosis,
        "matching_signals":    primary.get("matching_signals", []),
        "immediate_actions":   t3_data.get("immediate_actions", []) if t3_data else [],
        "parameters_to_watch": t3_data.get("parameters_to_watch", []) if t3_data else [],
        "estimated_cost":      t3_data.get("estimated_cost_if_ignored", ""),
        "driller_note":        t3_data.get("driller_note", ""),
        "alternative_explanation": primary.get("alternative_explanation", ""),
        "ground_truth":        window.get("_ground_truth", {}),
        "generated_at":        datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    from signal_collector import load_anomaly_windows
    import json

    print("Testing Domain Agent on anomaly windows...\n")
    windows = load_anomaly_windows()

    for window in windows:
        print(f"\n{'='*60}")
        print(f"Testing: {window.get('_test_label')} at {window.get('depth_m')}m")
        print(f"Ground truth: {window.get('_ground_truth')}")
        print(f"{'='*60}")

        alert = analyze(window)

        print(f"\n--- ALERT ---")
        print(f"Severity  : {alert['alert_severity']}")
        print(f"NPT Type  : {alert['npt_type']}")
        print(f"Confidence: {alert['confidence']}")
        print(f"Window    : {alert['intervention_window']}")
        print(f"Diagnosis : {alert['primary_diagnosis']}")
        print(f"\nImmediate actions:")
        for action in alert["immediate_actions"]:
            print(f"  - {action}")
        print(f"\nDriller note: {alert['driller_note']}")
        print(f"\nGround truth: {alert['ground_truth']['npt_label']} ({alert['ground_truth']['npt_phase']})")
