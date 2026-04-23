# ⚡ Drilling NPT Agent

> *A domain-expert AI agent that monitors offshore directional drilling sensor streams, detects early NPT warning signatures — stuck pipe, washout, bit balling, twist-off — using cross-parameter reasoning and operational context, not hardcoded thresholds. Built with Claude API. Powered by 20 years of drilling engineering expertise.*

---

## The Problem That Costs the Industry Billions

NPT — Non-Productive Time — is the single biggest cost driver in offshore drilling operations. A stuck pipe event on a deepwater well costs $500K–$2M per day. A twist-off requiring fishing operations can run $5M and weeks of lost time. A washout that goes undetected destroys the drill bit and contaminates the wellbore.

The data to prevent most of these events exists. Weight on bit, torque, hookload, standpipe pressure, rate of penetration — streaming in real time from every sensor on the rig floor. The patterns that precede each NPT event are well understood by experienced drilling engineers.

The gap is not data. The gap is attention.

A driller monitoring a screen at 3AM in the Gulf of Mexico, 45 degrees into a tangent section at 4,000 meters, cannot simultaneously track nine parameters, compute their interactions, compare them to formation baselines, and distinguish a washout from a surface leak from a pump change — all in real time, every two minutes, for a 12-hour shift.

This agent does exactly that.

---

## Why This Is Different From a Threshold Alarm

Every rig has threshold alarms. Torque above 900 kNm — alarm fires. Standpipe pressure below 200 bar — alarm fires. These alarms fire constantly, correctly and incorrectly, until drillers learn to ignore them.

This agent does not alarm on thresholds. It reasons about **parameter interactions** and **trends** in the context of **operational state** — exactly the way an experienced drilling engineer thinks.

> A traditional threshold alarm is like a smoke detector — it trips when a single value crosses a line.
> This agent is like a senior drilling engineer on the morning tour — it looks at how nine parameters are moving relative to each other, considers whether the string is drilling or making a connection, and says *"that torque oscillation combined with that ROP drop at this depth in this formation means the bit is balling up — back off WOB now before you pack it tighter."*

The intelligence is not in the sensors. It is in the reasoning about what the sensors mean together.

---

## The Domain Knowledge Moat

The system prompt that drives this agent encodes 20 years of offshore directional drilling experience across the Gulf of Mexico and Middle East. It is not a generic anomaly detection prompt — it is a structured expert knowledge base covering:

**NPT signature library** — the exact parameter combination signatures for each failure mode, including which signals confirm a hypothesis and which signals rule out alternative explanations:
- SPP dropping + SPM increasing + pit volume increasing + torque stable = washout (fluid escaping uphole)
- SPP dropping + pit volume decreasing = surface leak (NOT washout — don't pull the string)
- Torque oscillation amplitude growing + RPM instability = twist-off fatigue building (10-30 minute warning window)
- Torque + 20% + hookload increasing + ROP declining = mechanical stuck pipe developing

**Operational context gates** — the same parameter reading means completely different things in different operational states:
- 15-ton overpull during a CONNECTION = red alert
- 15-ton overpull during TRIPPING = investigate trend, not necessarily alarming
- High torque during DRILLING = expected
- High torque during REAMING = warning of hole instability

**Dynamic thresholds** — not hardcoded values but physics-based calculations:
- Overpull threshold = 10% of current hookload (scales with string weight and depth)
- Torque baseline = rolling 5-minute average for current formation, not a fixed number

No data engineer without drilling domain experience could write this system prompt. The knowledge is the moat.

---

## Architecture: Eyes → Brain → Hands

```
Sensor Stream (synthetic WITSML)
         │
         ▼
┌────────────────────┐
│  signal_collector  │  👁️ Eyes
│                    │  Reads CSV, computes rolling deltas,
│  Rolling 10-row    │  classifies parameter trends,
│  windows           │  packages context for Brain
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   domain_agent     │  🧠 Brain
│                    │
│  Turn 1 — Detect   │  Which parameter combinations are anomalous?
│  Turn 2 — Diagnose │  What physical phenomenon does this indicate?
│  Turn 3 — Recommend│  What should the driller do right now?
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   alert_writer     │  🤝 Hands
│                    │  Local JSON file for immediate review
│  workspace.drilling│  Delta tables for audit trail + post-well analysis
│  .npt_alerts       │
└────────────────────┘
```

| Module | Role | Key design |
|---|---|---|
| `generate_sensor_data.py` | Synthetic WITSML generator | Trend-based anomaly injection, not threshold spikes |
| `signal_collector.py` | Eyes | Rolling windows, delta computation, trend classification |
| `domain_agent.py` | Brain | 3-turn expert reasoning loop with domain knowledge system prompt |
| `alert_writer.py` | Hands | File + `workspace.drilling` Delta tables |
| `run_agent.py` | Entrypoint | Full stream, anomaly-focused, dry-run modes |

---

## The 3-Turn Reasoning Loop

```
Turn 1 — Detect
  Feed 10-row rolling window (20 minutes of sensor data) to Claude
  Claude identifies anomalous parameter combinations
  Considers operational context — is this drilling, connection, or reaming?
  Never alarms on a single parameter
        ↓
Turn 2 — Diagnose
  Cross-reference detected combinations against NPT signature library
  Form root cause hypothesis with confidence score
  Consider and explicitly state alternative explanations
  HIGH confidence: 2+ parameters confirming, context consistent, trend sustained
  MEDIUM: partial match, alternatives exist, monitor closely
  LOW: anomaly present but inconclusive — needs more data
        ↓
Turn 3 — Recommend
  Produce specific actions with specific parameter values
  Calibrate urgency to intervention window status
  Write a plain-English driller note for 3AM comprehension
  Flag whether the intervention window is OPEN, CLOSING, or CLOSED
```

---

## The Four NPT Signatures

### Mechanical Stuck Pipe
The most common NPT event in high-angle directional wells. Poor hole cleaning builds a cuttings bed that traps the drillstring.

Early warning (catch this):
- Delta hookload trending up at connections — overpull exceeding 10% of string weight
- Torque average elevated and climbing across window
- ROP declining — string cannot feed forward efficiently
- WOB increasing as driller compensates

### Washout
A hole in the drillstring or bit allows drilling fluid to bypass the bit, destroying hydraulic energy and bit cutting action.

Early warning (catch this):
- SPP steady decrease over 10+ readings — gradual, not sudden
- SPM increasing as pump compensates for lost pressure
- Pit volume increasing — fluid returning uphole through washout path
- Torque and hookload STABLE — the key discriminator from stuck pipe

Critical discriminator: SPP drop + pit volume UP = washout. SPP drop + pit volume DOWN = surface leak. Same SPP signal, completely different response required.

### Bit Balling
Sticky clay or shale packs around the bit cutters, preventing them from cutting formation. The driller's instinct — add more weight — makes it worse.

Early warning (catch this):
- ROP sudden drop >30% while WOB increasing
- Torque erratic or decreasing — bit spinning but not cutting
- SPP slight increase — hydraulic plugging at bit nozzles

### Twist-Off Risk
Cyclic fatigue failure of the drillstring — catastrophic and expensive. The early warning window is 10-30 minutes before the string fails.

Early warning — catch THIS phase:
- Torque oscillation amplitude INCREASING across window
- Torque variance growing (character, not magnitude)
- RPM instability building

Post-event (too late):
- Torque → near zero suddenly
- RPM spikes — top drive free-spinning
- Hookload drops — lighter string means bottom half is gone

---

## Sample Output

```
============================================================
  DRILLING NPT AGENT
  Well     : GULF-OF-MEXICO-001
  Mode     : ANOMALY FOCUSED
============================================================

  🟡 [MEDIUM] BIT_BALLING_RISK at 2352.9m [MEDIUM confidence | Window: OPEN]
     Possible bit balling developing - ROP dropped 18% while WOB
     increased 9%, with oscillating torque signature

  🟠 [HIGH] WASHOUT_CANDIDATE at 2952.9m [HIGH confidence | Window: OPEN]
     Early washout signature detected - SPP declining while driller
     unconsciously compensating with increased pump strokes

  🔴 [CRITICAL] TWIST_OFF_RISK at 3497.1m [HIGH confidence | Window: CLOSING]
     Extreme torque oscillation (401 kNm swing) with growing variance
     indicates drillstring fatigue failure risk — immediate action required

  🟡 [MEDIUM] TWIST_OFF_RISK at 3981.4m [MEDIUM confidence | Window: OPEN]
     Oscillating torque with wide variance suggests potential cyclic
     fatigue building — 10-30 minute prevention window open

============================================================
  RUN COMPLETE
  Windows     : 4 analyzed
  🔴 CRITICAL : 1
  🟠 HIGH     : 1
  🟡 MEDIUM   : 2
  NPT types   : BIT_BALLING_RISK, TWIST_OFF_RISK, WASHOUT_CANDIDATE
============================================================

⚠️  1 CRITICAL alert(s) — exit code 2
```

The exit code 2 on CRITICAL alerts enables integration with rig alerting systems — a non-zero exit can trigger a pager notification to the company man and drilling supervisor without any additional tooling.

---

## Synthetic Well Data

The sensor data simulates an offshore directional well with a 45-degree tangent section — the highest NPT risk zone in directional drilling.

| Parameter | Description |
|---|---|
| `wob_klbs` | Weight on Bit (1,000 lbs) |
| `rpm` | Rotary speed |
| `torque_ftlbs` | Rotation resistance (ft-lbs) |
| `torque_variance` | Rolling variance — twist-off fatigue indicator |
| `hookload_tons` | Surface string weight |
| `delta_hookload_tons` | Computed overpull vs expected |
| `overpull_threshold` | Dynamic — 10% of hookload (scales with depth) |
| `spp_bar` | Standpipe pressure |
| `rop_ft_hr` | Rate of penetration (ft/hr) |
| `ecd_sg` | Equivalent circulating density |
| `flow_rate_gal_min` | Drilling fluid flow rate (gal/min) |
| `spm` | Pump strokes per minute |
| `pit_volume_m3` | Mud pit volume — washout discriminator |
| `depth_ft` | Measured depth (feet) |
| `operation_state` | DRILLING / CONNECTION / REAMING / TRIPPING |

Four NPT anomaly windows are injected as **trends**, not threshold spikes — because real NPT events develop gradually, and the agent's value is catching them in the early phase, not after they have fully developed.

---

## Setup

### Prerequisites
- Python 3.8+
- Databricks workspace (for alert persistence)
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))

### Installation

```bash
git clone https://github.com/venkatchittoor/drilling-npt-agent.git
cd drilling-npt-agent
python3 -m venv venv
source venv/bin/activate
pip install databricks-sdk requests anthropic python-dotenv pandas numpy
```

### Configuration

```
DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
DATABRICKS_TOKEN=your-personal-access-token
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your-warehouse-id
ANTHROPIC_API_KEY=your-anthropic-api-key
```

### Generate sensor data

```bash
python generate_sensor_data.py
```

---

## Usage

```bash
# Focused mode — 4 NPT anomaly windows (best for demos)
python run_agent.py --anomaly-only --well "GULF-OF-MEXICO-001"

# Full stream — all 700 sensor rows
python run_agent.py --well "GULF-OF-MEXICO-001"

# Dry run — Eyes only, no agent calls
python run_agent.py --dry-run

# Skip Delta write (local file only)
python run_agent.py --anomaly-only --no-write
```

### Exit codes
| Code | Meaning |
|---|---|
| `0` | No CRITICAL alerts |
| `2` | One or more CRITICAL alerts — escalate to drilling supervisor |

---

## Post-Well Analysis Queries

```sql
-- Full alert history for this well
SELECT timestamp, depth_m, alert_severity, npt_type,
       confidence, intervention_window, primary_diagnosis
FROM workspace.drilling.npt_alerts
ORDER BY depth_m;

-- Did the agent catch the right events?
SELECT npt_type, ground_truth_label, confidence, alert_severity,
       CASE WHEN npt_type = ground_truth_label THEN 'CORRECT'
            WHEN alert_severity != 'MONITOR' THEN 'CAUGHT_DIFFERENT'
            ELSE 'MISSED' END AS detection_result
FROM workspace.drilling.npt_alerts;

-- Which NPT type appeared at greatest depth (highest risk zone)?
SELECT npt_type, MAX(depth_m) as max_depth,
       COUNT(*) as alert_count
FROM workspace.drilling.npt_alerts
GROUP BY npt_type ORDER BY max_depth DESC;

-- Intervention window status across all alerts
SELECT intervention_window, COUNT(*) as count,
       AVG(depth_m) as avg_depth
FROM workspace.drilling.npt_alerts
GROUP BY intervention_window;
```

---

## Why Not a Traditional ML Model?

Standard approach: train a classification model on historical drilling data, deploy it to flag anomalies against learned patterns.

The problem: labeled historical NPT data is scarce, proprietary, and well-specific. A model trained on North Sea wells may not generalize to Gulf of Mexico geology. Retraining requires data science cycles and labeled datasets.

This agent uses a different approach: encode domain expertise directly into the reasoning system. The knowledge travels with the agent — no training data required, no retraining when moving to a new well or formation. Update the system prompt with formation-specific context and the agent adapts immediately.

The trade-off: an ML model with sufficient data will outperform expert-encoded reasoning on pattern recognition at scale. This agent is designed for wells where labeled training data is sparse or unavailable — which describes most new wells.

---

## Portfolio Context

This is the fifth and most domain-specialized repo in a Databricks + Claude API portfolio:

| Repo | Type | What it demonstrates |
|---|---|---|
| [`ecommerce-pipeline`](https://github.com/venkatchittoor/ecommerce-pipeline) | Pipeline engineering | Medallion Architecture, DLT, streaming |
| [`data-incident-agent`](https://github.com/venkatchittoor/data-incident-agent) | Monitoring agent | Observe, diagnose, report |
| [`pricing-decision-agent`](https://github.com/venkatchittoor/pricing-decision-agent) | Decisioning agent | Reason, decide, act with confidence gating |
| [`customer-behavior-crew`](https://github.com/venkatchittoor/customer-behavior-crew) | Multi-agent crew | Orchestrate specialists, synthesize answers |
| [`drilling-npt-agent`](https://github.com/venkatchittoor/drilling-npt-agent) | Domain-expert agent | 20 years of drilling knowledge as a reasoning engine |

The progression: Build pipelines → Monitor with AI → Decide with AI → Coordinate AI agents → Encode domain expertise as AI

The drilling NPT agent is the answer to the question every recruiter eventually asks: *"What makes you different from other data engineers building AI systems?"*

The answer is encoded in the system prompt.

---

## Roadmap

- [ ] **v2 — Real WITSML integration:** Connect to live WITSML 2.0 feed instead of synthetic CSV
- [ ] **v3 — Formation context:** Inject formation tops and lithology into the reasoning window — agent knows it is entering a shale section before the sensors show it
- [ ] **v4 — Multi-well crew:** Deploy the customer-behavior-crew pattern — one Orchestrator coordinating NPT agents across multiple simultaneous wells
- [ ] **v5 — Offset well memory:** Agent reads alert history from previous wells in the same field — *"this formation at this depth caused bit balling on the last three wells"*

---

*Built by Venkat Chittoor in collaboration with Claude (Anthropic) — combining 20 years of offshore drilling engineering with modern agentic AI to bring domain expertise and AI reasoning together in a way that is practical, explainable, and deployable. The future belongs to those who adapt and adopt.*
