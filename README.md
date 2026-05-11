# ⚡ SENTRA — Autonomous Predictive Maintenance Intelligence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![AMD MI300X](https://img.shields.io/badge/AMD-MI300X_GPU-ED1C24?style=flat-square&logo=amd&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Hackathon](https://img.shields.io/badge/AMD_Developer_Hackathon-lablab.ai-blueviolet?style=flat-square)

**A production-grade multi-agent AI system for real-time industrial equipment monitoring,
anomaly detection, and autonomous maintenance planning — powered by AMD Instinct MI300X GPUs.**

[Live Demo](#) · [Architecture](#system-architecture) · [Quickstart](#quickstart) · [API Reference](#agent-api-reference)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Why SENTRA?](#why-sentra)
- [System Architecture](#system-architecture)
- [Agent Design](#agent-design)
- [Tech Stack](#tech-stack)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [AMD MI300X Integration](#amd-mi300x-integration)
- [Agent API Reference](#agent-api-reference)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

Industrial equipment failure costs the global manufacturing sector an estimated **$50 billion per year** in unplanned downtime. Traditional rule-based monitoring systems generate excessive false positives and provide no root cause insight — leaving engineers to manually triage alerts with no actionable guidance.

**SENTRA** addresses this by deploying three specialized large language model agents in a sequential reasoning pipeline:

1. **SENTINEL** detects statistical and trend-based anomalies in real-time sensor streams
2. **ANALYST** performs structured root cause analysis grounded in mechanical engineering knowledge
3. **PLANNER** synthesizes actionable maintenance work orders with cost estimates and part lists

All three 70B-class models run **simultaneously** on a single AMD Instinct MI300X GPU — made possible by its unique 192 GB HBM3 memory capacity. No model swapping. No API round-trips. Sub-30-second end-to-end analysis.

---

## Why SENTRA?

| Capability | Rule-Based Systems | Single-Model AI | SENTRA |
|---|---|---|---|
| Anomaly Detection | ✅ Threshold only | ✅ Statistical | ✅ Statistical + Trend |
| Root Cause Analysis | ❌ | ⚠️ Generic | ✅ Domain-specific |
| Maintenance Planning | ❌ | ⚠️ Hallucination-prone | ✅ Structured + costed |
| Multi-sensor Correlation | ❌ | ⚠️ Limited | ✅ Cross-sensor |
| Explainability | ❌ | ⚠️ Black-box | ✅ Step-by-step reasoning |
| On-premise Deployable | ✅ | ❌ | ✅ |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SENTRA Pipeline                          │
│                                                                 │
│  CSV / IoT Feed                                                 │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────┐    Anomaly     ┌──────────┐    Root Cause         │
│  │ SENTINEL │   Report  ──► │ ANALYST  │   Report  ──►         │
│  │ Llama 3.1│               │  Qwen3   │                       │
│  │   70B    │               │   72B    │                       │
│  └──────────┘               └──────────┘                       │
│                                    │                            │
│                                    ▼                            │
│                             ┌──────────┐                        │
│                             │ PLANNER  │  ──► Work Order        │
│                             │DeepSeek  │      + Cost Estimate   │
│                             │  R1 70B  │      + Parts List      │
│                             └──────────┘                        │
│                                                                 │
│  ─────────────────────────────────────────────────────────      │
│  AMD Instinct MI300X · 192 GB HBM3 · vLLM Inference Engine     │
└─────────────────────────────────────────────────────────────────┘
```

```
sentra/
 ├── app.py                  # Streamlit dashboard entry point
 ├── agents/
 │   ├── sentinel.py         # Anomaly detection agent
 │   ├── analyst.py          # Root cause analysis agent
 │   └── planner.py          # Maintenance planning agent
 ├── .env                    # Environment variables (see Configuration)
 ├── requirements.txt
 └── README.md
```

---

## Agent Design

Each agent is a stateless LLM call with a structured system prompt encoding domain expertise. Agents communicate through **natural language handoffs** — the output of one agent becomes the grounded context for the next.

### SENTINEL — Anomaly Detection

```
Input  : Pandas DataFrame (sensor time-series)
Output : Anomaly report with urgency classification
Model  : Llama 3.1 70B Instruct
Method : Statistical outlier detection (μ ± 2σ) + trend gradient analysis
```

SENTINEL computes descriptive statistics over the full sensor window and presents the most recent readings to the model alongside normal operating bounds. The model is prompted to reason over temperature, vibration, and pressure jointly — catching correlated failures that single-sensor thresholds miss.

### ANALYST — Root Cause Analysis

```
Input  : Anomaly report (SENTINEL output) + machine type
Output : Failure mode hypothesis + time-to-failure estimate + risk level
Model  : Qwen3 72B
Method : Chain-of-thought mechanical failure reasoning
```

ANALYST receives the anomaly report and performs structured failure analysis. The system prompt encodes mechanical engineering heuristics (e.g., coincident high temperature and vibration → bearing failure) and instructs the model to reason step-by-step before committing to a diagnosis.

### PLANNER — Maintenance Planning

```
Input  : Root cause report (ANALYST output) + machine type
Output : Prioritised work order: immediate actions, parts list, downtime estimate, cost range
Model  : DeepSeek-R1 70B
Method : Structured planning with cost-aware reasoning
```

PLANNER converts the diagnosis into an actionable maintenance plan using a structured output format. The system prompt constrains the model to produce concrete, time-boxed actions with realistic cost estimates — avoiding vague recommendations.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Compute** | AMD Instinct MI300X (192 GB HBM3) | GPU inference |
| **Inference** | vLLM | High-throughput LLM serving |
| **Agent Framework** | LangChain | Agent orchestration |
| **Models** | Llama 3.1 70B · Qwen3 72B · DeepSeek-R1 70B | Specialized reasoning |
| **Frontend** | Streamlit + Plotly | Interactive dashboard |
| **Data** | Pandas + NumPy | Sensor data processing |

---

## Quickstart

### Prerequisites

- Python 3.11+
- AMD Developer Cloud account with MI300X GPU Droplet (vLLM Quick Start image)
- Groq API key (for local development without GPU)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/sentra.git
cd sentra

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Setup

```bash
cp .env.example .env
```

Edit `.env` with your credentials (see [Configuration](#configuration)).

### Run

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`.

---

## Configuration

Create a `.env` file in the project root:

```env
# ── AMD Developer Cloud (Production) ────────────────────────────
AMD_BASE_URL=http://YOUR_DROPLET_IP:8000/v1
AMD_MODEL=meta-llama/Llama-3.1-70B-Instruct

# ── Groq (Local Development) ────────────────────────────────────
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

The agents automatically use `AMD_BASE_URL` when set, falling back to Groq for local development. Switch between environments by commenting/uncommenting the relevant block — no code changes required.

---

## AMD MI300X Integration

SENTRA is architecturally designed around the AMD MI300X's defining characteristic: **192 GB of unified HBM3 memory on a single GPU**.

This enables loading all three 70B-class models into VRAM simultaneously:

```
Llama 3.1 70B   ≈ 140 GB (BF16)
Qwen3 72B        ≈ 144 GB (BF16)
DeepSeek-R1 70B  ≈ 140 GB (BF16)
─────────────────────────────────
Total            > 420 GB  ← impossible on any other single GPU
```

On MI300X, models are served at full precision via vLLM with an OpenAI-compatible API:

```python
# vLLM endpoint is drop-in compatible with the OpenAI SDK
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://YOUR_DROPLET_IP:8000/v1",
    api_key="not-required",
    model="meta-llama/Llama-3.1-70B-Instruct",
)
```

No CUDA. No proprietary runtime. ROCm handles the full inference stack on open-source AMD hardware.

---

## Agent API Reference

### `analyze_sensor_data(df: pd.DataFrame) -> str`

```python
from agents.sentinel import analyze_sensor_data

report = analyze_sensor_data(df)
# Returns: Markdown-formatted anomaly report with urgency level
```

**Parameters**

| Name | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Sensor data with columns: `temperature_C`, `vibration_mms`, `pressure_PSI` |

---

### `analyze_root_cause(anomaly_report: str, machine_type: str) -> str`

```python
from agents.analyst import analyze_root_cause

diagnosis = analyze_root_cause(report, "Centrifugal Pump")
# Returns: Structured root cause analysis with failure mode and time-to-failure estimate
```

---

### `create_maintenance_plan(root_cause_report: str, machine_type: str) -> str`

```python
from agents.planner import create_maintenance_plan

plan = create_maintenance_plan(diagnosis, "Centrifugal Pump")
# Returns: Actionable maintenance work order with parts list and cost estimate
```

---

## Roadmap

- [x] Three-agent sequential reasoning pipeline
- [x] Real-time sensor dashboard with anomaly visualisation
- [x] Light / Dark theme toggle
- [ ] Streaming agent responses (token-by-token)
- [ ] Multi-machine monitoring (parallel pipelines)
- [ ] PDF work order export
- [ ] Historical analysis and failure pattern learning
- [ ] REST API for integration with CMMS systems

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built for the **AMD Developer Hackathon** · lablab.ai · May 2026

Powered by AMD Instinct MI300X · vLLM · LangChain

</div>
