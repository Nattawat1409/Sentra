import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from agents.sentinel import analyze_sensor_data
from agents.analyst import analyze_root_cause
from agents.planner import create_maintenance_plan

st.set_page_config(
    page_title="SENTRA · Predictive Maintenance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme before CSS block (used in f-string below)
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'white' #default to light theme
theme = st.session_state['theme']

# ── Theme-specific CSS (extracted to avoid multi-line string inside f-string) ──
_dark_css = """
.stApp { background: #0f172a; color: #e2e8f0; }
[data-testid='stSidebar'] { background: #0d1117 !important; border-right: 1px solid #1e293b; }
.metric-card { background: linear-gradient(145deg, #111827, #1e293b); border: 1px solid #1e293b; color: #e2e8f0; }
.metric-card:hover { border-color: #334155; }
.metric-label { color: #64748b; }
.metric-delta { color: #64748b; }
.section-title { color: #94a3b8; }
.section-title::after { background: #1e293b; }
.result-card { background: linear-gradient(145deg, #0f172a, #1a1f35); border: 1px solid #1e293b; }
.result-card-header { color: #94a3b8; border-bottom: 1px solid #1e293b; }
.sentra-header { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%); border: 1px solid #334155; }
.sentra-subtitle { color: #64748b; }
.sentra-badge { background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.4); color: #a5b4fc; border-radius: 4px; }
.sidebar-section { color: #475569; }
.stTabs [data-baseweb='tab-list'] { background: #0d1117; border: 1px solid #1e293b; }
.stTabs [data-baseweb='tab'] { color: #64748b; }
.stTabs [aria-selected='true'] { background: linear-gradient(135deg, #1e1b4b, #1e293b) !important; color: #a5b4fc !important; }
"""

_light_css = """
.stApp { background: #f8fafc; color: #0f172a; }
[data-testid='stSidebar'] { background: #ffffff !important; border-right: 1px solid #e2e8f0; }
.metric-card { background: #ffffff; border: 1px solid #e2e8f0; color: #0f172a; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.metric-card:hover { border-color: #cbd5e1; }
.metric-label { color: #64748b; }
.metric-delta { color: #94a3b8; }
.section-title { color: #475569; }
.section-title::after { background: #e2e8f0; }
.result-card { background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.result-card-header { color: #475569; border-bottom: 1px solid #e2e8f0; }
.sentra-header { background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%); border: 1px solid #4338ca; }
.sentra-subtitle { color: #a5b4fc; }
.sentra-badge { background: rgba(99,102,241,0.2); border: 1px solid rgba(99,102,241,0.5); color: #818cf8; border-radius: 4px;}
.sidebar-section { color: #94a3b8; }
.stTabs [data-baseweb='tab-list'] { background: #f1f5f9; border: 1px solid #e2e8f0; }
.stTabs [data-baseweb='tab'] { color: #94a3b8; }
.stTabs [aria-selected='true'] { background: #ffffff !important; color: #6366f1 !important; }
"""

_theme_css = _dark_css if theme == "dark" else _light_css

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }}

/* ── THEME VARIABLES ── */
{_theme_css}

/* ── Shared styles (ไม่เปลี่ยนตาม theme) ── */
.sentra-title {{
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px; margin: 0;
}}
.sentra-header {{
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; justify-content: space-between;
    position: relative; overflow: hidden;
}}
.sentra-header::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
}}
.metric-card {{
    border-radius: 12px; padding: 1.2rem 1.4rem; transition: border-color 0.2s;
}}
.metric-value {{
    font-size: 1.7rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; line-height: 1;
}}
.val-red   {{ color: #f87171; }}
.val-amber {{ color: #fbbf24; }}
.val-blue  {{ color: #60a5fa; }}
.val-green {{ color: #34d399; }}
.section-title {{
    font-size: 0.72rem; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; margin: 1.5rem 0 0.75rem 0;
    display: flex; align-items: center; gap: 0.5rem;
}}
.status-normal   {{ color: #34d399; background: rgba(52,211,153,0.1);  border: 1px solid rgba(52,211,153,0.3); }}
.status-warning  {{ color: #fbbf24; background: rgba(251,191,36,0.1);  border: 1px solid rgba(251,191,36,0.3); }}
.status-critical {{ color: #f87171; background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.3); }}
.status-pill {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.3rem 0.85rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
}}
.pulse {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
.pulse-red   {{ background: #f87171; box-shadow: 0 0 6px #f87171; }}
.pulse-green {{ background: #34d399; box-shadow: 0 0 6px #34d399; }}
.pulse-amber {{ background: #fbbf24; box-shadow: 0 0 6px #fbbf24; }}
.result-card {{ border-radius: 14px; padding: 1.5rem 1.8rem; margin-top: 0.5rem; }}
.result-card-header {{
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 0.85rem; font-weight: 600; letter-spacing: 0.5px;
    margin-bottom: 1rem; padding-bottom: 0.75rem;
}}
.sidebar-logo {{
    font-size: 1.3rem; font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 1.5rem;
}}
.sidebar-section {{
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    margin: 1.2rem 0 0.5rem 0;
}}
.stButton button {{
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.9rem !important; padding: 0.7rem 1.5rem !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
}}
.stButton button:hover {{ opacity: 0.9 !important; transform: translateY(-1px) !important; }}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚡ SENTRA</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Machine Configuration</div>', unsafe_allow_html=True)
    machine_type = st.selectbox(
        "Machine Type",
        ["Centrifugal Pump", "Compressor", "Electric Motor", "Conveyor Belt"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-section">Data Source</div>', unsafe_allow_html=True)
    use_sample = st.checkbox("Use sample data (with anomaly)", value=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Display</div>', unsafe_allow_html=True)
    selected_theme = st.radio(
        "Theme", ["dark", "light"],
        index=0 if theme == "dark" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    if selected_theme != theme:
        st.session_state['theme'] = selected_theme
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="color:#334155; font-size:0.72rem; line-height:1.7;">
    <b style="color:#475569">3 AI AGENTS ACTIVE</b><br>
    <span style="color:#6366f1">●</span> SENTINEL — Anomaly detection<br>
    <span style="color:#8b5cf6">●</span> ANALYST — Root cause analysis<br>
    <span style="color:#06b6d4">●</span> PLANNER — Maintenance planning
    </div>
    """, unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────────────────
if uploaded:
    df = pd.read_csv(uploaded)
else:
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min'),
        'temperature_C': np.random.normal(75, 2, n),
        'vibration_mms': np.random.normal(0.5, 0.1, n),
        'pressure_PSI': np.random.normal(100, 5, n),
    })
    if use_sample:
        df.loc[95:, 'temperature_C'] += 20
        df.loc[95:, 'vibration_mms'] *= 4

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="sentra-header">
  <div>
    <div class="sentra-title">⚡ SENTRA</div>
    <div class="sentra-subtitle">AUTONOMOUS PREDICTIVE MAINTENANCE INTELLIGENCE · {machine_type.upper()}</div>
  </div>
  <div class="sentra-badge">3 Agents Online</div>
</div>
""", unsafe_allow_html=True)

# ── Live Metric Cards ──────────────────────────────────────────────────────
temp_last  = df['temperature_C'].iloc[-1]
vib_last   = df['vibration_mms'].iloc[-1]
pres_last  = df['pressure_PSI'].iloc[-1]
temp_mean  = df['temperature_C'].mean()
vib_mean   = df['vibration_mms'].mean()
pres_mean  = df['pressure_PSI'].mean()

temp_status = "CRITICAL" if temp_last > 90 else ("WARNING" if temp_last > 82 else "NORMAL")
vib_status  = "CRITICAL" if vib_last > 1.5 else ("WARNING" if vib_last > 1.0 else "NORMAL")
pres_status = "WARNING"  if abs(pres_last - 100) > 10 else "NORMAL"

status_map = {
    "CRITICAL": ("pulse-red",   "status-critical", "🔴"),
    "WARNING":  ("pulse-amber", "status-warning",  "🟡"),
    "NORMAL":   ("pulse-green", "status-normal",   "🟢"),
}

st.markdown('<div class="section-title">Live Sensor Snapshot</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

def metric_card(col, label, value, unit, val_class, delta_label, status_key):
    pulse_cls, pill_cls, _ = status_map[status_key]
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {val_class}">{value:.1f}<span style="font-size:1rem;font-weight:400;color:#475569"> {unit}</span></div>
      <div class="metric-delta">{delta_label}</div>
      <div style="margin-top:0.7rem">
        <span class="status-pill {pill_cls}">
          <span class="pulse {pulse_cls}"></span>{status_key}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

metric_card(c1, "Temperature",  temp_last,  "°C",  "val-red",   f"avg {temp_mean:.1f} °C",  temp_status)
metric_card(c2, "Vibration",    vib_last,   "mm/s","val-amber", f"avg {vib_mean:.2f} mm/s",  vib_status)
metric_card(c3, "Pressure",     pres_last,  "PSI", "val-blue",  f"avg {pres_mean:.1f} PSI",  pres_status)

# 4th card — overall health
critical_count = [temp_status, vib_status, pres_status].count("CRITICAL")
warn_count     = [temp_status, vib_status, pres_status].count("WARNING")
if critical_count > 0:
    health_score, health_cls, health_status = 100 - critical_count * 30 - warn_count * 10, "val-red", "CRITICAL"
elif warn_count > 0:
    health_score, health_cls, health_status = 100 - warn_count * 15, "val-amber", "WARNING"
else:
    health_score, health_cls, health_status = 98, "val-green", "NORMAL"

pulse_cls, pill_cls, _ = status_map[health_status]
c4.markdown(f"""
<div class="metric-card">
  <div class="metric-label">System Health</div>
  <div class="metric-value {health_cls}">{health_score}<span style="font-size:1rem;font-weight:400;color:#475569"> %</span></div>
  <div class="metric-delta">{len(df)} data points · last {df['timestamp'].iloc[-1].strftime('%H:%M') if hasattr(df['timestamp'].iloc[-1], 'strftime') else ''}</div>
  <div style="margin-top:0.7rem">
    <span class="status-pill {pill_cls}">
      <span class="pulse {pulse_cls}"></span>{health_status}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Charts ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Sensor Time Series</div>', unsafe_allow_html=True)

CHART_BG   = "#ffffff" if theme == "white" else "#0d1117" ##0d1117#
GRID_COLOR = "#f1f5f9" if theme == "white" else "#1e293b" ##1e293b#
FONT_COLOR = "#64748b"

def hex_to_rgba(hex_color, alpha=0.08):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def make_chart(y_series, title, color, y_label, threshold=None, threshold_label="Threshold"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(y_series))), y=y_series,
        mode='lines', fill='tozeroy',
        line=dict(color=color, width=2),
        fillcolor=hex_to_rgba(color, 0.08),
        hovertemplate=f"<b>{y_label}:</b> %{{y:.2f}}<extra></extra>"
    ))
    if threshold:
        fig.add_hline(y=threshold, line_dash="dash", line_color="#475569",
                      annotation_text=threshold_label,
                      annotation_font_color="#475569", annotation_font_size=10)
    fig.update_layout(
        title=dict(text=title, font=dict(color="#94a3b8", size=13), x=0),
        height=220,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        margin=dict(t=40, b=20, l=40, r=10),
        font=dict(color=FONT_COLOR, family="JetBrains Mono"),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                   tickfont=dict(size=10), title=None),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                   tickfont=dict(size=10), title=None),
        showlegend=False,
        hoverlabel=dict(bgcolor="#1e293b", font_color="#e2e8f0", bordercolor="#334155")
    )
    return fig

col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(make_chart(df['temperature_C'],  "Temperature (°C)",  "#f87171", "Temp", 90, "⚠ 90°C"), use_container_width=True)
with col2:
    st.plotly_chart(make_chart(df['vibration_mms'],  "Vibration (mm/s)",  "#fbbf24", "Vib",  1.5, "⚠ 1.5 mm/s"), use_container_width=True)
with col3:
    st.plotly_chart(make_chart(df['pressure_PSI'],   "Pressure (PSI)",    "#60a5fa", "Pres"), use_container_width=True)

# ── Run Analysis ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">AI Analysis</div>', unsafe_allow_html=True)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run = st.button("⚡ Run AI Analysis", type="primary", use_container_width=True)
with col_info:
    st.markdown("""
    <div style="color:#475569; font-size:0.8rem; padding-top:0.6rem;">
    Runs <span style="color:#818cf8">SENTINEL → ANALYST → PLANNER</span> in sequence.
    Analysis typically takes 15–30 seconds.
    </div>
    """, unsafe_allow_html=True)

if run:
    prog = st.progress(0, text="Initializing agents...")

    with st.spinner("SENTINEL — Scanning sensor streams for anomalies..."):
        anomaly = analyze_sensor_data(df)
    prog.progress(33, text="ANALYST — Diagnosing root cause...")

    with st.spinner("ANALYST — Performing root cause analysis..."):
        root_cause = analyze_root_cause(anomaly, machine_type)
    prog.progress(66, text="PLANNER — Generating maintenance plan...")

    with st.spinner("PLANNER — Synthesizing maintenance recommendations..."):
        plan = create_maintenance_plan(root_cause, machine_type)
    prog.progress(100, text="Analysis complete.")

    st.markdown("""
    <div style="background:rgba(52,211,153,0.08); border:1px solid rgba(52,211,153,0.3);
         border-radius:10px; padding:0.75rem 1.2rem; margin:1rem 0;
         color:#34d399; font-weight:600; font-size:0.85rem; display:flex; align-items:center; gap:0.5rem;">
    ✅ &nbsp;Analysis complete — 3 agents finished successfully.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔍 Anomaly Report", "🧠 Root Cause Analysis", "📋 Maintenance Plan"])

    with tab1:
        st.markdown('<div class="result-card"><div class="result-card-header">🔍 SENTINEL · Anomaly Detection Report</div>', unsafe_allow_html=True)
        st.markdown(anomaly)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="result-card"><div class="result-card-header">🧠 ANALYST · Root Cause Analysis</div>', unsafe_allow_html=True)
        st.markdown(root_cause)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="result-card"><div class="result-card-header">📋 PLANNER · Maintenance Recommendations</div>', unsafe_allow_html=True)
        st.markdown(plan)
        st.markdown('</div>', unsafe_allow_html=True)
