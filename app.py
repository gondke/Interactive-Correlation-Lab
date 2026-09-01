import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Correlation Lab",
    page_icon="📊",
    layout="wide"
)

# --- Custom Styling & CSS Injection ---
st.markdown("""
    <style>
    /* Main Background and Font Settings */
    .main {
        background-color: #F8F9FA;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Styling */
    h1 {
        color: #1E293B;
        font-weight: 800;
        font-size: 2.2rem !important;
        margin-bottom: 0.5rem !important;
    }
    h3 {
        color: #334155;
        font-weight: 700;
        font-size: 1.25rem !important;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 0.85rem !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }

    /* Data Editor / Sidebar Tweaks */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
        color: #F8FAFC;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label {
        color: #F8FAFC !important;
    }
    
    /* Custom Dividers */
    hr {
        margin: 1.5rem 0;
        border-color: #CBD5E1;
    }
    </style>
""", unsafe_allow_dict=True)

st.title("📊 Interactive Correlation & Deviation Analyzer")

# 1. Initialize State Data
if "x" not in st.session_state:
    st.session_state.x = [2.0, 4.0, 6.0, 8.0, 10.0]
if "y" not in st.session_state:
    st.session_state.y = [3.0, 5.0, 7.0, 8.0, 11.0]

# Helper function to compute correlations sequentially
def compute_correlations(x_vals, y_vals):
    pearsons, spearmans = [], []
    for i in range(2, len(x_vals) + 1):
        r_val, _ = pearsonr(x_vals[:i], y_vals[:i])
        s_val, _ = spearmanr(x_vals[:i], y_vals[:i])
        pearsons.append(round(r_val, 3))
        spearmans.append(round(s_val, 3))
    return pearsons, spearmans

# --- SIDEBAR: Controls & Dynamic Table ---
st.sidebar.header("🕹️ Control Panel")

st.sidebar.subheader("Add Point")
col_x, col_y = st.sidebar.columns(2)
new_x = col_x.number_input("X Value", value=12.0, step=0.5)
new_y = col_y.number_input("Y Value", value=13.0, step=0.5)

if st.sidebar.button("➕ Add Data Point", use_container_width=True, type="primary"):
    st.session_state.x.append(new_x)
    st.session_state.y.append(new_y)
    st.rerun()

if st.sidebar.button("🗑️ Clear All Points", use_container_width=True):
    st.session_state.x = []
    st.session_state.y = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("✏️ Editable Data Table")

df_current = pd.DataFrame({
    "X": st.session_state.x,
    "Y": st.session_state.y
})

edited_df = st.sidebar.data_editor(
    df_current,
    num_rows="dynamic",
    key="data_editor",
    use_container_width=True
)

# Sync table modifications back to state
if not edited_df.equals(df_current):
    st.session_state.x = edited_df["X"].dropna().tolist()
    st.session_state.y = edited_df["Y"].dropna().tolist()
    st.rerun()


# --- MAIN PANEL: Live Metrics ---
x_vals = st.session_state.x
y_vals = st.session_state.y
n = len(x_vals)

st.subheader("⚡ Live Calculations")

if n >= 2:
    cov_xy = float(np.cov(x_vals, y_vals, ddof=0)[0, 1])
    sigma_x = float(np.std(x_vals, ddof=0))
    sigma_y = float(np.std(y_vals, ddof=0))
    
    r_calc = (cov_xy / (sigma_x * sigma_y)) if (sigma_x > 0 and sigma_y > 0) else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("1. Covariance: Cov(X,Y)", f"{cov_xy:.4f}")
    m2.metric("2. Std Dev: σ_X", f"{sigma_x:.4f}")
    m3.metric("3. Std Dev: σ_Y", f"{sigma_y:.4f}")
    m4.metric("4. Pearson Correlation (r)", f"{r_calc:.4f}")
    
    st.latex(r"r = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}")

else:
    st.info("💡 Add at least 2 points using the sidebar control panel to generate live statistical calculations.")

st.markdown("---")

# --- MAIN PANEL: High Contrast Visualizations ---
plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.subheader("📌 Scatter Plot & Mean Deviations")
    fig1 = go.Figure()

    if n > 0:
        mx, my = float(np.mean(x_vals)), float(np.mean(y_vals))

        # Deviation lines (Bold Colors with high opacity)
        for i in range(n):
            # Horizontal red line (X deviation: dx)
            fig1.add_trace(go.Scatter(
                x=[x_vals[i], mx], y=[y_vals[i], y_vals[i]],
                mode='lines',
                line=dict(color='#E11D48', dash='dash', width=2),  # Vivid Crimson Red
                showlegend=False, hoverinfo='skip'
            ))
            # Vertical blue line (Y deviation: dy)
            fig1.add_trace(go.Scatter(
                x=[x_vals[i], x_vals[i]], y=[y_vals[i], my],
                mode='lines',
                line=dict(color='#2563EB', dash='dash', width=2),  # Vivid Royal Blue
                showlegend=False, hoverinfo='skip'
            ))

        # High-visibility Mean Reference Crosshairs
        fig1.add_vline(x=mx, line_dash="dash", line_color="#0F172A", line_width=2,
                       annotation_text=f"Mean X: {mx:.2f}", annotation_position="top left",
                       annotation_font=dict(size=12, color="#0F172A"))
        
        fig1.add_hline(y=my, line_dash="dash", line_color="#0F172A", line_width=2,
                       annotation_text=f"Mean Y: {my:.2f}", annotation_position="bottom right",
                       annotation_font=dict(size=12, color="#0F172A"))

        # Primary Scatter Points
        fig1.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers+text',
            text=[f"  P{i+1}" for i in range(n)],
            textposition="top right",
            textfont=dict(size=13, color="#0F172A", family="Arial Black"),
            marker=dict(size=14, color='#0284C7', line=dict(width=2, color='#0F172A')),
            name="Points"
        ))

    # Darker background grid for optimal contrast
    fig1.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        xaxis=dict(
            showgrid=True, gridcolor='#CBD5E1', gridwidth=1.5,
            zeroline=True, zerolinecolor='#64748B', zerolinewidth=2,
            title=dict(text="X Axis", font=dict(size=14, color="#1E293B"))
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#CBD5E1', gridwidth=1.5,
            zeroline=True, zerolinecolor='#64748B', zerolinewidth=2,
            title=dict(text="Y Axis", font=dict(size=14, color="#1E293B"))
        ),
        height=480,
        margin=dict(l=40, r=40, t=30, b=40),
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

with plot_col2:
    st.subheader("📈 Correlation Trajectory Grid")
    fig2 = go.Figure()

    if n >= 2:
        pearsons, spearmans = compute_correlations(x_vals, y_vals)
        steps = list(range(2, n + 1))
        
        fig2.add_trace(go.Scatter(
            x=steps, y=pearsons, mode='lines+markers', name='Pearson (r)',
            line=dict(color='#16A34A', width=3),
            marker=dict(size=8, symbol='circle')
        ))
        fig2.add_trace(go.Scatter(
            x=steps, y=spearmans, mode='lines+markers', name='Spearman (ρ)',
            line=dict(color='#9333EA', width=3, dash='dot'),
            marker=dict(size=8, symbol='diamond')
        ))

    fig2.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        xaxis=dict(
            title=dict(text="Points Included in Calculation", font=dict(size=14, color="#1E293B")),
            showgrid=True, gridcolor='#CBD5E1', gridwidth=1.5,
            dtick=1
        ),
        yaxis=dict(
            title=dict(text="Correlation Coefficient Value", font=dict(size=14, color="#1E293B")),
            range=[-1.1, 1.1], showgrid=True, gridcolor='#CBD5E1', gridwidth=1.5,
            zeroline=True, zerolinecolor='#64748B', zerolinewidth=2
        ),
        height=480,
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=12, color="#0F172A"),
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#CBD5E1", borderwidth=1
        )
    )
    st.plotly_chart(fig2, use_container_width=True)
