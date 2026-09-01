import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
import streamlit as st

# Set page configuration to wide mode
st.set_page_config(page_title="Correlation Visualizer", layout="wide")

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

# --- SIDEBAR: Data Controls ---
st.sidebar.header("Control Panel")

# Section A: Add New Points
st.sidebar.subheader("Add Point")
col_x, col_y = st.sidebar.columns(2)
new_x = col_x.number_input("X Value", value=12.0, step=0.5)
new_y = col_y.number_input("Y Value", value=13.0, step=0.5)

if st.sidebar.button("➕ Add Point", use_container_width=True):
    st.session_state.x.append(new_x)
    st.session_state.y.append(new_y)
    st.rerun()

if st.sidebar.button("🗑️ Clear All Points", type="secondary", use_container_width=True):
    st.session_state.x = []
    st.session_state.y = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Live Editable Data")

# Section B: Dynamic Table Editing
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


# --- MAIN CONTENT: Metrics & Analysis ---
x_vals = st.session_state.x
y_vals = st.session_state.y
n = len(x_vals)

# --- LIVE STATISTICAL CALCULATIONS ---
st.subheader("⚡ Live Calculations")

if n >= 2:
    # Population metrics calculation (ddof=0)
    cov_xy = float(np.cov(x_vals, y_vals, ddof=0)[0, 1])
    sigma_x = float(np.std(x_vals, ddof=0))
    sigma_y = float(np.std(y_vals, ddof=0))
    
    # Compute Pearson's r
    if sigma_x > 0 and sigma_y > 0:
        r_calc = cov_xy / (sigma_x * sigma_y)
    else:
        r_calc = 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("1. Covariance: Cov(X, Y)", f"{cov_xy:.4f}")
    m2.metric("2. Std Dev X: σ_X", f"{sigma_x:.4f}")
    m3.metric("3. Std Dev Y: σ_Y", f"{sigma_y:.4f}")
    m4.metric("4. Correlation (r)", f"{r_calc:.4f}")
    
    st.latex(r"r = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}")

else:
    st.info("Add at least 2 points to calculate covariance, standard deviations, and correlation coefficient.")

st.markdown("---")

# --- PLOTS ---
plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.subheader("Scatter Plot (Grid & Deviations)")
    fig1 = go.Figure()

    if n > 0:
        mx, my = float(np.mean(x_vals)), float(np.mean(y_vals))

        # Deviation lines from X and Y means
        for i in range(n):
            # Horizontal red line (X deviation)
            fig1.add_trace(go.Scatter(
                x=[x_vals[i], mx], y=[y_vals[i], y_vals[i]],
                mode='lines', line=dict(color='rgba(230, 0, 0, 0.4)', dash='dot', width=1.5),
                showlegend=False, hoverinfo='skip'
            ))
            # Vertical blue line (Y deviation)
            fig1.add_trace(go.Scatter(
                x=[x_vals[i], x_vals[i]], y=[y_vals[i], my],
                mode='lines', line=dict(color='rgba(0, 102, 204, 0.4)', dash='dot', width=1.5),
                showlegend=False, hoverinfo='skip'
            ))

        # Mean Crosshairs
        fig1.add_vline(x=mx, line_dash="dash", line_color="black", annotation_text=f"Mean X: {mx:.2f}")
        fig1.add_hline(y=my, line_dash="dash", line_color="black", annotation_text=f"Mean Y: {my:.2f}")

        # Main Scatter Points
        fig1.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers+text',
            text=[f"P{i+1}" for i in range(n)],
            textposition="top right",
            marker=dict(size=12, color='#1f77b4', line=dict(width=1, color='black')),
            name="Points"
        ))

    # Grid lines & Layout adjustments
    fig1.update_layout(
        xaxis=dict(showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True),
        yaxis=dict(showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True),
        height=450,
        margin=dict(l=40, r=40, t=20, b=40),
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

with plot_col2:
    st.subheader("Correlation Trajectory Grid")
    fig2 = go.Figure()

    if n >= 2:
        pearsons, spearmans = compute_correlations(x_vals, y_vals)
        steps = list(range(2, n + 1))
        fig2.add_trace(go.Scatter(x=steps, y=pearsons, mode='lines+markers', name='Pearson (r)', line=dict(color='green')))
        fig2.add_trace(go.Scatter(x=steps, y=spearmans, mode='lines+markers', name='Spearman (ρ)', line=dict(color='purple')))

    fig2.update_layout(
        xaxis=dict(title="Points Included", showgrid=True, gridcolor='LightGray', gridwidth=1),
        yaxis=dict(title="Correlation Value", range=[-1.1, 1.1], showgrid=True, gridcolor='LightGray', gridwidth=1),
        height=450,
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig2, use_container_width=True)
