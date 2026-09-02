import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata
import plotly.graph_objects as go
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Statistical Analysis Lab",
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
        font-size: 2rem !important;
        margin-bottom: 0.5rem !important;
    }
    h2, h3 {
        color: #334155;
        font-weight: 700;
    }

    /* Compact Metric Card Containers */
    .metric-card {
        background-color: #0F172A;
        border: 1.5px solid #334155;
        border-radius: 8px;
        padding: 8px 12px;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
    .metric-symbol {
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 1.35rem;
        font-weight: 800;
        font-family: 'Courier New', Courier, monospace;
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
        margin: 1.2rem 0;
        border-color: #CBD5E1;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Statistical Correlation & Regression Suite")

# 1. Initialize State Data
if "x" not in st.session_state:
    st.session_state.x = [2.0, 4.0, 6.0, 8.0, 10.0]
if "y" not in st.session_state:
    st.session_state.y = [3.0, 5.0, 7.0, 8.0, 11.0]

# SAFEGUARD: Ensure X and Y lists stay synchronized
min_len = min(len(st.session_state.x), len(st.session_state.y))
st.session_state.x = st.session_state.x[:min_len]
st.session_state.y = st.session_state.y[:min_len]

# Helper function to compute trajectory correlations
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

# Synchronize table changes while filtering incomplete rows
cleaned_df = edited_df.dropna(subset=["X", "Y"])
new_x_list = cleaned_df["X"].astype(float).tolist()
new_y_list = cleaned_df["Y"].astype(float).tolist()

if new_x_list != st.session_state.x or new_y_list != st.session_state.y:
    st.session_state.x = new_x_list
    st.session_state.y = new_y_list
    st.rerun()

# Export CSV Setup
x_vals = st.session_state.x
y_vals = st.session_state.y
n = len(x_vals)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export Options")

if n >= 2:
    cov_xy = float(np.cov(x_vals, y_vals, ddof=0)[0, 1])
    sigma_x = float(np.std(x_vals, ddof=0))
    sigma_y = float(np.std(y_vals, ddof=0))
    r_calc = (cov_xy / (sigma_x * sigma_y)) if (sigma_x > 0 and sigma_y > 0) else 0.0

    export_df = pd.DataFrame({"X": x_vals, "Y": y_vals})
    export_df["Covariance_XY"] = cov_xy
    export_df["StdDev_X"] = sigma_x
    export_df["StdDev_Y"] = sigma_y
    export_df["Pearson_r"] = r_calc

    csv_data = export_df.to_csv(index=False).encode('utf-8')

    st.sidebar.download_button(
        label="📄 Download Data & Metrics (CSV)",
        data=csv_data,
        file_name="correlation_analysis_data.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.sidebar.info("Add at least 2 points to enable CSV export.")


# --- MAIN INTERFACE NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs([
    "📈 Interactive Correlation Lab", 
    "📑 Detailed Calculation Tables", 
    "📉 Regression Analysis Lab"
])

# ==========================================
# TAB 1: INTERACTIVE CORRELATION LAB
# ==========================================
with tab1:
    st.subheader("⚡ Live Calculations")

    if n >= 2:
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">1. Covariance</div>
                    <div class="metric-symbol" style="color: #38BDF8;">Cov(X, Y)</div>
                    <div class="metric-value">{cov_xy:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">2. Std Dev X</div>
                    <div class="metric-symbol" style="color: #F43F5E;">σ<sub>X</sub></div>
                    <div class="metric-value">{sigma_x:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">3. Std Dev Y</div>
                    <div class="metric-symbol" style="color: #60A5FA;">σ<sub>Y</sub></div>
                    <div class="metric-value">{sigma_y:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">4. Pearson Correlation</div>
                    <div class="metric-symbol" style="color: #4ADE80;">r</div>
                    <div class="metric-value">{r_calc:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.latex(r"r = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}")

    else:
        st.info("💡 Add at least 2 points using the sidebar control panel to generate live statistical calculations.")

    st.markdown("---")

    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.subheader("📌 Scatter Plot & Mean Deviations")
        fig1 = go.Figure()

        if n > 0:
            mx, my = float(np.mean(x_vals)), float(np.mean(y_vals))

            for i in range(n):
                fig1.add_trace(go.Scatter(
                    x=[x_vals[i], mx], y=[y_vals[i], y_vals[i]],
                    mode='lines',
                    line=dict(color='#E11D48', dash='dash', width=2),
                    showlegend=False, hoverinfo='skip'
                ))
                fig1.add_trace(go.Scatter(
                    x=[x_vals[i], x_vals[i]], y=[y_vals[i], my],
                    mode='lines',
                    line=dict(color='#2563EB', dash='dash', width=2),
                    showlegend=False, hoverinfo='skip'
                ))

            fig1.add_vline(x=mx, line_dash="dash", line_color="#0F172A", line_width=2,
                           annotation_text=f"Mean X: {mx:.2f}", annotation_position="top left",
                           annotation_font=dict(size=12, color="#0F172A"))
            
            fig1.add_hline(y=my, line_dash="dash", line_color="#0F172A", line_width=2,
                           annotation_text=f"Mean Y: {my:.2f}", annotation_position="bottom right",
                           annotation_font=dict(size=12, color="#0F172A"))

            fig1.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='markers+text',
                text=[f"  P{i+1}" for i in range(n)],
                textposition="top right",
                textfont=dict(size=13, color="#0F172A", family="Arial Black"),
                marker=dict(size=14, color='#0284C7', line=dict(width=2, color='#0F172A')),
                name="Points"
            ))

        fig1.update_layout(
            plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
            xaxis=dict(
                showgrid=True, gridcolor='#CBD5E1', gridwidth=1.5,
                zeroline=True, zerolinecolor='#64748B', zerolinewidth=2,
                title=dict(text="X Axis", font=dict(size=14, color="#0F172A", weight="bold")),
                tickfont=dict(color="#0F172A", size=12, weight="bold")
            ),
            yaxis=dict(
                showgrid=True, gridcolor='#CBD5E1', gridwidth=1.5,
                zeroline=True, zerolinecolor='#64748B', zerolinewidth=2,
                title=dict(text="Y Axis", font=dict(size=14, color="#0F172A", weight="bold")),
                tickfont=dict(color="#0F172A", size=12, weight="bold")
            ),
            height=480, margin=dict(l=40, r=40, t=30, b=40), showlegend=False
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
                line=dict(color='#16A34A', width=3), marker=dict(size=8, symbol='circle')
            ))
            fig2.add_trace(go.Scatter(
                x=steps, y=spearmans, mode='lines+markers', name='Spearman (ρ)',
                line=dict(color='#9333EA', width=3, dash='dot'), marker=dict(size=8, symbol='diamond')
            ))

        fig2.update_layout(
            plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
            xaxis=dict(
                title=dict(text="Points Included in Calculation", font=dict(size=14, color="#0F172A", weight="bold")),
                showgrid=True, gridcolor='#CBD5E1', gridwidth=1.5, dtick=1,
                tickfont=dict(color="#0F172A", size=12, weight="bold")
            ),
            yaxis=dict(
                title=dict(text="Correlation Coefficient Value", font=dict(size=14, color="#0F172A", weight="bold")),
                range=[-1.1, 1.1], showgrid=True, gridcolor='#CBD5E1', gridwidth=1.5,
                zeroline=True, zerolinecolor='#64748B', zerolinewidth=2,
                tickfont=dict(color="#0F172A", size=12, weight="bold")
            ),
            height=480, margin=dict(l=40, r=40, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color="#0F172A"), bgcolor="rgba(255,255,255,0.8)", bordercolor="#CBD5E1", borderwidth=1)
        )
        st.plotly_chart(fig2, use_container_width=True)


# ==========================================
# TAB 2: DETAILED CALCULATION TABLES
# ==========================================
with tab2:
    st.header("📑 Detailed Calculation Tables")

    if n < 2:
        st.warning("⚠️ Please enter at least 2 data points in the sidebar to generate calculation tables.")
    else:
        # 1. KARL PEARSON'S CORRELATION CALCULATION TABLE
        st.subheader("1. Karl Pearson's Correlation Coefficient Table")
        st.latex(r"r = \frac{\sum xy}{\sqrt{\sum x^2} \sqrt{\sum y^2}}")
        
        mean_x = float(np.mean(x_vals))
        mean_y = float(np.mean(y_vals))
        
        dev_x = [x - mean_x for x in x_vals]
        dev_y = [y - mean_y for y in y_vals]
        dev_xy = [dx * dy for dx, dy in zip(dev_x, dev_y)]
        dev_x2 = [dx ** 2 for dx in dev_x]
        dev_y2 = [dy ** 2 for dy in dev_y]

        df_pearson = pd.DataFrame({
            "X": x_vals,
            "Y": y_vals,
            "x = X - X̄": dev_x,
            "y = Y - Ȳ": dev_y,
            "xy": dev_xy,
            "x²": dev_x2,
            "y²": dev_y2
        })

        totals_pearson = {
            "X": sum(x_vals),
            "Y": sum(y_vals),
            "x = X - X̄": sum(dev_x),
            "y = Y - Ȳ": sum(dev_y),
            "xy": sum(dev_xy),
            "x²": sum(dev_x2),
            "y²": sum(dev_y2)
        }
        df_pearson_display = pd.concat([df_pearson, pd.DataFrame([totals_pearson], index=["Total (Σ)"])])

        def highlight_total_row(row):
            if row.name == "Total (Σ)":
                return ['background-color: #F1F5F9; color: #0F172A; font-weight: bold; border-top: 2px solid #94A3B8;'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_pearson_display.style.format("{:.4f}").apply(highlight_total_row, axis=1),
            use_container_width=True
        )

        sum_xy = totals_pearson["xy"]
        sum_x2 = totals_pearson["x²"]
        sum_y2 = totals_pearson["y²"]
        denom = np.sqrt(sum_x2 * sum_y2)
        kp_r = sum_xy / denom if denom != 0 else 0.0

        st.markdown(f"""
        * **Mean of X ($\overline{{X}}$):** `{mean_x:.4f}` | **Mean of Y ($\overline{{Y}}$):** `{mean_y:.4f}`
        * **$\sum xy$:** `{sum_xy:.4f}` | **$\sum x^2$:** `{sum_x2:.4f}` | **$\sum y^2$:** `{sum_y2:.4f}`
        * **Calculated $r$:** **`{kp_r:.4f}`**
        """)

        st.markdown("---")

        # 2. SPEARMAN'S RANK CORRELATION CALCULATION TABLE
        st.subheader("2. Spearman's Rank Correlation Coefficient Table")
        st.latex(r"\rho = 1 - \frac{6 \left[\sum D^2 + \text{Correction}\right]}{N^3 - N}")

        ranks_x = rankdata(x_vals, method='average')
        ranks_y = rankdata(y_vals, method='average')
        diff_d = ranks_x - ranks_y
        diff_d2 = diff_d ** 2

        df_spearman = pd.DataFrame({
            "X": x_vals,
            "Y": y_vals,
            "R_1 (Rank X)": ranks_x,
            "R_2 (Rank Y)": ranks_y,
            "D = R_1 - R_2": diff_d,
            "D²": diff_d2
        })

        def calc_tie_correction(ranks):
            correction = 0.0
            _, counts = np.unique(ranks, return_counts=True)
            for count in counts:
                if count > 1:
                    correction += (count**3 - count) / 12.0
            return correction

        corr_x = calc_tie_correction(ranks_x)
        corr_y = calc_tie_correction(ranks_y)
        total_correction = corr_x + corr_y

        totals_spearman = {
            "X": sum(x_vals),
            "Y": sum(y_vals),
            "R_1 (Rank X)": sum(ranks_x),
            "R_2 (Rank Y)": sum(ranks_y),
            "D = R_1 - R_2": sum(diff_d),
            "D²": sum(diff_d2)
        }
        df_spearman_display = pd.concat([df_spearman, pd.DataFrame([totals_spearman], index=["Total (Σ)"])])

        def style_spearman_table(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            if "Total (Σ)" in df.index:
                styles.loc["Total (Σ)"] = 'background-color: #F1F5F9; color: #0F172A; font-weight: bold; border-top: 2px solid #94A3B8;'
            return styles

        st.dataframe(
            df_spearman_display.style.format("{:.4f}").apply(style_spearman_table, axis=None),
            use_container_width=True
        )

        sum_d2 = totals_spearman["D²"]
        numerator = 6 * (sum_d2 + total_correction)
        denominator = (n ** 3) - n
        spearman_rho = 1.0 - (numerator / denominator) if denominator != 0 else 0.0

        st.markdown(f"""
        * **$\sum D^2$:** `{sum_d2:.4f}` | **Total Tie Correction:** `{total_correction:.4f}`
        * **Number of observations ($N$):** `{n}`
        * **Calculated $\\rho$:** **`{spearman_rho:.4f}`**
        """)


# ==========================================
# TAB 3: REGRESSION ANALYSIS LAB
# ==========================================
with tab3:
    st.header("📉 Linear Regression Analysis")

    if n < 2:
        st.warning("⚠️ Please enter at least 2 data points in the sidebar to generate regression lines.")
    else:
        mx, my = float(np.mean(x_vals)), float(np.mean(y_vals))
        var_x = float(np.var(x_vals, ddof=0))
        var_y = float(np.var(y_vals, ddof=0))
        cov_xy = float(np.cov(x_vals, y_vals, ddof=0)[0, 1])

        # Regression Coefficients
        b_yx = (cov_xy / var_x) if var_x > 0 else 0.0
        b_xy = (cov_xy / var_y) if var_y > 0 else 0.0

        # Intercepts
        a_yx = my - (b_yx * mx)
        a_xy = mx - (b_xy * my)

        # Compact Live Coefficients Metric Displays
        reg_m1, reg_m2, reg_m3, reg_m4 = st.columns(4)

        with reg_m1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Coeff: Y on X</div>
                    <div class="metric-symbol" style="color: #F43F5E;">b<sub>yx</sub></div>
                    <div class="metric-value">{b_yx:.4f}</div>
                </div>
            """, unsafe_allow_html=True)

        with reg_m2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Equation: Y on X</div>
                    <div class="metric-symbol" style="color: #F43F5E;">Y = a + bX</div>
                    <div class="metric-value" style="font-size: 1.05rem; line-height: 1.8rem;">Y = {a_yx:.2f} + {b_yx:.2f}X</div>
                </div>
            """, unsafe_allow_html=True)

        with reg_m3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Coeff: X on Y</div>
                    <div class="metric-symbol" style="color: #0EA5E9;">b<sub>xy</sub></div>
                    <div class="metric-value">{b_xy:.4f}</div>
                </div>
            """, unsafe_allow_html=True)

        with reg_m4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Equation: X on Y</div>
                    <div class="metric-symbol" style="color: #0EA5E9;">X = a + bY</div>
                    <div class="metric-value" style="font-size: 1.05rem; line-height: 1.8rem;">X = {a_xy:.2f} + {b_xy:.2f}Y</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Range setup for continuous line rendering
        x_min, x_max = min(x_vals) - 3, max(x_vals) + 3
        y_min, y_max = min(y_vals) - 3, max(y_vals) + 3

        x_range = np.linspace(x_min, x_max, 100)
        y_range = np.linspace(y_min, y_max, 100)

        reg_col1, reg_col2 = st.columns(2)

        # Plot 1: Regression of Y on X
        with reg_col1:
            st.subheader("🔴 Regression Line of Y on X")
            st.caption("Predicts Dependent Variable **Y** given Independent Variable **X**")
            
            y_pred_line = a_yx + b_yx * x_range

            fig_reg1 = go.Figure()
            fig_reg1.add_trace(go.Scatter(
                x=x_range, y=y_pred_line, mode='lines',
                name=f"Y = {a_yx:.2f} + {b_yx:.2f}X",
                line=dict(color='#E11D48', width=3)
            ))
            fig_reg1.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='markers',
                marker=dict(size=12, color='#0F172A', line=dict(width=2, color='#FFFFFF')),
                name="Data Points"
            ))

            fig_reg1.update_layout(
                plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
                xaxis=dict(
                    title=dict(text="X Axis", font=dict(size=14, color="#0F172A", weight="bold")),
                    showgrid=True, gridcolor='#CBD5E1', gridwidth=1.5,
                    tickfont=dict(color="#0F172A", size=12, weight="bold")
                ),
                yaxis=dict(
                    title=dict(text="Y Axis", font=dict(size=14, color="#0F172A", weight="bold")),
                    showgrid=True, gridcolor='#CBD5E1', gridwidth=1.5,
                    tickfont=dict(color="#0F172A", size=12, weight="bold")
                ),
                height=460, margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_reg1, use_container_width=True)

        # Plot 2: Regression of X on Y
        with reg_col2:
            st.subheader("🔵 Regression Line of X on Y")
            st.caption("Predicts Dependent Variable **X** given Independent Variable **Y**")
            
            x_pred_line = a_xy + b_xy * y_range

            fig_reg2 = go.Figure()
            fig_reg2.add_trace(go.Scatter(
                x=x_pred_line, y=y_range, mode='lines',
                name=f"X = {a_xy:.2f} + {b_xy:.2f}Y",
                line=dict(color='#0284C7', width=3)
            ))
            fig_reg2.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='markers',
                marker=dict(size=12, color='#0F172A', line=dict(width=2, color='#FFFFFF')),
                name="Data Points"
            ))

            fig_reg2.update_layout(
                plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
                xaxis=dict(
                    title=dict(text="X Axis", font=dict(size=14, color="#0F172A", weight="bold")),
                    showgrid=True, gridcolor='#CBD5E1', gridwidth=1.5,
                    tickfont=dict(color="#0F172A", size=12, weight="bold")
                ),
                yaxis=dict(
                    title=dict(text="Y Axis", font=dict(size=14, color="#0F172A", weight="bold")),
                    showgrid=True, gridcolor='#CBD5E1', gridwidth=1.5,
                    tickfont=dict(color="#0F172A", size=12, weight="bold")
                ),
                height=460, margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_reg2, use_container_width=True)

        st.markdown("---")

        # SECTION MOVED UNDERNEATH GRAPHS: Interactive Value Prediction & Perpendicular Projection
        st.subheader("🎯 Interactive Value Prediction & Perpendicular Projection")
        pred_col1, pred_col2 = st.columns(2)

        with pred_col1:
            given_x = st.number_input(
                "Predict Y from known X (Line Y on X)", 
                value=0.0, 
                step=0.5, 
                key="input_x_pred"
            )
            calc_y_pred = a_yx + b_yx * given_x
            st.success(f"📌 Predicted **Y** for X = `{given_x:.2f}`: **`{calc_y_pred:.4f}`**")

        with pred_col2:
            given_y = st.number_input(
                "Predict X from known Y (Line X on Y)", 
                value=0.0, 
                step=0.5, 
                key="input_y_pred"
            )
            calc_x_pred = a_xy + b_xy * given_y
            st.info(f"📌 Predicted **X** for Y = `{given_y:.2f}`: **`{calc_x_pred:.4f}`**")
