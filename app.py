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

        # Style Spearman Table: Keep D² uniform with other columns, highlighting only the Total row
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
