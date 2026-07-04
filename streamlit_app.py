# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="US Recession Probability Model", layout="wide")

@st.cache_data
def load_data():
    rp = pd.read_csv("data/processed/recession_probabilities.csv", index_col=0, parse_dates=True)
    wf = pd.read_csv("data/processed/walkforward_probabilities.csv", index_col=0, parse_dates=True)
    df = rp.join(wf["recession_prob_walkforward"], how="left")
    return df

df = load_data()
latest = df.iloc[-1]
latest_date = df.index[-1].strftime("%B %Y")

st.title("US Recession Probability Model")
st.caption(f"Last updated: {latest_date} · Data refreshed monthly via automated pipeline")
st.markdown(
    "Personal project by **Vihaan Bakshi** estimating near-term US recession probability from "
    "8 macroeconomic indicators (unemployment, housing starts, consumer sentiment, industrial "
    "production, jobless claims, yield spread, credit spread, S&P 500) using logistic regression, "
    "validated against NBER recession dates since 1986. Methodology draws on the OECD's Composite "
    "Leading Indicator framework. "
    "[View source on GitHub](https://github.com/vihaan-bakshi-23/cli_recession_model)"
)

# --- Headline metrics ---
col1, col2 = st.columns(2)
col1.metric("Multivariate Model", f"{latest['recession_prob_multi']:.2%}")
# col2.metric("CLI Equal Weighted", f"{latest['recession_prob_cli_equal']:.0%}")
# col3.metric("CLI PCA Weighted", f"{latest['recession_prob_cli_weighted']:.0%}")
col2.metric("Walk-Forward (OOS)", f"{latest['recession_prob_walkforward']:.2%}")

with st.expander("Other model estimates (CLI-based)"):
    col3, col4 = st.columns(2)
    col3.metric("CLI Equal Weighted", f"{latest['recession_prob_cli_equal']:.0%}")
    col4.metric("CLI PCA Weighted", f"{latest['recession_prob_cli_weighted']:.0%}")
    st.caption("These composite-CLI approaches showed weaker validated performance (AUC ~0.67) compared to the multivariate model (AUC ~0.95), so are shown here for reference rather than as primary signals.")

st.divider()

# --- Historical chart ---
fig = go.Figure()

# Recession shading: detect contiguous blocks where recession_indicator == 1
in_recession = df["recession_indicator"] == 1
blocks = (in_recession != in_recession.shift()).cumsum()
for _, block in df[in_recession].groupby(blocks[in_recession]):
    fig.add_vrect(
        x0=block.index[0], x1=block.index[-1],
        fillcolor="gray", opacity=0.2, layer="below", line_width=0
    )

fig.add_trace(go.Scatter(x=df.index, y=df["recession_prob_multi"], name="Multivariate Model", line=dict(color="crimson")))
fig.add_trace(go.Scatter(x=df.index, y=df["recession_prob_cli_equal"], name="CLI Equal Weighted", line=dict(color="seagreen")))
fig.add_trace(go.Scatter(x=df.index, y=df["recession_prob_cli_weighted"], name="CLI PCA Weighted", line=dict(color="orchid")))
fig.add_trace(go.Scatter(x=df.index, y=df["recession_prob_walkforward"], name="Walk-Forward (OOS)", line=dict(color="slateblue")))

fig.add_hline(y=0.4, line_dash="dash", line_color="pink", annotation_text="Alarm threshold (0.4)")

fig.update_layout(
    height=550,
    yaxis_title="Recession Probability",
    yaxis_tickformat=".0%",
    xaxis_title="Date",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Methodology: composite leading indicators built from 8 macro series (unemployment, housing starts, "
    "consumer sentiment, industrial production, jobless claims, yield spread, credit spread, S&P 500), "
    "feeding into logistic regression models validated against NBER recession dates since 1986."
)
