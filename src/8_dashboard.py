# src/8_dashboard.py

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Load data --- 
df = pd.read_csv("data/processed/recession_probabilities.csv", index_col=0, parse_dates=True)
print(f"Loaded: {df.shape}")

THRESHOLD = 0.4

# --- 2. Define NBER recession bands --- 
recessions = [
    ("1990-08-01", "1991-03-01", "1990–91"),
    ("2001-04-01", "2001-11-01", "2001"),
    ("2008-01-01", "2009-06-01", "2008–09"),
    ("2020-03-01", "2020-04-01", "2020"),
]

# --- 3. Build figure with dual y-axes --- 
fig = make_subplots(specs=[[{"secondary_y": True}]])

# --- 4. NBER recession shading --- 
for i, (start, end, label) in enumerate(recessions):
    fig.add_vrect(
        x0=start, x1=end,
        fillcolor="grey",
        opacity=0.25,
        layer="below",
        line_width=0,
        annotation_text=label,
        annotation_position="top left",
        annotation_font_size=10,
    )

# --- 5. Recession probability (primary y-axis) --- 
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["recession_prob_multi"],
        name="Recession Probability",
        line=dict(color="crimson", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(220, 20, 60, 0.15)",
    ),
    secondary_y=False,
)

# --- 6. Threshold line --- 
fig.add_hline(
    y=THRESHOLD,
    line_dash="dash",
    line_color="crimson",
    line_width=1,
    opacity=0.5,
    annotation_text=f"Alarm threshold ({THRESHOLD})",
    annotation_position="bottom right",
    annotation_font_size=10,
)

# --- 7. CLI equal (secondary y-axis) --- 
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["cli_equal"],
        name="CLI (equal weighted)",
        line=dict(color="steelblue", width=1.2, dash="dot"),
        opacity=0.7,
    ),
    secondary_y=True,
)

# --- 8. Layout --- 
fig.update_layout(
    title=dict(
        text="US Recession Probability Model (1986–2024)",
        font=dict(size=18)
    ),
    xaxis=dict(title="Date", showgrid=False),
    yaxis=dict(
        title="Recession Probability",
        range=[0, 1],
        tickformat=".0%",
        showgrid=True,
        gridcolor="rgba(200,200,200,0.3)"
    ),
    yaxis2=dict(
        title="CLI Score",
        showgrid=False,
        zeroline=False
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=1200,
    height=550,
)

# --- 9. Save --- 
fig.write_html("outputs/recession_dashboard.html")
print("Saved: outputs/recession_dashboard.html")