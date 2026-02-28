"""
CSIS 4260 - Assignment 1, Part 3
Stock Price Prediction Dashboard (Streamlit)

Select / search a company ticker in the sidebar to view:
 - Actual vs predicted next-day closing price (Linear Regression & Random Forest)
 - Historical closing price with SMA(20) overlay
 - RSI(14) indicator
 - Per-company error metrics
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


DATA_DIR = Path(__file__).parent
PREDICTIONS_PATH = DATA_DIR / "model_outputs" / "predictions.csv"
ENRICHED_PATH = DATA_DIR / "model_outputs" / "enriched_stocks.csv"


@st.cache_data
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])


@st.cache_data
def load_enriched() -> pd.DataFrame:
    if not ENRICHED_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(ENRICHED_PATH, parse_dates=["date"])


def main():
    st.set_page_config(
        page_title="Stock Price Prediction Dashboard",
        page_icon="📈",
        layout="wide",
    )
    st.title("📈 Stock Price Prediction Dashboard")
    st.caption("CSIS 4260 Assignment 1 — Next-day closing price predictions for S&P 500 companies")

    predictions = load_predictions()
    enriched = load_enriched()

    if predictions.empty:
        st.error("Predictions file not found. Run `python part2_data_analysis.py` first.")
        st.stop()

    companies = sorted(predictions["name"].unique().tolist())

    # ---- Sidebar ----
    st.sidebar.header("Company Selection")
    selected = st.sidebar.selectbox(
        "Search / Select Ticker",
        options=companies,
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{len(companies)}** companies available")
    st.sidebar.markdown(
        "**Models:** Linear Regression, Random Forest  \n"
        "**Indicators:** SMA(20), RSI(14)  \n"
        "**Split:** 80 % train / 20 % test"
    )

    # ---- Filter to selected company ----
    comp = predictions[predictions["name"] == selected].sort_values("date")

    if comp.empty:
        st.warning(f"No data for {selected}")
        return

    # ---- KPI row ----
    lr_mae = (comp["target"] - comp["Linear Regression_pred"]).abs().mean()
    rf_mae = (comp["target"] - comp["Random Forest_pred"]).abs().mean()
    date_min = comp["date"].min().strftime("%Y-%m-%d")
    date_max = comp["date"].max().strftime("%Y-%m-%d")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Points", f"{len(comp):,}")
    c2.metric("Linear Reg MAE", f"${lr_mae:.2f}")
    c3.metric("Random Forest MAE", f"${rf_mae:.2f}")
    c4.metric("Date Range", f"{date_min} to {date_max}")

    # ---- Chart 1: Actual vs Predicted ----
    st.subheader(f"Next-Day Close: Actual vs Predicted — {selected}")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=comp["date"], y=comp["target"],
        name="Actual (Next Day)", mode="lines",
        line=dict(color="#1f77b4", width=2),
    ))
    fig1.add_trace(go.Scatter(
        x=comp["date"], y=comp["Linear Regression_pred"],
        name="Linear Regression", mode="lines",
        line=dict(color="#ff7f0e", width=1.5, dash="dash"),
    ))
    fig1.add_trace(go.Scatter(
        x=comp["date"], y=comp["Random Forest_pred"],
        name="Random Forest", mode="lines",
        line=dict(color="#2ca02c", width=1.5, dash="dot"),
    ))
    fig1.update_layout(
        xaxis_title="Date", yaxis_title="Price ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white", height=450,
        margin=dict(l=60, r=40, t=60, b=40),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ---- Chart 2: Technical Indicators (Close + SMA + RSI) ----
    if not enriched.empty and selected in enriched["name"].values:
        te = enriched[enriched["name"] == selected].sort_values("date")

        st.subheader(f"Technical Indicators — {selected}")

        fig2 = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35], vertical_spacing=0.06,
            subplot_titles=("Close & SMA(20)", "RSI(14)"),
        )
        fig2.add_trace(go.Scatter(
            x=te["date"], y=te["close"], name="Close",
            line=dict(color="#9467bd", width=2),
        ), row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=te["date"], y=te["sma"], name="SMA(20)",
            line=dict(color="#d62728", width=1.5, dash="dash"),
        ), row=1, col=1)

        fig2.add_trace(go.Scatter(
            x=te["date"], y=te["rsi"], name="RSI(14)",
            line=dict(color="#17becf", width=1.5),
        ), row=2, col=1)
        fig2.add_hline(y=70, line_dash="dot", line_color="red",
                       annotation_text="Overbought (70)", row=2, col=1)
        fig2.add_hline(y=30, line_dash="dot", line_color="green",
                       annotation_text="Oversold (30)", row=2, col=1)

        fig2.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig2.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig2.update_xaxes(title_text="Date", row=2, col=1)
        fig2.update_layout(
            template="plotly_white", height=550,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=60, r=40, t=80, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
