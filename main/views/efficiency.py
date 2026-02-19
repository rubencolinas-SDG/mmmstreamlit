# main/views/efficiency.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def _inject_css():
    st.markdown(
        """
        <style>
          div.stButton > button, div.stDownloadButton > button {
            justify-content: flex-start !important;
            text-align: left !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _roi_demo_data() -> pd.DataFrame:
    """
    Demo ROI dataset aligned with your request:
    - Only one YouTube and one Facebook
    - Add Sampling and Sponsorship
    """
    rows = [
        # Offline
        ("Corporate Social Media", "Media", 0.40, 0.60),
        #("Outdoor", "Offline Media", 0.18, 0.42),
        ("Medical magazines", "Media", 0.25, 0.49),
        #("Press", "Offline Media", 0.20, 0.52),

        # Digital / Performance
        ("Sales representative", "Field Marketing", 3.85, 0.79),
        #("Display", "Display", 0.15, 0.00),
        ("HCPs emailing", "Field Marketing", 3.20, 0.68),

        # Social / Video (single entries)
        ("Referrals & Content", "Leadership & Advocacy", 1.55, 0.76),
        ("KOL campaigns", "Leadership & Advocacy", 1.65, 1.24),

        # Added channels
        ("HCPs platforms", "Field Marketing", 0.35, 0.90),
        ("Sponsorship", "Events", 0.55, 1.10),

        # Other
        ("Webinars & Congresses", "Events", 2.10, 0.89),
        #("Online Radio", "Other", 0.22, 0.00),
    ]
    df = pd.DataFrame(rows, columns=["Channel", "Group", "ROI_short", "ROI_long"])
    df["ROI_total"] = df["ROI_short"] + df["ROI_long"]
    return df


def _order_channels(df: pd.DataFrame, sort_mode: str) -> list[str]:
    if sort_mode == "By name (A→Z)":
        return sorted(df["Channel"].astype(str).unique().tolist())
    return (
        df.groupby("Channel", as_index=False)["ROI_total"]
        .sum()
        .sort_values("ROI_total", ascending=False)["Channel"]
        .astype(str)
        .tolist()
    )


def _roi_total_by_medium_chart(df: pd.DataFrame, channel_order: list[str]) -> go.Figure:
    teal = "#2AA6B5"
    grey = "#BFBFBF"

    agg = df.groupby("Channel", as_index=False)[["ROI_total"]].sum()
    agg["Channel"] = agg["Channel"].astype(str)
    top_channels = set(agg.sort_values("ROI_total", ascending=False).head(3)["Channel"].tolist())

    y = []
    colors = []
    for ch in channel_order:
        val = float(agg.loc[agg["Channel"] == ch, "ROI_total"].iloc[0])
        y.append(val)
        colors.append(teal if ch in top_channels else grey)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=channel_order,
            y=y,
            marker_color=colors,
            showlegend=False,
            hovertemplate="%{x}<br>Total ROI: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_traces(text=[f"{v:.2f}" for v in y], textposition="outside", cliponaxis=False)

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=50),
        xaxis=dict(title="", tickangle=0,tickfont=dict(size=17)),
        yaxis=dict(title="ROI (total)"),
    )
    return fig


def _roi_short_long_stacked_chart(df: pd.DataFrame, channel_order: list[str]) -> go.Figure:
    # Keep the same color family as your example
    c_long = "rgba(42, 166, 181, 0.35)"
    c_short = "rgba(42, 166, 181, 0.85)"

    agg = df.groupby("Channel", as_index=False)[["ROI_short", "ROI_long"]].sum()
    agg["Channel"] = agg["Channel"].astype(str)

    long_vals = [float(agg.loc[agg["Channel"] == ch, "ROI_long"].iloc[0]) for ch in channel_order]
    short_vals = [float(agg.loc[agg["Channel"] == ch, "ROI_short"].iloc[0]) for ch in channel_order]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=channel_order,
            y=long_vals,
            name="LONG-TERM",
            marker_color=c_long,
            hovertemplate="%{x}<br>Long-term ROI: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=channel_order,
            y=short_vals,
            name="SHORT-TERM",
            marker_color=c_short,
            hovertemplate="%{x}<br>Short-term ROI: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="stack",
        height=320,
        margin=dict(l=10, r=10, t=10, b=50),
        xaxis=dict(title="", tickangle=0,tickfont=dict(size=17)),
        yaxis=dict(title="ROI (split)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def show_view():
    _inject_css()

    st.title("Marketing ROI")
    st.markdown("**Efficiency = ROI by channel** (total) and its split into **Short-term** vs **Long-term**.")

    # Data (demo fallback)
    df = None
    if isinstance(st.session_state.get("efficiency_df"), pd.DataFrame):
        df = st.session_state["efficiency_df"].copy()
    if df is None:
        df = _roi_demo_data()

    required = {"Channel", "ROI_short", "ROI_long"}
    if not required.issubset(set(df.columns)):
        st.error(f"Missing required columns: {sorted(list(required))}")
        return

    if "ROI_total" not in df.columns:
        df["ROI_total"] = df["ROI_short"].astype(float) + df["ROI_long"].astype(float)

    # Controls
    st.subheader("Settings")
    col1, col2 = st.columns([1.2, 2.0])

    with col1:
        sort_mode = st.selectbox("Sort channels", ["By total ROI (desc)", "By name (A→Z)"], index=0)

    with col2:
        groups = df["Group"].astype(str).unique().tolist() if "Group" in df.columns else []
        if groups:
            selected_groups = st.multiselect("Groups", options=groups, default=groups)
            df_f = df[df["Group"].astype(str).isin(selected_groups)].copy()
        else:
            df_f = df.copy()

    if df_f.empty:
        st.warning("No data after filtering.")
        return

    # Build consistent channel order ONCE (used for both charts)
    channel_order = _order_channels(df_f, sort_mode)

    # 1) Total ROI per channel (FIRST)
    st.subheader("ROI by channel (total)")
    fig_total = _roi_total_by_medium_chart(df_f, channel_order)
    st.plotly_chart(fig_total, use_container_width=True)

    # 2) Short vs Long (SECOND) – consistent order and palette family
    st.subheader("Short-term vs Long-term (consistent with total ROI)")
    fig_split = _roi_short_long_stacked_chart(df_f, channel_order)
    st.plotly_chart(fig_split, use_container_width=True)

     # Table
    st.subheader("ROI table")

    cols = ["Channel", "ROI_short", "ROI_long", "ROI_total"]
    if "Group" in df_f.columns:
        cols = ["Group"] + cols

    df_table = (
        df_f[cols]
        .groupby(cols[: (2 if "Group" in df_f.columns else 1)], as_index=False)
        .sum()
        .sort_values("ROI_total", ascending=False)
    )

    # 👇 Renombrado SOLO para visualización
    rename_map = {
        "Group": "Marketing Channel",
        "Channel": "Channel",
        "ROI_short": "Short-term ROI",
        "ROI_long": "Long-term ROI",
        "ROI_total": "Full ROI",
    }

    df_table = df_table.rename(columns=rename_map)

    fmt = {
        "Short-term ROI": "{:.2f}",
        "Long-term ROI": "{:.2f}",
        "Full ROI": "{:.2f}",
    }

    st.dataframe(df_table.style.format(fmt), use_container_width=True)
