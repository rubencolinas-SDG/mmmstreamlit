# main/views/boost.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def _inject_css():
    # Left-align buttons (same trick as Efficiency)
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


def show_view():
    _inject_css()

    st.title("MM Optimization")

    st.markdown(
        """
        Optimize the marketing budget allocation based on model insights.
        Choose to keep the current budget or test a new total budget and
        get recommended investment ranges by channel.
        """
    )

    # -----------------------------
    # Budget scenario (LEFT-aligned labels)
    # -----------------------------
    st.subheader("Budget scenario")

    st.markdown("**Budget option**")
    scenario = st.radio(
        label="Budget option",
        options=["Keep current budget", "Set new total budget"],
        horizontal=True,
        label_visibility="collapsed",
    )

    CURRENT_TOTAL = 40_000_000  # demo annual marketing budget

    if scenario == "Set new total budget":
        st.markdown("**New total annual budget (€)**")
        total_budget = st.number_input(
            label="New total annual budget (€)",
            min_value=10_000_000,
            max_value=80_000_000,
            step=1_000_000,
            value=CURRENT_TOTAL,
            format="%d",
            label_visibility="collapsed",
        )
    else:
        total_budget = CURRENT_TOTAL

    # -----------------------------
    # Constraints
    # -----------------------------
    st.subheader("Constraints")

    st.markdown("**Include constraints**")
    include_constraints = st.toggle(
        "Include constraints",
        value=False,
        label_visibility="collapsed",
    )

#    # Channels aligned with main/views/efficiency.py
#    all_channels = [
#        "TV",
#        "Outdoor",
#        "Radio",
#        "Press",
#        "Paid Search",
#        "Display",
#        "Retargeting",
#        "Facebook",
#        "YouTube",
#        "Sampling",
#        "Sponsorship",
#        "Affiliate",
#        "Online Radio",
#    ]
#
    all_channels = [
        "Corporate Social Media",
        "Medical magazines",
        "Sales representative",
        "HCPs emailing",
        "Referrals & Content Generation",
        "KOL campaigns",
        "HCPs platforms",
        "Sponsorship",
        "Webinars & Congresses",
    ]

    if include_constraints:
        st.markdown("**Include channels in the optimization**")
        selected_channels = st.multiselect(
            label="Include channels in the optimization",
            options=all_channels,
            default=all_channels,
            label_visibility="collapsed",
            help="Unselected channels are excluded from the recommended allocation.",
        )
        if len(selected_channels) == 0:
            st.warning("Select at least one channel to run the optimization.")
            return
    else:
        selected_channels = all_channels

    st.markdown("")
    if not st.button("Run optimization"):
        st.info("Select a budget scenario (and optional constraints) then run the optimization.")
        return

    # -----------------------------
    # Fake current budgets (shares)
    # -----------------------------
    # Baseline shares for the full channel set (must sum to 1.0)
#    base_shares = pd.Series(
#        {
#            # Offline
#            "TV": 0.30,
#            "Outdoor": 0.07,
#            "Radio": 0.06,
#            "Press": 0.04,
#            # Digital / Performance
#            "Paid Search": 0.16,
#            "Display": 0.05,
#            "Retargeting": 0.05,
#            # Social / Video
#            "Facebook": 0.08,
#            "YouTube": 0.09,
#            # Others
#            "Sampling": 0.03,
#            "Sponsorship": 0.03,
#            "Affiliate": 0.03,
#            "Online Radio": 0.01,
#        },
#        dtype=float,
#    )
    base_shares = pd.Series(
{
    # ↓↓↓ Ajuste demo: bajar CSM a ~1M€ con total 40M€ (share ~0.025)
    # (Revertible: vuelve a 0.30 si quieres el comportamiento anterior)
    "Corporate Social Media": 0.025,

    "Medical magazines": 0.08,
    "Sales representative": 0.18,
    "HCPs emailing": 0.09,
    "Referrals & Content Generation": 0.10,
    "KOL campaigns": 0.11,
    "HCPs platforms": 0.05,
    "Sponsorship": 0.04,
    "Webinars & Congresses": 0.05,
},
dtype=float,
)

    shares = base_shares[selected_channels].copy()
    shares = shares / shares.sum()
    current = total_budget * shares.to_numpy()

    df = pd.DataFrame({"Channel": selected_channels, "Current": current})

    # -----------------------------
    # Fake optimal ranges (multipliers vs current)
    # -----------------------------
#    ranges = {
#        # Offline
#        "TV": (0.85, 0.90),
#        "Outdoor": (0.95, 1.08),
#        "Radio": (0.98, 1.10),
#        "Press": (0.90, 1.05),
#        # Digital / Performance
#        "Paid Search": (1.03, 1.10),
#        "Display": (0.95, 1.10),
#        "Retargeting": (1.05, 1.18),
#        # Social / Video
#        "Facebook": (0.98, 1.12),
#        "YouTube": (1.05, 1.20),
#        # Others
#        "Sampling": (0.95, 1.15),
#        "Sponsorship": (0.92, 1.10),
#        "Affiliate": (1.02, 1.15),
#        "Online Radio": (0.95, 1.12),
#    }
#
    ranges = {
    "Corporate Social Media": (0.85, 0.90),
    "Medical magazines": (0.98, 1.10),
    "Sales representative": (1.03, 1.10),
    "HCPs emailing": (1.05, 1.18),
    "Referrals & Content Generation": (0.98, 1.12),
    "KOL campaigns": (1.05, 1.20),
    "HCPs platforms": (0.95, 1.15),
    "Sponsorship": (0.92, 1.10),
    "Webinars & Congresses": (1.02, 1.15),
    }

    df["Optimal min"] = df.apply(lambda r: r["Current"] * ranges[str(r["Channel"])][0], axis=1)
    df["Optimal max"] = df.apply(lambda r: r["Current"] * ranges[str(r["Channel"])][1], axis=1)

    def recommendation(row):
        if row["Current"] < row["Optimal min"]:
            return "Increase"
        if row["Current"] > row["Optimal max"]:
            return "Reduce"
        return "Keep"

    df["Recommendation"] = df.apply(recommendation, axis=1)

    order = [c for c in all_channels if c in selected_channels]
    df["Channel"] = pd.Categorical(df["Channel"].astype(str), categories=order, ordered=True)
    df = df.sort_values("Channel")

    # -----------------------------
    # Visualization (split: TV in GRPs, others in €)
    # Current = bar, Optimal = box (min-max)
    # -----------------------------
    st.subheader("Recommended investment ranges")

    #EUR_PER_GRP = 20_000  # demo conversion € -> GRPs

    TV_CHANNEL = "Sales representative"
    df_tv = df[df["Channel"].astype(str) == TV_CHANNEL].copy()
    df_other = df[df["Channel"].astype(str) != TV_CHANNEL].copy()
   # Palette: keep TV dark; others in the turquoise family
    colors = {ch: "#2AA6B5" for ch in all_channels}
    colors[TV_CHANNEL] = "#0B1F3B"

    col_left, col_right = st.columns([1, 2])

    # =========
    # TV (GRPs)
    # =========
    with col_left:
        st.markdown("**Sales representative (€)**")
        fig_tv = go.Figure()

        if not df_tv.empty:
            r = df_tv.iloc[0]

            # 👉 Ahora todo en euros, sin dividir
            tv_curr = float(r["Current"])
            tv_min = float(r["Optimal min"])
            tv_max = float(r["Optimal max"])
            tv_med = (tv_min + tv_max) / 2

            fig_tv.add_trace(
                go.Bar(
                    x=[TV_CHANNEL],
                    y=[tv_curr],
                    width=0.42,
                    marker_color=colors[TV_CHANNEL],
                    showlegend=False,
                    hovertemplate="{TV_CHANNEL}<br>Current: %{y:,.0f}€<extra></extra>",
                )
            )

            fig_tv.add_trace(
                go.Box(
                    x=[TV_CHANNEL],
                    q1=[tv_min],
                    median=[tv_med],
                    q3=[tv_max],
                    lowerfence=[tv_min],
                    upperfence=[tv_max],
                    boxpoints=False,
                    whiskerwidth=0.6,
                    marker_color="rgba(42, 166, 181, 0.35)",
                    fillcolor="rgba(42, 166, 181, 0.35)",
                    line=dict(width=1, color="rgba(42,166,181,0.35)"),
                    showlegend=False,
                    hovertemplate="{TV_CHANNEL}<br>Optimal: %{customdata}<extra></extra>",
                    customdata=[f"€{tv_min:,.0f}–€{tv_max:,.0f}"],
                )
            )

        fig_tv.update_layout(
            height=380,
            yaxis_title="Annual budget (€)",
            xaxis_title="",
            margin=dict(l=10, r=10, t=10, b=10),
            barmode="overlay",
        )

        st.plotly_chart(fig_tv, use_container_width=True)

    # =====================
    # Rest (euros)
    # =====================
    with col_right:
        st.markdown("**Digital / Other channels (€)**")
        fig_eur = go.Figure()

        if not df_other.empty:
            fig_eur.add_trace(
                go.Bar(
                    x=df_other["Channel"].astype(str),
                    y=df_other["Current"].astype(float),
                    width=0.42,
                    marker_color="rgba(42, 166, 181, 0.55)",
                    showlegend=False,
                    hovertemplate="%{x}<br>Current: %{y:,.0f}€<extra></extra>",
                )
            )

            for _, r in df_other.iterrows():
                ch = str(r["Channel"])
                o_min = float(r["Optimal min"])
                o_max = float(r["Optimal max"])
                o_med = (o_min + o_max) / 2

                fig_eur.add_trace(
                    go.Box(
                        x=[ch],
                        q1=[o_min],
                        median=[o_med],
                        q3=[o_max],
                        lowerfence=[o_min],
                        upperfence=[o_max],
                        boxpoints=False,
                        whiskerwidth=0.6,
                        marker_color=colors.get(ch, "#2AA6B5"),
                        fillcolor="rgba(42, 166, 181, 0.20)",
                        line=dict(width=2),
                        showlegend=False,
                        hovertemplate=f"{ch}<br>Optimal: %{{customdata}}<extra></extra>",
                        customdata=[f"€{o_min:,.0f}–€{o_max:,.0f}"],
                    )
                )

        fig_eur.update_layout(
            height=380,
            yaxis_title="Annual budget (€)",
            xaxis_title="",
            margin=dict(l=10, r=10, t=10, b=10),
            barmode="overlay",
        )

        st.plotly_chart(fig_eur, use_container_width=True)

    # -----------------------------
    # Table (TV in GRPs, rest in €)
    # -----------------------------
    st.subheader("Budget recommendation summary")

    df_table = df.copy()

    def _format_row(row):
        return pd.Series(
            {
                "Current": f'€{row["Current"]:,.0f}',
                "Optimal min": f'€{row["Optimal min"]:,.0f}',
                "Optimal max": f'€{row["Optimal max"]:,.0f}',
            }
        )

    formatted = df_table.apply(_format_row, axis=1)

    df_display = pd.concat(
        [
            df_table[["Channel", "Recommendation"]].reset_index(drop=True),
            formatted.reset_index(drop=True),
        ],
        axis=1,
    )

    df_display = df_display[["Channel", "Current", "Optimal min", "Optimal max", "Recommendation"]]
    st.dataframe(df_display, use_container_width=True)

    # -----------------------------
    # Saturation / optimization curves (demo) - consistent turquoise palette
    # -----------------------------
    # -----------------------------
    # Saturation / optimization curves (demo) - consistent turquoise palette
    # -----------------------------
    with st.expander("View saturation curves"):
        st.caption(
            "Demo saturation curves (Hill function). "
            "Shows diminishing returns and overlays Current + Optimal range."
        )

        TURQ_DARK = "rgba(42,166,181,0.95)"
        TURQ_MED = "rgba(42,166,181,0.65)"
        TURQ_LITE = "rgba(42,166,181,0.25)"
        RED_LITE = "rgba(220,0,0,0.12)"

        def hill(x, alpha, gamma, cap=1.0):
            x = np.maximum(x, 0.0)
            return cap * (x**gamma) / (x**gamma + alpha**gamma)

        def build_curve(channel: str):
            # Simple demo params per channel (all in € now)
            params = {
                "Corporate Social Media": dict(alpha=900_000, gamma=1.6, cap=1.0),
                "Medical magazines": dict(alpha=3_000_000, gamma=1.25, cap=1.0),
                "Sales representative": dict(alpha=12_000_000, gamma=1.40, cap=1.0),
                "HCPs emailing": dict(alpha=5_000_000, gamma=1.30, cap=1.0),
                "Referrals & Content Generation": dict(alpha=7_000_000, gamma=1.25, cap=1.0),
                "KOL campaigns": dict(alpha=8_000_000, gamma=1.30, cap=1.0),
                "HCPs platforms": dict(alpha=2_000_000, gamma=1.15, cap=1.0),
                "Sponsorship": dict(alpha=2_500_000, gamma=1.15, cap=1.0),
                "Webinars & Congresses": dict(alpha=4_500_000, gamma=1.30, cap=1.0),
            }

            p = params.get(channel, dict(alpha=1.0, gamma=1.3, cap=1.0))

            curr = float(df[df["Channel"].astype(str) == channel]["Current"].iloc[0])
            x = np.linspace(0, 1.6 * curr, 200)
            y = hill(x, **p) * 100
            return x, y

        ch_options = df["Channel"].astype(str).tolist()
        selected_curve_channel = st.selectbox("Channel", ch_options, index=0)

        unit = "€"

        # 👉 aquí definimos row correctamente
        row = df[df["Channel"].astype(str) == selected_curve_channel].iloc[0]
        curr_x = float(row["Current"])
        opt_min_x = float(row["Optimal min"])
        opt_max_x = float(row["Optimal max"])
        opt_mid_x = (opt_min_x + opt_max_x) / 2

        x, y = build_curve(selected_curve_channel)

        fig_curve = go.Figure()

        fig_curve.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=TURQ_DARK, width=3),
                hovertemplate=(
                    f"{selected_curve_channel}"
                    "<br>Investment: %{x:,.0f} " + unit +
                    "<br>Response: %{y:,.1f}"
                    "<extra></extra>"
                ),
            )
        )

        fig_curve.add_vrect(
            x0=opt_min_x,
            x1=opt_max_x,
            fillcolor=TURQ_LITE if opt_min_x <= curr_x <= opt_max_x else RED_LITE,
            line_width=0,
        )

        fig_curve.add_vline(
            x=curr_x,
            line_width=2,
            line_color=TURQ_MED,
            annotation_text="Current",
            annotation_position="top right",
        )

        fig_curve.add_trace(
            go.Scatter(
                x=[opt_mid_x],
                y=[np.interp(opt_mid_x, x, y)],
                mode="markers",
                marker=dict(size=10, color=TURQ_DARK),
                hovertemplate=f"Optimal mid: %{{x:,.0f}} {unit}<extra></extra>",
                showlegend=False,
            )
        )

        fig_curve.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=f"Investment ({unit})",
            yaxis_title="Response index",
            showlegend=False,
        )

        st.plotly_chart(fig_curve, use_container_width=True)
