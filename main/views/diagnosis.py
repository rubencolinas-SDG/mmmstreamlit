# main/views/diagnosis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fake_data import build_fake_datastore


# -----------------------------
# Channel list aligned with Efficiency / Boost
# (Display name, dataframe column key)
# -----------------------------
CHANNELS = [
    # Offline
    ("Corporate Social Media", "tv"),
    ("Medical magazines", "radio"),
    # Digital / Performance
    ("Sales representative", "paid_search"),
    ("HCPs emailing", "retargeting"),
    # Social / Video
    ("Referrals & Content Generation", "facebook"),
    ("KOL campaigns", "youtube"),
    # Other
    ("HCPs platforms", "sampling"),
    ("Sponsorship", "sponsorship"),
    ("Webinars & Congresses", "affiliate"),
]

COUNTRIES = ["United States", "Spain", "Portugal", "France", "United Kingdom"]

# -----------------------------
# Visual tuning (requested)
# -----------------------------
BASELINE_SCALE = 0.65
NEG_SEASONALITY_SCALE = 0.45

# -----------------------------
# Shape tuning (requested)
# -----------------------------
SMOOTH_SPAN = 2
PRICE_VOLATILITY_SCALE = 1.35
PRICE_WIGGLE_STRENGTH = 0.52


# -----------------------------
# Fixed colors per channel (consistent across charts)
# -----------------------------
def _build_color_map() -> dict[str, str]:
    # Plotly-esque qualitative palette (fixed ordering)
    palette = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC949", "#AF7AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
        "#17BECF", "#9467BD"
    ]
    names = [name for name, _ in CHANNELS]
    return {names[i]: palette[i % len(palette)] for i in range(len(names))}


COLOR_MAP = _build_color_map()


def _smooth(x: np.ndarray, span: int = SMOOTH_SPAN) -> np.ndarray:
    s = pd.Series(x.astype(float))
    return s.ewm(span=span, adjust=False).mean().to_numpy()


def _inject_demo_patterns(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    # --- BASELINE suave ---
    base = np.zeros(n, dtype=float)
    base[0] = 1_320_000.0
    drift = rng.normal(0, 1_200, size=n)
    eps = rng.normal(0, 8_000, size=n)
    for t in range(1, n):
        base[t] = 0.5785 * base[t - 1] + 0.215 * 1_350_000.0 + drift[t] + eps[t]
    out["baseline"] = base

    # --- SAMPLING: tramos largos suaves ---
    k = max(6, n // 40)
    knots_x = np.linspace(0, n - 1, k).astype(int)
    levels = rng.uniform(280_000, 460_000, size=k)
    for i in range(1, k):
        levels[i] = 0.75 * levels[i - 1] + 0.25 * levels[i]
    sampling = np.interp(np.arange(n), knots_x, levels)
    sampling = pd.Series(sampling).rolling(5, center=True, min_periods=1).mean().to_numpy()
    out["sampling"] = sampling

    # --- FACEBOOK: flights con decaimiento ---
    fb = np.zeros(n, dtype=float)
    n_flights = 4 if n >= 120 else 3
    starts = rng.integers(low=5, high=max(6, n - 25), size=n_flights)
    starts = np.sort(starts)

    for s in starts:
        dur = int(rng.integers(4, 10))
        amp = float(rng.uniform(60_000, 220_000))
        decay = float(rng.uniform(0.65, 0.85))
        end = min(n, s + dur)
        fb[s:end] += amp

        tail_len = min(n - end, 18)
        for i in range(tail_len):
            fb[end + i] += amp * (decay ** (i + 1))

    fb = pd.Series(fb).rolling(3, center=True, min_periods=1).mean().to_numpy()
    out["facebook"] = fb

    # --- TV: casi todo 0 + bloque corto ---
    tv = np.zeros(n, dtype=float)
    block_len = 4 if n < 120 else 5
    block_start = int(n * 0.65)
    block_start = min(block_start, n - block_len - 1)
    tv_value = 28.411
    tv[block_start : block_start + block_len] = tv_value
    out["tv"] = tv

    return out


def _diag_numbers(product: str) -> dict:
    maps = {
        "Product 1": {"r2": 0.91, "mape": 6.7, "dw": 1.96, "cv": "High", "vif": "Low"},
        "Product 2": {"r2": 0.89, "mape": 7.5, "dw": 1.88, "cv": "Medium", "vif": "Medium"},
        "Product 3": {"r2": 0.92, "mape": 6.1, "dw": 2.01, "cv": "High", "vif": "Low"},
        "Product 4": {"r2": 0.68, "mape": 8.0, "dw": 1.83, "cv": "Medium", "vif": "Medium"},
        "Product 5": {"r2": 0.90, "mape": 6.4, "dw": 1.94, "cv": "High", "vif": "Low"},
    }
    return maps.get(product, {"r2": 0.90, "mape": 7.0, "dw": 1.95, "cv": "High", "vif": "Low"})


def _stacked_area_absolute(
    df_plot: pd.DataFrame,
    components: list[str],
    negatives: set[str],
    y_title: str = "Value (in M€)",
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    fig = go.Figure()

    cols = [c for c in components if c in df_plot.columns and c != "date"]
    if not cols:
        return fig

    mean_abs = {c: float(np.nanmean(np.abs(df_plot[c].astype(float).values))) for c in cols}
    pos = [c for c in cols if c not in negatives]
    neg = [c for c in cols if c in negatives]

    pos_sorted = sorted(pos, key=lambda c: mean_abs.get(c, 0.0), reverse=True)
    neg_sorted = sorted(neg, key=lambda c: mean_abs.get(c, 0.0), reverse=True)

    def _c(name: str) -> str | None:
        if color_map and name in color_map:
            return color_map[name]
        return None

    for c in pos_sorted:
        fig.add_trace(
            go.Scatter(
                x=df_plot["date"],
                y=df_plot[c],
                mode="lines",
                name=c,
                stackgroup="positive",
                line=dict(width=1, color=_c(c)),
                opacity=0.95,
                hovertemplate="%{x|%b %Y}<br>%{fullData.name}: %{y:,.2f} M€<extra></extra>",
            )
        )

    for c in neg_sorted:
        fig.add_trace(
            go.Scatter(
                x=df_plot["date"],
                y=df_plot[c],
                mode="lines",
                name=c,
                stackgroup="negative",
                line=dict(width=1, color=_c(c)),
                opacity=0.95,
                hovertemplate="%{x|%b %Y}<br>%{fullData.name}: %{y:,.2f} M€<extra></extra>",
            )
        )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            x=0,
            title="",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        xaxis_title="",
        yaxis_title=y_title,
        hovermode="x unified",
    )
    fig.update_yaxes(zeroline=True, zerolinewidth=1)
    return fig


def _assign_dummy_country(df: pd.DataFrame, countries: list[str], product: str = "") -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    day_key = out["date"].dt.strftime("%Y%m%d").astype(int)
    prod_offset = sum(ord(c) for c in product) % len(countries) if product else 0

    idx = (day_key + prod_offset) % len(countries)
    out["country"] = idx.map(lambda i: countries[int(i)])
    return out


def _get_sales_series(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "sales", "Sales", "y", "Y", "target", "Target",
        "revenue", "Revenue", "net_sales", "Net Sales", "value", "Value",
    ]
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s

    base = pd.to_numeric(df.get("baseline", 0.0), errors="coerce").fillna(0.0)
    promo = pd.to_numeric(df.get("promo", 0.0), errors="coerce").fillna(0.0) * 0.25
    tv = pd.to_numeric(df.get("tv", 0.0), errors="coerce").fillna(0.0) * 0.10
    fb = pd.to_numeric(df.get("facebook", 0.0), errors="coerce").fillna(0.0) * 0.08
    return base + promo + tv + fb


def _build_dummy_monthly_investment_by_channel(
    month_starts: pd.Series,
    product: str,
    country: str,
    channel_names: list[str],
) -> pd.DataFrame:
    """
    Inversión dummy mensual por canal (stacked):
    - Total mensual aprox 2.5M–4.5M (anual ~40M)
    - Determinístico por (mes, producto, país)
    - Reparte el total por shares (con un poco de variación suave)
    """
    # Shares base (suma ~1)
    base_shares = {
        "Corporate Social Media": 0.30,
        "Medical magazines": 0.08,
        "Sales representative": 0.18,
        "HCPs emailing": 0.09,
        "Referrals & Content Generation": 0.10,
        "KOL campaigns": 0.11,
        "HCPs platforms": 0.05,
        "Sponsorship": 0.04,
        "Webinars & Congresses": 0.05,
    }
    # Fallback si faltara algún canal
    for ch in channel_names:
        base_shares.setdefault(ch, 1.0 / max(1, len(channel_names)))

    rows = []
    month_idx = pd.to_datetime(month_starts)

    for d in month_idx:
        key = f"{d.strftime('%Y-%m')}-{product}-{country}"
        seed = abs(hash(key)) % (2**32)
        rng = np.random.default_rng(seed)

        total = rng.uniform(2_500_000, 4_500_000)
        season = 1.0 + 0.08 * np.sin(2 * np.pi * (d.month / 12.0))
        total *= season

        # shares con pequeña variación y renormalización
        raw = np.array([base_shares[ch] for ch in channel_names], dtype=float)
        noise = rng.normal(0.0, 0.04, size=len(channel_names))  # +-4% aprox
        raw = np.clip(raw * (1.0 + noise), 0.001, None)
        raw = raw / raw.sum()

        vals = total * raw
        row = {"month": d}
        row.update({ch: float(vals[i]) for i, ch in enumerate(channel_names)})
        rows.append(row)

    return pd.DataFrame(rows)


def show_view() -> None:
    if "data_store" not in st.session_state:
        st.session_state["data_store"] = build_fake_datastore(
            years=5,
            annual_sales_eur=200_000_000,
            mkt_share=0.20,
            tv_grps_per_year=900,
        )

    data_store = st.session_state["data_store"]

    st.title("Sales Drivers")

    # -----------------------------
    # Top controls (default widget labels)
    # -----------------------------
    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

    with c1:
        product = st.selectbox("Product", options=data_store["products"], index=0)

    df = data_store["raw"][product].copy()
    df["date"] = pd.to_datetime(df["date"])

    # Patterns + dummy country
    df = _inject_demo_patterns(df, seed=7)
    df = _assign_dummy_country(df, COUNTRIES, product=product)

    min_d = df["date"].min().date()
    max_d = df["date"].max().date()

    with c2:
        date_range = st.date_input(
            "Date range",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
        )

    with c3:
        country_opt = st.selectbox("Country", options=["All"] + COUNTRIES, index=0)

    # Parse date_range robustly
    if isinstance(date_range, tuple) and len(date_range) == 2 and date_range[0] and date_range[1]:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_d, max_d

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    # Filter
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
    if country_opt != "All":
        df = df[df["country"].astype(str) == country_opt].copy()

    if df.empty:
        st.warning("No data for the selected filters.")
        return

    betas = data_store["betas"][product]

    # Ensure channels exist
    for _, col in CHANNELS:
        if col not in df.columns:
            df[col] = 0.0

    # -----------------------------
    # Contributions (M€)
    # -----------------------------
    contrib = pd.DataFrame({"date": df["date"]})
    n = len(contrib)
    t = np.arange(n)

    baseline = df.get("baseline", 0.0).to_numpy() * BASELINE_SCALE
    neg_seas = df.get("neg_seasonality", 0.0).to_numpy() * NEG_SEASONALITY_SCALE
    pos_events = df.get("pos_events", 0.0).to_numpy()

    contrib["Baseline"] = _smooth(np.asarray(baseline))
    contrib["Negative seasonality"] = _smooth(np.asarray(neg_seas))
    contrib["Positive events"] = _smooth(np.asarray(pos_events))

    for name, col in CHANNELS:
        raw = df[col].to_numpy().astype(float) * float(betas.get(col, 0.0))
        contrib[name] = _smooth(np.asarray(raw))

    promo = df.get("promo", 0.0).to_numpy() * float(betas.get("promo", 300000.0))
    temp = df.get("temperature", 0.0).to_numpy() * float(betas.get("temperature", 1000.0))
    comp = df.get("competitors", 0.0).to_numpy() * float(betas.get("competitors", -180000.0))

    contrib["Promotion"] = _smooth(np.asarray(promo))
    contrib["Temperature"] = _smooth(np.asarray(temp))
    contrib["Competitors"] = _smooth(np.asarray(comp))

    price_base = df.get("price", 0.0).to_numpy() * float(betas.get("price", -1500.0))
    price_more = np.asarray(price_base, dtype=float) * PRICE_VOLATILITY_SCALE
    price_std = float(np.nanstd(price_more)) if np.isfinite(np.nanstd(price_more)) else 0.0
    wiggle = np.sin(2 * np.pi * t / 10.0) * (price_std * PRICE_WIGGLE_STRENGTH)
    contrib["Price"] = price_more + wiggle

    contrib_m = contrib.copy()
    for col in contrib_m.columns:
        if col != "date":
            contrib_m[col] = contrib_m[col] / 1e6

    # Merge non-channel into channel names (so values get summed)
    MERGE_MAP = {
        "Promotion": "Referrals & Content Generation",
        "Temperature": "Corporate Social Media",
        "Positive events": "Webinars & Congresses",
        "Competitors": "Medical magazines",
    }
    for old, new in MERGE_MAP.items():
        if old in contrib_m.columns:
            if new not in contrib_m.columns:
                contrib_m[new] = 0.0
            contrib_m[new] = contrib_m[new].astype(float) + contrib_m[old].astype(float)
            contrib_m.drop(columns=[old], inplace=True)

    # Layout columns (keep option B)
    left, right = st.columns([2.2, 1], gap="large")

    with left:
        # =========================================================
        # 1) Monthly investment vs sales (STACKED bars by channel)
        # =========================================================
        st.subheader("Monthly investment vs sales")

        df_sales = df.copy()
        df_sales["sales"] = _get_sales_series(df_sales)
        df_sales["month"] = df_sales["date"].dt.to_period("M").dt.to_timestamp()

        monthly_sales = df_sales.groupby("month", as_index=False)["sales"].sum()
        month_starts = monthly_sales["month"]

        inv_country_key = country_opt if country_opt != "All" else "All"
        channel_names = [name for name, _ in CHANNELS]

        df_inv = _build_dummy_monthly_investment_by_channel(
            month_starts=month_starts,
            product=product,
            country=inv_country_key,
            channel_names=channel_names,
        )

        df_month = monthly_sales.merge(df_inv, on="month", how="left")

        fig_inv = make_subplots(specs=[[{"secondary_y": True}]])

        # Stacked bars (one trace per channel)
        for ch in channel_names:
            fig_inv.add_trace(
                go.Bar(
                    x=df_month["month"],
                    y=df_month[ch],
                    name=ch,
                    marker_color=COLOR_MAP.get(ch, None),
                    hovertemplate="%{x|%b %Y}<br>" + ch + ": €%{y:,.0f}<extra></extra>",
                ),
                secondary_y=False,
            )

        # Sales line
        fig_inv.add_trace(
            go.Scatter(
                x=df_month["month"],
                y=df_month["sales"],
                name="Sales",
                mode="lines+markers",
                hovertemplate="%{x|%b %Y}<br>Sales: %{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )

        fig_inv.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                itemclick="toggle",
                itemdoubleclick="toggleothers",
            ),
            hovermode="x unified",
            barmode="stack",
            xaxis_title="",
        )
        fig_inv.update_yaxes(title_text="Investment (€)", secondary_y=False)
        fig_inv.update_yaxes(title_text="Sales", secondary_y=True)
        fig_inv.update_xaxes(tickangle=0, tickfont=dict(size=13))

        st.plotly_chart(fig_inv, use_container_width=True, config={"displayModeBar": False})

        # =========================================================
        # 2) Contribution decomposition (stacked area)
        # =========================================================
        st.subheader("Contribution decomposition")

        full_order = [
            "Baseline",
            "Sales representative",
            "Referrals & Content Generation",
            "KOL campaigns",
            "HCPs emailing",
            "HCPs platforms",
            "Webinars & Congresses",
            "Sponsorship",
            "Corporate Social Media",
            "Medical magazines",
            "Negative seasonality",
            "Price",
        ]

        negatives = {"Price", "Negative seasonality", "Competitors"}  # "Competitors" may be merged away

        fig = _stacked_area_absolute(
            contrib_m,
            components=full_order,
            negatives=negatives,
            y_title="Value (in M€)",
            color_map=COLOR_MAP,  # <- same colors as investment bars
        )
        fig.update_xaxes(tickangle=0, tickfont=dict(size=13))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # =========================================================
        # 3) Donut (same colors)
        # =========================================================
        ordered_cols = [c for c in full_order if c in contrib_m.columns]
        if len(ordered_cols) == 0:
            st.info("No hay drivers disponibles para el desglose por ahora.")
        else:
            totals = contrib_m[ordered_cols].sum(numeric_only=True)
            pie_values = totals.abs()
            pie_values = pie_values[pie_values.fillna(0) > 0]

            st.subheader("Contribution split by driver")

            pie_colors = [COLOR_MAP.get(lbl, None) for lbl in pie_values.index]

            pie_fig = go.Figure(
                go.Pie(
                    labels=pie_values.index,
                    values=pie_values.values,
                    hole=0.4,
                    marker=dict(colors=pie_colors),
                    hovertemplate="%{label}<br>Contribution: %{value:.2f} M€<extra></extra>",
                )
            )

            pie_fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    x=0,
                    title_text="",
                    itemclick="toggle",
                    itemdoubleclick="toggleothers",
                ),
            )

            st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.subheader("Model diagnosis")

        if "show_diag" not in st.session_state:
            st.session_state["show_diag"] = False

        st.button(
            "Run model diagnosis",
            use_container_width=True,
            on_click=lambda: st.session_state.update({"show_diag": True}),
        )

        if st.session_state["show_diag"]:
            nums = _diag_numbers(product)
            tabs = st.tabs(["Statistical diagnosis", "Main Highlights"])

            with tabs[0]:
                st.markdown("**Fit & errors**")
                st.write(f"- **R²:** {nums['r2']:.2f} (weekly)")
                st.write(f"- **MAPE:** {nums['mape']:.1f}% (backtest)")
                st.write(f"- **Durbin–Watson:** {nums['dw']:.2f}")
                st.markdown("**Robustness**")
                st.write(f"- **Coefficient stability (CV):** {nums['cv']}")
                st.write(f"- **Multicollinearity (VIF proxy):** {nums['vif']}")

            with tabs[1]:
                st.markdown("**What the decomposition suggests (demo)**")
                st.write(
                    "- Since the price has experienced little variation over the historical period, "
                    "sales contribution remains very stable, with limited visibility regarding price sensitivity."
                )
                st.write(
                    "- The impact of competition remains stable as there have been no strategic changes or major "
                    "high-impact launches in the category."
                )
                st.write(
                    "- The marketing channels with the highest contribution have been sales representatives, delivering "
                    "the best ROI over the past year (reflecting improved optimization of this channel’s strategy), "
                    "along with other direct-to-HCP channels such as email, congresses, and webinars. Campaigns involving "
                    "Key Opinion Leaders have been increasing in effectiveness since being integrated into the marketing mix."
                )
                st.write(
                    "- The seasonal component has not changed its pattern throughout the historical period. "
                    "Reinforcing this impact through greater marketing activation is recommended."
                )
