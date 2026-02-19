# main/views/diagnosis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from fake_data import build_fake_datastore


# -----------------------------
# Channel list aligned with Efficiency / Boost
# (Display name, dataframe column key)
# -----------------------------
CHANNELS = [
    # Offline
    ("Corporate Social Media", "tv"),
    #("Outdoor", "outdoor"),
    ("Medical magazines", "radio"),
    #("Press", "press"),
    # Digital / Performance
    ("Sales representative", "paid_search"),
    #("Display", "display"),
    ("HCPs emailing", "retargeting"),
    # Social / Video
    ("Referrals & Content Generation", "facebook"),
    ("KOL campaigns", "youtube"),
    # Other
    ("HCPs platforms", "sampling"),
    ("Sponsorship", "sponsorship"),
    ("Webinars & Congresses", "affiliate"),
    #("Online Radio", "online_radio"),
]

COUNTRIES = ["United States", "Spain", "Portugal", "France", "United Kingdom"]

# -----------------------------
# Visual tuning (requested)
# -----------------------------
BASELINE_SCALE = 0.65             # baseline más bajo
NEG_SEASONALITY_SCALE = 0.45      # estacionalidad negativa más baja

# -----------------------------
# Shape tuning (requested)
# -----------------------------
SMOOTH_SPAN = 2                   # más alto = más suave (EWMA)
PRICE_VOLATILITY_SCALE = 1.35     # price oscila algo más
PRICE_WIGGLE_STRENGTH = 0.52      # 0..0.5 aprox; añade “wiggle” relativo a su std


def _smooth(x: np.ndarray, span: int = SMOOTH_SPAN) -> np.ndarray:
    """Suavizado exponencial: reduce picos y hace las curvas más suaves."""
    s = pd.Series(x.astype(float))
    return s.ewm(span=span, adjust=False).mean().to_numpy()


def _inject_demo_patterns(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """
    Fuerza patrones tipo Excel:
    - baseline: suave, mean-reverting
    - sampling: suave en tramos (piecewise trend)
    - facebook: flights con decaimiento
    - tv: casi todo 0 + bloque corto constante
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    # --- BASELINE suave (mean-reverting + deriva lenta) ---
    base = np.zeros(n, dtype=float)
    base[0] = 1_320_000.0
    drift = rng.normal(0, 1_200, size=n)  # deriva lenta
    eps = rng.normal(0, 8_000, size=n)    # ruido suave
    for t in range(1, n):
        base[t] = 0.5785 * base[t - 1] + 0.215 * 1_350_000.0 + drift[t] + eps[t]
    out["baseline"] = base

    # --- SAMPLING: tramos largos suaves ---
    k = max(6, n // 40)  # nº de tramos según longitud
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

    # --- TV: casi todo 0 + bloque corto constante ---
    tv = np.zeros(n, dtype=float)
    block_len = 4 if n < 120 else 5
    block_start = int(n * 0.65)
    block_start = min(block_start, n - block_len - 1)
    tv_value = 28.411
    tv[block_start:block_start + block_len] = tv_value
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


def _distinct_colors(n: int) -> list[str]:
    base = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC949",
        "#AF7AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B",
        "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
        "#393B79", "#637939", "#8C6D31", "#843C39", "#7B4173", "#5254A3",
        "#6B6ECF", "#9C9EDE", "#8CA252", "#B5CF6B", "#CEDB9C", "#BD9E39",
    ]
    if n <= len(base):
        return base[:n]
    colors = base[:]
    extra = n - len(base)
    for i in range(extra):
        hue = int((i * (360 / max(1, extra))) % 360)
        colors.append(f"hsl({hue},65%,50%)")
    return colors


def _stacked_area_absolute(
    df_plot: pd.DataFrame,
    components: list[str],
    negatives: set[str],
    y_title: str = "Value (in M€)",
) -> go.Figure:
    fig = go.Figure()

    cols = [c for c in components if c in df_plot.columns and c != "date"]

    mean_abs = {
        c: float(np.nanmean(np.abs(df_plot[c].astype(float).values)))
        for c in cols
    }

    pos = [c for c in cols if c not in negatives]
    neg = [c for c in cols if c in negatives]

    pos_sorted = sorted(pos, key=lambda c: mean_abs.get(c, 0.0), reverse=True)
    neg_sorted = sorted(neg, key=lambda c: mean_abs.get(c, 0.0), reverse=True)

    ordered_all = pos_sorted + neg_sorted
    colors = _distinct_colors(len(ordered_all))
    color_map = {c: colors[i] for i, c in enumerate(ordered_all)}

    for c in pos_sorted:
        fig.add_trace(
            go.Scatter(
                x=df_plot["date"],
                y=df_plot[c],
                mode="lines",
                name=c,
                stackgroup="positive",
                line=dict(width=1, color=color_map[c]),
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
                line=dict(width=1, color=color_map[c]),
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
    """
    Asigna un país dummy de forma determinística:
    - estable por día
    - opcionalmente depende también del producto para variar entre productos
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    # "día" como entero (YYYYMMDD) para que sea estable por día
    day_key = out["date"].dt.strftime("%Y%m%d").astype(int)

    # offset por producto (estable)
    prod_offset = sum(ord(c) for c in product) % len(countries) if product else 0

    idx = (day_key + prod_offset) % len(countries)
    out["country"] = idx.map(lambda i: countries[int(i)])

    return out


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
    # Top controls (use default widget labels)
    # -----------------------------
    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

    with c1:
        product = st.selectbox(
            "Product",
            options=data_store["products"],
            index=0,
        )

    df = data_store["raw"][product].copy()

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])

    # ✅ Inyecta patrones tipo Excel (baseline/sampling/facebook/tv)
    df = _inject_demo_patterns(df, seed=7)

    # ✅ Country dummy (estable)
    df = _assign_dummy_country(df, COUNTRIES)

    # Date bounds
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
        country_opt = st.selectbox(
            "Country",
            options=["All"] + COUNTRIES,
            index=0,
        )

    # Robust parsing of Streamlit date_input return value
    if isinstance(date_range, tuple) and len(date_range) == 2 and date_range[0] and date_range[1]:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_d, max_d

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    # Filter by date
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()

    # Filter by country
    if country_opt != "All":
        df = df[df["country"].astype(str) == country_opt].copy()

    if df.empty:
        st.warning("No data for the selected filters.")
        return

    betas = data_store["betas"][product]

    # Ensure all channels exist in dataframe (fallback to 0 if missing)
    for _, col in CHANNELS:
        if col not in df.columns:
            df[col] = 0.0

    # -----------------------------
    # Contributions in € (NO adstock now)
    # -----------------------------
    contrib = pd.DataFrame({"date": df["date"]})
    n = len(contrib)
    t = np.arange(n)

    # baseline + seasonal (scaled) + smooth
    baseline = df.get("baseline", 0.0).to_numpy() * BASELINE_SCALE
    neg_seas = df.get("neg_seasonality", 0.0).to_numpy() * NEG_SEASONALITY_SCALE
    pos_events = df.get("pos_events", 0.0).to_numpy()

    contrib["Baseline"] = _smooth(np.asarray(baseline))
    contrib["Negative seasonality"] = _smooth(np.asarray(neg_seas))
    contrib["Positive events"] = _smooth(np.asarray(pos_events))

    # Media contributions: raw * beta + smooth (no carryover/adstock)
    for name, col in CHANNELS:
        raw = df[col].to_numpy().astype(float) * float(betas.get(col, 0.0))
        contrib[name] = _smooth(np.asarray(raw))

    # Other drivers (smooth)
    promo = df.get("promo", 0.0).to_numpy() * float(betas.get("promo", 300000.0))
    temp = df.get("temperature", 0.0).to_numpy() * float(betas.get("temperature", 1000.0))
    comp = df.get("competitors", 0.0).to_numpy() * float(betas.get("competitors", -180000.0))

    contrib["Promotion"] = _smooth(np.asarray(promo))
    contrib["Temperature"] = _smooth(np.asarray(temp))
    contrib["Competitors"] = _smooth(np.asarray(comp))

    # Price: más oscilación (y menos suavizado)
    price_base = df.get("price", 0.0).to_numpy() * float(betas.get("price", -1500.0))
    price_more = np.asarray(price_base, dtype=float) * PRICE_VOLATILITY_SCALE

    price_std = float(np.nanstd(price_more)) if np.isfinite(np.nanstd(price_more)) else 0.0
    wiggle = np.sin(2 * np.pi * t / 10.0) * (price_std * PRICE_WIGGLE_STRENGTH)

    contrib["Price"] = price_more + wiggle

    # Convert to M€
    contrib_m = contrib.copy()
    for col in contrib_m.columns:
        if col != "date":
            contrib_m[col] = contrib_m[col] / 1e6

    # -----------------------------
    # Merge certain non-channel drivers into channel names (minimal, reversible)
    # -----------------------------
    MERGE_MAP = {
        "Promotion": "Referrals & Content Generation",
        "Temperature": "Corporate Social Media",
        "Positive events": "Webinars & Congresses",
        "Competitors": "Medical magazines",
        # "Price": "Sales representative",
        # "Negative seasonality": "Corporate Social Media",
    }

    for old, new in MERGE_MAP.items():
        if old in contrib_m.columns:
            if new not in contrib_m.columns:
                contrib_m[new] = 0.0
            contrib_m[new] = contrib_m[new].astype(float) + contrib_m[old].astype(float)
            contrib_m.drop(columns=[old], inplace=True)

    left, right = st.columns([2.2, 1], gap="large")

    with left:
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
            "Positive events",
            "Promotion",
            "Temperature",
            "Competitors",
            "Negative seasonality",
            "Price",
        ]

        negatives = {"Price", "Negative seasonality", "Competitors"}

        fig = _stacked_area_absolute(
            contrib_m,
            components=full_order,
            negatives=negatives,
            y_title="Value (in M€)",
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # --- Pie chart: split of total contribution by driver ---
        ordered_cols = [c for c in full_order if c in contrib_m.columns]

        if len(ordered_cols) == 0:
            st.info("No hay drivers disponibles para el desglose por ahora.")
        else:
            totals = contrib_m[ordered_cols].sum(numeric_only=True)
            pie_values = totals.abs()
            pie_values = pie_values[pie_values.fillna(0) > 0]

            st.subheader("Contribution split by driver")

            pie_fig = go.Figure(
                go.Pie(
                    labels=pie_values.index,
                    values=pie_values.values,
                    hole=0.4,
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
                st.write("- Since the price has experienced little variation over the historical period, sales contribution remains very stable, with limited visibility regarding price sensitivity.")
                st.write("- The impact of competition remains stable as there have been no strategic changes or major high-impact launches in the category.")
                st.write("- The marketing channels with the highest contribution have been sales representatives, delivering the best ROI over the past year (reflecting improved optimization of this channel’s strategy), along with other direct-to-HCP channels such as email, congresses, and webinars. Campaigns involving Key Opinion Leaders have been increasing in effectiveness since being integrated into the marketing mix.")
                st.write("- The seasonal component has not changed its pattern throughout the historical period. Reinforcing this impact through greater marketing activation is recommended.")
