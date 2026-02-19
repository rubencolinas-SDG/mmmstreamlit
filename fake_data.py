import numpy as np
import pandas as pd

def _smooth(x, k=7):
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")

def _adstock_geometric(x, theta):
    y = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        y[i] = x[i] + (theta * y[i-1] if i > 0 else 0.0)
    return y

def build_fake_datastore(
    years=5,
    annual_sales_eur=200_000_000,
    mkt_share=0.20,
    tv_grps_per_year=900,
    seed=7,
):
    rng = np.random.default_rng(seed)
    weeks = years * 52
    dates = pd.date_range("2017-01-02", periods=weeks, freq="W-MON")
    year = dates.year
    t = np.arange(weeks)

    products = ["Product 1","Product 2","Product 3","Product 4","Product 5"]
    raw, betas, auto = {}, {}, {}

    # Target total ~200M€/year (=> ~4 M€/week)
    target_annual = 200_000_000

    for i,p in enumerate(products):
        # Baseline ~60% of total
        baseline_annual = target_annual * 0.60
        baseline_weekly = baseline_annual / 52
        baseline = baseline_weekly * (
            1
            + 0.015 * np.sin(2*np.pi*t/52)
            + rng.normal(0, 0.002, weeks)
        )
        baseline = _smooth(baseline, 5)

        # Negative seasonality (Christmas)
        neg_season = np.zeros(weeks)
        for w in [50,51,0,1]:
            neg_season[w::52] = -baseline_weekly * 0.25

        # Positive events
        pos_events = np.zeros(weeks)
        for y in np.unique(year):
            idx = np.where(year==y)[0]
            s = rng.integers(idx[10], idx[-10])
            pos_events[s:s+3] += baseline_weekly * 0.15

        # Media
        tv = np.zeros(weeks)
        for y in np.unique(year):
            idx = np.where(year==y)[0]
            base = np.zeros(len(idx))
            for a,b in [(6,10),(18,22),(34,38),(46,50)]:
                base[a:b] += rng.uniform(8,15)
            base *= tv_grps_per_year / base.sum()
            tv[idx] = base

        annual_mkt = target_annual * mkt_share
        weekly_mkt = annual_mkt / 52
        search = _smooth(weekly_mkt * 0.35 * rng.lognormal(0,0.15,weeks),5)
        social = _smooth(weekly_mkt * 0.25 * rng.lognormal(0,0.18,weeks),5)
        display = _smooth(weekly_mkt * 0.20 * rng.lognormal(0,0.20,weeks),5)

        price = 100 + np.cumsum(rng.normal(0.01,0.03,weeks))
        promo = (rng.random(weeks)<0.18).astype(int)
        temperature = 15 + 10*np.sin(2*np.pi*t/52)
        competitors = _smooth(rng.normal(0,1,weeks),7)

        auto[p] = {"tv":0.7,"search":0.35,"social":0.45,"display":0.3}

        betas[p] = {
            "tv": rng.uniform(4500,7000),
            "search": rng.uniform(0.04,0.06),
            "social": rng.uniform(0.03,0.05),
            "display": rng.uniform(0.02,0.04),
            "price": rng.uniform(-2000,-800),
            "promo": rng.uniform(200000,350000),
            "temperature": rng.uniform(5000,9000),
            "competitors": rng.uniform(-150000,-90000),
        }

        y = (
            baseline
            + neg_season
            + pos_events
            + _adstock_geometric(tv, auto[p]["tv"]) * betas[p]["tv"]
            + _adstock_geometric(search, auto[p]["search"]) * betas[p]["search"]
            + _adstock_geometric(social, auto[p]["social"]) * betas[p]["social"]
            + _adstock_geometric(display, auto[p]["display"]) * betas[p]["display"]
            + price * betas[p]["price"]
            + promo * betas[p]["promo"]
            + temperature * betas[p]["temperature"]
            + competitors * betas[p]["competitors"]
            + rng.normal(0, baseline_weekly * 0.03, weeks)
        )

        # scale exactly to target
        scale = target_annual / (pd.Series(y).groupby(year).sum().mean())
        y *= scale
        baseline *= scale
        neg_season *= scale
        pos_events *= scale

        raw[p] = pd.DataFrame({
            "date":dates,
            "y":y,
            "baseline":baseline,
            "neg_seasonality":neg_season,
            "pos_events":pos_events,
            "tv":tv,
            "search":search,
            "social":social,
            "display":display,
            "price":price,
            "promo":promo,
            "temperature":temperature,
            "competitors":competitors,
        })

    return {
        "products":products,
        "raw":raw,
        "media_cols":["tv","search","social","display"],
        "exog_cols":["price","promo","temperature","competitors","neg_seasonality","pos_events"],
        "betas":betas,
        "auto_adstock":auto,
    }
