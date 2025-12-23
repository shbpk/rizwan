import itertools
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats

# ----------------------------
# Helpers
# ----------------------------
def add_season(dt_series: pd.Series) -> pd.Series:
    """Meteorological seasons (DJF, MAM, JJA, SON)."""
    m = dt_series.dt.month
    return pd.Series(
        np.select(
            [
                m.isin([12, 1, 2]),
                m.isin([3, 4, 5]),
                m.isin([6, 7, 8]),
                m.isin([9, 10, 11]),
            ],
            ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Fall (SON)"],
            default="Unknown",
        ),
        index=dt_series.index,
    )

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-FDR adjustment (returns q-values)."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out

def safe_shapiro(x: np.ndarray):
    """Shapiro-Wilk is sensitive; only run on 3..5000 samples."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or len(x) > 5000:
        return np.nan, np.nan
    try:
        w, p = stats.shapiro(x)
        return w, p
    except Exception:
        return np.nan, np.nan

def group_test(df: pd.DataFrame, group_col: str, value_col: str = "n2og"):
    """Returns omnibus test result + pairwise table (MWU + BH-FDR)."""
    tmp = df[[group_col, value_col]].dropna()
    tmp[group_col] = tmp[group_col].astype(str)

    groups = [g for g, d in tmp.groupby(group_col) if len(d) >= 3]
    if len(groups) < 2:
        return None, None, "Not enough groups (need ≥2 groups with ≥3 points each)."

    # Build group arrays
    arrays = []
    ns = []
    for g in groups:
        x = tmp.loc[tmp[group_col] == g, value_col].values.astype(float)
        x = x[~np.isnan(x)]
        arrays.append(x)
        ns.append(len(x))

    # Omnibus (default nonparametric)
    if len(groups) == 2:
        u, p = stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
        omnibus = {
            "test": "Mann–Whitney U",
            "groups": groups,
            "statistic": float(u),
            "p_value": float(p),
        }
        pairwise_df = None
        return omnibus, pairwise_df, None

    # >2 groups: Kruskal–Wallis
    h, p = stats.kruskal(*arrays)
    omnibus = {
        "test": "Kruskal–Wallis H",
        "groups": groups,
        "statistic": float(h),
        "p_value": float(p),
    }

    # Pairwise MWU + BH-FDR (post-hoc)
    pairs = list(itertools.combinations(range(len(groups)), 2))
    rows = []
    pvals = []
    for i, j in pairs:
        gi, gj = groups[i], groups[j]
        xi, xj = arrays[i], arrays[j]
        u, pv = stats.mannwhitneyu(xi, xj, alternative="two-sided")
        pvals.append(pv)
        rows.append({"group_1": gi, "group_2": gj, "U": float(u), "p_raw": float(pv)})

    qvals = benjamini_hochberg(np.array(pvals))
    for r, qv in zip(rows, qvals):
        r["p_FDR(BH)"] = float(qv)

    pairwise_df = pd.DataFrame(rows).sort_values("p_FDR(BH)")
    return omnibus, pairwise_df, None


# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="N2OG Dashboard", layout="wide")
st.title("N2OG Difference Dataset Explorer")

st.caption(
    "Explore N2OG by plot / season / crop phase / planting / cover crop, "
    "filter by timeline, and run significance tests + correlation matrix."
)

# Data input
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    default_path = "Data for Dashboard Rizwan.csv"
    use_default = st.checkbox("Use default file in app folder", value=(uploaded is None))

@st.cache_data(show_spinner=False)
def load_data_from_filelike(filelike):
    return pd.read_csv(filelike)

@st.cache_data(show_spinner=False)
def load_data_from_path(path):
    return pd.read_csv(path)

if uploaded is not None:
    df = load_data_from_filelike(uploaded)
elif use_default:
    try:
        df = load_data_from_path(default_path)
    except Exception as e:
        st.error(
            f"Couldn't load '{default_path}'. Put the CSV next to app.py or upload it. Error: {e}"
        )
        st.stop()
else:
    st.info("Upload a CSV or enable 'Use default file in app folder'.")
    st.stop()

# Standardize + derived fields
df = df.copy()
df.columns = [c.strip() for c in df.columns]
if "date" not in df.columns:
    st.error("Expected a 'date' column in the CSV.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["season"] = add_season(df["date"])

# Make sure expected categorical columns exist
expected_cats = ["plot", "crop phase", "planting", "cc", "rep"]
for c in expected_cats:
    if c in df.columns:
        df[c] = df[c].astype(str)

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    min_d, max_d = df["date"].min(), df["date"].max()
    date_range = st.slider(
        "Timeline",
        min_value=min_d.to_pydatetime(),
        max_value=max_d.to_pydatetime(),
        value=(min_d.to_pydatetime(), max_d.to_pydatetime()),
    )

    def multisel(col, label):
        if col not in df.columns:
            return None
        opts = sorted(df[col].dropna().astype(str).unique().tolist())
        return st.multiselect(label, options=opts, default=opts)

    sel_plot = multisel("plot", "Plot")
    sel_crop_phase = multisel("crop phase", "Crop phase")
    sel_planting = multisel("planting", "Planting")
    sel_cc = multisel("cc", "Cover crop (cc)")
    sel_season = st.multiselect(
        "Season", options=sorted(df["season"].unique().tolist()),
        default=sorted(df["season"].unique().tolist())
    )

    y_scale = st.selectbox("Y scale", ["linear", "log"], index=0)
    show_points = st.checkbox("Show individual points on boxplots", value=True)

# Apply filters
mask = (df["date"] >= pd.to_datetime(date_range[0])) & (df["date"] <= pd.to_datetime(date_range[1]))
if sel_plot is not None:
    mask &= df["plot"].isin(sel_plot)
if sel_crop_phase is not None:
    mask &= df["crop phase"].isin(sel_crop_phase)
if sel_planting is not None:
    mask &= df["planting"].isin(sel_planting)
if sel_cc is not None:
    mask &= df["cc"].isin(sel_cc)
mask &= df["season"].isin(sel_season)

dff = df.loc[mask].copy()

# Summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(dff):,}")
c2.metric("Date min", str(dff["date"].min().date()) if len(dff) else "—")
c3.metric("Date max", str(dff["date"].max().date()) if len(dff) else "—")
c4.metric("N2OG mean", f"{dff['n2og'].mean():.3f}" if ("n2og" in dff.columns and len(dff)) else "—")

if len(dff) == 0:
    st.warning("No data after filtering. Widen the filters.")
    st.stop()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Plots", "Season / Phase Views", "Stats Tests", "Correlation"])

# ---- Tab 1: Plots
with tab1:
    st.subheader("Timeline views")

    # Daily mean line (if multiple points per day)
    daily = dff.groupby("date", as_index=False)["n2og"].agg(["mean", "median", "count"]).reset_index()
    fig = px.line(daily, x="date", y="mean", markers=True, title="Daily mean N2OG")
    fig.update_yaxes(type=y_scale)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scatter: N2OG vs DOY")
    if "doy" in dff.columns:
        fig2 = px.scatter(
            dff, x="doy", y="n2og",
            color="season", symbol="planting" if "planting" in dff.columns else None,
            hover_data=["plot", "crop phase", "cc", "rep"] if all(c in dff.columns for c in ["plot","crop phase","cc","rep"]) else None,
            title="N2OG vs Day-of-Year (colored by season)"
        )
        fig2.update_yaxes(type=y_scale)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No 'doy' column found for DOY scatter.")

# ---- Tab 2: Season / Phase Views
with tab2:
    st.subheader("Distributions by key groupings")

    group_col = st.selectbox(
        "Choose grouping for boxplot",
        options=[c for c in ["plot", "season", "crop phase", "planting", "cc", "rep"] if c in dff.columns],
        index=0
    )

    figb = px.box(
        dff, x=group_col, y="n2og",
        points="all" if show_points else False,
        title=f"N2OG distribution by {group_col}"
    )
    figb.update_yaxes(type=y_scale)
    st.plotly_chart(figb, use_container_width=True)

    st.subheader("Faceted view (optional)")
    facet = st.selectbox(
        "Facet by (optional)",
        options=["None"] + [c for c in ["season", "crop phase", "planting", "cc"] if c in dff.columns],
        index=0
    )
    if facet != "None":
        figf = px.box(
            dff, x=group_col, y="n2og",
            points=False,
            facet_col=facet,
            title=f"N2OG by {group_col} (faceted by {facet})"
        )
        figf.update_yaxes(type=y_scale)
        st.plotly_chart(figf, use_container_width=True)

# ---- Tab 3: Stats Tests
with tab3:
    st.subheader("Significance testing (group differences)")

    test_group = st.selectbox(
        "Test differences across:",
        options=[c for c in ["plot", "season", "crop phase", "planting", "cc", "rep"] if c in dff.columns],
        index=0
    )

    st.write("Normality check (Shapiro–Wilk) is shown for context; the app uses nonparametric tests by default.")

    # Shapiro per group
    sh_rows = []
    for g, dd in dff.groupby(test_group):
        w, p = safe_shapiro(dd["n2og"].values)
        sh_rows.append({"group": str(g), "n": int(dd["n2og"].notna().sum()), "W": w, "p": p})
    sh_df = pd.DataFrame(sh_rows).sort_values("p")
    st.dataframe(sh_df, use_container_width=True)

    omnibus, pairwise, err = group_test(dff, test_group, "n2og")
    if err:
        st.warning(err)
    else:
        st.markdown("### Omnibus test")
        st.json(omnibus)

        if pairwise is not None:
            st.markdown("### Post-hoc (pairwise Mann–Whitney U with BH-FDR)")
            st.dataframe(pairwise, use_container_width=True)

            sig = pairwise[pairwise["p_FDR(BH)"] <= 0.05].copy()
            st.markdown("### Significant pairs (q ≤ 0.05)")
            st.dataframe(sig if len(sig) else pd.DataFrame({"note": ["None at q≤0.05"]}), use_container_width=True)

# ---- Tab 4: Correlation
with tab4:
    st.subheader("Correlation matrix (Spearman)")
    numeric_cols = []
    for c in dff.columns:
        if pd.api.types.is_numeric_dtype(dff[c]):
            numeric_cols.append(c)

    # Also attempt to coerce a few known numeric-like IDs
    for c in ["plot", "rep", "doy"]:
        if c in dff.columns and c not in numeric_cols:
            coerced = pd.to_numeric(dff[c], errors="coerce")
            if coerced.notna().sum() > 0:
                dff[f"{c}__num"] = coerced
                numeric_cols.append(f"{c}__num")

    numeric_cols = [c for c in numeric_cols if c != "month" and c != "year"] + [c for c in ["month","year"] if c in dff.columns and pd.api.types.is_numeric_dtype(dff[c])]

    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for a correlation matrix.")
    else:
        corr = dff[numeric_cols].corr(method="spearman")
        figc = px.imshow(corr, text_auto=True, title="Spearman correlation matrix")
        st.plotly_chart(figc, use_container_width=True)

st.caption("Tip: Use filters to isolate specific plots/seasons/phases before running the stats tab.")
