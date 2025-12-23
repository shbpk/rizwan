import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd



st.set_page_config(page_title="N2O Explorer Dashboard", layout="wide")
st.title("N2O Dataset Explorer")
st.caption("Upload CSV → explore by time, season, crop phase, planting, plot num.")


def add_season(dt_series: pd.Series) -> pd.Series:
    """Meteorological seasons (DJF, MAM, JJA, SON)."""
    m = dt_series.dt.month
    return pd.Series(
        np.select(
            [m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8]), m.isin([9, 10, 11])],
            ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Fall (SON)"],
            default="Unknown",
        ),
        index=dt_series.index,
    )

def bh_fdr(pvals):
    """Benjamini–Hochberg FDR adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj_ranked = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj_ranked[i] = prev

    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj_ranked, 0, 1)
    return out

def pairwise_mwu(df, group_col, value_col):
    """Pairwise Mann–Whitney U with BH-FDR correction."""
    groups = sorted([g for g in df[group_col].dropna().unique()])
    rows = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            x = df.loc[df[group_col] == g1, value_col].dropna().values
            y = df.loc[df[group_col] == g2, value_col].dropna().values
            if len(x) < 2 or len(y) < 2:
                continue
            U, p = stats.mannwhitneyu(x, y, alternative="two-sided")  # Mann & Whitney, 1947
            rows.append((g1, g2, U, p))

    if not rows:
        return pd.DataFrame(columns=["group1", "group2", "U", "p", "p_adj(BH)"])

    out = pd.DataFrame(rows, columns=["group1", "group2", "U", "p"])
    out["p_adj(BH)"] = bh_fdr(out["p"].values)  # Benjamini & Hochberg, 1995
    return out.sort_values("p_adj(BH)")

def effect_size_anova_eta_sq(anova_tbl):
    """Eta-squared (η²) from ANOVA table (one-way)."""
    ss_between = float(anova_tbl["sum_sq"].iloc[0])
    ss_total = float(anova_tbl["sum_sq"].sum())
    return ss_between / ss_total if ss_total > 0 else np.nan

def effect_size_kruskal_epsilon_sq(H, n, k):
    """Epsilon-squared (ε²) for Kruskal–Wallis."""
    denom = (n - k)
    if denom <= 0:
        return np.nan
    return (H - k + 1) / denom

def require_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    return missing


st.sidebar.header("1) Upload data")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload CSV Please")
    st.stop()

@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)

df = load_data(uploaded_file)

st.sidebar.success("File loaded!")


st.sidebar.header("2) Parse settings")

all_cols = list(df.columns)

date_col = st.sidebar.selectbox("Date column", options=all_cols, index=all_cols.index("date") if "date" in all_cols else 0)
value_col = st.sidebar.selectbox("Value column (n2og)", options=all_cols, index=all_cols.index("n2og") if "n2og" in all_cols else 0)

plot_col = st.sidebar.selectbox("Plot column", options=["(none)"] + all_cols, index=(["(none)"] + all_cols).index("plot") if "plot" in all_cols else 0)
rep_col = st.sidebar.selectbox("Rep column", options=["(none)"] + all_cols, index=(["(none)"] + all_cols).index("rep") if "rep" in all_cols else 0)
crop_phase_col = st.sidebar.selectbox("Crop phase column", options=["(none)"] + all_cols, index=(["(none)"] + all_cols).index("crop phase") if "crop phase" in all_cols else 0)
planting_col = st.sidebar.selectbox("Planting column", options=["(none)"] + all_cols, index=(["(none)"] + all_cols).index("planting") if "planting" in all_cols else 0)
cc_col = st.sidebar.selectbox("CC column", options=["(none)"] + all_cols, index=(["(none)"] + all_cols).index("cc") if "cc" in all_cols else 0)
doy_col = st.sidebar.selectbox("DOY column", options=["(none)"] + all_cols, index=(["(none)"] + all_cols).index("doy") if "doy" in all_cols else 0)

w = df.copy()

w["__date__"] = pd.to_datetime(w[date_col], errors="coerce")
w["__value__"] = pd.to_numeric(w[value_col], errors="coerce")

if plot_col != "(none)":
    w["__plot__"] = pd.to_numeric(w[plot_col], errors="coerce")
if rep_col != "(none)":
    w["__rep__"] = pd.to_numeric(w[rep_col], errors="coerce")
if doy_col != "(none)":
    w["__doy__"] = pd.to_numeric(w[doy_col], errors="coerce")

def as_cat(src, target):
    if src != "(none)" and src in w.columns:
        w[target] = w[src].astype(str).str.strip()
as_cat(crop_phase_col, "__crop_phase__")
as_cat(planting_col, "__planting__")
as_cat(cc_col, "__cc__")

w = w.dropna(subset=["__date__", "__value__"])
w["__season__"] = add_season(w["__date__"])
w["__year__"] = w["__date__"].dt.year.astype("Int64")
w["__month__"] = w["__date__"].dt.month.astype("Int64")


st.sidebar.header("3) Filters")

min_date = w["__date__"].min()
max_date = w["__date__"].max()

date_range = st.sidebar.slider(
    "Date range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
)

f = w[(w["__date__"] >= pd.to_datetime(date_range[0])) & (w["__date__"] <= pd.to_datetime(date_range[1]))].copy()

def multi_filter(df_in, col, label):
    if col not in df_in.columns:
        return df_in
    opts = sorted(df_in[col].dropna().unique())
    if len(opts) == 0:
        return df_in
    sel = st.sidebar.multiselect(label, opts, default=opts)
    return df_in[df_in[col].isin(sel)]

f = multi_filter(f, "__season__", "Season")
f = multi_filter(f, "__crop_phase__", "Crop phase")
f = multi_filter(f, "__planting__", "Planting")
f = multi_filter(f, "__cc__", "Cover crop (cc)")
if "__plot__" in f.columns:
    plot_opts = sorted(f["__plot__"].dropna().unique())
    sel_plot = st.sidebar.multiselect("Plot", plot_opts, default=plot_opts[: min(30, len(plot_opts))])
    f = f[f["__plot__"].isin(sel_plot)]
if "__rep__" in f.columns:
    rep_opts = sorted(f["__rep__"].dropna().unique())
    sel_rep = st.sidebar.multiselect("Rep", rep_opts, default=rep_opts)
    f = f[f["__rep__"].isin(sel_rep)]


c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (filtered)", f"{len(f):,}")
c2.metric("Mean", f"{f['__value__'].mean():.4g}")
c3.metric("Median", f"{f['__value__'].median():.4g}")
c4.metric("Std dev", f"{f['__value__'].std():.4g}")

tabs = st.tabs(["Plots", "Season & Phase", "Significance Tests", "Correlation", "Data"])


with tabs[0]:
    st.subheader("Time series and distributions")

    daily = f.groupby("__date__", as_index=False)["__value__"].agg(["mean", "median", "count"]).reset_index()
    fig_ts = px.line(daily, x="__date__", y="mean", title="Daily mean (filtered)")
    st.plotly_chart(fig_ts, use_container_width=True)

    candidates = []
    if "__plot__" in f.columns: candidates.append("__plot__")
    candidates += ["__season__"]
    if "__crop_phase__" in f.columns: candidates.append("__crop_phase__")
    if "__planting__" in f.columns: candidates.append("__planting__")
    if "__cc__" in f.columns: candidates.append("__cc__")
    if "__rep__" in f.columns: candidates.append("__rep__")

    group_col = st.selectbox("Group distributions by", options=candidates, index=0)

    chart_type = st.radio("Distribution chart", options=["Box", "Violin"], horizontal=True)

    if chart_type == "Box":
        fig = px.box(f, x=group_col, y="__value__", points="outliers", title=f"Value by {group_col}")
    else:
        fig = px.violin(f, x=group_col, y="__value__", box=True, points="outliers", title=f"Value by {group_col}")

    fig.update_layout(xaxis_title=group_col, yaxis_title=value_col)
    st.plotly_chart(fig, use_container_width=True)


with tabs[1]:
    st.subheader("Faceted views")

    facet_row_opts = ["__season__"]
    if "__crop_phase__" in f.columns: facet_row_opts.append("__crop_phase__")
    if "__planting__" in f.columns: facet_row_opts.append("__planting__")
    if "__cc__" in f.columns: facet_row_opts.append("__cc__")

    facet_col_opts = []
    if "__crop_phase__" in f.columns: facet_col_opts.append("__crop_phase__")
    if "__planting__" in f.columns: facet_col_opts.append("__planting__")
    if "__cc__" in f.columns: facet_col_opts.append("__cc__")
    if "__rep__" in f.columns: facet_col_opts.append("__rep__")

    facet_row = st.selectbox("Facet row", options=facet_row_opts, index=0)
    facet_col = st.selectbox("Facet color", options=facet_col_opts if facet_col_opts else ["__season__"], index=0)

    agg = f.groupby(["__date__", facet_row, facet_col], as_index=False)["__value__"].mean().rename(columns={"__value__": "mean_value"})
    fig_fac = px.line(
        agg,
        x="__date__",
        y="mean_value",
        color=facet_col,
        facet_row=facet_row,
        title="Mean value over time (faceted)",
    )
    st.plotly_chart(fig_fac, use_container_width=True)

    st.markdown("**Season summary**")
    seas_sum = f.groupby("__season__")["__value__"].agg(["count", "mean", "median", "std"]).reset_index()
    st.dataframe(seas_sum, use_container_width=True)


with tabs[2]:
    st.subheader("Significance tests for group differences")

    test_factors = ["__season__"]
    if "__crop_phase__" in f.columns: test_factors.append("__crop_phase__")
    if "__planting__" in f.columns: test_factors.append("__planting__")
    if "__cc__" in f.columns: test_factors.append("__cc__")
    if "__plot__" in f.columns: test_factors.append("__plot__")
    if "__rep__" in f.columns: test_factors.append("__rep__")

    factor = st.selectbox("Factor to test", options=test_factors, index=0)

    test_mode = st.radio(
        "Test selection",
        options=["Auto (ANOVA if ~normal; else Kruskal)", "Force ANOVA", "Force Kruskal-Wallis"],
        horizontal=True,
    )

    tdf = f[[factor, "__value__"]].dropna()

    # Keep only groups with n>=5 for stability
    counts = tdf.groupby(factor)["__value__"].size()
    keep = counts[counts >= 5].index
    tdf = tdf[tdf[factor].isin(keep)]

    st.caption("Only groups with n≥5 are included (reduces unstable inference).")

    if tdf[factor].nunique() < 2:
        st.warning("Not enough groups after filtering to run a test.")
    else:
        vals = tdf["__value__"].values
        vals_shap = np.random.choice(vals, size=5000, replace=False) if len(vals) > 5000 else vals

        shapiro_p = np.nan
        try:
            shapiro_p = stats.shapiro(vals_shap).pvalue  # Shapiro & Wilk, 1965
        except Exception:
            pass

        st.write(f"Shapiro–Wilk p-value (pooled, approximate): **{shapiro_p:.4g}**")

        if test_mode.startswith("Auto"):
            chosen = "anova" if (not np.isnan(shapiro_p) and shapiro_p >= 0.05) else "kruskal"
        elif test_mode.startswith("Force ANOVA"):
            chosen = "anova"
        else:
            chosen = "kruskal"

        st.write(f"Chosen test: **{ 'One-way ANOVA' if chosen=='anova' else 'Kruskal–Wallis' }**")

        if chosen == "anova":
            tmp = tdf.rename(columns={factor: "factor", "__value__": "value"})
            model = ols("value ~ C(factor)", data=tmp).fit()  # Fisher, 1925 framework
            anova_tbl = sm.stats.anova_lm(model, typ=2)

            pval = float(anova_tbl["PR(>F)"].iloc[0])
            eta2 = effect_size_anova_eta_sq(anova_tbl)

            st.markdown(f"**ANOVA p-value:** `{pval:.4g}`")
            st.markdown(f"**Effect size (η²):** `{eta2:.4f}`")
            st.dataframe(anova_tbl.reset_index(), use_container_width=True)

            st.markdown("**Posthoc: Tukey HSD (pairwise)**")
            try:
                tuk = pairwise_tukeyhsd(endog=tmp["value"], groups=tmp["factor"], alpha=0.05)
                tuk_df = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
                st.dataframe(tuk_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not run Tukey HSD: {e}")

        else:
            grouped = [g["__value__"].values for _, g in tdf.groupby(factor)]
            H, pval = stats.kruskal(*grouped)  # Kruskal & Wallis, 1952
            n = len(tdf)
            k = tdf[factor].nunique()
            eps2 = effect_size_kruskal_epsilon_sq(H, n, k)

            st.markdown(f"**Kruskal–Wallis H:** `{H:.4f}`")
            st.markdown(f"**Kruskal p-value:** `{pval:.4g}`")
            st.markdown(f"**Effect size (ε²):** `{eps2:.4f}`")

            st.markdown("**Posthoc: pairwise Mann–Whitney U + BH-FDR**")
            post = pairwise_mwu(tdf.rename(columns={factor: "factor"}), "factor", "__value__")
            st.dataframe(post, use_container_width=True)


with tabs[3]:
    st.subheader("Correlation matrix")

    method = st.radio("Correlation method", options=["Pearson", "Spearman"], horizontal=True)  # Pearson 1895; Spearman 1904

    numeric_cols = [c for c in f.columns if pd.api.types.is_numeric_dtype(f[c])]
    # Remove derived IDs if you want (still available)
    default_cols = [c for c in numeric_cols if c not in ["__plot__", "__rep__"]]
    sel_num = st.multiselect("Numeric columns", numeric_cols, default=default_cols)

    if len(sel_num) < 2:
        st.warning("Select at least 2 numeric columns.")
    else:
        corr = f[sel_num].corr(method="pearson" if method == "Pearson" else "spearman")
        fig_corr = px.imshow(corr, text_auto=True, title=f"{method} correlation matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.dataframe(corr, use_container_width=True)

with tabs[4]:
    st.subheader("Filtered data preview")
    st.dataframe(f.sort_values("__date__"), use_container_width=True)
