"""
Drug Screen Luminescence Analysis — Streamlit App
Run with: streamlit run drug_screen_app.py
Requires: pandas, numpy, scipy, plotly, streamlit, openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Screen Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.75; font-size: 0.95rem; }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
    .metric-card .sub   { font-size: 0.78rem; color: #888; margin-top: 2px; }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a1a2e;
        border-left: 4px solid #0f3460;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem;
    }
    .zprime-excellent { color: #1a7a1a; font-weight: 700; }
    .zprime-good      { color: #3a7a00; font-weight: 700; }
    .zprime-marginal  { color: #c47a00; font-weight: 700; }
    .zprime-poor      { color: #c40000; font-weight: 700; }
    .zprime-pending   { color: #666; font-weight: 700; }
    div[data-testid="stTabs"] button { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
    else:
        # Try default path
        df = pd.read_excel("Automation_Specialist_HW_-_data.xlsx")
    return df

@st.cache_data
def process_data(_df, anchor_condition):
    df = _df.copy()

    # ── Normalisation factor (mean of selected anchor wells) ─────────────────
    anchor_mean = df[df["drug"] == anchor_condition]["raw_luminescence"].mean()

    # ── Per-well normalised luminescence ─────────────────────────────────────
    if pd.isna(anchor_mean) or anchor_mean == 0:
        df["norm_luminescence"] = np.nan
    else:
        df["norm_luminescence"] = df["raw_luminescence"] / anchor_mean

    # ── Drug conditions that have dose-response data ─────────────────────────
    dose_mask = df["drug_dose_in_nanomolar"].notna()
    drug_drugs = sorted(df.loc[dose_mask, "drug"].unique())

    # ── Aggregate triplicates ─────────────────────────────────────────────────
    agg_rows = []
    for drug in drug_drugs:
        sub = df[df["drug"] == drug].dropna(subset=["drug_dose_in_nanomolar"])
        for dose, grp in sub.groupby("drug_dose_in_nanomolar"):
            agg_rows.append({
                "drug": drug,
                "dose_nM": dose,
                "raw_mean":   grp["raw_luminescence"].mean(),
                "raw_std":    grp["raw_luminescence"].std(ddof=1) if len(grp) > 1 else 0,
                "norm_mean":  grp["norm_luminescence"].mean(),
                "norm_std":   grp["norm_luminescence"].std(ddof=1) if len(grp) > 1 else 0,
                "n":          len(grp),
            })

    agg = pd.DataFrame(agg_rows)

    # ── CV per condition ──────────────────────────────────────────────────────
    cv_rows = []
    for drug in df["drug"].unique():
        sub = df[df["drug"] == drug]
        for dose, grp in sub.groupby("drug_dose_in_nanomolar", dropna=False):
            label = drug if pd.isna(dose) else f"{drug} ({dose:.4f} nM)"
            if len(grp) > 1:
                mean_v = grp["raw_luminescence"].mean()
                std_v  = grp["raw_luminescence"].std(ddof=1)
                cv     = (std_v / mean_v) * 100 if mean_v != 0 else np.nan
                cv_rows.append({"Condition": label, "Drug": drug, "Dose (nM)": dose,
                                 "Mean": mean_v, "SD": std_v, "CV (%)": cv, "n": len(grp)})

    cv_df = pd.DataFrame(cv_rows)
    return df, agg, cv_df, anchor_mean

# ── 4PL Sigmoid fit ───────────────────────────────────────────────────────────
def four_pl(x, bottom, top, ec50, hillslope):
    return bottom + (top - bottom) / (1 + (ec50 / x) ** hillslope)

def fit_dose_response(doses, responses):
    try:
        x = np.array(doses, dtype=float)
        y = np.array(responses, dtype=float)
        p0    = [y.min(), y.max(), np.median(x), 1.0]
        bounds = ([-np.inf, -np.inf, 1e-6, 0.01], [np.inf, np.inf, 1e6, 10])
        popt, _ = curve_fit(four_pl, x, y, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except Exception:
        return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data")
    uploaded = st.file_uploader("Upload XLSX file", type=["xlsx", "xls"],
                                help="Upload the plate reader results file")
    st.markdown("---")
    st.markdown("### ⚙️ Options")
    show_fit = st.toggle("Show 4PL sigmoid fit", value=True)
    show_raw_points = st.toggle("Show individual data points", value=False)
    log_x = st.toggle("Log₁₀ X-axis (dose)", value=True)

# ── Load & process ────────────────────────────────────────────────────────────
try:
    raw_df = load_data(uploaded)
except FileNotFoundError:
    st.error("No data file found. Please upload your XLSX file using the sidebar.")
    st.stop()

all_conditions = sorted(raw_df["drug"].dropna().unique())
if len(all_conditions) == 0:
    st.error("No conditions found in the 'drug' column.")
    st.stop()

def _default_index(options, preferred):
    return options.index(preferred) if preferred in options else 0

default_anchor_idx = _default_index(all_conditions, "0.1% DMSO")
default_pos_idx = _default_index(all_conditions, "0.1% DMSO")
default_neg_idx = _default_index(all_conditions, "Empty Control")
if default_neg_idx == default_pos_idx and len(all_conditions) > 1:
    default_neg_idx = 1 if default_pos_idx == 0 else 0

if "zprime_ready" not in st.session_state:
    st.session_state["zprime_ready"] = False
if "zprime_pos" not in st.session_state:
    st.session_state["zprime_pos"] = all_conditions[default_pos_idx]
if "zprime_neg" not in st.session_state:
    st.session_state["zprime_neg"] = all_conditions[default_neg_idx]
if st.session_state["zprime_pos"] not in all_conditions:
    st.session_state["zprime_pos"] = all_conditions[default_pos_idx]
    st.session_state["zprime_ready"] = False
if st.session_state["zprime_neg"] not in all_conditions:
    st.session_state["zprime_neg"] = all_conditions[default_neg_idx]
    st.session_state["zprime_ready"] = False

with st.sidebar:
    st.markdown("---")
    st.markdown("### 🧭 Normalisation")
    anchor_condition = st.selectbox(
        "Anchoring condition",
        all_conditions,
        index=default_anchor_idx,
        help="Condition used to normalise raw luminescence."
    )

    st.markdown("---")
    st.markdown("### ✅ Z′ Controls")
    pos_choice = st.selectbox(
        "Positive control",
        all_conditions,
        index=default_pos_idx,
        key="zprime_pos_choice",
    )
    neg_choice = st.selectbox(
        "Negative control",
        all_conditions,
        index=default_neg_idx,
        key="zprime_neg_choice",
    )
    apply_z = st.button("Apply controls", use_container_width=True)
    if apply_z:
        if pos_choice == neg_choice:
            st.error("Positive and negative controls must be different.")
            st.session_state["zprime_ready"] = False
        else:
            st.session_state["zprime_pos"] = pos_choice
            st.session_state["zprime_neg"] = neg_choice
            st.session_state["zprime_ready"] = True

df, agg, cv_df, anchor_mean = process_data(raw_df, anchor_condition)

if pd.isna(anchor_mean) or anchor_mean == 0:
    st.warning(
        f"Normalisation anchor '{anchor_condition}' has no data or mean=0. "
        "Normalised values will be blank."
    )

cv_condition_options = sorted(cv_df["Condition"].dropna().unique().tolist())
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Pooled CV")
    pooled_cv_exclude = st.multiselect(
        "Exclude conditions",
        options=cv_condition_options,
        help="Exclude specific conditions from the pooled CV calculation."
    )

cv_pool = cv_df[~cv_df["Condition"].isin(pooled_cv_exclude)] if pooled_cv_exclude else cv_df
pooled_cv = cv_pool["CV (%)"].dropna().mean()
pooled_cv_n = cv_pool["CV (%)"].dropna().shape[0]

zprime = np.nan
pos_ctrl_name = st.session_state.get("zprime_pos")
neg_ctrl_name = st.session_state.get("zprime_neg")
pos = df[df["drug"] == pos_ctrl_name]["raw_luminescence"] if pos_ctrl_name else pd.Series(dtype=float)
neg = df[df["drug"] == neg_ctrl_name]["raw_luminescence"] if neg_ctrl_name else pd.Series(dtype=float)
if st.session_state.get("zprime_ready") and len(pos) > 0 and len(neg) > 0:
    denom = abs(pos.mean() - neg.mean())
    if denom and not pd.isna(denom):
        zprime = 1 - (3 * (pos.std() + neg.std())) / denom

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 Drug Screen Luminescence Analysis</h1>
    <p>Dose–response curves · Plate heatmap · Quality metrics · Z′-factor</p>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
drug_list = sorted(agg["drug"].unique())
n_drugs   = raw_df["drug"].dropna().nunique()
n_wells   = len(df)

def zprime_class(z):
    if pd.isna(z):
        return "zprime-pending", "Not computed"
    if z >= 0.5:  return "zprime-excellent", "Excellent (≥0.5)"
    if z >= 0.0:  return "zprime-good",      "Acceptable (0–0.5)"
    if z >= -0.5: return "zprime-marginal",  "Marginal"
    return "zprime-poor", "Poor"

zclass, zlabel = zprime_class(zprime)
zprime_display = "—" if pd.isna(zprime) else f"{zprime:.4f}"
pooled_cv_display = "—" if pd.isna(pooled_cv) else f"{pooled_cv:.2f}%"
anchor_display = "—" if pd.isna(anchor_mean) else f"{anchor_mean:,.0f}"

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Drugs screened</div>
        <div class="value">{n_drugs}</div>
        <div class="sub">unique entries in drug column</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Total wells</div>
        <div class="value">{n_wells}</div>
        <div class="sub">across all conditions</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Anchor mean (raw)</div>
        <div class="value">{anchor_display}</div>
        <div class="sub">{anchor_condition}</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Pooled CV</div>
        <div class="value">{pooled_cv_display}</div>
        <div class="sub">n={pooled_cv_n} after exclusions</div>
    </div>""", unsafe_allow_html=True)
with col5:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Z′-factor</div>
        <div class="value"><span class="{zclass}">{zprime_display}</span></div>
        <div class="sub">{zlabel}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Dose–Response Curves", "🟦 Plate Heatmap", "📊 CV Summary", "📋 Quality Metrics"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dose–Response Curves
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Dose–Response Curves</div>', unsafe_allow_html=True)

    # View toggle
    c1, c2, c3 = st.columns([2, 2, 6])
    with c1:
        view_mode = st.radio("Luminescence", ["Raw", "Normalised"],
                             horizontal=True, key="lum_mode")
    with c2:
        display = st.radio("Display", ["All drugs", "Select drug"],
                           horizontal=True, key="disp_mode")

    y_col  = "raw_mean"  if view_mode == "Raw" else "norm_mean"
    sd_col = "raw_std"   if view_mode == "Raw" else "norm_std"
    y_lbl  = "Raw Luminescence (RLU)" if view_mode == "Raw" else f"Normalised Luminescence (vs {anchor_condition})"

    COLORS = px.colors.qualitative.Plotly

    # ── Colour palette
    drug_colors = {d: COLORS[i % len(COLORS)] for i, d in enumerate(drug_list)}

    def make_drug_trace(drug, color, agg_df, show_fit, show_raw_points,
                        y_col, sd_col, log_x, raw_df_full, view_mode):
        sub = agg_df[agg_df["drug"] == drug].sort_values("dose_nM")
        doses = sub["dose_nM"].values
        means = sub[y_col].values
        stds  = sub[sd_col].values
        traces = []

        # Individual replicates (raw scatter)
        if show_raw_points:
            raw_sub = raw_df_full[raw_df_full["drug"] == drug].dropna(subset=["drug_dose_in_nanomolar"])
            rep_y = raw_sub["norm_luminescence"] if view_mode == "Normalised" else raw_sub["raw_luminescence"]
            traces.append(go.Scatter(
                x=raw_sub["drug_dose_in_nanomolar"], y=rep_y,
                mode="markers",
                marker=dict(color=color, size=5, opacity=0.35, symbol="circle-open"),
                name=f"{drug} (reps)", showlegend=False,
                hovertemplate="Replicate<br>Dose: %{x:.4f} nM<br>Value: %{y:,.0f}<extra></extra>",
            ))

        # Error-bar scatter
        traces.append(go.Scatter(
            x=doses, y=means,
            error_y=dict(type="data", array=stds, visible=True, thickness=1.5, width=5),
            mode="markers+lines",
            line=dict(color=color, width=1.5, dash="dot"),
            marker=dict(color=color, size=8),
            name=drug,
            hovertemplate=(
                f"<b>{drug}</b><br>"
                "Dose: %{x:.4f} nM<br>"
                "Mean: %{y:,.2f}<br>"
                f"SD: " + "<extra></extra>"
            ),
        ))

        # 4PL fit
        if show_fit and len(doses) >= 4:
            popt = fit_dose_response(doses, means)
            if popt is not None:
                x_fit = np.logspace(np.log10(doses.min()), np.log10(doses.max()), 300)
                y_fit = four_pl(x_fit, *popt)
                traces.append(go.Scatter(
                    x=x_fit, y=y_fit,
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    name=f"{drug} (4PL fit)",
                    hoverinfo="skip",
                    showlegend=False,
                ))

        return traces

    # ── All drugs panel ───────────────────────────────────────────────────────
    if display == "All drugs":
        ncols = 2
        nrows = int(np.ceil(len(drug_list) / ncols))
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=[str(d) for d in drug_list],
            vertical_spacing=0.12, horizontal_spacing=0.08,
        )

        for i, drug in enumerate(drug_list):
            r, c = divmod(i, ncols)
            color = drug_colors[drug]
            for trace in make_drug_trace(drug, color, agg, show_fit, show_raw_points,
                                          y_col, sd_col, log_x, df, view_mode):
                fig.add_trace(trace, row=r+1, col=c+1)
            if log_x:
                fig.update_xaxes(type="log", row=r+1, col=c+1)

        fig.update_layout(
            height=420 * nrows,
            showlegend=False,
            template="plotly_white",
            font=dict(family="Arial", size=11),
        )
        fig.update_xaxes(title_text="Dose (nM)")
        fig.update_yaxes(title_text=y_lbl)
        st.plotly_chart(fig, use_container_width=True)

    # ── Single drug panel ─────────────────────────────────────────────────────
    else:
        sel_drug = st.selectbox("Select drug", drug_list)
        color = drug_colors[sel_drug]
        fig = go.Figure()
        for trace in make_drug_trace(sel_drug, color, agg, show_fit, show_raw_points,
                                      y_col, sd_col, log_x, df, view_mode):
            fig.add_trace(trace)

        # Add anchor reference line
        if view_mode == "Normalised":
            fig.add_hline(y=1.0, line_dash="dash", line_color="grey",
                          annotation_text=f"{anchor_condition} (1.0)", annotation_position="right")
        else:
            if not pd.isna(anchor_mean):
                fig.add_hline(y=anchor_mean, line_dash="dash", line_color="grey",
                              annotation_text=f"{anchor_condition} mean ({anchor_display})",
                              annotation_position="right")

        if log_x:
            fig.update_xaxes(type="log")
        fig.update_layout(
            title=dict(text=sel_drug,
                       font=dict(size=16)),
            xaxis_title="Dose (nM)",
            yaxis_title=y_lbl,
            height=520,
            template="plotly_white",
            font=dict(family="Arial"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # EC50 annotation
        sub = agg[agg["drug"] == sel_drug].sort_values("dose_nM")
        popt = fit_dose_response(sub["dose_nM"].values, sub[y_col].values)
        if popt is not None:
            ec50 = popt[2]
            st.info(f"**4PL EC₅₀ estimate:** {ec50:.4f} nM  |  "
                    f"Bottom: {popt[0]:.2f}  |  Top: {popt[1]:.2f}  |  Hill slope: {popt[3]:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Plate Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Plate Layout Heatmap</div>', unsafe_allow_html=True)

    hmap_mode = st.radio("Luminescence values", ["Raw", "Normalised"],
                         horizontal=True, key="hmap_mode")
    hmap_col = "raw_luminescence" if hmap_mode == "Raw" else "norm_luminescence"

    def _condition_label(drug, dose):
        if pd.isna(dose):
            return str(drug)
        dose_str = np.format_float_positional(float(dose), unique=True, trim="-")
        return f"{drug} ({dose_str} nM)"

    hmap_df = df.copy()
    hmap_df["condition_label"] = hmap_df.apply(
        lambda r: _condition_label(r["drug"], r["drug_dose_in_nanomolar"]),
        axis=1
    )
    condition_options = sorted(hmap_df["condition_label"].unique().tolist())
    hmap_exclude = st.multiselect(
        "Exclude conditions from heatmap",
        options=condition_options,
        help="Excluded conditions (drug + dose) are removed from the color scale calculation."
    )
    if hmap_exclude:
        hmap_df = hmap_df[~hmap_df["condition_label"].isin(hmap_exclude)].copy()

    rows_order = sorted(df["row"].unique())
    cols_order  = sorted(df["column"].unique())

    pivot = hmap_df.pivot_table(index="row", columns="column",
                                values=hmap_col, aggfunc="mean")
    pivot = pivot.reindex(index=rows_order, columns=cols_order)

    # Annotation text: drug name + dose
    def short_name(r):
        d = str(r["drug"])
        dose = r["drug_dose_in_nanomolar"]
        if pd.isna(dose):
            return d
        dose_str = np.format_float_positional(float(dose), unique=True, trim="-")
        return f"{d}\n{dose_str}"

    hmap_df["label"] = hmap_df.apply(short_name, axis=1)
    pivot_labels = hmap_df.pivot_table(index="row", columns="column",
                                       values="label", aggfunc="first")
    pivot_labels = pivot_labels.reindex(index=rows_order, columns=cols_order)

    z_vals  = pivot.values.tolist()
    ann_txt = pivot_labels.values.tolist()

    if hmap_df.empty:
        st.warning("No data to display after exclusions.")
    else:
        # Build heatmap
        fig_hm = go.Figure(go.Heatmap(
            z=z_vals,
            x=[str(c) for c in cols_order],
            y=rows_order,
            customdata=ann_txt,
            colorscale=[[0, "#e8f4f8"], [0.5, "#2980b9"], [1, "#0a1a6e"]],
            colorbar=dict(title=dict(text=hmap_col.replace("_", " ").title(), side="right")),
            hoverongaps=False,
            hovertemplate=(
                "Row %{y} · Col %{x}<br>"
                "Value: %{z:,.1f}<br>"
                "Condition: %{customdata}<extra></extra>"
            ),
            reversescale=False,
        ))
        fig_hm.update_layout(
            height=600,
            xaxis=dict(title="Column", side="top", tickmode="array",
                       tickvals=[str(c) for c in cols_order],
                       ticktext=[str(c) for c in cols_order]),
            yaxis=dict(title="Row", autorange="reversed"),
            template="plotly_white",
            font=dict(family="Arial", size=11),
            margin=dict(t=80, b=20),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        st.caption(
            "Colour scale: lightest blue = lowest luminescence · darkest blue = highest luminescence. "
            "Labels show drug and dose in nM."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CV Summary
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Coefficient of Variation (CV) per Condition</div>',
                unsafe_allow_html=True)

    cl, cr = st.columns([3, 1])
    with cr:
        cv_sort = st.selectbox("Sort by", ["CV (%)", "Condition", "Mean"], key="cv_sort")
        asc = st.checkbox("Ascending", value=False, key="cv_asc")

    cv_display = cv_df[["Condition", "Mean", "SD", "CV (%)", "n"]].copy()
    cv_display = cv_display.sort_values(cv_sort, ascending=asc).reset_index(drop=True)
    cv_display["Mean"] = cv_display["Mean"].map("{:,.1f}".format)
    cv_display["SD"]   = cv_display["SD"].map("{:,.1f}".format)
    cv_display["CV (%)"] = cv_display["CV (%)"].map("{:.2f}".format)

    st.dataframe(cv_display, use_container_width=True, height=500)

    # Bar chart
    cv_plot = cv_df.sort_values("CV (%)", ascending=False).copy()
    colors_bar = ["#c0392b" if v > 15 else "#e67e22" if v > 10
                  else "#f1c40f" if v > 5 else "#2ecc71"
                  for v in cv_plot["CV (%)"]]

    fig_cv = go.Figure(go.Bar(
        x=cv_plot["Condition"],
        y=cv_plot["CV (%)"],
        marker_color=colors_bar,
        hovertemplate="%{x}<br>CV: %{y:.2f}%<extra></extra>",
    ))
    if not pd.isna(pooled_cv):
        fig_cv.add_hline(y=pooled_cv, line_dash="dash", line_color="#2980b9",
                         annotation_text=f"Pooled CV: {pooled_cv:.2f}%",
                         annotation_position="top right")
    fig_cv.add_hline(y=10, line_dash="dot", line_color="#e67e22",
                     annotation_text="10% threshold", annotation_position="top right")
    fig_cv.update_layout(
        xaxis=dict(tickangle=-55, tickfont=dict(size=8)),
        yaxis_title="CV (%)",
        height=480,
        template="plotly_white",
        font=dict(family="Arial"),
        showlegend=False,
        margin=dict(b=160),
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    # Pooled CV callout
    excluded_count = len(pooled_cv_exclude)
    excluded_txt = f" — excluding {excluded_count} condition(s)" if excluded_count else ""
    pooled_cv_text = f"{pooled_cv:.2f}%" if not pd.isna(pooled_cv) else "—"
    st.markdown(f"""
    <div style="background:#eaf3fb; border-left:4px solid #2980b9; padding:0.9rem 1.2rem;
                border-radius:6px; margin-top:0.5rem;">
        <b>Pooled CV (selected conditions):</b>
        &nbsp;&nbsp;<span style="font-size:1.2rem; font-weight:700; color:#0a1a6e;">{pooled_cv_text}</span>
        <span style="font-size:0.85rem; color:#555; margin-left:8px;">
            — mean of per-condition CVs (n={pooled_cv_n} conditions){excluded_txt}
        </span>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Quality Metrics (Z′, controls)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Assay Quality Metrics</div>', unsafe_allow_html=True)

    if not st.session_state.get("zprime_ready"):
        st.info("Select positive and negative controls in the sidebar and click **Apply controls** to compute Z′.")
    elif len(pos) == 0 or len(neg) == 0 or pd.isna(zprime):
        st.warning("Z′ could not be computed for the selected controls. Check that both controls have data and different means.")
    else:
        pos_label = pos_ctrl_name or "Positive control"
        neg_label = neg_ctrl_name or "Negative control"

        zc, zc2 = st.columns(2)
        with zc:
            st.markdown("#### Z′-Factor")
            st.latex(r"Z' = 1 - \frac{3\sigma_{pos} + 3\sigma_{neg}}{|\mu_{pos} - \mu_{neg}|}")
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| μ positive ({pos_label}) | {pos.mean():,.1f} |
| σ positive | {pos.std():,.1f} |
| μ negative ({neg_label}) | {neg.mean():,.1f} |
| σ negative | {neg.std():,.1f} |
| **Z′-factor** | **{zprime:.4f}** |
| **Classification** | **{zlabel}** |
            """)

            if zprime >= 0.5:
                st.success(f"✅ Z′ = {zprime:.4f} — Excellent assay quality. A Z′ ≥ 0.5 indicates a robust, high-quality screening assay.")
            elif zprime >= 0.0:
                st.warning(f"⚠️ Z′ = {zprime:.4f} — Acceptable but borderline. A Z′ between 0 and 0.5 suggests the assay can be used with caution.")
            else:
                st.error(f"❌ Z′ = {zprime:.4f} — Poor assay quality. Assay may need optimisation.")

        with zc2:
            st.markdown("#### Z′-Factor Interpretation")
            fig_z = go.Figure()
            # Distribution for pos ctrl
            x_range = np.linspace(
                min(pos.min(), neg.min()) - 3*max(pos.std(), neg.std()),
                max(pos.max(), neg.max()) + 3*max(pos.std(), neg.std()),
                400
            )
            from scipy.stats import norm as sp_norm
            fig_z.add_trace(go.Scatter(
                x=x_range,
                y=sp_norm.pdf(x_range, pos.mean(), pos.std()),
                fill="tozeroy", fillcolor="rgba(41,128,185,0.25)",
                line=dict(color="#2980b9", width=2),
                name=f"Positive ctrl ({pos_label})",
            ))
            fig_z.add_trace(go.Scatter(
                x=x_range,
                y=sp_norm.pdf(x_range, neg.mean(), neg.std()),
                fill="tozeroy", fillcolor="rgba(231,76,60,0.25)",
                line=dict(color="#e74c3c", width=2),
                name=f"Negative ctrl ({neg_label})",
            ))
            # 3σ bands
            for m, s, c in [(pos.mean(), pos.std(), "#2980b9"), (neg.mean(), neg.std(), "#e74c3c")]:
                for sign in [-1, 1]:
                    fig_z.add_vline(x=m + sign*3*s, line_dash="dot",
                                    line_color=c, line_width=1)
            fig_z.update_layout(
                height=300,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis_title="Raw Luminescence",
                yaxis_title="Density",
                font=dict(family="Arial", size=11),
                margin=dict(t=50),
            )
            st.plotly_chart(fig_z, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Control Well Summary")
    ctrl_rows = []
    control_names = []
    if anchor_condition:
        control_names.append(anchor_condition)
    if st.session_state.get("zprime_ready"):
        if pos_ctrl_name:
            control_names.append(pos_ctrl_name)
        if neg_ctrl_name:
            control_names.append(neg_ctrl_name)
    # De-duplicate while preserving order
    control_names = list(dict.fromkeys(control_names))

    for ctrl_name in control_names:
        grp = df[df["drug"] == ctrl_name]
        if len(grp) == 0:
            continue
        ctrl_rows.append({
            "Control": ctrl_name,
            "n wells": len(grp),
            "Raw Mean": f"{grp['raw_luminescence'].mean():,.1f}",
            "Raw SD":   f"{grp['raw_luminescence'].std(ddof=1):,.1f}",
            "Raw CV (%)": f"{(grp['raw_luminescence'].std(ddof=1)/grp['raw_luminescence'].mean()*100):.2f}",
            "Norm Mean": f"{grp['norm_luminescence'].mean():.4f}",
        })
    if ctrl_rows:
        st.table(pd.DataFrame(ctrl_rows).set_index("Control"))
    else:
        st.info("No control summary available for the current selections.")

    st.markdown("---")
    st.markdown("#### Z-Factor Scale Reference")
    ref_data = {
        "Z′ Range": ["≥ 0.5", "0 – 0.5", "< 0", "= 1"],
        "Classification": ["Excellent", "Acceptable", "Not suitable", "Ideal (theoretical)"],
        "Interpretation": [
            "High-quality HTS assay",
            "Marginal — use with caution",
            "Poor separation; assay needs optimisation",
            "Perfect separation between controls"
        ]
    }
    st.table(pd.DataFrame(ref_data).set_index("Z′ Range"))
