"""RNA or protein? Interactive web tool for a proteogenomic comparison across five cancers (CPTAC).
Research use only - not a diagnostic device."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

st.set_page_config(page_title="RNA or Protein?", page_icon="🧬", layout="wide")
BASE = Path(__file__).parent; D, M = BASE / "data", BASE / "models"

BLUE, ORANGE, AQUA, YELLOW, PINK, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"
COHORTS = ["BRCA", "LUAD", "UCEC", "COAD", "GBM"]
NAME = {"BRCA": "Breast", "LUAD": "Lung adenocarcinoma", "UCEC": "Endometrial", "COAD": "Colon", "GBM": "Glioblastoma"}
VIEWS = ["RNA", "Protein", "RNA (matched genes)", "Protein (matched genes)", "Early fusion", "Late fusion"]
VIEW_C = dict(zip(VIEWS, [BLUE, ORANGE, "#86b6ef", "#f4a582", AQUA, YELLOW]))
MODEL_N = {"LR": "Logistic regression", "RF": "Random forest", "XGB": "XGBoost"}
VERDICT_C = {"Protein favoured": ORANGE, "RNA favoured": BLUE, "No consistent difference": "#9a9893",
             "Not predictable (best AUC < 0.65)": "#cfcdc6", "Control task": "#52514e"}


def style(fig, h=400):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="Inter, Arial, sans-serif", size=13),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    fig.update_xaxes(gridcolor="#ececea"); fig.update_yaxes(gridcolor="#ececea")
    return fig


@st.cache_data
def tables():
    t = {f.stem: pd.read_csv(f) for f in D.glob("*.csv") if not f.stem.startswith("demo")}
    t["oof"] = pd.read_parquet(D / "oof_predictions.parquet")
    t["genes"] = pd.read_parquet(D / "gene_level_auc_top.parquet")
    t["key"] = json.loads((D / "key_numbers.json").read_text())
    return t


@st.cache_data
def layer_matrix(c, layer):
    return pd.read_parquet(D / f"{c}_{'rna' if layer == 'RNA' else 'protein'}.parquet")


@st.cache_data
def labels(c):
    return pd.read_csv(D / f"{c}_labels.csv", index_col=0)


@st.cache_resource
def predictor():
    return pd.read_parquet(M / "lr_coefficients.parquet"), json.loads((M / "lr_info.json").read_text())


T = tables(); K = T["key"]
CONS = T["task_consensus"].assign(Cancer=lambda d: d.cohort.map(NAME))
TASKS = {c: labels(c).columns.tolist() for c in COHORTS}

st.sidebar.title("🧬 RNA or Protein?")
st.sidebar.caption("Proteogenomic ML comparison · 5 cancers · 522 tumours")
PAGES = ["Overview", "Data & RNA–protein coupling", "Benchmark", "Task explorer", "Mutation effects",
         "Clusters & pathways", "Predict"]
page = st.sidebar.radio("Section", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Research use only. Data: CPTAC breast (Krug 2020), lung adenocarcinoma (Gillette 2020), endometrial (Dou 2020), "
                   "colon (Vasaikar 2019) and glioblastoma (Wang 2021) cohorts via cBioPortal.")

# ======================================================================= Overview
if page == "Overview":
    st.title("When does proteomics add to RNA?")
    st.markdown("RNA sequencing is routine; mass-spectrometry proteomics is costly but measures the molecules that act. "
                "This tool accompanies a study that compares **RNA**, **protein** and **both combined** for predicting "
                "clinical and molecular features of tumours in which both layers were measured from the same tissue.")
    c = st.columns(4)
    c[0].metric("Tumours", K["n_tumours"]); c[1].metric("Cancer types", 5)
    c[2].metric("Prediction tasks", f"{K['n_tasks_total']}", help="28 clinical/molecular tasks + 1 control task")
    c[3].metric("Informative tasks", K["n_informative"], help="Best AUC ≥ 0.65 in any configuration, control excluded")
    st.subheader("Main findings")
    v = K["verdicts"]
    st.markdown(
        f"1. **No overall winner.** Across {K['n_informative']} informative tasks, protein and RNA perform alike with every model "
        f"(Wilcoxon p = {K['protein_vs_rna_wilcoxon_p']['LR']}, {K['protein_vs_rna_wilcoxon_p']['RF']}, {K['protein_vs_rna_wilcoxon_p']['XGB']} "
        f"for logistic regression, random forest, XGBoost).\n"
        f"2. **Task-specific advantages.** Protein is favoured by all three models in {v.get('Protein favoured', 0)} tasks "
        f"(e.g. breast HER2 status, luminal A vs B), RNA in {v.get('RNA favoured', 0)} (e.g. lung KRAS mutation, smoking status); "
        f"{v.get('No consistent difference', 0)} show no consistent difference.\n"
        f"3. **Combining layers helps a little.** Late fusion improves on RNA alone by a median "
        f"{K['late_vs_rna']['LR']['median_delta']:+.3f} AUC (logistic regression, p = {K['late_vs_rna']['LR']['p_Wilcoxon']}).\n"
        f"4. **Protein reveals what mutations do to proteins.** For tumour-suppressor mutations, the mutated gene's own protein "
        f"identifies the mutation better than its RNA in {K['cis_TS_protein_better']}/{K['cis_TS_n']} cases "
        f"(median +{K['cis_TS_median_delta']} AUC, p = {K['cis_TS_wilcoxon_p']}); for oncogenes it does not "
        f"(difference between groups p = {K['cis_TS_vs_OG_mannwhitney_p']}).")
    vc = CONS.verdict.value_counts().reset_index(); vc.columns = ["Verdict", "Tasks"]
    fig = px.bar(vc, x="Tasks", y="Verdict", orientation="h", color="Verdict", color_discrete_map=VERDICT_C, text="Tasks")
    fig.update_layout(showlegend=False); fig.update_yaxes(title=None)
    st.plotly_chart(style(fig, 260), width="stretch")
    st.subheader("Design")
    st.markdown("- **Data:** matched RNA-seq, global proteomics (TMT mass spectrometry) and somatic mutations from five CPTAC cohorts.\n"
                "- **Views:** RNA, protein, each restricted to the same genes, early fusion (concatenated) and late fusion (averaged predictions).\n"
                "- **Models:** L2 logistic regression (inner-CV tuned), random forest, XGBoost.\n"
                "- **Validation:** 5-fold cross-validation × 3 repeats; preprocessing (missing-value handling, top-2,000 variance selection, scaling) fitted inside training folds only.\n"
                "- **Statistics:** paired DeLong tests and bootstrap CIs per task; Wilcoxon signed-rank tests across tasks; Benjamini–Hochberg correction.\n"
                "- **Interpretation:** SHAP, gene-level discrimination, mRNA–protein correlation, Hallmark enrichment, unsupervised clustering.")

# ======================================================================= Data & coupling
elif page == "Data & RNA–protein coupling":
    st.title("Cohorts and mRNA–protein coupling")
    t1 = T["table1_cohorts"]
    st.dataframe(t1.drop(columns=["Task list"]), width="stretch", hide_index=True)
    with st.expander("Task list per cancer (positives / negatives)"):
        for _, r in t1.iterrows(): st.markdown(f"**{r.Cancer}:** {r['Task list']}")
    corr = T["rna_protein_correlation"].assign(Cancer=lambda d: d.cohort.map(NAME))
    c1, c2 = st.columns(2)
    fig = px.violin(corr, x="rho", y="Cancer", box=True, points=False, color_discrete_sequence=[BLUE],
                    category_orders={"Cancer": [NAME[c] for c in COHORTS]})
    fig.update_xaxes(title="Gene-wise mRNA–protein Spearman ρ"); fig.update_yaxes(title=None)
    c1.subheader("How well does RNA track protein?"); c1.plotly_chart(style(fig, 420), width="stretch")
    hc = T["hallmark_coupling"]; h = pd.concat([hc.head(8), hc.tail(8)])
    fig = px.bar(h, x="median_rho", y="set", orientation="h", color=np.where(h.median_rho < h.median_rho_others, "Weakly coupled", "Strongly coupled"),
                 color_discrete_map={"Weakly coupled": ORANGE, "Strongly coupled": BLUE}, hover_data={"q": ":.2g", "n_genes": True})
    fig.add_vline(x=float(hc.median_rho_others.median()), line_dash="dash", line_color="#52514e")
    fig.update_xaxes(title="Median ρ of pathway genes"); fig.update_yaxes(title=None, categoryorder="total ascending")
    c2.subheader("Hallmark pathways"); c2.plotly_chart(style(fig, 420), width="stretch")
    st.subheader("Gene lookup")
    c1, c2, c3 = st.columns(3)
    coh = c1.selectbox("Cancer", COHORTS, format_func=NAME.get)
    R, P = layer_matrix(coh, "RNA"), layer_matrix(coh, "Protein")
    default = R.columns.get_loc("ERBB2") if coh == "BRCA" and "ERBB2" in R.columns else 0
    gene = c2.selectbox("Gene", R.columns.tolist(), index=default)
    lab = labels(coh); colour = c3.selectbox("Colour by", ["(none)"] + lab.columns.tolist())
    d = pd.DataFrame({"RNA": R[gene], "Protein": P[gene]})
    if colour != "(none)": d["group"] = lab[colour].map({1: "Positive", 0: "Negative"}).reindex(d.index).fillna("Unknown")
    rho = corr[(corr.cohort == coh) & (corr.gene == gene)].rho
    fig = px.scatter(d, x="RNA", y="Protein", color="group" if colour != "(none)" else None,
                     color_discrete_map={"Positive": ORANGE, "Negative": BLUE, "Unknown": "#cfcdc6"}, hover_name=d.index)
    fig.update_xaxes(title=f"{gene} RNA (log scale, as provided)"); fig.update_yaxes(title=f"{gene} protein (log2 ratio)")
    c1, c2 = st.columns([2, 1])
    c1.plotly_chart(style(fig, 420), width="stretch")
    c2.metric("mRNA–protein Spearman ρ", f"{rho.iloc[0]:.2f}" if len(rho) else "n/a")
    pct = (corr[corr.cohort == coh].rho < (rho.iloc[0] if len(rho) else 0)).mean() * 100
    c2.caption(f"More tightly coupled than {pct:.0f}% of genes in this cancer.")
    if colour != "(none)":
        y = lab[colour].reindex(d.index); m_ = y.notna()
        if m_.sum() > 10 and y[m_].nunique() == 2:
            for layer in ["RNA", "Protein"]:
                v_ = d.loc[m_, layer]; ok = v_.notna()
                if ok.sum() > 10:
                    a = roc_auc_score(y[m_][ok], v_[ok]); c2.metric(f"{layer} AUC for '{colour}'", f"{max(a, 1 - a):.2f}",
                                                                   help="Single-gene discrimination; direction ignored")

# ======================================================================= Benchmark
elif page == "Benchmark":
    st.title("RNA vs protein across all tasks")
    met, pw = T["benchmark_metrics"], T["pairwise_tests"]
    c1, c2 = st.columns([1, 2])
    mdl = c1.radio("Model", list(MODEL_N), format_func=MODEL_N.get, horizontal=True)
    cmp_ = c2.selectbox("Comparison", sorted(pw.comparison.unique()), index=sorted(pw.comparison.unique()).index("Protein vs RNA"))
    a_name, b_name = cmp_.split(" vs ")
    q = pw[(pw.model == mdl) & (pw.comparison == cmp_)].merge(CONS[["cohort", "task", "verdict"]], on=["cohort", "task"])
    q["Cancer"] = q.cohort.map(NAME); q["label"] = q.cohort + " · " + q.task
    fig = px.scatter(q, x="AUC_b", y="AUC_a", color="verdict", color_discrete_map=VERDICT_C, hover_name="label",
                     hover_data={"delta": ":.3f", "p_DeLong": ":.3g", "q_BH": ":.3g", "AUC_a": ":.3f", "AUC_b": ":.3f", "verdict": False})
    fig.add_shape(type="line", x0=0.3, y0=0.3, x1=1, y1=1, line=dict(dash="dash", color="#52514e"))
    fig.update_traces(marker_size=11); fig.update_xaxes(title=f"AUC: {b_name}", range=[0.3, 1.02]); fig.update_yaxes(title=f"AUC: {a_name}", range=[0.3, 1.02])
    fig.update_layout(legend=dict(orientation="v", y=0.02, x=0.98, xanchor="right", yanchor="bottom", bgcolor="rgba(255,255,255,0.8)"))
    c1, c2 = st.columns([1.1, 1])
    c1.plotly_chart(style(fig, 520), width="stretch")
    q2 = q.sort_values("delta")
    fig = go.Figure()
    for _, r in q2.iterrows():
        col = ORANGE if r.delta > 0 else BLUE
        fig.add_trace(go.Scatter(x=[r.delta_lo, r.delta_hi], y=[r.label] * 2, mode="lines", line=dict(color=col, width=2), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r.delta], y=[r.label], mode="markers", showlegend=False,
                                 marker=dict(color=col if r.p_DeLong < 0.05 else "white", line=dict(color=col, width=2), size=9),
                                 hovertemplate=f"{r.label}<br>Δ={r.delta:.3f} [{r.delta_lo:.3f}, {r.delta_hi:.3f}]<br>p={r.p_DeLong:.3g}, q={r.q_BH:.3g}<extra></extra>"))
    fig.add_vline(x=0, line_color="#52514e"); fig.update_xaxes(title=f"ΔAUC ({a_name} − {b_name}), 95% CI")
    c2.caption("Filled markers: DeLong p < 0.05 (uncorrected).")
    c2.plotly_chart(style(fig, 520 + 6 * len(q2)), width="stretch")
    st.subheader("Across informative tasks")
    acr = T["across_task_tests"]; acr = acr[acr.subset == "informative tasks"].drop(columns="subset")
    st.dataframe(acr.round(4), width="stretch", hide_index=True)
    st.subheader("Per-task verdicts (consensus of three models)")
    st.dataframe(CONS[["Cancer", "task", "class", "delta_LR", "delta_RF", "delta_XGB", "mean_delta", "n_models_p05", "best_AUC_any", "verdict"]].round(3),
                 width="stretch", hide_index=True)
    st.caption("Protein/RNA favoured = same sign in all three models and mean |ΔAUC| ≥ 0.02. With ~100 tumours per cancer few "
               "single-task differences survive multiple-testing correction; patterns across tasks and models carry the evidence.")

# ======================================================================= Task explorer
elif page == "Task explorer":
    st.title("Task explorer")
    c1, c2, c3 = st.columns(3)
    coh = c1.selectbox("Cancer", COHORTS, format_func=NAME.get)
    task = c2.selectbox("Task", TASKS[coh])
    mdl = c3.radio("Model", list(MODEL_N), format_func=MODEL_N.get, horizontal=True, index=0)
    r = CONS[(CONS.cohort == coh) & (CONS.task == task)].iloc[0]
    k = st.columns(4)
    k[0].markdown(f"<div style='font-size:14px'>Verdict</div><div style='font-size:22px;font-weight:600;color:{VERDICT_C.get(r.verdict, '#0b0b0b')}'>{r.verdict}</div>", unsafe_allow_html=True); k[1].metric("Mean ΔAUC (protein − RNA)", f"{r.mean_delta:+.3f}")
    k[2].metric("Best AUC (any view/model)", f"{r.best_AUC_any:.3f}"); k[3].metric("Positives / total", f"{int(T['benchmark_metrics'].query('cohort==@coh and task==@task').n_pos.iloc[0])} / {int(T['benchmark_metrics'].query('cohort==@coh and task==@task').n.iloc[0])}")
    met = T["benchmark_metrics"].query("cohort == @coh and task == @task")
    fig = px.bar(met, x="view", y="AUC", color="model", barmode="group", error_y=met.AUC_hi - met.AUC, error_y_minus=met.AUC - met.AUC_lo,
                 color_discrete_sequence=[BLUE, ORANGE, AQUA], category_orders={"view": VIEWS})
    fig.update_yaxes(range=[0.3, 1.02], title="AUC (95% bootstrap CI)"); fig.update_xaxes(title=None)
    c1, c2 = st.columns([1.3, 1])
    c1.plotly_chart(style(fig, 400), width="stretch")
    oof = T["oof"].query("cohort == @coh and task == @task and model == @mdl")
    fig = go.Figure()
    for v in ["RNA", "Protein", "Late fusion"]:
        o = oof[oof.view == v]; fpr, tpr, _ = roc_curve(o.y, o.p)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{v} ({roc_auc_score(o.y, o.p):.3f})", line=dict(color=VIEW_C[v], width=3 if v == "Late fusion" else 2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="#52514e"), showlegend=False))
    fig.update_xaxes(title="False positive rate"); fig.update_yaxes(title="True positive rate", scaleanchor="x")
    c2.plotly_chart(style(fig, 400), width="stretch")
    st.dataframe(T["pairwise_tests"].query("cohort == @coh and task == @task and model == @mdl")[["comparison", "AUC_a", "AUC_b", "delta", "delta_lo", "delta_hi", "p_DeLong", "q_BH"]].round(4),
                 width="stretch", hide_index=True)
    st.subheader("What drives the models (SHAP, XGBoost fitted to all tumours)")
    sh = T["shap_top20"].query("cohort == @coh and task == @task")
    c1, c2 = st.columns(2)
    for col_, layer, colr in [(c1, "RNA", BLUE), (c2, "Protein", ORANGE)]:
        s = sh[sh.layer == layer].sort_values("mean_abs_shap")
        other = set(sh[sh.layer != layer].gene)
        fig = px.bar(s, x="mean_abs_shap", y="gene", orientation="h", color=np.where(s.gene.isin(other), "In both top-20", f"{layer} only"),
                     color_discrete_map={"In both top-20": YELLOW, f"{layer} only": colr})
        fig.update_xaxes(title="mean |SHAP|"); fig.update_yaxes(title=None)
        col_.markdown(f"**{layer} model**"); col_.plotly_chart(style(fig, 520), width="stretch")
    st.subheader("Gene by gene: discrimination as RNA vs as protein")
    g = T["genes"].query("cohort == @coh and task == @task")
    fig = px.scatter(g, x="AUC_rna", y="AUC_protein", color="rho", color_continuous_scale="RdBu", range_color=[-0.4, 1], hover_name="gene",
                     hover_data={"rho": ":.2f", "AUC_rna": ":.3f", "AUC_protein": ":.3f"})
    fig.add_shape(type="line", x0=0.5, y0=0.5, x1=1, y1=1, line=dict(dash="dash", color="#52514e"))
    fig.update_xaxes(title="Single-gene AUC as RNA", range=[0.5, 1]); fig.update_yaxes(title="Single-gene AUC as protein", range=[0.5, 1], scaleanchor="x")
    fig.update_layout(coloraxis_colorbar_title="mRNA–protein ρ")
    st.plotly_chart(style(fig, 520), width="stretch")
    st.caption("Shows the 300 most discriminative genes in each layer. Genes far from the diagonal carry layer-specific information.")

# ======================================================================= Mutation effects
elif page == "Mutation effects":
    st.title("Does the mutated gene's own protein flag its mutation?")
    st.markdown("For each mutation task, we ask how well the **mutated gene's own RNA** and **own protein** separate mutant from wild-type tumours.")
    cis = T["cis_mutation_rna_vs_protein"].assign(Cancer=lambda d: d.cohort.map(NAME))
    cis["label"] = cis.gene + " · " + cis.Cancer
    k = st.columns(3)
    k[0].metric("Tumour suppressors: protein better", f"{K['cis_TS_protein_better']}/{K['cis_TS_n']}", help=f"Wilcoxon p = {K['cis_TS_wilcoxon_p']}")
    k[1].metric("Median gain (tumour suppressors)", f"+{K['cis_TS_median_delta']:.2f} AUC")
    k[2].metric("Tumour suppressors vs oncogenes", f"p = {K['cis_TS_vs_OG_mannwhitney_p']}")
    s = cis.sort_values("delta_protein_minus_rna")
    fig = go.Figure()
    for _, r in s.iterrows():
        fig.add_trace(go.Scatter(x=[r.RNA_AUC, r.Protein_AUC], y=[r.label] * 2, mode="lines", line=dict(color="#e4e3df", width=4), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=s.RNA_AUC, y=s.label, mode="markers", name="Own RNA", marker=dict(color=BLUE, size=11)))
    fig.add_trace(go.Scatter(x=s.Protein_AUC, y=s.label, mode="markers", name="Own protein", marker=dict(color=ORANGE, size=11, symbol="diamond"),
                             customdata=np.c_[s.role, s.q_BH], hovertemplate="%{y}<br>protein AUC %{x:.3f}<br>%{customdata[0]}, q=%{customdata[1]:.3g}<extra></extra>"))
    fig.update_xaxes(title="AUC", range=[0.45, 1])
    c1, c2 = st.columns([1.2, 1])
    c1.plotly_chart(style(fig, 560), width="stretch")
    pick = c2.selectbox("Show tumours for", s.label[::-1].tolist())
    r = s[s.label == pick].iloc[0]; lab = labels(r.cohort)[f"{r.gene} mutation"].dropna()
    rows = []
    for layer in ["RNA", "Protein"]:
        X = layer_matrix(r.cohort, layer)
        if r.gene in X.columns:
            v = X.loc[lab.index, r.gene]; v = (v - v.mean()) / v.std()
            rows.append(pd.DataFrame({"layer": layer, "status": lab.map({1: "Mutant", 0: "Wild-type"}), "z": v}))
    if rows:
        d = pd.concat(rows).dropna()
        fig = px.box(d, x="layer", y="z", color="status", points="all", color_discrete_map={"Mutant": ORANGE, "Wild-type": "#9a9893"})
        fig.update_yaxes(title=f"{r.gene} level (z-score)"); fig.update_xaxes(title=None)
        c2.plotly_chart(style(fig, 420), width="stretch")
        c2.caption(f"RNA AUC {r.RNA_AUC:.2f} ({r.RNA_direction}) · protein AUC {r.Protein_AUC:.2f} ({r.Protein_direction}) · {r.role}")
    else:
        c2.info("This gene is not measured with ≥70% coverage in both layers for the gene lookup; see the table below.")
    st.dataframe(cis[["Cancer", "gene", "role", "n", "n_mut", "RNA_AUC", "RNA_direction", "Protein_AUC", "Protein_direction",
                      "delta_protein_minus_rna", "delta_lo", "delta_hi", "p_DeLong", "q_BH"]].round(3), width="stretch", hide_index=True)
    st.caption("Mutant TP53 protein is higher in mutated tumours (missense mutations stabilise it), while ARID1A and PTEN protein are lost. "
               "β-catenin (CTNNB1) mutations block its degradation, so protein accumulates. Oncogenes activated through expression "
               "(EGFR in lung) are better read from RNA.")

# ======================================================================= Clusters & pathways
elif page == "Clusters & pathways":
    st.title("Unsupervised structure and pathways")
    cl = T["clustering"]
    ari = cl[cl.layer.astype(str).str.startswith("ARI")][["cohort", "layer", "ARI"]].assign(Cancer=lambda d: d.cohort.map(NAME))
    c1, c2 = st.columns([1, 1])
    a = ari[ari.layer.str.contains(" vs ") & ~ari.layer.str.contains("PAM50")]
    fig = px.bar(a, x="Cancer", y="ARI", color="layer", barmode="group", color_discrete_sequence=[BLUE, AQUA, ORANGE])
    fig.update_yaxes(title="Adjusted Rand index (k = 3)", range=[-0.05, 1])
    c1.subheader("Do RNA and protein group tumours the same way?"); c1.plotly_chart(style(fig, 420), width="stretch")
    pam = ari[ari.layer.str.contains("PAM50")]
    c2.subheader("Recovery of breast PAM50 subtypes (k = 5)")
    c2.dataframe(pam[["layer", "ARI"]].round(3), width="stretch", hide_index=True)
    c2.caption("Endometrial RNA and protein clusters are essentially unrelated (ARI ≈ 0), the lowest agreement of the five cancers.")
    st.subheader("Hallmark pathways enriched among genes that discriminate better in one layer")
    en = T["hallmark_enrichment_task_signals"].assign(Cancer=lambda d: d.cohort.map(NAME))
    qmax = st.select_slider("q-value cut-off", [0.01, 0.05, 0.1, 0.25, 1.0], value=0.05)
    st.dataframe(en[en.q <= qmax][["Cancer", "task", "direction", "set", "hits", "size_in_background", "p", "q", "genes"]].round(4),
                 width="stretch", hide_index=True)
    st.caption("Exploratory: hypergeometric test of the 150 genes with the largest protein (or RNA) advantage per task, against all genes measured in both layers.")

# ======================================================================= Predict
elif page == "Predict":
    st.title("Score your own tumours")
    st.caption("L2 logistic regression models trained on all CPTAC tumours of the chosen cancer. Research use only.")
    coefs, info = predictor()
    c1, c2 = st.columns(2)
    coh = c1.selectbox("Cancer", COHORTS, format_func=NAME.get)
    task = c2.selectbox("Task", [t for t in TASKS[coh] if not t.startswith("Sex")])
    r = CONS[(CONS.cohort == coh) & (CONS.task == task)].iloc[0]
    st.info(f"Cross-validated AUC in CPTAC (logistic regression): RNA {T['benchmark_metrics'].query('cohort==@coh and task==@task and model==\"LR\" and view==\"RNA\"').AUC.iloc[0]:.3f}, "
            f"protein {T['benchmark_metrics'].query('cohort==@coh and task==@task and model==\"LR\" and view==\"Protein\"').AUC.iloc[0]:.3f}, "
            f"late fusion {T['benchmark_metrics'].query('cohort==@coh and task==@task and model==\"LR\" and view==\"Late fusion\"').AUC.iloc[0]:.3f} · verdict: {r.verdict}")
    st.markdown("Upload one CSV per layer: **rows = tumours, first column `sample_id`, other columns = gene symbols**. "
                "RNA on any log scale (raw counts/FPKM are log2-transformed automatically); protein as log2 ratios or intensities. "
                "Values are standardised **within your batch**, so upload at least **10 tumours**, ideally with both classes present.")
    c1, c2, c3 = st.columns(3)
    up_r = c1.file_uploader("RNA CSV", type="csv"); up_p = c2.file_uploader("Protein CSV", type="csv")
    demo = c3.button("Use demo: 30 CPTAC breast tumours") or st.session_state.get("demo_on")
    if demo: st.session_state["demo_on"] = True
    data = {}
    if up_r is not None: data["RNA"] = pd.read_csv(up_r, index_col=0)
    if up_p is not None: data["Protein"] = pd.read_csv(up_p, index_col=0)
    if not data and demo and coh == "BRCA":
        data = {"RNA": pd.read_csv(D / "demo_BRCA_rna.csv", index_col=0), "Protein": pd.read_csv(D / "demo_BRCA_protein.csv", index_col=0)}
        st.warning("Demo tumours were part of the training data, so these scores are optimistic; see Benchmark for honest performance.")
    elif not data and demo:
        st.info("The demo uses breast tumours: choose Breast as the cancer.")
    if data:
        probs, contribs = {}, {}
        for layer, X in data.items():
            X = X.apply(pd.to_numeric, errors="coerce")
            if len(X) < 10: st.error(f"{layer}: only {len(X)} tumours; at least 10 are needed."); st.stop()
            cf = coefs[(coefs.cohort == coh) & (coefs.task == task) & (coefs.layer == layer)].set_index("gene").coef
            have = [g for g in cf.index if g in X.columns]
            st.write(f"**{layer}:** {len(X)} tumours · {len(have)}/{len(cf)} model genes found")
            if len(have) < 0.5 * len(cf): st.warning(f"{layer}: fewer than half of the model genes are present; scores will be unreliable.")
            Z = X.reindex(columns=cf.index)
            if layer == "RNA" and np.nanmax(Z.values) > 100: Z = np.log2(Z.clip(lower=0) + 1)
            Z = Z.fillna(Z.min() - 1) if layer == "RNA" else Z.fillna(Z.median())
            Z = ((Z - Z.mean()) / Z.std().replace(0, 1)).fillna(0)
            lin = info[f"{coh}|{task}|{layer}"]["intercept"] + Z.values @ cf.values
            probs[layer] = pd.Series(1 / (1 + np.exp(-lin)), index=X.index)
            contribs[layer] = pd.DataFrame(Z.values * cf.values, index=X.index, columns=cf.index)
        res = pd.DataFrame({f"P({task}) {k_}": v for k_, v in probs.items()})
        if len(probs) == 2:
            common = probs["RNA"].index.intersection(probs["Protein"].index)
            res.loc[common, f"P({task}) late fusion"] = (probs["RNA"][common] + probs["Protein"][common]) / 2
        st.subheader("Scores"); st.dataframe(res.round(3), width="stretch")
        st.download_button("Download scores", res.to_csv().encode(), f"{coh}_{task.replace(' ', '_')}_scores.csv", "text/csv")
        st.subheader("Explain one tumour")
        c1, c2 = st.columns(2)
        sid = c1.selectbox("Tumour", res.index.tolist()); layer = c2.radio("Layer", list(contribs), horizontal=True)
        s = contribs[layer].loc[sid]; s = s.reindex(s.abs().sort_values(ascending=False).index).head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=[ORANGE if v > 0 else BLUE for v in s.values]))
        fig.add_vline(x=0, line_color="#52514e"); fig.update_xaxes(title=f"Contribution to log-odds of '{task}'  ← negative | positive →")
        st.plotly_chart(style(fig, 520), width="stretch")
