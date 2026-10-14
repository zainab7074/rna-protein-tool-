# RNA or Protein? — proteogenomic ML comparison across five cancers

Interactive web tool accompanying the study **"When does proteomics add to RNA? A systematic machine-learning comparison of transcriptomic and proteomic profiles for clinical and molecular prediction across five cancers."**

**Research use only.** Not a diagnostic device.

## Sections
| Section | Content |
|---|---|
| Overview | Question, design and main findings |
| Data & RNA–protein coupling | Cohorts, gene-wise mRNA–protein correlation, Hallmark coupling, lookup for any gene |
| Benchmark | RNA vs protein vs fusion for all 29 tasks, three models, per-task CIs and across-task tests |
| Task explorer | Per-task AUCs, ROC curves, DeLong tests, SHAP features, gene-by-gene comparison |
| Mutation effects | Whether a mutated gene's own protein flags the mutation better than its RNA |
| Clusters & pathways | RNA vs protein clustering agreement, PAM50 recovery, Hallmark enrichment |
| Predict | Score your own tumours (RNA and/or protein CSV, ≥10 tumours) with per-tumour explanations |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Create a GitHub repository and upload everything in this folder, keeping `data/`, `models/` and `.streamlit/`.
2. At https://share.streamlit.io choose **Create app**, select the repository, branch `main`, file `app.py`.

## Data
CPTAC cohorts obtained from cBioPortal: breast (Krug et al., Cell 2020), lung adenocarcinoma (Gillette et al., Cell 2020),
endometrial (Dou et al., Cell 2020), colon (Vasaikar et al., Cell 2019), glioblastoma (Wang et al., Cancer Cell 2021).
Hallmark gene sets: MSigDB (Liberzon et al., Cell Systems 2015).

## Models
Benchmark: L2 logistic regression (inner-CV tuned), random forest and XGBoost; 5-fold CV × 3 repeats with all preprocessing inside training folds.
Predictor: L2 logistic regression per cancer, task and layer, trained on all tumours; uploaded data are standardised within the uploaded batch.
