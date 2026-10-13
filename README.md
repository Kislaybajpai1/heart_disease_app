# Heart Disease Risk Explorer (DMDW Project)

A Streamlit app that applies **classification, clustering and association rule mining**
to the UCI Heart Disease (Cleveland) dataset.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
Keep `app.py` and `heart.csv` in the same folder. The app opens at http://localhost:8501.

## What each page does
| Page | Technique | What you see |
|---|---|---|
| Overview | EDA | dataset preview, class balance, correlation heatmap |
| Classification | Logistic Regression, KNN, Random Forest | accuracy / precision / recall / F1 / ROC-AUC, ROC curves, confusion matrix, feature importance |
| Clustering | K-Means (+ PCA view) | elbow and silhouette plots, 2D cluster plot, risk profile of each cluster |
| Association Rules | Apriori / FP-Growth (mlxtend) | rules leading to heart disease, with support, confidence, lift |
| Patient Risk Check | all three | enter one patient: predicted probability, cluster, matching rules |

All thresholds (k, min support, min confidence, test size) are in the sidebar Settings.

## Data notes
- 303 rows originally; 7 rows with placeholder values for `ca` (4) or `thal` (0) are dropped, leaving 296.
- In the bundled CSV, `target = 1` means *no* disease. The app flips it so 1 = disease.
- You can upload another CSV with the same columns from the sidebar.

## Disclaimer
Classroom project on a small public dataset. Not for medical diagnosis.
