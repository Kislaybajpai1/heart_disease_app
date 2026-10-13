"""
Heart Disease Risk Explorer
---------------------------
DMDW project: Classification + Clustering + Association Rule Mining
on the UCI Heart Disease (Cleveland) dataset, wrapped in a Streamlit app.

Run:  streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Heart Disease Risk Explorer", page_icon="🫀", layout="wide")

DATA_PATH = Path(__file__).parent / "heart.csv"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "target"  # after loading: 1 = heart disease present, 0 = absent

# Human-readable names for the input form
FRIENDLY = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thalassemia (thal)",
}
SEX_OPTS = {0: "Female", 1: "Male"}
CP_OPTS = {0: "Asymptomatic", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Typical angina"}
FBS_OPTS = {0: "No", 1: "Yes"}
ECG_OPTS = {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}
EXANG_OPTS = {0: "No", 1: "Yes"}
SLOPE_OPTS = {0: "Downsloping", 1: "Flat", 2: "Upsloping"}
THAL_OPTS = {0: "Unknown (0)", 1: "Fixed defect", 2: "Normal", 3: "Reversible defect"}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def clean(df):
    """Drop rows where 'ca' = 4 or 'thal' = 0 (placeholders for missing values in this dataset)."""
    df = df.dropna(subset=FEATURES + [TARGET])
    return df[(df["ca"] != 4) & (df["thal"] != 0)].reset_index(drop=True)


@st.cache_data
def load_default():
    df = pd.read_csv(DATA_PATH)
    # In this public copy of the data, target 1 means NO disease. Flip it so 1 = disease.
    df[TARGET] = 1 - df[TARGET]
    return clean(df)


def validate(df):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        return f"Missing columns: {', '.join(missing)}"
    return None


def get_data():
    up = st.sidebar.file_uploader("Use your own CSV (optional)", type="csv",
                                  help="Must have the same columns as the UCI Heart Disease dataset.")
    if up is not None:
        df = pd.read_csv(up)
        err = validate(df)
        if err:
            st.sidebar.error(err + ". Using the bundled dataset instead.")
        else:
            if st.sidebar.checkbox("In my file, target 1 means NO disease"):
                df[TARGET] = 1 - df[TARGET]
            df = clean(df)
            st.sidebar.success(f"Loaded {len(df)} rows from your file.")
            return df
    return load_default()


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training classifiers...")
def train_classifiers(df, test_size, seed):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed),
    }
    results, fitted, probs, preds = [], {}, {}, {}
    for name, m in models.items():
        m.fit(Xs_tr, y_tr)
        p = m.predict(Xs_te)
        pr = m.predict_proba(Xs_te)[:, 1]
        fitted[name], probs[name], preds[name] = m, pr, p
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, p),
            "Precision": precision_score(y_te, p),
            "Recall": recall_score(y_te, p),
            "F1": f1_score(y_te, p),
            "ROC-AUC": roc_auc_score(y_te, pr),
        })
    return {
        "scaler": scaler, "models": fitted, "probs": probs, "preds": preds,
        "y_te": y_te.values, "table": pd.DataFrame(results).set_index("Model"),
    }


# ----------------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Scanning k values...")
def k_scan(df, seed):
    Xs = StandardScaler().fit_transform(df[FEATURES])
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        rows.append({"k": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(Xs, km.labels_)})
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Clustering patients...")
def fit_clusters(df, k, seed):
    scaler = StandardScaler().fit(df[FEATURES])
    Xs = scaler.transform(df[FEATURES])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
    labels = km.labels_
    pca = PCA(n_components=2, random_state=seed).fit(Xs)
    prof = df[FEATURES + [TARGET]].copy()
    prof["cluster"] = labels
    summary = prof.groupby("cluster").agg(
        patients=(TARGET, "size"),
        disease_rate=(TARGET, "mean"),
        avg_age=("age", "mean"),
        avg_chol=("chol", "mean"),
        avg_bp=("trestbps", "mean"),
        avg_max_hr=("thalach", "mean"),
        avg_oldpeak=("oldpeak", "mean"),
        exercise_angina=("exang", "mean"),
    )
    order = summary["disease_rate"].sort_values().index.tolist()
    tags = {2: ["Lower risk", "Higher risk"], 3: ["Lower risk", "Medium risk", "Higher risk"]}
    names = {}
    for rank, cid in enumerate(order):
        tag = tags[k][rank] if k in tags else f"Risk rank {rank + 1} of {k}"
        names[cid] = f"Cluster {cid} - {tag}"
    summary.insert(0, "group", [names[c] for c in summary.index])
    return {"scaler": scaler, "km": km, "pca": pca, "labels": labels, "summary": summary,
            "names": names, "coords": pca.transform(Xs), "silhouette": silhouette_score(Xs, labels)}


# ----------------------------------------------------------------------------
# Association rules
# ----------------------------------------------------------------------------
def to_items(df):
    """Turn numeric patient rows into readable categorical items (for Apriori)."""
    out = pd.DataFrame(index=df.index)
    out["Age"] = "Age " + pd.cut(df["age"], [0, 39, 54, 64, 200],
                                 labels=["<40", "40-54", "55-64", "65+"]).astype(str)
    out["Sex"] = df["sex"].map(SEX_OPTS)
    out["ChestPain"] = "Chest pain: " + df["cp"].map(CP_OPTS)
    out["BP"] = "BP " + pd.cut(df["trestbps"], [0, 119, 139, 400],
                               labels=["normal", "elevated", "high"]).astype(str)
    out["Cholesterol"] = "Cholesterol " + pd.cut(df["chol"], [0, 199, 239, 1000],
                                                 labels=["desirable", "borderline", "high"]).astype(str)
    out["Sugar"] = "Fasting sugar " + df["fbs"].map({0: "normal", 1: "high"})
    out["ECG"] = "Resting ECG: " + df["restecg"].map(ECG_OPTS)
    out["MaxHR"] = "Max heart rate " + pd.cut(df["thalach"], [0, 119, 150, 300],
                                             labels=["low", "medium", "high"]).astype(str)
    out["Angina"] = "Exercise angina: " + df["exang"].map(EXANG_OPTS)
    out["STdep"] = "ST depression " + pd.cut(df["oldpeak"], [-1, 0.99, 2, 20],
                                            labels=["low", "moderate", "high"]).astype(str)
    out["Slope"] = "ST slope: " + df["slope"].map(SLOPE_OPTS)
    out["Vessels"] = "Vessels coloured: " + df["ca"].astype(int).astype(str)
    out["Thal"] = "Thal: " + df["thal"].map(THAL_OPTS)
    return out


@st.cache_data(show_spinner="Mining association rules...")
def mine_rules(df, algo, min_support, min_conf, max_len):
    items = to_items(df)
    items["Outcome"] = df[TARGET].map({1: "Heart disease", 0: "No heart disease"})
    onehot = pd.get_dummies(items, prefix="", prefix_sep="").astype(bool)
    fn = apriori if algo == "Apriori" else fpgrowth
    freq = fn(onehot, min_support=min_support, use_colnames=True, max_len=max_len)
    if freq.empty:
        return freq, pd.DataFrame()
    rules = association_rules(freq, num_itemsets=len(onehot), metric="confidence", min_threshold=min_conf)
    return freq, rules


def fmt_set(s):
    return " AND ".join(sorted(s))


def rules_table(rules, outcome):
    if rules.empty:
        return rules
    sel = rules[rules["consequents"].apply(lambda c: c == frozenset([outcome]))]
    sel = sel[sel["lift"] > 1].sort_values(["lift", "confidence"], ascending=False)
    t = pd.DataFrame({
        "If patient has...": sel["antecedents"].apply(fmt_set),
        "...then": outcome,
        "Support": sel["support"].round(3),
        "Confidence": sel["confidence"].round(3),
        "Lift": sel["lift"].round(2),
    })
    return t.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Small plot helpers
# ----------------------------------------------------------------------------
def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
df = get_data()

st.sidebar.title("🫀 Heart Disease Explorer")
page = st.sidebar.radio("Go to", [
    "🏠 Overview", "🎯 Classification", "🧩 Clustering",
    "🔗 Association Rules", "🩺 Patient Risk Check"])

with st.sidebar.expander("⚙️ Settings", expanded=False):
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    k = st.slider("Number of clusters (k)", 2, 6, 3)
    algo = st.selectbox("Rule mining algorithm", ["Apriori", "FP-Growth"])
    min_support = st.slider("Min support", 0.05, 0.5, 0.10, 0.01)
    min_conf = st.slider("Min confidence", 0.3, 0.95, 0.70, 0.05)
    SEED = 42

st.sidebar.caption("Educational project. Not a medical diagnosis tool.")

clf = train_classifiers(df, test_size, SEED)
clu = fit_clusters(df, k, SEED)

# ---------------------------------------------------------------- Overview
if page.startswith("🏠"):
    st.title("🫀 Heart Disease Risk Explorer")
    st.write("A data mining project that looks at the same patients in three ways: "
             "**classification** predicts disease, **clustering** finds risk groups, and "
             "**association rule mining** explains which feature combinations go with disease.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", len(df))
    c2.metric("Features", len(FEATURES))
    c3.metric("With heart disease", f"{df[TARGET].mean():.0%}")
    c4.metric("Best model AUC", f"{clf['table']['ROC-AUC'].max():.2f}")

    st.subheader("How the pieces fit")
    st.markdown(
        "- 🎯 **Classification** - Logistic Regression, KNN and Random Forest compared on accuracy, recall and ROC-AUC.\n"
        "- 🧩 **Clustering** - K-Means groups patients without seeing the label; we then check each group's disease rate.\n"
        "- 🔗 **Association rules** - Apriori / FP-Growth find rules like *Exercise angina AND Asymptomatic chest pain → Heart disease*.\n"
        "- 🩺 **Patient Risk Check** - enter one patient and see all three results together.")

    st.subheader("Dataset preview")
    st.dataframe(df.head(20), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Class balance")
        fig, ax = plt.subplots(figsize=(4, 3))
        counts = df[TARGET].map({0: "No disease", 1: "Disease"}).value_counts()
        ax.bar(counts.index, counts.values, color=["#4C9F70", "#D1495B"][: len(counts)])
        ax.set_ylabel("Patients")
        show(fig)
    with col2:
        st.caption("Feature correlation")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.heatmap(df[FEATURES + [TARGET]].corr(), cmap="coolwarm", center=0, ax=ax,
                    cbar_kws={"shrink": 0.7}, xticklabels=True, yticklabels=True)
        show(fig)

# ---------------------------------------------------------- Classification
elif page.startswith("🎯"):
    st.title("🎯 Classification")
    st.write("Three models are trained on the same split. In a medical setting **recall** matters most, "
             "because missing a patient who has the disease is worse than a false alarm.")

    st.dataframe(clf["table"].style.format("{:.3f}").highlight_max(axis=0, color="#cfe8d5"),
                 width="stretch")

    model_name = st.selectbox("Inspect a model", list(clf["models"].keys()), index=2)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("ROC curves")
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, pr in clf["probs"].items():
            fpr, tpr, _ = roc_curve(clf["y_te"], pr)
            ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(clf['y_te'], pr):.2f})",
                    lw=3 if name == model_name else 1.5)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=8)
        show(fig)
    with c2:
        st.caption(f"Confusion matrix - {model_name}")
        cm = confusion_matrix(clf["y_te"], clf["preds"][model_name])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No disease", "Disease"], yticklabels=["No disease", "Disease"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        show(fig)

    st.caption("Which features matter most (Random Forest)")
    imp = pd.Series(clf["models"]["Random Forest"].feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color="#3D5A80")
    ax.set_xlabel("Importance")
    show(fig)

# -------------------------------------------------------------- Clustering
elif page.startswith("🧩"):
    st.title("🧩 Clustering")
    st.write("K-Means groups patients using only their measurements (the diagnosis is hidden). "
             "Afterwards we check how much disease each group actually has.")

    scan = k_scan(df, SEED)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Elbow method (lower inertia = tighter clusters)")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(scan["k"], scan["Inertia"], "o-", color="#3D5A80")
        ax.axvline(k, color="#D1495B", ls="--")
        ax.set_xlabel("k")
        ax.set_ylabel("Inertia")
        show(fig)
    with c2:
        st.caption("Silhouette score (higher = better separated)")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(scan["k"], scan["Silhouette"], "o-", color="#4C9F70")
        ax.axvline(k, color="#D1495B", ls="--")
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette")
        show(fig)
    st.caption(f"Currently using k = {k} (change it in the sidebar Settings). "
               f"Silhouette for this k: {clu['silhouette']:.3f}")

    st.subheader("Patient groups (2D view using PCA)")
    coords = clu["coords"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    palette = sns.color_palette("Set2", k)
    for cid in range(k):
        mask = clu["labels"] == cid
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=22, color=palette[cid],
                        label=clu["names"][cid])
    axes[0].set_title("Coloured by cluster")
    axes[0].legend(fontsize=7)
    axes[1].scatter(coords[:, 0], coords[:, 1], s=22,
                    c=df[TARGET].map({0: "#4C9F70", 1: "#D1495B"}))
    axes[1].set_title("Coloured by actual diagnosis (green = none, red = disease)")
    for a in axes:
        a.set_xlabel("PC1")
        a.set_ylabel("PC2")
    show(fig)

    st.subheader("Cluster profiles")
    summ = clu["summary"].copy()
    summ["disease_rate"] = (summ["disease_rate"] * 100).round(1).astype(str) + "%"
    summ["exercise_angina"] = (summ["exercise_angina"] * 100).round(0).astype(int).astype(str) + "%"
    for col in ["avg_age", "avg_chol", "avg_bp", "avg_max_hr", "avg_oldpeak"]:
        summ[col] = summ[col].round(1)
    st.dataframe(summ, width="stretch")
    st.info("Groups are named by their disease rate: the cluster with the lowest share of patients "
            "with disease is 'Lower risk', and so on.")

# -------------------------------------------------------- Association rules
elif page.startswith("🔗"):
    st.title("🔗 Association Rule Mining")
    st.write("Numeric measurements are converted into readable items (e.g. *Cholesterol high*, "
             "*Exercise angina: Yes*) and then mined for combinations that go with the outcome.")

    max_len = st.slider("Max items per rule", 2, 4, 3)
    outcome = st.radio("Show rules that lead to", ["Heart disease", "No heart disease"], horizontal=True)
    freq, rules = mine_rules(df, algo, min_support, min_conf, max_len)
    table = rules_table(rules, outcome)

    c1, c2, c3 = st.columns(3)
    c1.metric("Frequent itemsets", len(freq))
    c2.metric("Rules (all)", len(rules))
    c3.metric(f"Rules → {outcome}", len(table))

    if table.empty:
        st.warning("No rules found. Try lowering min support or min confidence in the sidebar.")
    else:
        top_n = st.slider("Rows to show", 5, min(50, max(5, len(table))), min(15, len(table)))
        st.dataframe(table.head(top_n), width="stretch")

        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(table["Support"], table["Confidence"], c=table["Lift"], cmap="viridis", s=40)
        plt.colorbar(sc, label="Lift")
        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        ax.set_title("Rules: support vs confidence (colour = lift)")
        show(fig)

    with st.expander("How to read these numbers"):
        st.markdown(
            "- **Support**: share of all patients who match the whole rule.\n"
            "- **Confidence**: of patients matching the 'if' part, the share who also have the outcome.\n"
            "- **Lift**: how much more likely the outcome is than normal. Above 1 means a positive link.")

# ------------------------------------------------------ Patient risk check
else:
    st.title("🩺 Patient Risk Check")
    st.write("Enter the measurements for one patient to see the classifier's prediction, "
             "the patient's cluster, and any mined rules that match.")

    med = df[FEATURES].median()
    with st.form("patient"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input(FRIENDLY["age"], 20, 100, int(med["age"]))
        sex = c1.selectbox(FRIENDLY["sex"], list(SEX_OPTS), format_func=SEX_OPTS.get, index=1)
        cp = c1.selectbox(FRIENDLY["cp"], list(CP_OPTS), format_func=CP_OPTS.get)
        trestbps = c1.number_input(FRIENDLY["trestbps"], 80, 220, int(med["trestbps"]))
        chol = c2.number_input(FRIENDLY["chol"], 100, 600, int(med["chol"]))
        fbs = c2.selectbox(FRIENDLY["fbs"], list(FBS_OPTS), format_func=FBS_OPTS.get)
        restecg = c2.selectbox(FRIENDLY["restecg"], list(ECG_OPTS), format_func=ECG_OPTS.get)
        thalach = c2.number_input(FRIENDLY["thalach"], 60, 220, int(med["thalach"]))
        exang = c3.selectbox(FRIENDLY["exang"], list(EXANG_OPTS), format_func=EXANG_OPTS.get)
        oldpeak = c3.number_input(FRIENDLY["oldpeak"], 0.0, 7.0, float(med["oldpeak"]), step=0.1)
        slope = c3.selectbox(FRIENDLY["slope"], list(SLOPE_OPTS), format_func=SLOPE_OPTS.get, index=1)
        ca = c3.selectbox(FRIENDLY["ca"], [0, 1, 2, 3])
        thal = c3.selectbox(FRIENDLY["thal"], [1, 2, 3], format_func=THAL_OPTS.get, index=1)
        model_name = st.selectbox("Classifier", list(clf["models"].keys()), index=2)
        go = st.form_submit_button("Check risk", type="primary")

    if go:
        patient = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                 thalach, exang, oldpeak, slope, ca, thal]], columns=FEATURES)

        # 1) Classification
        prob = clf["models"][model_name].predict_proba(clf["scaler"].transform(patient))[0, 1]
        # 2) Clustering
        cid = int(clu["km"].predict(clu["scaler"].transform(patient))[0])
        row = clu["summary"].loc[cid]
        # 3) Association rules
        _, rules = mine_rules(df, algo, min_support, min_conf, 3)
        p_items = set(to_items(patient).iloc[0].dropna().astype(str))
        matched = pd.DataFrame()
        if not rules.empty:
            tbl = rules[rules["consequents"].apply(lambda c: c == frozenset(["Heart disease"]))]
            tbl = tbl[tbl["antecedents"].apply(lambda a: a.issubset(p_items)) & (tbl["lift"] > 1)]
            matched = tbl.sort_values("lift", ascending=False)

        level = "Low" if prob < 0.35 else "Moderate" if prob < 0.65 else "High"
        icon = {"Low": "🟢", "Moderate": "🟠", "High": "🔴"}[level]

        st.subheader("Results")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown("**🎯 Classification**")
            st.metric(f"{model_name}", f"{prob:.0%} disease probability")
            st.progress(float(prob))
            st.write(f"{icon} Estimated risk level: **{level}**")
        with r2:
            st.markdown("**🧩 Clustering**")
            st.metric("Patient group", clu["names"][cid].split(" - ")[1] if " - " in clu["names"][cid] else clu["names"][cid])
            st.write(f"{clu['names'][cid]}: {int(row['patients'])} similar patients, "
                     f"{row['disease_rate']:.0%} of them have heart disease.")
        with r3:
            st.markdown("**🔗 Association rules**")
            st.metric("Matching rules", len(matched))
            st.write("Rules (leading to heart disease) whose conditions this patient meets.")

        if not matched.empty:
            show_tbl = pd.DataFrame({
                "Matched pattern": matched["antecedents"].apply(fmt_set),
                "Confidence": matched["confidence"].round(2),
                "Lift": matched["lift"].round(2),
            }).head(10).reset_index(drop=True)
            st.dataframe(show_tbl, width="stretch")

        st.caption("This is a classroom demonstration built on a small public dataset. "
                   "It is not medical advice and must not be used for diagnosis.")
