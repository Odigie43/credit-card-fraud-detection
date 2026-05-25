"""
Credit Card Fraud Detection — Streamlit App
Converted from Clement_ML.ipynb
"""

import matplotlib
matplotlib.use("Agg")          # must come BEFORE pyplot import — fixes Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detector — Clement ML",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour map (matches notebook) ─────────────────────────────
COLOURS = {
    "Logistic Regression": "#2196F3",
    "Decision Tree":       "#FF9800",
    "Random Forest":       "#4CAF50",
    "XGBoost":             "#F44336",
}
FEATURES = [
    "amt", "log_amt", "hour", "day_of_week", "month", "age",
    "distance_km", "city_pop", "is_night", "is_weekend",
    "category_enc", "state_enc",
]
MODEL_FILE = "xgb_model.pkl"


# ═══════════════════════════════════════════════════════════
# Helper — feature engineering (same logic as notebook)
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Engineering features…")
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"]                   = pd.to_datetime(df["dob"])
    df["hour"]         = df["trans_date_trans_time"].dt.hour
    df["day_of_week"]  = df["trans_date_trans_time"].dt.dayofweek
    df["month"]        = df["trans_date_trans_time"].dt.month
    df["age"]          = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    df["distance_km"]  = np.sqrt(
        (df["lat"] - df["merch_lat"]) ** 2 +
        (df["long"] - df["merch_long"]) ** 2
    ) * 111
    df["log_amt"]      = np.log1p(df["amt"])
    df["is_night"]     = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["category_enc"] = LabelEncoder().fit_transform(df["category"])
    df["state_enc"]    = LabelEncoder().fit_transform(df["state"])
    return df


# ═══════════════════════════════════════════════════════════
# Helper — train all 4 models (cached so only runs once)
# ═══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Training models — this takes ~60 s…")
def train_models(df: pd.DataFrame):
    df = engineer_features(df)
    X, y = df[FEATURES], df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sm = SMOTE(random_state=42, sampling_strategy=0.1)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=10, eval_metric="logloss",
            n_jobs=-1, random_state=42),
    }

    results, probas, preds = {}, {}, {}
    for name, model in models.items():
        model.fit(X_sm, y_sm)
        proba = model.predict_proba(X_test)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        probas[name] = proba
        preds[name]  = pred
        results[name] = {
            "AUC-ROC":   round(roc_auc_score(y_test, proba), 4),
            "PR-AUC":    round(average_precision_score(y_test, proba), 4),
            "F1":        round(f1_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred), 4),
            "Recall":    round(recall_score(y_test, pred), 4),
        }

    # Save XGBoost for the live predictor
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(models["XGBoost"], f)

    return models, results, probas, preds, X_test, y_test


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/"
    "Credit-cards.jpg/320px-Credit-cards.jpg",
    use_column_width=True,
)
st.sidebar.title("🔍 Fraud Detector")
st.sidebar.caption("Converted from Clement_ML.ipynb")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 EDA", "🤖 Model Training", "📈 Evaluation", "⚡ Live Predictor"],
)

# ── File upload ───────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader("Upload dataset")
uploaded = st.sidebar.file_uploader(
    "credit_card_frauds.csv", type="csv",
    help="Upload the same CSV file used in your notebook"
)

if uploaded:
    df_raw = pd.read_csv(uploaded)
    st.sidebar.success(f"✅ {len(df_raw):,} rows loaded")
else:
    st.sidebar.info("Upload your CSV to begin")
    df_raw = None


# ═══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Credit Card Fraud Detection")
    st.markdown("**Machine Learning System** — deployed from `Clement_ML.ipynb`")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset rows",    "339,607")
    c2.metric("Fraud cases",     "1,782",     "0.52%")
    c3.metric("Best model",      "XGBoost")
    c4.metric("Best AUC-ROC",    "0.9956")

    st.divider()
    st.subheader("How this app works")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Step 1 — Upload your CSV**
Upload `credit_card_frauds.csv` in the sidebar.

**Step 2 — EDA**
Explore class imbalance, amount distributions,
hourly fraud patterns, and correlations.

**Step 3 — Model Training**
All 4 models are trained with SMOTE balancing.
Results are cached so training only runs once.
        """)
    with col2:
        st.markdown("""
**Step 4 — Evaluation**
ROC curves, Precision-Recall curves, confusion
matrices, feature importances, and threshold
optimisation charts — all from your notebook.

**Step 5 — Live Predictor**
Enter a transaction's details and get a real-time
fraud probability score from the trained XGBoost model.
        """)

    st.divider()
    st.subheader("Taxonomy structure — 7 engineered features")
    feat_df = pd.DataFrame({
        "Feature": FEATURES,
        "Description": [
            "Raw transaction amount ($)",
            "log(1 + amt) — reduces right skew",
            "Hour of day (0–23)",
            "Day of week (0=Mon, 6=Sun)",
            "Month of year",
            "Customer age in years",
            "Distance between customer and merchant (km)",
            "Population of customer's city",
            "1 if hour < 6 or >= 22",
            "1 if Saturday or Sunday",
            "Encoded merchant category",
            "Encoded US state",
        ],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.title("Exploratory Data Analysis")

    if df_raw is None:
        st.warning("⬅️ Please upload your CSV file in the sidebar first.")
        st.stop()

    df = engineer_features(df_raw)
    fraud_amt = df[df["is_fraud"] == 1]["amt"]
    legit_amt = df[df["is_fraud"] == 0]["amt"]

    # ── Row 1: Class distribution + Amount hist ───────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class distribution")
        counts = df["is_fraud"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Legitimate", "Fraud"], counts.values,
               color=["#2196F3", "#F44336"], edgecolor="white", linewidth=1.5)
        for i, v in enumerate(counts.values):
            ax.text(i, v + 800, f"{v:,}\n({v/len(df)*100:.2f}%)",
                    ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(counts) * 1.18)
        ax.set_title("Class Distribution", fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.subheader("Amount distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(legit_amt.clip(upper=500), bins=50, alpha=0.6,
                color="#2196F3", label="Legitimate", density=True)
        ax.hist(fraud_amt.clip(upper=500), bins=50, alpha=0.8,
                color="#F44336", label="Fraud", density=True)
        ax.set_xlabel("Amount ($)")
        ax.set_ylabel("Density")
        ax.set_title("Transaction Amount (clipped $500)", fontweight="bold")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Row 2: Hourly fraud + Category fraud ─────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Fraud rate by hour")
        hourly = df.groupby("hour")["is_fraud"].agg(["sum", "count"])
        hourly["rate"] = hourly["sum"] / hourly["count"] * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(hourly.index, hourly["rate"], color="#EF5350", alpha=0.85)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Fraud Rate (%)")
        ax.set_title("Fraud Rate by Hour of Day", fontweight="bold")
        ax.set_xticks(range(0, 24, 2))
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:
        st.subheader("Fraud rate by category")
        cat_fraud = (df.groupby("category")["is_fraud"]
                       .mean().sort_values(ascending=True) * 100)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(cat_fraud.index, cat_fraud.values, color="#42A5F5")
        ax.set_xlabel("Fraud Rate (%)")
        ax.set_title("Fraud Rate by Category", fontweight="bold")
        for i, v in enumerate(cat_fraud.values):
            ax.text(v + 0.01, i, f"{v:.2f}%", va="center", fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Row 3: Boxplot + Correlation heatmap ─────────────────
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Amount boxplot (log scale)")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.boxplot([legit_amt, fraud_amt],
                   labels=["Legitimate", "Fraud"],
                   patch_artist=True,
                   boxprops=dict(facecolor="#BBDEFB"),
                   medianprops=dict(color="#1565C0", linewidth=2))
        ax.set_ylabel("Amount ($)")
        ax.set_yscale("log")
        ax.set_title("Amount Distribution (log scale)", fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col6:
        st.subheader("Feature correlation heatmap")
        corr = df[FEATURES + ["is_fraud"]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax,
                    annot_kws={"size": 7}, linewidths=0.5)
        ax.set_title("Feature Correlation Heatmap", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════
# PAGE 3 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.title("Model Training")

    if df_raw is None:
        st.warning("⬅️ Please upload your CSV file in the sidebar first.")
        st.stop()

    st.info("Click the button below to train all 4 models. Results are cached — "
            "training only runs once per session.")

    if st.button("🚀 Train all 4 models", type="primary"):
        models, results, probas, preds, X_test, y_test = train_models(df_raw)
        st.session_state["trained"] = True
        st.session_state["results"] = results

    if "results" in st.session_state:
        results = st.session_state["results"]

        st.success("✅ All models trained successfully!")
        st.divider()
        st.subheader("Performance summary (threshold = 0.5)")

        res_df = pd.DataFrame(results).T.reset_index()
        res_df.columns = ["Model", "AUC-ROC", "PR-AUC", "F1", "Precision", "Recall"]

        def highlight_best(col):
            is_best = col == col.max()
            return ["background-color: #E8F5E9; color: #2E7D32; font-weight: bold"
                    if v else "" for v in is_best]

        st.dataframe(
            res_df.style.apply(highlight_best, subset=["AUC-ROC", "PR-AUC", "F1",
                                                        "Precision", "Recall"]),
            use_container_width=True, hide_index=True,
        )

        st.divider()
        st.subheader("Metrics comparison chart")
        metric_cols  = ["AUC-ROC", "PR-AUC", "F1", "Precision", "Recall"]
        bar_colours  = ["#1565C0", "#6A1B9A", "#2E7D32", "#E65100", "#C62828"]
        res_plot     = pd.DataFrame(results).T
        x = np.arange(len(res_plot))

        fig, ax = plt.subplots(figsize=(13, 5))
        for i, (col, colour) in enumerate(zip(metric_cols, bar_colours)):
            bars = ax.bar(x + i * 0.14, res_plot[col], 0.14,
                          label=col, color=colour, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold", color=colour)
        ax.set_xticks(x + 2 * 0.14)
        ax.set_xticklabels(res_plot.index, fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison — All Metrics",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor("#FAFAFA")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════
# PAGE 4 — EVALUATION
# ═══════════════════════════════════════════════════════════
elif page == "📈 Evaluation":
    st.title("Model Evaluation")

    if df_raw is None:
        st.warning("⬅️ Please upload your CSV file in the sidebar first.")
        st.stop()
    if "results" not in st.session_state:
        st.warning("⬅️ Please train the models first (Model Training page).")
        st.stop()

    models, results, probas, preds, X_test, y_test = train_models(df_raw)

    # ── ROC + PR curves ───────────────────────────────────────
    st.subheader("ROC and Precision-Recall curves")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[0].plot(fpr, tpr,
                     label=f"{name} (AUC={roc_auc_score(y_test,proba):.4f})",
                     color=COLOURS[name], linewidth=2)
        pr, rc, _ = precision_recall_curve(y_test, proba)
        axes[1].plot(rc, pr,
                     label=f"{name} (AP={average_precision_score(y_test,proba):.4f})",
                     color=COLOURS[name], linewidth=2)
    axes[0].plot([0,1],[0,1],"--",color="grey",alpha=0.5,label="Random")
    for ax, title, xl, yl in zip(
        axes,
        ["ROC Curves", "Precision-Recall Curves"],
        ["False Positive Rate", "Recall"],
        ["True Positive Rate", "Precision"],
    ):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Confusion matrices ────────────────────────────────────
    st.divider()
    st.subheader("Confusion matrices (threshold = 0.5)")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (name, pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_test, pred)
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if cm[i,j] > cm.max()/2 else "navy")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred\nLegit","Pred\nFraud"], fontsize=9)
        ax.set_yticklabels(["Actual\nLegit","Actual\nFraud"], fontsize=9)
        ax.set_title(name, fontweight="bold", fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Feature importances ───────────────────────────────────
    st.divider()
    st.subheader("XGBoost feature importances")
    xgb_model   = models["XGBoost"]
    importances = xgb_model.feature_importances_
    idx         = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(np.array(FEATURES)[idx], importances[idx],
                   color="#42A5F5", alpha=0.85)
    for bar, val in zip(bars, importances[idx]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost Feature Importances", fontweight="bold")
    ax.grid(axis="x", alpha=0.3); ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Threshold optimisation ────────────────────────────────
    st.divider()
    st.subheader("Decision threshold optimisation (XGBoost)")
    xgb_proba  = probas["XGBoost"]
    thresholds = np.arange(0.05, 0.96, 0.05)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        p = (xgb_proba >= t).astype(int)
        f1s.append(  f1_score(y_test, p, zero_division=0))
        precs.append(precision_score(y_test, p, zero_division=0))
        recs.append( recall_score(y_test, p, zero_division=0))
    best_idx = int(np.argmax(f1s))
    best_t   = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(thresholds, f1s,   "-o", label="F1 Score",  color="#1565C0", lw=2, ms=5)
    ax.plot(thresholds, precs, "-s", label="Precision",  color="#C62828", lw=2, ms=5)
    ax.plot(thresholds, recs,  "-^", label="Recall",     color="#2E7D32", lw=2, ms=5)
    ax.axvline(x=best_t, color="darkorange", linestyle="--", lw=2,
               label=f"Optimal t={best_t:.2f}  F1={f1s[best_idx]:.4f}")
    ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
    ax.set_title("XGBoost — F1 / Precision / Recall vs Threshold", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal threshold", f"{best_t:.2f}")
    c2.metric("Best F1 score",     f"{f1s[best_idx]:.4f}")
    c3.metric("Precision @ opt.",  f"{precs[best_idx]:.4f}")


# ═══════════════════════════════════════════════════════════
# PAGE 5 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════
elif page == "⚡ Live Predictor":
    st.title("⚡ Live Fraud Predictor")
    st.markdown("Enter a transaction's details below and get an instant fraud risk score.")

    if df_raw is None:
        st.warning("⬅️ Please upload your CSV file first — "
                   "the model needs to be trained on it.")
        st.stop()
    if "results" not in st.session_state:
        st.warning("⬅️ Please train the models first (Model Training page).")
        st.stop()

    # Load saved model
    models, *_ = train_models(df_raw)
    xgb = models["XGBoost"]

    # Category and state lists (from notebook data)
    categories = sorted([
        "entertainment", "food_dining", "gas_transport", "grocery_net",
        "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net",
        "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel",
    ])
    states = sorted([
        "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
        "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
        "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
        "TX","UT","VT","VA","WA","WV","WI","WY",
    ])
    cat_map   = {c: i for i, c in enumerate(categories)}
    state_map = {s: i for i, s in enumerate(states)}

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Transaction details")
        amt         = st.number_input("Transaction amount ($)", 1.0, 30000.0, 150.0, step=10.0)
        hour        = st.slider("Hour of day", 0, 23, 14)
        day_of_week = st.selectbox("Day of week", range(7),
                                   format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        month       = st.slider("Month", 1, 12, 6)

    with col2:
        st.subheader("Customer details")
        age         = st.slider("Customer age", 18, 90, 40)
        distance_km = st.number_input("Distance to merchant (km)", 0.0, 5000.0, 25.0, step=5.0)
        city_pop    = st.number_input("City population", 500, 10_000_000, 250_000, step=10_000)

    with col3:
        st.subheader("Merchant details")
        category  = st.selectbox("Merchant category", categories)
        state     = st.selectbox("Customer state", states, index=states.index("TX"))
        threshold = st.slider("Decision threshold", 0.05, 0.99, 0.95, step=0.01,
                               help="Lower = catch more fraud but more false alarms. "
                                    "Optimal = 0.95 (best F1)")

    st.divider()

    if st.button("🔍 Analyse this transaction", type="primary", use_container_width=True):
        log_amt    = float(np.log1p(amt))
        is_night   = int(hour < 6 or hour >= 22)
        is_weekend = int(day_of_week >= 5)
        cat_enc    = cat_map.get(category, 0)
        state_enc  = state_map.get(state, 0)

        X_input = np.array([[
            amt, log_amt, hour, day_of_week, month, age,
            distance_km, city_pop, is_night, is_weekend,
            cat_enc, state_enc
        ]])

        prob     = float(xgb.predict_proba(X_input)[0, 1])
        is_fraud = prob >= threshold

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Fraud probability",   f"{prob:.1%}")
        r2.metric("Decision threshold",  f"{threshold:.2f}")
        r3.metric("Verdict",
                  "🚨 FRAUD" if is_fraud else "✅ LEGITIMATE",
                  delta=None)

        if is_fraud:
            st.error(f"⚠️  HIGH FRAUD RISK — Score {prob:.3f} exceeds threshold {threshold:.2f}")
        else:
            st.success(f"✅  Transaction appears legitimate — Score {prob:.3f} is below threshold {threshold:.2f}")

        # Visual gauge
        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.barh(["Risk"], [prob],    color="#F44336" if is_fraud else "#4CAF50", height=0.5)
        ax.barh(["Risk"], [1 - prob], left=[prob], color="#E0E0E0", height=0.5)
        ax.axvline(x=threshold, color="navy", linestyle="--", linewidth=2,
                   label=f"Threshold={threshold:.2f}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraud probability")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"Risk gauge — {prob:.1%}", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.caption(f"Features used: amt={amt}, log_amt={log_amt:.3f}, "
                   f"hour={hour}, day_of_week={day_of_week}, month={month}, "
                   f"age={age}, distance_km={distance_km}, city_pop={city_pop}, "
                   f"is_night={is_night}, is_weekend={is_weekend}, "
                   f"category_enc={cat_enc}, state_enc={state_enc}")
