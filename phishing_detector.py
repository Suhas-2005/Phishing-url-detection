"""
Phishing URL Detection System
==============================
Trains ML models on pre-extracted features, then predicts phishing risk
for raw URLs by extracting features on-the-fly.

Author : Auto-generated
Date   : 2026-04-19
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from urllib.parse import urlparse

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# FEATURE COLUMNS (must be identical everywhere)
# ──────────────────────────────────────────────
FEATURE_COLUMNS = [
    "url_length",
    "valid_url",
    "at_symbol",
    "sensitive_words_count",
    "path_length",
    "isHttps",
    "nb_dots",
    "nb_hyphens",
    "nb_and",
    "nb_or",
    "nb_www",
    "nb_com",
    "nb_underscore",
]

# Resolve paths relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH       = os.path.join(BASE_DIR, "phishing_url_dataset.csv")
MODEL_PATH         = os.path.join(BASE_DIR, "phishing_model.pkl")
FEATURE_COL_PATH   = os.path.join(BASE_DIR, "feature_columns.pkl")
IMPORTANCE_IMG_PATH = os.path.join(BASE_DIR, "feature_importance.png")


# ═══════════════════════════════════════════════
# STEP 1 — LOAD TRAINING DATA
# ═══════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    """Load the phishing URL dataset and print basic info."""
    print("=" * 60)
    print("STEP 1: LOADING TRAINING DATA")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"  Shape          : {df.shape}")
    print(f"  Class distribution:\n{df['target'].value_counts().to_string()}")
    print(f"\n  First 3 rows:\n{df.head(3).to_string(index=False)}\n")
    return df


# ═══════════════════════════════════════════════
# STEP 2 — DATA CLEANING
# ═══════════════════════════════════════════════
def clean_data(df: pd.DataFrame):
    """Drop rows where target is null, keep duplicates. Separate X and y."""
    print("=" * 60)
    print("STEP 2: DATA CLEANING")
    print("=" * 60)

    before = df.shape[0]
    # Only drop rows where the target label is missing — duplicates are
    # legitimate training samples and must be kept.
    df = df.dropna(subset=["target"])
    after = df.shape[0]
    print(f"  Rows before cleaning : {before}")
    print(f"  Rows after cleaning  : {after}")
    print(f"  Dropped (null target): {before - after}\n")

    X = df[FEATURE_COLUMNS]
    y = df["target"]
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}\n")
    return X, y


# ═══════════════════════════════════════════════
# STEP 3 — MODEL TRAINING
# ═══════════════════════════════════════════════
def train_models(X: pd.DataFrame, y: pd.Series):
    """Train RF & GB pipelines, pick the best, and persist it."""
    print("=" * 60)
    print("STEP 3: MODEL TRAINING")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"  Train size : {X_train.shape[0]}")
    print(f"  Test  size : {X_test.shape[0]}\n")

    # --- Pipeline A: Random Forest ---
    pipe_rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipe_rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, pipe_rf.predict(X_test))
    print(f"  Random Forest  accuracy : {acc_rf:.4f}")

    # --- Pipeline B: Gradient Boosting ---
    pipe_gb = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ])
    pipe_gb.fit(X_train, y_train)
    acc_gb = accuracy_score(y_test, pipe_gb.predict(X_test))
    print(f"  Gradient Boost accuracy : {acc_gb:.4f}")

    # Pick the winner
    if acc_gb >= acc_rf:
        best_model = pipe_gb
        best_name  = "GradientBoostingClassifier"
        best_acc   = acc_gb
    else:
        best_model = pipe_rf
        best_name  = "RandomForestClassifier"
        best_acc   = acc_rf

    print(f"\n  [BEST] Best model: {best_name}  (accuracy {best_acc:.4f})")

    # Persist
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURE_COL_PATH)
    print(f"  Saved model           → {MODEL_PATH}")
    print(f"  Saved feature columns → {FEATURE_COL_PATH}\n")

    return best_model, X_test, y_test


# ═══════════════════════════════════════════════
# STEP 4 — EVALUATION
# ═══════════════════════════════════════════════
def evaluate_model(model, X_test, y_test):
    """Print metrics & save a feature-importance bar chart."""
    print("=" * 60)
    print("STEP 4: EVALUATION")
    print("=" * 60)

    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy : {acc:.4f}\n")

    # Classification report
    print("  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe (0)", "Phishing (1)"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n{cm}\n")

    # Feature importance chart
    clf = model.named_steps["clf"]
    importances = clf.feature_importances_
    sorted_idx  = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx)))
    ax.barh(
        [FEATURE_COLUMNS[i] for i in sorted_idx],
        importances[sorted_idx],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Feature Importances — Best Model", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(IMPORTANCE_IMG_PATH, dpi=150)
    plt.close(fig)
    print(f"  [CHART] Feature-importance chart saved -> {IMPORTANCE_IMG_PATH}\n")


# ═══════════════════════════════════════════════
# STEP 5 — FEATURE EXTRACTION FOR RAW URLs
# ═══════════════════════════════════════════════
SENSITIVE_WORDS = [
    "login", "verify", "secure", "update", "bank",
    "account", "confirm", "password", "signin", "webscr",
]


def extract_features(url: str) -> dict:
    """
    Extract the 13 features from a raw URL string.
    Returns a dict whose keys match FEATURE_COLUMNS exactly.

    The training CSV was built from bare URLs (no http/https scheme).
    We strip the scheme before computing length / dot / hyphen counts
    so the feature distribution matches what the model was trained on.

    valid_url is a *suspiciousness* flag in the CSV (valid_url=1 →
    phishing 97% of the time).  We approximate it with a heuristic
    that fires on IP-based hosts, @ symbols, hyphens in the domain,
    sensitive words in the domain, or non-standard TLDs.
    """
    import re

    try:
        low = url.lower()

        # --- strip scheme to match CSV encoding ---
        if low.startswith("https://"):
            bare_url = url[len("https://"):]
        elif low.startswith("http://"):
            bare_url = url[len("http://"):]
        else:
            bare_url = url

        bare_low = bare_url.lower()

        # Split into domain and path
        slash_pos = bare_url.find("/")
        if slash_pos != -1:
            domain = bare_url[:slash_pos]
            path   = bare_url[slash_pos:]
        else:
            domain = bare_url
            path   = ""

        domain_low = domain.lower()

        # ---------- compute features ----------
        features = {
            "url_length":           len(bare_url),
            "valid_url":            0,                  # computed below
            "at_symbol":            1 if "@" in bare_url else 0,
            "sensitive_words_count": sum(1 for w in SENSITIVE_WORDS if w in bare_low),
            "path_length":          len(path),
            "isHttps":              0,                  # default safe
            "nb_dots":              bare_url.count("."),
            "nb_hyphens":           bare_url.count("-"),
            "nb_and":               bare_url.count("&"),
            "nb_or":                bare_url.count("|"),
            "nb_www":               1 if "www" in bare_low else 0,
            "nb_com":               bare_low.count(".com"),
            "nb_underscore":        bare_url.count("_"),
        }

        # ---------- valid_url heuristic ----------
        # In the CSV valid_url=1 is a strong phishing indicator (r=+0.63).
        # We flag URLs as suspicious (valid_url=1) when they exhibit
        # common phishing patterns:
        is_ip         = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}", domain))
        has_at        = "@" in bare_url
        hyphen_domain = "-" in domain_low    # hyphens in domain are suspicious
        has_sensitive  = any(w in domain_low for w in SENSITIVE_WORDS)
        many_dots     = domain_low.count(".") > 3

        # Well-known safe TLDs (bare, no hyphens, no sensitive words)
        known_tlds = {".com", ".org", ".net", ".edu", ".gov", ".io", ".co"}
        tld = ""
        last_dot = domain_low.rfind(".")
        if last_dot != -1:
            tld = domain_low[last_dot:]
        unusual_tld = tld not in known_tlds

        if is_ip or has_at or many_dots or (hyphen_domain and has_sensitive) or (has_sensitive and unusual_tld):
            features["valid_url"] = 1

    except Exception:
        features = {col: 0 for col in FEATURE_COLUMNS}

    return features


# ═══════════════════════════════════════════════
# STEP 6 — PREDICTION WITH CONFIDENCE
# ═══════════════════════════════════════════════
def predict_url(url: str, model) -> dict:
    """
    Extract features from a raw URL, predict with the trained model,
    and return a result dict with label, confidence and features.
    """
    feats = extract_features(url)
    df    = pd.DataFrame([feats], columns=FEATURE_COLUMNS)
    proba = model.predict_proba(df)[0]          # [P(safe), P(phishing)]

    pred_class = int(np.argmax(proba))
    confidence = round(float(proba[pred_class]) * 100, 2)

    return {
        "url":        url,
        "label":      "Phishing" if pred_class == 1 else "Safe",
        "confidence": confidence,
        "features":   feats,
    }


# ═══════════════════════════════════════════════
# STEP 7 & 8 — INTERACTIVE LOOP (with auto-tests)
# ═══════════════════════════════════════════════
SAMPLE_URLS = [
    "https://google.com",
    "http://paypa1-login.com",
    "http://secure-verify-bank.net/account",
    "https://github.com",
    "http://amazon-account-update.xyz/confirm",
]


def auto_test(model):
    """STEP 8: Run predictions on pre-defined sample URLs."""
    print("=" * 60)
    print("STEP 8: AUTO-TEST ON SAMPLE URLs")
    print("=" * 60)

    for url in SAMPLE_URLS:
        result = predict_url(url, model)
        print(f"  URL        : {result['url']}")
        print(f"  Verdict    : {result['label']}")
        print(f"  Confidence : {result['confidence']}%")
        print(f"  Features   : {result['features']}")
        print("  " + "-" * 40)
    print()


def interactive_loop(model):
    """STEP 7: Continuously ask the user for URLs to classify."""
    print("=" * 60)
    print("STEP 7: INTERACTIVE PREDICTION")
    print("=" * 60)

    while True:
        url = input("\nEnter a URL to check (or 'quit' to exit): ").strip()
        if url.lower() == "quit":
            print("Goodbye!")
            break

        result = predict_url(url, model)
        print(f"  URL        : {result['url']}")
        print(f"  Verdict    : {result['label']}")
        print(f"  Confidence : {result['confidence']}%")
        print("  " + "-" * 40)


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    # Step 1
    df = load_data(DATASET_PATH)

    # Step 2
    X, y = clean_data(df)

    # Step 3
    best_model, X_test, y_test = train_models(X, y)

    # Step 4
    evaluate_model(best_model, X_test, y_test)

    # Step 8 (before the interactive loop)
    auto_test(best_model)

    # Step 7
    interactive_loop(best_model)


if __name__ == "__main__":
    main()
