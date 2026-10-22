"""Train and serialize production-ready models for the Streamlit web application.
This script trains the ECG classification models and Hospital Length of Stay (LOS)
regression pipeline, saving compact model artifacts and evaluation metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_absolute_error, r2_score
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "sample_data"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def train_ecg_classifiers():
    print("=" * 60)
    print("Training ECG Arrhythmia Classifiers...")
    print("=" * 60)

    preprocessed_dir = ROOT_DIR / "preprocessed_dataset"
    x_signals_path = preprocessed_dir / "X_signals.npy"
    y_class_path = preprocessed_dir / "y_classification.npy"
    y_reg_path = preprocessed_dir / "y_regression.npy"
    csv_features_path = preprocessed_dir / "ecg_features_dataset.csv"

    # Load data
    if x_signals_path.exists() and y_class_path.exists():
        print("Loading signals and labels from preprocessed_dataset/...")
        x_signals = np.load(x_signals_path, mmap_mode="r")
        y_classification = np.load(y_class_path, allow_pickle=True)
        y_regression = np.load(y_reg_path) if y_reg_path.exists() else None

        mean_amp = np.mean(x_signals, axis=1)
        std_amp = np.std(x_signals, axis=1)
        r_peak_amp = x_signals[:, 150]
        energy = np.sum(x_signals**2, axis=1)

        X = np.column_stack([mean_amp, std_amp, r_peak_amp, energy])

        # Also extract sample waveforms for the web app
        classes = ["N", "S", "V", "F", "Q"]
        samples = {}
        for c in classes:
            indices = np.where(y_classification == c)[0]
            selected_idx = indices[:25]
            class_samples = []
            for idx in selected_idx:
                sig = x_signals[idx].tolist()
                rr = float(y_regression[idx]) if y_regression is not None else 0.8
                hr = float(60.0 / rr) if rr > 0 else 75.0
                class_samples.append({
                    "index": int(idx),
                    "class": str(c),
                    "rr_interval": rr,
                    "heart_rate": hr,
                    "signal": sig,
                    "mean_amplitude": float(np.mean(sig)),
                    "std_amplitude": float(np.std(sig)),
                    "r_peak_amplitude": float(sig[150]),
                    "signal_energy": float(np.sum(np.array(sig)**2))
                })
            samples[c] = class_samples

        sample_signals_file = DATA_DIR / "sample_ecg_signals.json"
        with open(sample_signals_file, "w") as f:
            json.dump(samples, f)
        print(f"Saved sample ECG signals to {sample_signals_file}")

    elif csv_features_path.exists():
        print("Loading tabular features from preprocessed_dataset/ecg_features_dataset.csv...")
        df = pd.read_csv(csv_features_path)
        feature_cols = ["mean_amplitude", "std_amplitude", "r_amplitude", "signal_energy"]
        X = df[feature_cols].values
        y_classification = df["class_target"].values
    else:
        raise FileNotFoundError("Could not find preprocessed ECG dataset files.")

    feature_names = ["mean_amplitude", "std_amplitude", "r_peak_amplitude", "signal_energy"]

    le = LabelEncoder()
    y = le.fit_transform(y_classification)
    class_names = [str(c) for c in le.classes_]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier
    # Instantiate models matching the case study parameters
    model_defs = {
        "MLP Neural Network": (MLPClassifier(hidden_layer_sizes=(128, 64), activation="tanh", solver="adam", alpha=1e-3, max_iter=300, random_state=42), True),
        "Logistic Regression": (LogisticRegression(C=0.1, max_iter=1000, random_state=42), True),
        "Decision Tree": (DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_leaf=5, min_samples_split=20, random_state=42), False),
        "Random Forest": (RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_split=15, random_state=42, n_jobs=-1), False),
        "Gaussian Naive Bayes": (GaussianNB(var_smoothing=0.1), True),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1), True),
    }

    trained_models = {}
    metrics = {}

    for name, (clf, use_scaled) in model_defs.items():
        t0 = time.time()
        train_data = X_train_scaled if use_scaled else X_train
        test_data = X_test_scaled if use_scaled else X_test

        clf.fit(train_data, y_train)
        preds = clf.predict(test_data)
        prob = clf.predict_proba(test_data) if hasattr(clf, "predict_proba") else None

        acc = float(accuracy_score(y_test, preds))
        f1_weighted = float(f1_score(y_test, preds, average="weighted"))
        f1_macro = float(f1_score(y_test, preds, average="macro"))
        prec_weighted = float(precision_score(y_test, preds, average="weighted", zero_division=0))
        rec_weighted = float(recall_score(y_test, preds, average="weighted"))
        cm = confusion_matrix(y_test, preds).tolist()
        report = classification_report(y_test, preds, target_names=class_names, output_dict=True)

        trained_models[name] = clf
        metrics[name] = {
            "accuracy": acc,
            "weighted_f1": f1_weighted,
            "macro_f1": f1_macro,
            "weighted_precision": prec_weighted,
            "weighted_recall": rec_weighted,
            "confusion_matrix": cm,
            "classification_report": report,
            "fit_time_sec": round(time.time() - t0, 3),
            "use_scaled": use_scaled
        }

        print(f" -> {name:<22}: Acc={acc:.4f} | Weighted F1={f1_weighted:.4f} | Macro F1={f1_macro:.4f}")

    # Package and save
    ecg_bundle = {
        "models": trained_models,
        "scaler": scaler,
        "label_encoder": le,
        "class_names": class_names,
        "feature_names": feature_names,
        "metrics": metrics,
        "class_descriptions": {
            "N": "Normal Beat (Normal sinus rhythm)",
            "S": "Supraventricular Ectopic Beat (Atrial premature, nodal escape)",
            "V": "Ventricular Ectopic Beat (Premature ventricular contraction)",
            "F": "Fusion Beat (Ventricular fusion beat)",
            "Q": "Unknown / Paced Beat (Unclassifiable or paced)"
        },
        "class_risk": {
            "N": ("Low Risk", "Normal sinus rhythm pattern without acute pathology."),
            "S": ("Moderate Risk", "Supraventricular ectopic activity. May warrant Holter monitoring if recurrent."),
            "V": ("High Risk", "Ventricular ectopic activity (e.g. PVC). May trigger ventricular tachycardia if frequent."),
            "F": ("High Risk", "Fusion of ventricular and supraventricular impulse; indicates conduction abnormality."),
            "Q": ("Monitor", "Unclassified or cardiac pacemaker artifact. Clinical verification recommended.")
        }
    }

    joblib.dump(ecg_bundle, MODELS_DIR / "ecg_classifier_bundle.joblib", compress=3)
    print(f"Saved ECG bundle to {MODELS_DIR / 'ecg_classifier_bundle.joblib'}")


def train_los_regressor():
    print("\n" + "=" * 60)
    print("Training Hospital Length of Stay (LOS) Regressor...")
    print("=" * 60)

    csv_path = ROOT_DIR / "regression_dataset" / "hospital-inpatient-discharges-sparcs-de-identified-2010-1.csv"
    if not csv_path.exists():
        print(f"Regression CSV not found at {csv_path}, skipping LOS training.")
        return

    df = pd.read_csv(csv_path)
    target = "Length of Stay"
    df[target] = pd.to_numeric(df[target].astype(str).str.replace("+", "", regex=False).str.strip(), errors="coerce")
    df = df.dropna(subset=[target])
    df = df[df[target] > 0]

    features = [
        "Age Group",
        "Gender",
        "Type of Admission",
        "APR Severity of Illness Description",
        "APR Risk of Mortality",
        "Emergency Department Indicator",
        "APR Medical Surgical Description"
    ]

    # Save a small sample of representative inpatient discharge rows for Streamlit demo
    sample_df = df[features + [target]].dropna().sample(min(50, len(df)), random_state=42)
    sample_df.to_csv(DATA_DIR / "sample_sparcs_records.csv", index=False)
    print(f"Saved sample SPARCS records to {DATA_DIR / 'sample_sparcs_records.csv'}")

    X = df[features].fillna("Unknown")
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features)
        ]
    )

    regressor = HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    print(f" -> LOS HistGradientBoosting: MAE={mae:.2f} days | R2={r2:.4f}")

    los_bundle = {
        "pipeline": pipeline,
        "features": features,
        "target": target,
        "metrics": {
            "mae_days": round(mae, 2),
            "r2_score": round(r2, 4)
        },
        "feature_options": {
            col: sorted(df[col].dropna().unique().tolist()) for col in features
        }
    }

    joblib.dump(los_bundle, MODELS_DIR / "los_regressor_bundle.joblib", compress=3)
    print(f"Saved LOS bundle to {MODELS_DIR / 'los_regressor_bundle.joblib'}")


if __name__ == "__main__":
    train_ecg_classifiers()
    train_los_regressor()
    print("\nModel training and asset extraction completed successfully!")
