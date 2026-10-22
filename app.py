"""CardioPulse AI — ECG Arrhythmia Classification & Clinical LOS Case Study
Streamlit Web Application for Healthcare Machine Learning
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import streamlit as st

# -----------------------------------------------------------------------------
# App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioPulse AI — ECG & Hospital Case Study",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
SAMPLE_DATA_DIR = ROOT_DIR / "sample_data"

# -----------------------------------------------------------------------------
# Theme Management (Switchable: Light by default)
# -----------------------------------------------------------------------------
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Light"

with st.sidebar:
    st.markdown("### Settings")
    selected_theme = st.radio(
        "Appearance Theme",
        options=["Light", "Dark"],
        index=0 if st.session_state["theme_mode"] == "Light" else 1,
        horizontal=True,
        help="Select between clean light clinical aesthetic and dark mode."
    )
    st.session_state["theme_mode"] = selected_theme

is_dark = st.session_state["theme_mode"] == "Dark"

# Theme-specific color palette
if is_dark:
    t_bg = "#0F172A"
    t_card_bg = "#1E293B"
    t_border = "rgba(255, 255, 255, 0.08)"
    t_text = "#F8FAFC"
    t_subtext = "#94A3B8"
    t_accent = "#10B981"
    t_chart_template = "plotly_dark"
    t_plot_bg = "#0F172A"
    t_grid = "rgba(255, 255, 255, 0.08)"
else:
    t_bg = "#F8FAFC"
    t_card_bg = "#FFFFFF"
    t_border = "#E2E8F0"
    t_text = "#0F172A"
    t_subtext = "#64748B"
    t_accent = "#059669"
    t_chart_template = "plotly_white"
    t_plot_bg = "#FFFFFF"
    t_grid = "#F1F5F9"

# Clean CSS without excessive icons or clunky decorations
st.markdown(f"""
<style>
    /* Metric Cards */
    div[data-testid="stMetricValue"] {{
        font-size: 1.6rem !important;
        font-weight: 700;
        color: {t_accent};
    }}
    .metric-card {{
        background: {t_card_bg};
        border-radius: 8px;
        padding: 16px 20px;
        border: 1px solid {t_border};
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 14px;
        color: {t_text};
    }}
    .status-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }}
    .badge-normal {{
        background: rgba(16, 185, 129, 0.12);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    .badge-warning {{
        background: rgba(245, 158, 11, 0.12);
        color: #D97706;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }}
    .badge-danger {{
        background: rgba(239, 68, 68, 0.12);
        color: #DC2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }}
    .badge-info {{
        background: rgba(14, 165, 233, 0.12);
        color: #0284C7;
        border: 1px solid rgba(14, 165, 233, 0.3);
    }}
    .hero-banner {{
        background: {t_card_bg};
        border: 1px solid {t_border};
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 1px solid {t_border};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.95rem;
    }}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Asset & Model Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading machine learning models...")
def load_ecg_bundle():
    bundle_path = MODELS_DIR / "ecg_classifier_bundle.joblib"
    if not bundle_path.exists():
        from scripts.train_and_export_models import train_ecg_classifiers
        train_ecg_classifiers()
    return joblib.load(bundle_path)


@st.cache_resource(show_spinner="Loading hospital regression model...")
def load_los_bundle():
    bundle_path = MODELS_DIR / "los_regressor_bundle.joblib"
    if bundle_path.exists():
        return joblib.load(bundle_path)
    return None


@st.cache_data(show_spinner=False)
def load_sample_signals():
    file_path = SAMPLE_DATA_DIR / "sample_ecg_signals.json"
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_sample_sparcs():
    file_path = SAMPLE_DATA_DIR / "sample_sparcs_records.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()


ecg_bundle = load_ecg_bundle()
los_bundle = load_los_bundle()
sample_signals_db = load_sample_signals()
sample_sparcs_df = load_sample_sparcs()


# -----------------------------------------------------------------------------
# Waveform Synthesis & Helper Functions
# -----------------------------------------------------------------------------
def synthesize_ecg(heart_rate=75, r_peak_amp=1.2, noise_level=0.03, points=300):
    """Synthesize a canonical P-Q-R-S-T ECG window of 300 points centered at index 150."""
    t = np.linspace(-1.0, 1.0, points)
    r_wave = r_peak_amp * np.exp(-((t) ** 2) / 0.0015)
    q_wave = -0.15 * r_peak_amp * np.exp(-((t + 0.08) ** 2) / 0.001)
    s_wave = -0.25 * r_peak_amp * np.exp(-((t - 0.08) ** 2) / 0.0012)
    p_wave = 0.2 * np.exp(-((t + 0.35) ** 2) / 0.008)
    t_wave = 0.35 * np.exp(-((t - 0.40) ** 2) / 0.015)
    baseline = 0.02 * np.sin(2 * np.pi * 0.5 * t)
    noise = np.random.normal(0, noise_level, points) if noise_level > 0 else 0
    signal = r_wave + q_wave + s_wave + p_wave + t_wave + baseline + noise
    return signal


def extract_signal_features(signal_array: np.ndarray):
    """Extract standard features matching the case study feature vector."""
    signal = np.asarray(signal_array, dtype=float)
    if len(signal) != 300:
        orig_indices = np.linspace(0, 1, len(signal))
        new_indices = np.linspace(0, 1, 300)
        signal = np.interp(new_indices, orig_indices, signal)

    mean_amp = float(np.mean(signal))
    std_amp = float(np.std(signal))
    r_peak_amp = float(signal[150])
    energy = float(np.sum(signal**2))
    return {
        "mean_amplitude": mean_amp,
        "std_amplitude": std_amp,
        "r_peak_amplitude": r_peak_amp,
        "signal_energy": energy,
        "raw_signal": signal
    }


def predict_ecg_class(features_dict, model_name):
    """Run model prediction and probability estimation."""
    model = ecg_bundle["models"][model_name]
    scaler = ecg_bundle["scaler"]
    le = ecg_bundle["label_encoder"]
    use_scaled = ecg_bundle["metrics"][model_name]["use_scaled"]

    feat_vector = np.array([[
        features_dict["mean_amplitude"],
        features_dict["std_amplitude"],
        features_dict["r_peak_amplitude"],
        features_dict["signal_energy"]
    ]])

    input_data = scaler.transform(feat_vector) if use_scaled else feat_vector
    pred_encoded = model.predict(input_data)[0]
    pred_class = le.inverse_transform([pred_encoded])[0]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        for c, p in zip(le.classes_, probs):
            probabilities[c] = float(p)
    else:
        for c in le.classes_:
            probabilities[c] = 1.0 if c == pred_class else 0.0

    return pred_class, probabilities


# -----------------------------------------------------------------------------
# Sidebar Navigation & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("CardioPulse")
    st.caption("Clinical ECG & Hospital Inpatient Case Study")
    st.markdown("---")

    # Place MLP as default selection or first option
    model_options = list(ecg_bundle["models"].keys())
    default_idx = model_options.index("MLP Neural Network") if "MLP Neural Network" in model_options else 0

    selected_model = st.selectbox(
        "Primary Classifier Model",
        options=model_options,
        index=default_idx,
        help="Select which regularized model algorithm processes the ECG signals. MLP Neural Network ranked #1 among all 10 evaluated algorithms."
    )

    if selected_model == "MLP Neural Network":
        st.success("Selected Top Performer: MLP Neural Network (91.45% Acc, 0.9073 Weighted F1)")

    st.markdown("---")
    st.markdown("### Case Study Overview")
    st.markdown(
        "**Track 1: MIT-BIH Arrhythmia**\n"
        "- 109,423 Ambulatory Beats\n"
        "- 5 AAMI EC57 Rhythm Classes\n"
        "- 10 Supervised Algorithms & Clustering\n\n"
        "**Track 2: NY SPARCS 2010**\n"
        "- Inpatient Discharge Modeling\n"
        "- Target: Hospital Length of Stay\n"
        "- Gradient Boosting Pipeline"
    )

    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: {t_subtext}; font-size: 0.8rem;'>"
        "PhysioNet MIT-BIH & NY Open Data"
        "</div>",
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# Header / Hero Section
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0; font-size: 1.8rem; color: {t_text}; font-weight: 700;">
                CardioPulse Analytics
            </h1>
            <p style="margin: 4px 0 0 0; color: {t_subtext}; font-size: 0.95rem;">
                Clinical Machine Learning: ECG Arrhythmia Classification, Unsupervised Clustering, and Hospital Inpatient Length of Stay
            </p>
        </div>
        <div style="margin-top: 8px;">
            <span class="status-badge badge-normal">MIT-BIH Arrhythmia</span>
            <span class="status-badge badge-info" style="margin-left: 6px;">NY SPARCS 2010</span>
            <span class="status-badge badge-warning" style="margin-left: 6px;">Theme: {st.session_state['theme_mode']}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Navigation Tabs (Clean labels without heavy icons)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "ECG Classification",
    "ECG Beat Clustering",
    "Hospital Stay Prediction",
    "Model Benchmarks & EDA",
    "Methodology & Documentation"
])


# =============================================================================
# TAB 1: ECG Arrhythmia Classification & Waveform Studio
# =============================================================================
with tab1:
    col_input, col_viz = st.columns([1, 2], gap="large")

    with col_input:
        st.subheader("1. ECG Signal Source")
        input_mode = st.radio(
            "Select Input Mode",
            ["Preset Clinical Beats (MIT-BIH)", "Synthetic ECG Generator", "Upload Signal (CSV/NPY)"],
            index=0
        )

        current_signal = None
        true_label = None
        est_hr = 75.0
        est_rr = 0.80

        if input_mode == "Preset Clinical Beats (MIT-BIH)":
            class_choice = st.selectbox(
                "Heartbeat Rhythm Type (AAMI Class)",
                options=["N - Normal Sinus Beat", "V - Ventricular Ectopic (PVC)", "S - Supraventricular Ectopic", "F - Fusion Beat", "Q - Unknown / Paced Beat"],
                index=0
            )
            c_key = class_choice[0]
            avail_samples = sample_signals_db.get(c_key, [])
            sample_idx = st.slider("Select Sample Recording", 1, max(1, len(avail_samples)), 1) - 1
            
            if avail_samples:
                chosen = avail_samples[sample_idx]
                current_signal = np.array(chosen["signal"])
                true_label = chosen["class"]
                est_hr = chosen.get("heart_rate", 75.0)
                est_rr = chosen.get("rr_interval", 0.80)
            else:
                current_signal = synthesize_ecg()

        elif input_mode == "Synthetic ECG Generator":
            st.caption("Synthesize a custom ECG beat and observe model classification decisions:")
            syn_hr = st.slider("Simulated Heart Rate (BPM)", 40, 160, 75, 1)
            syn_r_amp = st.slider("R-Peak Amplitude (mV)", 0.2, 3.5, 1.25, 0.05)
            syn_noise = st.slider("Baseline Noise Level", 0.0, 0.15, 0.02, 0.01)
            current_signal = synthesize_ecg(heart_rate=syn_hr, r_peak_amp=syn_r_amp, noise_level=syn_noise)
            est_hr = float(syn_hr)
            est_rr = 60.0 / syn_hr

        else:
            uploaded_file = st.file_uploader("Upload 300-point ECG Signal (.csv or .npy)", type=["csv", "npy", "txt"])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".npy"):
                        current_signal = np.load(uploaded_file)
                    else:
                        uploaded_df = pd.read_csv(uploaded_file, header=None)
                        current_signal = uploaded_df.values.flatten()
                    st.success(f"Loaded signal with {len(current_signal)} samples.")
                except Exception as e:
                    st.error(f"Error parsing uploaded file: {e}")
                    current_signal = synthesize_ecg()
            else:
                st.info("No file uploaded. Displaying default normal sinus heartbeat sample.")
                current_signal = synthesize_ecg()

        # Extract features
        extracted = extract_signal_features(current_signal)

        st.markdown("---")
        st.subheader("2. Extracted Beat Features")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.metric("R-Peak Amplitude", f"{extracted['r_peak_amplitude']:.3f} mV")
            st.metric("Mean Amplitude", f"{extracted['mean_amplitude']:.3f} mV")
        with f_col2:
            st.metric("Signal Energy", f"{extracted['signal_energy']:.2f}")
            st.metric("Std Deviation", f"{extracted['std_amplitude']:.3f}")

        st.markdown("---")
        st.subheader("3. Interactive Feature Adjustment")
        st.caption("Fine-tune values to examine model decision boundaries:")
        adj_r_peak = st.slider("R-Peak Amplitude (mV)", -1.0, 4.0, float(extracted["r_peak_amplitude"]), 0.05)
        adj_energy = st.slider("Signal Energy", 0.0, 200.0, float(extracted["signal_energy"]), 1.0)

        inference_features = {
            "mean_amplitude": extracted["mean_amplitude"],
            "std_amplitude": extracted["std_amplitude"],
            "r_peak_amplitude": adj_r_peak,
            "signal_energy": adj_energy,
            "raw_signal": extracted["raw_signal"]
        }

    with col_viz:
        st.subheader("ECG Waveform (300 Samples Centered on R-Peak)")

        samples_x = np.arange(len(current_signal))
        time_sec = (samples_x - 150) / 360.0

        fig = go.Figure()

        # ECG Trace
        fig.add_trace(go.Scatter(
            x=time_sec,
            y=current_signal,
            mode="lines",
            name="Lead MLII",
            line=dict(color=t_accent, width=2.2),
            hovertemplate="Time: %{x:.3f}s<br>Voltage: %{y:.3f} mV<extra></extra>"
        ))

        # R-Peak marker
        r_val = current_signal[150]
        fig.add_trace(go.Scatter(
            x=[0],
            y=[r_val],
            mode="markers+text",
            name="R-Peak (t=0s)",
            marker=dict(color="#EF4444", size=9, symbol="diamond"),
            text=["R-Peak"],
            textposition="top center",
            textfont=dict(color=t_text, size=11)
        ))

        fig.add_vrect(x0=-0.06, x1=0.06, fillcolor="rgba(239, 68, 68, 0.08)", layer="below", line_width=0, annotation_text="QRS", annotation_position="bottom right")

        fig.update_layout(
            template=t_chart_template,
            paper_bgcolor=t_card_bg,
            plot_bgcolor=t_plot_bg,
            margin=dict(l=40, r=20, t=30, b=40),
            height=320,
            xaxis=dict(
                title="Time relative to R-peak (seconds)",
                gridcolor=t_grid
            ),
            yaxis=dict(
                title="Amplitude (mV)",
                gridcolor=t_grid
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Run Prediction
        pred_class, class_probs = predict_ecg_class(inference_features, selected_model)
        class_desc = ecg_bundle["class_descriptions"].get(pred_class, "Arrhythmia Classification")
        risk_level, clinical_note = ecg_bundle["class_risk"].get(pred_class, ("Unknown", "Further clinical evaluation required."))

        badge_style = "badge-normal" if pred_class == "N" else ("badge-danger" if pred_class in ["V", "F"] else "badge-warning")

        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.82rem; color: {t_subtext}; text-transform: uppercase; letter-spacing: 0.5px;">Classifier: {selected_model}</span>
                    <h3 style="margin: 4px 0; color: {t_text}; font-size: 1.3rem;">Class [{pred_class}]: {class_desc}</h3>
                    <span class="status-badge {badge_style}">{risk_level}</span>
                    {f'<span class="status-badge badge-info" style="margin-left: 8px;">Ground Truth: {true_label}</span>' if true_label else ''}
                </div>
                <div style="text-align: right; margin-top: 6px;">
                    <span style="color: {t_subtext}; font-size: 0.82rem;">Estimated Rhythm</span>
                    <div style="font-size: 1.4rem; font-weight: 700; color: {t_accent};">{est_hr:.1f} <span style="font-size: 0.85rem; color: {t_subtext};">BPM</span></div>
                    <span style="color: {t_subtext}; font-size: 0.8rem;">RR-Interval: {est_rr:.3f}s</span>
                </div>
            </div>
            <p style="margin: 12px 0 0 0; color: {t_text}; font-size: 0.9rem; border-top: 1px solid {t_border}; padding-top: 8px;">
                <strong>Clinical Notes:</strong> {clinical_note}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Probabilities Horizontal Bar Chart
        st.markdown("##### Multi-Class Probability Distribution")
        prob_df = pd.DataFrame({
            "Class": list(class_probs.keys()),
            "Probability": [class_probs[c] for c in class_probs.keys()],
            "Description": [ecg_bundle["class_descriptions"].get(c, "") for c in class_probs.keys()]
        })

        color_map = {
            "N": "#10B981",
            "S": "#F59E0B",
            "V": "#EF4444",
            "F": "#F97316",
            "Q": "#0EA5E9"
        }

        fig_prob = px.bar(
            prob_df,
            x="Probability",
            y="Class",
            orientation="h",
            color="Class",
            color_discrete_map=color_map,
            text=prob_df["Probability"].apply(lambda p: f"{p*100:.1f}%"),
            hover_data={"Description": True, "Probability": ":.3f"}
        )
        fig_prob.update_layout(
            template=t_chart_template,
            paper_bgcolor=t_card_bg,
            plot_bgcolor=t_plot_bg,
            height=200,
            margin=dict(l=20, r=20, t=10, b=30),
            xaxis=dict(range=[0, 1.05], title="Predicted Probability", gridcolor=t_grid),
            yaxis=dict(title="Class"),
            showlegend=False
        )
        st.plotly_chart(fig_prob, use_container_width=True)


# =============================================================================
# TAB 2: Unsupervised ECG Beat Clustering
# =============================================================================
with tab2:
    st.subheader("Unsupervised ECG Heartbeat Clustering")
    st.markdown(
        "Explore morphological grouping of ECG beats without prior label supervision using **K-Means Clustering** "
        "and **Principal Component Analysis (PCA)** on extracted beat features."
    )

    # Compile feature dataset from sample records
    cluster_records = []
    for c_key, samples in sample_signals_db.items():
        for s in samples:
            cluster_records.append({
                "record_id": s.get("index", len(cluster_records)),
                "true_class": c_key,
                "mean_amplitude": s["mean_amplitude"],
                "std_amplitude": s["std_amplitude"],
                "r_peak_amplitude": s["r_peak_amplitude"],
                "signal_energy": s["signal_energy"],
                "heart_rate": s["heart_rate"],
                "rr_interval": s["rr_interval"]
            })
    cluster_df = pd.DataFrame(cluster_records)

    feat_cols = ["mean_amplitude", "std_amplitude", "r_peak_amplitude", "signal_energy"]
    scaler = ecg_bundle["scaler"]
    X_features = cluster_df[feat_cols].values
    X_scaled = scaler.transform(X_features)

    c_ctrl1, c_ctrl2 = st.columns([1, 2], gap="large")

    with c_ctrl1:
        st.markdown("##### Clustering Controls")
        k_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=8, value=5, step=1,
                               help="Standard AAMI EC57 recognizes 5 distinct heartbeat categories.")
        random_seed = st.number_input("Random State Seed", value=42, min_value=0, max_value=999, step=1)

        # Run K-Means
        kmeans_model = KMeans(n_clusters=k_clusters, random_state=int(random_seed), n_init=10)
        cluster_labels = kmeans_model.fit_predict(X_scaled)
        cluster_df["cluster"] = [f"Cluster {i}" for i in cluster_labels]

        sil_avg = silhouette_score(X_scaled, cluster_labels)
        inertia_val = kmeans_model.inertia_

        st.metric("Silhouette Score", f"{sil_avg:.3f}", help="Ranges from -1 to 1; higher indicates well-separated clusters.")
        st.metric("Inertia (Within-cluster sum of squares)", f"{inertia_val:.2f}")

        # Map current inference beat into the cluster space
        cur_feat = np.array([[
            inference_features["mean_amplitude"],
            inference_features["std_amplitude"],
            inference_features["r_peak_amplitude"],
            inference_features["signal_energy"]
        ]])
        cur_scaled = scaler.transform(cur_feat)
        assigned_cluster = kmeans_model.predict(cur_scaled)[0]
        st.info(f"Current Beat from Studio: **Assigned to Cluster {assigned_cluster}**")

    with c_ctrl2:
        st.markdown("##### 2D PCA Cluster Projection")
        pca_engine = PCA(n_components=2, random_state=42)
        coords_2d = pca_engine.fit_transform(X_scaled)
        cluster_df["PCA 1"] = coords_2d[:, 0]
        cluster_df["PCA 2"] = coords_2d[:, 1]

        cur_pca = pca_engine.transform(cur_scaled)

        pca_color_by = st.radio("Color Data Points By:", ["K-Means Cluster", "True Rhythm Class (AAMI)"], horizontal=True)
        color_col = "cluster" if pca_color_by == "K-Means Cluster" else "true_class"

        fig_pca = px.scatter(
            cluster_df,
            x="PCA 1",
            y="PCA 2",
            color=color_col,
            hover_data=["true_class", "cluster", "r_peak_amplitude", "signal_energy"],
            color_discrete_sequence=px.colors.qualitative.Safe,
            title=f"PCA Latent Space (Explains {pca_engine.explained_variance_ratio_.sum()*100:.1f}% Variance)"
        )

        # Overlay current inspected beat
        fig_pca.add_trace(go.Scatter(
            x=[cur_pca[0, 0]],
            y=[cur_pca[0, 1]],
            mode="markers",
            marker=dict(size=14, color="#EF4444", symbol="star", line=dict(width=1.5, color="#000")),
            name="Current Studio Beat"
        ))

        fig_pca.update_layout(
            template=t_chart_template,
            paper_bgcolor=t_card_bg,
            plot_bgcolor=t_plot_bg,
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(gridcolor=t_grid),
            yaxis=dict(gridcolor=t_grid)
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("---")
    c_sub1, c_sub2 = st.columns(2, gap="large")

    with c_sub1:
        st.markdown("##### Cluster vs True Class Contingency Table")
        st.caption("Inspect overlap between unsupervised clusters and clinical ground truth annotations:")
        contingency = pd.crosstab(cluster_df["true_class"], cluster_df["cluster"])
        st.dataframe(contingency, use_container_width=True)

    with c_sub2:
        st.markdown("##### Elbow Curve & Silhouette Analysis")
        # Compute elbow data across k=2..8
        k_eval_range = list(range(2, 9))
        elbow_inertias = []
        elbow_silhouettes = []
        for test_k in k_eval_range:
            km_test = KMeans(n_clusters=test_k, random_state=42, n_init=5).fit(X_scaled)
            elbow_inertias.append(km_test.inertia_)
            elbow_silhouettes.append(silhouette_score(X_scaled, km_test.labels_))

        eval_df = pd.DataFrame({
            "k": k_eval_range,
            "Inertia": elbow_inertias,
            "Silhouette": elbow_silhouettes
        })

        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=eval_df["k"],
            y=eval_df["Inertia"],
            name="Inertia (Elbow)",
            mode="lines+markers",
            line=dict(color="#0EA5E9", width=2)
        ))
        fig_eval.add_trace(go.Scatter(
            x=eval_df["k"],
            y=eval_df["Silhouette"],
            name="Silhouette Score",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#10B981", width=2, dash="dot")
        ))
        fig_eval.update_layout(
            template=t_chart_template,
            paper_bgcolor=t_card_bg,
            plot_bgcolor=t_plot_bg,
            height=280,
            margin=dict(l=30, r=30, t=20, b=30),
            xaxis=dict(title="Number of Clusters (k)", gridcolor=t_grid),
            yaxis=dict(title="Inertia", gridcolor=t_grid),
            yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_eval, use_container_width=True)


# =============================================================================
# TAB 3: Hospital Inpatient Length of Stay (LOS) Predictor
# =============================================================================
with tab3:
    st.subheader("Hospital Inpatient Length of Stay (LOS) Prediction")
    st.markdown(
        "Estimate inpatient stay duration using clinical admission risk parameters "
        "trained on the **New York SPARCS Inpatient Discharges** dataset."
    )

    if los_bundle is None:
        st.warning("Hospital Length of Stay model bundle is currently unavailable.")
    else:
        los_pipeline = los_bundle["pipeline"]
        feature_options = los_bundle["feature_options"]

        st.markdown("##### Quick Patient Case Profiles")
        preset_cols = st.columns(4)
        preset_chosen = None

        with preset_cols[0]:
            if st.button("Emergency Cardiac Adult"):
                preset_chosen = {
                    "Age Group": "50 to 69",
                    "Gender": "M",
                    "Type of Admission": "Emergency",
                    "APR Severity of Illness Description": "Major",
                    "APR Risk of Mortality": "Major",
                    "Emergency Department Indicator": "Y",
                    "APR Medical Surgical Description": "Medical"
                }
        with preset_cols[1]:
            if st.button("Elective Joint Surgery"):
                preset_chosen = {
                    "Age Group": "50 to 69",
                    "Gender": "F",
                    "Type of Admission": "Elective",
                    "APR Severity of Illness Description": "Moderate",
                    "APR Risk of Mortality": "Minor",
                    "Emergency Department Indicator": "N",
                    "APR Medical Surgical Description": "Surgical"
                }
        with preset_cols[2]:
            if st.button("Routine Delivery"):
                preset_chosen = {
                    "Age Group": "0 to 17",
                    "Gender": "F",
                    "Type of Admission": "Newborn",
                    "APR Severity of Illness Description": "Minor",
                    "APR Risk of Mortality": "Minor",
                    "Emergency Department Indicator": "N",
                    "APR Medical Surgical Description": "Medical"
                }
        with preset_cols[3]:
            if st.button("Elderly High-Risk ICU"):
                preset_chosen = {
                    "Age Group": "70 or Older",
                    "Gender": "F",
                    "Type of Admission": "Emergency",
                    "APR Severity of Illness Description": "Extreme",
                    "APR Risk of Mortality": "Extreme",
                    "Emergency Department Indicator": "Y",
                    "APR Medical Surgical Description": "Medical"
                }

        # Form Inputs
        with st.form("los_form"):
            r_col1, r_col2, r_col3 = st.columns(3)

            with r_col1:
                age_group = st.selectbox(
                    "Patient Age Group",
                    options=feature_options["Age Group"],
                    index=feature_options["Age Group"].index(preset_chosen["Age Group"]) if preset_chosen else 2
                )
                gender = st.selectbox(
                    "Gender",
                    options=feature_options["Gender"],
                    index=feature_options["Gender"].index(preset_chosen["Gender"]) if preset_chosen else 0
                )
                admission_type = st.selectbox(
                    "Type of Admission",
                    options=feature_options["Type of Admission"],
                    index=feature_options["Type of Admission"].index(preset_chosen["Type of Admission"]) if preset_chosen else 0
                )

            with r_col2:
                severity = st.selectbox(
                    "APR Severity of Illness",
                    options=feature_options["APR Severity of Illness Description"],
                    index=feature_options["APR Severity of Illness Description"].index(preset_chosen["APR Severity of Illness Description"]) if preset_chosen else 1
                )
                mortality = st.selectbox(
                    "APR Risk of Mortality",
                    options=feature_options["APR Risk of Mortality"],
                    index=feature_options["APR Risk of Mortality"].index(preset_chosen["APR Risk of Mortality"]) if preset_chosen else 0
                )
                med_surg = st.selectbox(
                    "Procedure Type",
                    options=feature_options["APR Medical Surgical Description"],
                    index=feature_options["APR Medical Surgical Description"].index(preset_chosen["APR Medical Surgical Description"]) if preset_chosen else 0
                )

            with r_col3:
                ed_indicator = st.selectbox(
                    "Admitted Via Emergency Dept?",
                    options=feature_options["Emergency Department Indicator"],
                    index=feature_options["Emergency Department Indicator"].index(preset_chosen["Emergency Department Indicator"]) if preset_chosen else 0
                )

            submit_los = st.form_submit_button("Calculate Estimated Stay", use_container_width=True)

        # Run Prediction
        input_patient = pd.DataFrame([{
            "Age Group": age_group,
            "Gender": gender,
            "Type of Admission": admission_type,
            "APR Severity of Illness Description": severity,
            "APR Risk of Mortality": mortality,
            "Emergency Department Indicator": ed_indicator,
            "APR Medical Surgical Description": med_surg
        }])

        pred_days = float(los_pipeline.predict(input_patient)[0])
        pred_days = max(1.0, pred_days)

        res_col1, res_col2 = st.columns([1, 2], gap="large")

        with res_col1:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center; padding: 24px 20px;">
                <span style="color: {t_subtext}; text-transform: uppercase; font-size: 0.82rem;">Predicted Inpatient Stay</span>
                <div style="font-size: 2.8rem; font-weight: 800; color: {t_accent}; margin: 8px 0;">
                    {pred_days:.1f} <span style="font-size: 1.1rem; color: {t_subtext}; font-weight: 400;">Days</span>
                </div>
                <div style="color: {t_text}; font-size: 0.88rem;">
                    Estimated Window: <strong>{max(1.0, pred_days - 2.5):.1f} – {pred_days + 3.0:.1f} days</strong>
                </div>
                <div style="margin-top: 12px;">
                    <span class="status-badge {'badge-danger' if pred_days > 8 else ('badge-warning' if pred_days > 4 else 'badge-normal')}">
                        {'Extended Stay' if pred_days > 8 else ('Moderate Stay' if pred_days > 4 else 'Standard Stay')}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("##### Clinical Care & Discharge Advisory")
            st.write(
                f"- **Severity Tier:** `{severity}` with `{mortality}` Risk of Mortality.\n"
                f"- **Admission Pathway:** `{admission_type}` ({'Via Emergency' if ed_indicator == 'Y' else 'Non-ED'}).\n"
                f"- **Target Planning:** Pre-discharge assessments recommended to initiate by **Day {max(1, int(pred_days * 0.6))}**."
            )
            st.info(
                "In accordance with clinical machine learning guidelines, post-discharge variables (Total Charges and Costs) are excluded to prevent target leakage."
            )


# =============================================================================
# TAB 4: Model Benchmarks & Exploratory Data Analysis (EDA)
# =============================================================================
with tab4:
    st.subheader("Cross-Model Performance Benchmarks")
    st.markdown(
        "Consolidated benchmark comparing evaluated classifiers on the held-out test split (21,885 beats). "
        "The **MLP Neural Network** achieved the highest overall Weighted F1 score (0.9073) and Accuracy (91.45%)."
    )

    # Build comparison dataframe
    bench_rows = []
    for m_name, m_stats in ecg_bundle["metrics"].items():
        bench_rows.append({
            "Model": m_name,
            "Accuracy": f"{m_stats['accuracy']*100:.2f}%",
            "Weighted F1": f"{m_stats['weighted_f1']:.4f}",
            "Macro F1": f"{m_stats['macro_f1']:.4f}",
            "Precision (W)": f"{m_stats['weighted_precision']:.4f}",
            "Recall (W)": f"{m_stats['weighted_recall']:.4f}",
            "Fit Time (s)": f"{m_stats.get('fit_time_sec', 0.0):.2f}s"
        })
    bench_df = pd.DataFrame(bench_rows).sort_values(by="Weighted F1", ascending=False).reset_index(drop=True)
    bench_df.index = bench_df.index + 1
    bench_df.index.name = "Rank"
    st.dataframe(bench_df, use_container_width=True)

    st.markdown("---")
    cm_col, eda_col = st.columns(2, gap="large")

    with cm_col:
        st.subheader("Confusion Matrix Analysis")
        selected_cm_model = st.selectbox("Select Model for Matrix View", list(ecg_bundle["models"].keys()), index=0)
        cm_data = np.array(ecg_bundle["metrics"][selected_cm_model]["confusion_matrix"])
        class_labels = ecg_bundle["class_names"]

        fig_cm = px.imshow(
            cm_data,
            x=class_labels,
            y=class_labels,
            color_continuous_scale="Blues" if not is_dark else "Viridis",
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            text_auto=True
        )
        fig_cm.update_layout(
            template=t_chart_template,
            paper_bgcolor=t_card_bg,
            plot_bgcolor=t_plot_bg,
            height=320,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with eda_col:
        st.subheader("MIT-BIH Class Distribution Imbalance")
        dist_df = pd.DataFrame({
            "Class": ["N (Normal)", "Q (Unknown)", "V (Ventricular)", "S (Supravent.)", "F (Fusion)"],
            "Count": [90571, 8036, 7235, 2779, 802]
        })
        fig_dist = px.bar(
            dist_df,
            x="Class",
            y="Count",
            color="Class",
            color_discrete_sequence=px.colors.qualitative.Safe,
            text="Count"
        )
        fig_dist.update_layout(
            template=t_chart_template,
            paper_bgcolor=t_card_bg,
            plot_bgcolor=t_plot_bg,
            height=320,
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(type="log", title="Beat Count (Log Scale)", gridcolor=t_grid),
            xaxis=dict(gridcolor=t_grid)
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# =============================================================================
# TAB 5: Case Study Documentation & Methodology
# =============================================================================
with tab5:
    st.subheader("Clinical & Machine Learning Methodology")

    st.markdown(r"""
    ### AAMI EC57 Arrhythmia Heartbeat Standard
    The Association for the Advancement of Medical Instrumentation (AAMI) EC57 standard organizes cardiac arrhythmias into five clinical categories:

    | AAMI Code | Clinical Beat Category | Typical ECG Findings | Clinical Risk Level |
    | :---: | :--- | :--- | :--- |
    | **N** | **Normal Sinus Beat** | Regular narrow QRS complex, clear P wave, uniform ST-T segment | Low / Baseline |
    | **S** | **Supraventricular Ectopic Beat** | Atrial premature beat (APC), nodal escape, aberrant ventricular conduction | Moderate |
    | **V** | **Ventricular Ectopic Beat** | Premature ventricular contraction (PVC), widened QRS (>120ms), discordant T-wave | High (VTach precursor) |
    | **F** | **Fusion Beat** | Hybrid morphology resulting from concurrent ventricular & supraventricular impulse | High / Conduction defect |
    | **Q** | **Unknown / Paced Beat** | Cardiac pacemaker stimulation spikes or unclassifiable waveforms | Clinical Verification |

    ---

    ### Multi-Layer Perceptron (MLP) Architecture
    The Multi-Layer Perceptron neural network demonstrated the highest classification performance across all 10 algorithms benchmarked in the study:
    - **Architecture:** Standardized 4-dimensional feature input vector $\to$ Dense Layer 1 (128 neurons, $\tanh$) $\to$ Dense Layer 2 (64 neurons, $\tanh$) $\to$ Softmax output layer (5 classes).
    - **Regularization:** $L_2$ weight penalty ($\alpha = 10^{-3}$) combined with early stopping validation to prevent over-parametrization.
    - **Optimization:** Adam optimizer with adaptive learning rate scheduling.

    ---

    ### Regularization & Experimental Controls
    1. **Stratified 80/20 Train-Test Partitioning:** Preserves exact class frequency ratios across evaluation splits.
    2. **Strict Preprocessing Isolation:** Feature scalers (`StandardScaler`) and categorical encoders (`OneHotEncoder`) are fitted strictly on the training partition.
    3. **Unsupervised Clustering:** K-Means clustering and PCA allow validation of feature representations without label dependence.
    4. **Leakage Prevention:** Post-discharge variables like Total Charges and Costs are excluded from the hospital stay regression pipeline.

    ---

    ### References & Data Attribution
    - **PhysioNet MIT-BIH Arrhythmia Database:** Mark RG, Moody GB. *PhysioNet: The MIT-BIH Arrhythmia Database*. IEEE Engineering in Medicine and Biology Magazine, 2001.
    - **New York State SPARCS:** Hospital Inpatient Discharges: 2010. NY State Department of Health Open Data.
    """)
