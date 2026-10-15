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
import streamlit as st

# -----------------------------------------------------------------------------
# App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioPulse AI — ECG & Hospital Case Study",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
SAMPLE_DATA_DIR = ROOT_DIR / "sample_data"

# Custom Styling (Dark Medical Dashboard Aesthetics)
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #00D26A;
    }
    .metric-card {
        background: #1A1F2C;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 15px;
    }
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-normal {
        background: rgba(0, 210, 106, 0.15);
        color: #00D26A;
        border: 1px solid #00D26A;
    }
    .badge-warning {
        background: rgba(255, 179, 0, 0.15);
        color: #FFB300;
        border: 1px solid #FFB300;
    }
    .badge-danger {
        background: rgba(255, 75, 75, 0.15);
        color: #FF4B4B;
        border: 1px solid #FF4B4B;
    }
    .badge-info {
        background: rgba(0, 173, 237, 0.15);
        color: #00ADED;
        border: 1px solid #00ADED;
    }
    .hero-banner {
        background: linear-gradient(135deg, #111827 0%, #1F2937 50%, #111827 100%);
        border: 1px solid rgba(0, 210, 106, 0.25);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 25px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Asset & Model Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading machine learning models...")
def load_ecg_bundle():
    bundle_path = MODELS_DIR / "ecg_classifier_bundle.joblib"
    if not bundle_path.exists():
        # Fallback: run training script if bundle is missing
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
    # R-peak centered at 0 (index 150)
    r_wave = r_peak_amp * np.exp(-((t) ** 2) / 0.0015)
    # Q wave (negative deflection right before R)
    q_wave = -0.15 * r_peak_amp * np.exp(-((t + 0.08) ** 2) / 0.001)
    # S wave (negative deflection right after R)
    s_wave = -0.25 * r_peak_amp * np.exp(-((t - 0.08) ** 2) / 0.0012)
    # P wave (atrial depolarization)
    p_wave = 0.2 * np.exp(-((t + 0.35) ** 2) / 0.008)
    # T wave (ventricular repolarization)
    t_wave = 0.35 * np.exp(-((t - 0.40) ** 2) / 0.015)
    
    baseline = 0.02 * np.sin(2 * np.pi * 0.5 * t)
    noise = np.random.normal(0, noise_level, points) if noise_level > 0 else 0
    signal = r_wave + q_wave + s_wave + p_wave + t_wave + baseline + noise
    return signal


def extract_signal_features(signal_array: np.ndarray):
    """Extract standard features matching the case study feature vector."""
    signal = np.asarray(signal_array, dtype=float)
    if len(signal) != 300:
        # Interpolate or slice to 300 points
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
    st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=64)
    st.title("CardioPulse AI")
    st.caption("Clinical ECG & Hospital Inpatient Case Study")
    st.markdown("---")

    selected_model = st.selectbox(
        "🧠 Primary Classifier Model",
        options=list(ecg_bundle["models"].keys()),
        index=0,
        help="Select which regularized model algorithm processes the ECG signals."
    )

    st.markdown("### 📋 Case Study Overview")
    st.info(
        "**Track 1: MIT-BIH Arrhythmia**\n"
        "- 109,423 Ambulatory Beats\n"
        "- 5 AAMI EC57 Rhythm Classes\n"
        "- 5-Fold Regularized Tuning\n\n"
        "**Track 2: NY SPARCS 2010**\n"
        "- Inpatient Discharge Modeling\n"
        "- Target: Hospital Length of Stay\n"
        "- Gradient Boosting Pipeline"
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.8rem;'>"
        "Designed for Streamlit Community Cloud<br>PhysioNet & NY Open Data"
        "</div>",
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# Header / Hero Section
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; color: #FAFAFA; display: flex; align-items: center; gap: 10px;">
                <span>🫀 CardioPulse AI</span>
            </h1>
            <p style="margin: 6px 0 0 0; color: #9CA3AF; font-size: 1.05rem;">
                Machine Learning Case Study: ECG Arrhythmia Classification & Hospital Inpatient Length-of-Stay
            </p>
        </div>
        <div style="margin-top: 10px;">
            <span class="status-badge badge-normal">MIT-BIH Arrhythmia</span>
            <span class="status-badge badge-info" style="margin-left: 6px;">NY SPARCS 2010</span>
            <span class="status-badge badge-warning" style="margin-left: 6px;">Anti-Overfitting</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🫀 ECG Waveform Studio",
    "🏥 Hospital Stay Predictor",
    "📊 Benchmarks & EDA",
    "📘 Case Study Documentation"
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
                "Select Heartbeat Rhythm Type (AAMI Class)",
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
                    st.success(f"Loaded signal with {len(current_signal)} samples!")
                except Exception as e:
                    st.error(f"Error parsing uploaded file: {e}")
                    current_signal = synthesize_ecg()
            else:
                st.info("No file uploaded yet. Showing standard normal heartbeat sample.")
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
        st.subheader("3. Fine-Tune Feature Parameters")
        st.caption("Adjust extracted features interactively to test decision boundaries:")
        adj_r_peak = st.slider("R-Peak Amplitude (mV)", -1.0, 4.0, float(extracted["r_peak_amplitude"]), 0.05)
        adj_energy = st.slider("Signal Energy", 0.0, 200.0, float(extracted["signal_energy"]), 1.0)

        # Update features with adjusted values
        inference_features = {
            "mean_amplitude": extracted["mean_amplitude"],
            "std_amplitude": extracted["std_amplitude"],
            "r_peak_amplitude": adj_r_peak,
            "signal_energy": adj_energy,
            "raw_signal": extracted["raw_signal"]
        }

    with col_viz:
        st.subheader("Interactive ECG Waveform (300 Samples Centered on R-Peak)")

        # Plot ECG Waveform with Plotly
        samples_x = np.arange(len(current_signal))
        time_sec = (samples_x - 150) / 360.0  # MIT-BIH 360Hz sampling rate

        fig = go.Figure()

        # ECG Trace
        fig.add_trace(go.Scatter(
            x=time_sec,
            y=current_signal,
            mode="lines",
            name="ECG Lead MLII",
            line=dict(color="#00D26A", width=2.5),
            hovertemplate="Time: %{x:.3f}s<br>Voltage: %{y:.3f} mV<extra></extra>"
        ))

        # Highlight R-Peak at sample 150 (t=0)
        r_val = current_signal[150]
        fig.add_trace(go.Scatter(
            x=[0],
            y=[r_val],
            mode="markers+text",
            name="R-Peak (t=0s)",
            marker=dict(color="#FF4B4B", size=11, symbol="diamond"),
            text=["R-Peak"],
            textposition="top center",
            textfont=dict(color="#FAFAFA", size=12)
        ))

        # Annotate standard clinical segments
        fig.add_vrect(x0=-0.06, x1=0.06, fillcolor="rgba(255, 75, 75, 0.08)", layer="below", line_width=0, annotation_text="QRS", annotation_position="bottom right")

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1A1F2C",
            plot_bgcolor="#111827",
            margin=dict(l=40, r=20, t=30, b=40),
            height=340,
            xaxis=dict(
                title="Time relative to R-peak (seconds)",
                gridcolor="rgba(255, 255, 255, 0.08)",
                zerolinecolor="rgba(255, 255, 255, 0.15)"
            ),
            yaxis=dict(
                title="Amplitude (mV)",
                gridcolor="rgba(255, 255, 255, 0.08)",
                zerolinecolor="rgba(255, 255, 255, 0.15)"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Run Prediction
        pred_class, class_probs = predict_ecg_class(inference_features, selected_model)
        class_desc = ecg_bundle["class_descriptions"].get(pred_class, "Arrhythmia Classification")
        risk_level, clinical_note = ecg_bundle["class_risk"].get(pred_class, ("Unknown", "Further clinical evaluation required."))

        # Prediction Summary Banner
        badge_style = "badge-normal" if pred_class == "N" else ("badge-danger" if pred_class in ["V", "F"] else "badge-warning")

        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.85rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px;">Model Prediction ({selected_model})</span>
                    <h2 style="margin: 4px 0; color: #FAFAFA;">Class [{pred_class}]: {class_desc}</h2>
                    <span class="status-badge {badge_style}">{risk_level}</span>
                    {f'<span class="status-badge badge-info" style="margin-left: 8px;">Ground Truth: {true_label}</span>' if true_label else ''}
                </div>
                <div style="text-align: right; margin-top: 8px;">
                    <span style="color: #9CA3AF; font-size: 0.85rem;">Estimated Rhythm</span>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #00D26A;">{est_hr:.1f} <span style="font-size: 0.9rem; color: #9CA3AF;">BPM</span></div>
                    <span style="color: #6B7280; font-size: 0.8rem;">RR-Interval: {est_rr:.3f}s</span>
                </div>
            </div>
            <p style="margin: 14px 0 0 0; color: #D1D5DB; font-size: 0.95rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;">
                💡 <strong>Clinical Guidance:</strong> {clinical_note}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Probabilities Bar Chart
        st.markdown("#### Multi-Class Probability Distribution")
        prob_df = pd.DataFrame({
            "Class": list(class_probs.keys()),
            "Probability": [class_probs[c] for c in class_probs.keys()],
            "Description": [ecg_bundle["class_descriptions"].get(c, "") for c in class_probs.keys()]
        })

        color_map = {
            "N": "#00D26A",
            "S": "#FFB300",
            "V": "#FF4B4B",
            "F": "#FF7043",
            "Q": "#00ADED"
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
            template="plotly_dark",
            paper_bgcolor="#1A1F2C",
            plot_bgcolor="#111827",
            height=220,
            margin=dict(l=20, r=20, t=10, b=30),
            xaxis=dict(range=[0, 1.05], title="Predicted Probability", gridcolor="rgba(255, 255, 255, 0.08)"),
            yaxis=dict(title="AAMI Class"),
            showlegend=False
        )
        st.plotly_chart(fig_prob, use_container_width=True)


# =============================================================================
# TAB 2: Hospital Inpatient Length of Stay (LOS) Predictor
# =============================================================================
with tab2:
    st.subheader("🏥 Hospital Inpatient Length of Stay (LOS) Prediction")
    st.markdown(
        "Predict estimated inpatient duration using clinical admission risk parameters "
        "trained on the **New York SPARCS Inpatient Discharges** dataset."
    )

    if los_bundle is None:
        st.warning("Hospital Length of Stay model bundle is currently generating. Please refresh in a moment.")
    else:
        los_pipeline = los_bundle["pipeline"]
        feature_options = los_bundle["feature_options"]

        # Preset Patients
        st.markdown("##### Quick Patient Case Profiles")
        preset_cols = st.columns(4)
        preset_chosen = None

        with preset_cols[0]:
            if st.button("🚨 Emergency Cardiac Adult"):
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
            if st.button("🩺 Elective Joint Surgery"):
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
            if st.button("👶 Routine Newborn Delivery"):
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
            if st.button("👵 Elderly ICU Admission"):
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
            <div class="metric-card" style="text-align: center; padding: 28px 20px;">
                <span style="color: #9CA3AF; text-transform: uppercase; font-size: 0.85rem;">Predicted Inpatient Stay</span>
                <div style="font-size: 3rem; font-weight: 800; color: #00D26A; margin: 10px 0;">
                    {pred_days:.1f} <span style="font-size: 1.2rem; color: #9CA3AF;">Days</span>
                </div>
                <div style="color: #D1D5DB; font-size: 0.9rem;">
                    Estimated Range: <strong>{max(1.0, pred_days - 2.5):.1f} – {pred_days + 3.0:.1f} days</strong>
                </div>
                <div style="margin-top: 15px;">
                    <span class="status-badge {'badge-danger' if pred_days > 8 else ('badge-warning' if pred_days > 4 else 'badge-normal')}">
                        {'Extended Stay' if pred_days > 8 else ('Moderate Stay' if pred_days > 4 else 'Standard Stay')}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### Clinical Care & Discharge Advisory")
            st.write(
                f"- **Severity Tier:** `{severity}` with `{mortality}` Risk of Mortality.\n"
                f"- **Admission Pathway:** `{admission_type}` ({'Via ED' if ed_indicator == 'Y' else 'Non-ED'}).\n"
                f"- **Target Planning:** Pre-discharge assessments should initiate by **Day {max(1, int(pred_days * 0.6))}**."
            )
            st.info(
                "💡 **Model Integrity Note:** In accordance with clinical machine learning guidelines, "
                "post-discharge variables like Total Charges and Total Costs are excluded to prevent data leakage."
            )


# =============================================================================
# TAB 3: Model Benchmarks & Exploratory Data Analysis (EDA)
# =============================================================================
with tab3:
    st.subheader("📊 Cross-Model Performance Benchmarks")
    st.markdown(
        "Evaluation results across all 5 regularized classifiers on the stratified 80/20 test split (21,885 beats):"
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
            "Fit Time (s)": f"{m_stats['fit_time_sec']:.2f}s"
        })
    bench_df = pd.DataFrame(bench_rows)
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

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
            color_continuous_scale="Viridis",
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            text_auto=True
        )
        fig_cm.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1A1F2C",
            plot_bgcolor="#111827",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with eda_col:
        st.subheader("MIT-BIH Class Distribution Imbalance")
        # Class distribution numbers from the dataset
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
            template="plotly_dark",
            paper_bgcolor="#1A1F2C",
            plot_bgcolor="#111827",
            height=340,
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(type="log", title="Beat Count (Log Scale)")
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# =============================================================================
# TAB 4: Case Study Documentation & Methodology
# =============================================================================
with tab4:
    st.subheader("📘 Clinical & Machine Learning Methodology")

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

    ### Anti-Overfitting & Regularization Protocols
    To satisfy rigorous academic and clinical machine learning standards:
    1. **Stratified 80/20 Train-Test Partitioning:** Preserves exact class frequency ratios across the training and hold-out evaluation splits.
    2. **Strict Preprocessing Isolation:** Feature scalers (`StandardScaler`) and categorical encoders (`OneHotEncoder`) are fitted strictly on the training partition to eliminate data leakage.
    3. **Hyperparameter Regularization:**
       - **Logistic Regression:** Inverse penalty $C=0.1$ with $L_2$ shrinkage.
       - **Decision Tree:** Restricting `max_depth=6`, `min_samples_split=20`, and `min_samples_leaf=5`.
       - **Random Forest:** Ensemble of regularized trees with bounded leaf samples.
       - **Gaussian Naive Bayes:** Variance smoothing ($\sigma=0.1$) to prevent zero-variance degradation.
       - **K-Nearest Neighbors:** Constrained to $k=15$ with distance weighting to prevent 1-NN memorization.

    ---

    ### References & Data Attribution
    - **PhysioNet MIT-BIH Arrhythmia Database:** Mark RG, Moody GB. *PhysioNet: The MIT-BIH Arrhythmia Database*. IEEE Engineering in Medicine and Biology Magazine, 2001.
    - **New York State SPARCS:** Hospital Inpatient Discharges: 2010. NY State Department of Health Open Data.
    """)
