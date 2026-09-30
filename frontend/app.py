import os
from datetime import datetime
import pandas as pd
import requests
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
    }
    .card-fail {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        color: #991B1B;
    }
    .card-success {
        background-color: #F0FDF4;
        border-left: 6px solid #22C55E;
        color: #166534;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    backend_url = st.text_input(
        "FastAPI Backend URL",
        value=os.getenv("BACKEND_URL", "http://127.0.0.1:8000"),
        help="Address of the running FastAPI server"
    ).rstrip("/")

    st.markdown("---")
    st.subheader("Server Status")
    
    # Test connection button / check
    server_online = False
    try:
        health_resp = requests.get(f"{backend_url}/health", timeout=2)
        if health_resp.status_code == 200:
            st.success("🟢 Backend Connected & Healthy")
            server_online = True
        else:
            st.warning(f"🟡 Server responded with status {health_resp.status_code}")
    except requests.exceptions.RequestException:
        st.error("🔴 Backend Offline / Unreachable")
        st.caption("Start backend with: `uvicorn backend.main:app --reload`")

    st.markdown("---")
    st.markdown("### 📋 Quick Presets")
    preset_choice = st.radio(
        "Load Parameter Presets:",
        ["None (Custom)", "Normal Machine Operation", "High Risk of Failure"],
        index=0
    )

# Set input defaults according to selected preset
if preset_choice == "Normal Machine Operation":
    default_type = "M"
    default_air = 300.0
    default_proc = 310.0
    default_speed = 1500
    default_torque = 38.0
    default_wear = 60
elif preset_choice == "High Risk of Failure":
    default_type = "L"
    default_air = 304.5
    default_proc = 313.2
    default_speed = 1250
    default_torque = 72.0
    default_wear = 235
else:
    default_type = "M"
    default_air = 300.0
    default_proc = 310.0
    default_speed = 1500
    default_torque = 40.0
    default_wear = 100

# Main Header
st.markdown('<div class="main-header">⚙️ Machine Failure Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Based Predictive Maintenance System | Real-time Failure Risk Assessment</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_predict, tab_analytics, tab_history = st.tabs(["🔮 Live Prediction", "📊 Analytics & Model Info", "📜 Prediction History"])

# ==============================================================================
# TAB 1: LIVE PREDICTION
# ==============================================================================
with tab_predict:
    col_input, col_result = st.columns([1.1, 1], gap="large")

    with col_input:
        st.subheader("Machine Operating Parameters")
        
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                machine_type = st.selectbox(
                    "Machine Type",
                    options=["L", "M", "H"],
                    index=["L", "M", "H"].index(default_type),
                    help="L = Low quality (50%), M = Medium quality (30%), H = High quality (20%)"
                )
            with c2:
                air_temp = st.number_input(
                    "Air Temperature [K]",
                    min_value=280.0,
                    max_value=350.0,
                    value=float(default_air),
                    step=0.1,
                    help="Ambient temperature in Kelvin"
                )

            c3, c4 = st.columns(2)
            with c3:
                process_temp = st.number_input(
                    "Process Temperature [K]",
                    min_value=280.0,
                    max_value=350.0,
                    value=float(default_proc),
                    step=0.1,
                    help="Operational temperature in Kelvin"
                )
            with c4:
                rot_speed = st.number_input(
                    "Rotational Speed [rpm]",
                    min_value=500,
                    max_value=3500,
                    value=int(default_speed),
                    step=10,
                    help="Spindle rotational speed in RPM"
                )

            c5, c6 = st.columns(2)
            with c5:
                torque = st.number_input(
                    "Torque [Nm]",
                    min_value=0.0,
                    max_value=150.0,
                    value=float(default_torque),
                    step=0.5,
                    help="Torque generated in Newton-meters"
                )
            with c6:
                tool_wear = st.number_input(
                    "Tool Wear [min]",
                    min_value=0,
                    max_value=350,
                    value=int(default_wear),
                    step=1,
                    help="Cumulative tool usage in minutes"
                )

            submit_btn = st.form_submit_button("🚀 Predict Machine Failure", use_container_width=True)

    with col_result:
        st.subheader("Prediction Result")
        
        if submit_btn:
            payload = {
                "type": machine_type,
                "air_temperature": float(air_temp),
                "process_temperature": float(process_temp),
                "rotational_speed": float(rot_speed),
                "torque": float(torque),
                "tool_wear": float(tool_wear)
            }

            try:
                with st.spinner("Analyzing machine operational telemetry..."):
                    res = requests.post(f"{backend_url}/predict", json=payload, timeout=5)

                if res.status_code == 200:
                    data = res.json()
                    pred = data["prediction"]
                    result_text = data["result"]
                    prob = data["failure_probability"]
                    prob_pct = prob * 100

                    # Record to Session History
                    st.session_state.history.append({
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Machine Type": machine_type,
                        "Torque": f"{torque:.1f} Nm",
                        "Tool Wear": f"{tool_wear} min",
                        "Prediction": result_text,
                        "Failure Probability": f"{prob_pct:.2f}%"
                    })

                    # Result Card
                    if pred == 1:
                        st.markdown(f"""
                        <div class="status-card card-fail">
                            <h3 style="margin-top:0; color:#B91C1C;">⚠️ Machine Failure Predicted</h3>
                            <p style="font-size:1.1rem; margin-bottom:5px;"><b>Status:</b> Maintenance Required</p>
                            <p style="font-size:1.4rem; font-weight:700;">Failure Risk: {prob_pct:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="status-card card-success">
                            <h3 style="margin-top:0; color:#15803D;">✅ No Machine Failure Predicted</h3>
                            <p style="font-size:1.1rem; margin-bottom:5px;"><b>Status:</b> Normal Healthy Operation</p>
                            <p style="font-size:1.4rem; font-weight:700;">Failure Risk: {prob_pct:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.progress(min(prob, 1.0))
                    st.caption("ℹ️ Probability reflects model confidence based on Random Forest ensemble votes.")

                    # Entered Parameters Summary
                    st.markdown("#### Input Telemetry Summary")
                    param_df = pd.DataFrame({
                        "Parameter": [
                            "Machine Type",
                            "Air Temperature",
                            "Process Temperature",
                            "Rotational Speed",
                            "Torque",
                            "Tool Wear"
                        ],
                        "Value": [
                            f"Type {machine_type}",
                            f"{air_temp} K",
                            f"{process_temp} K",
                            f"{rot_speed} rpm",
                            f"{torque} Nm",
                            f"{tool_wear} min"
                        ]
                    })
                    st.table(param_df)

                else:
                    st.error(f"Prediction failed with status {res.status_code}: {res.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: Unable to reach FastAPI backend at `{backend_url}`.")
                st.info("Please verify the backend server is running via `uvicorn backend.main:app --reload`.")
        else:
            st.info("👈 Enter machine sensor values on the left and click **'Predict Machine Failure'** to get an instant diagnosis.")

# ==============================================================================
# TAB 2: ANALYTICS & MODEL INFORMATION
# ==============================================================================
with tab_analytics:
    st.subheader("Model Architecture & Performance Analytics")
    
    # Model Metadata Overview
    meta_c1, meta_c2, meta_c3 = st.columns(3)
    with meta_c1:
        st.markdown("""
        <div class="metric-box">
            <small style="color:#64748B;">MODEL</small>
            <h4 style="margin:4px 0 0 0; color:#1E293B;">Tuned Random Forest</h4>
        </div>
        """, unsafe_allow_html=True)
    with meta_c2:
        st.markdown("""
        <div class="metric-box">
            <small style="color:#64748B;">TASK</small>
            <h4 style="margin:4px 0 0 0; color:#1E293B;">Binary Classification</h4>
        </div>
        """, unsafe_allow_html=True)
    with meta_c3:
        st.markdown("""
        <div class="metric-box">
            <small style="color:#64748B;">TARGET</small>
            <h4 style="margin:4px 0 0 0; color:#1E293B;">Machine Failure (0 / 1)</h4>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Final Untouched Test Set Evaluation Metrics")
    
    # Final untouched metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", "97.40%")
    m2.metric("Precision", "58.89%")
    m3.metric("Recall", "77.94%")
    m4.metric("F1 Score", "67.09%")
    m5.metric("ROC-AUC", "96.26%")

    st.markdown("---")
    
    cm_col, fi_col = st.columns([1, 1.2], gap="large")

    with cm_col:
        st.subheader("Test Confusion Matrix")
        st.caption("Evaluation on 2,000 holdout test samples")
        
        cm_data = pd.DataFrame({
            "Predicted: No Failure (0)": [1895, 15],
            "Predicted: Failure (1)": [37, 53]
        }, index=["Actual: No Failure (0)", "Actual: Failure (1)"])
        st.dataframe(cm_data, use_container_width=True)

        st.markdown("""
        * **True Negatives (TN):** `1895`
        * **False Positives (FP):** `37` (Type I error / False alarms)
        * **False Negatives (FN):** `15` (Type II error / Missed failures)
        * **True Positives (TP):** `53` (Successfully flagged failures)
        """)

    with fi_col:
        st.subheader("Model Feature Importances")
        st.caption("Extracted from the trained Random Forest ensemble")

        # Try to load live from backend or use verified extracted values
        feature_data = {
            "Torque [Nm]": 0.3161,
            "Rotational speed [rpm]": 0.2955,
            "Tool wear [min]": 0.2067,
            "Air temperature [K]": 0.0982,
            "Process temperature [K]": 0.0667,
            "Type (L/M/H)": 0.0168
        }
        
        if server_online:
            try:
                info_res = requests.get(f"{backend_url}/model-info", timeout=2)
                if info_res.status_code == 200:
                    raw_fi = info_res.json().get("feature_importances", {})
                    if raw_fi:
                        # Aggregate Type
                        type_sum = sum(v for k, v in raw_fi.items() if "Type" in k)
                        feature_data = {k: v for k, v in raw_fi.items() if "Type" not in k}
                        feature_data["Type (L/M/H)"] = round(type_sum, 4)
            except Exception:
                pass

        fi_df = pd.DataFrame(
            list(feature_data.items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)

        st.bar_chart(fi_df.set_index("Feature"), horizontal=True)

# ==============================================================================
# TAB 3: PREDICTION HISTORY
# ==============================================================================
with tab_history:
    st.subheader("Session Prediction History")
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        h_col1, h_col2 = st.columns([1, 4])
        with h_col1:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()
        with h_col2:
            csv_data = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export History as CSV",
                data=csv_data,
                file_name="machine_failure_predictions.csv",
                mime="text/csv"
            )
    else:
        st.info("No predictions recorded in this session yet. Run a prediction in the 'Live Prediction' tab to see history here.")
