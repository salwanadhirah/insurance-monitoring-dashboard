# predictive_app.py
import time

import joblib
import pandas as pd
import streamlit as st

from log_utils import log_prediction

st.set_page_config(page_title="Insurance Charges Prediction App with Monitoring",
                   layout="centered")

st.title("Insurance Charges Prediction App with Live Monitoring")

@st.cache_resource
def load_models():
    v1_model = joblib.load("charges_model_v1.pkl")  # baseline
    v2_model = joblib.load("charges_model_v2.pkl")  # improved
    return v1_model, v2_model

v1_model, v2_model = load_models()

# ---------- Initialise session state ----------
if "pred_ready" not in st.session_state:
    st.session_state["pred_ready"] = False
if "v1_pred" not in st.session_state:
    st.session_state["v1_pred"] = None
if "v2_pred" not in st.session_state:
    st.session_state["v2_pred"] = None
if "latency_ms" not in st.session_state:
    st.session_state["latency_ms"] = None
if "input_summary" not in st.session_state:
    st.session_state["input_summary"] = ""

# ---------- INPUT SECTION ----------
st.sidebar.header("Input Parameters")

age = st.sidebar.slider("Age", min_value=18, max_value=64, value=30)
bmi = st.sidebar.slider("BMI", min_value=15.0, max_value=55.0, value=27.5, step=0.1)
children = st.sidebar.slider("Children", min_value=0, max_value=5, value=0)

sex = st.sidebar.selectbox("Sex", ["male", "female"])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Canonical input dataframe
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region],
})

st.subheader("Input Summary")
st.write(input_df)

# ---------- BUTTON 1: RUN PREDICTION ----------
if st.button("Run Prediction"):
    start_time = time.time()

    # v1: baseline – uses age, bmi, children
    input_v1 = input_df[["age", "bmi", "children"]]
    v1_pred = v1_model.predict(input_v1)[0]

    # v2: improved – uses all three features
    input_v2 = input_df[["age", "sex", "bmi", "children", "smoker", "region"]]
    v2_pred = v2_model.predict(input_v2)[0]

    latency_ms = (time.time() - start_time) * 1000.0

    # Store in session_state so they survive reruns
    st.session_state["v1_pred"] = float(v1_pred)
    st.session_state["v2_pred"] = float(v2_pred)
    st.session_state["latency_ms"] = float(latency_ms)
    st.session_state["input_summary"] = f"age={age}, sex={sex}, bmi={bmi}, children={children}, smoker={smoker}, region={region}"
    st.session_state["pred_ready"] = True

# ---------- SHOW PREDICTIONS IF READY ----------
if st.session_state["pred_ready"]:
    st.subheader("Predictions")
    st.write(f"v1 Model (v1 - baseline): **${st.session_state['v1_pred']:,.2f}**")
    st.write(f"v2 Model (v2 - improved): **${st.session_state['v2_pred']:,.2f}**")
    st.write(f"Latency: {st.session_state['latency_ms']:.1f} ms")
else:
    st.info("Click **Run Prediction** to see model outputs before giving feedback.")

# ---------- FEEDBACK SECTION ----------
st.subheader("Your Feedback on These Predictions")

feedback_score = st.slider(
    "How useful were these predictions? (1 = Poor, 5 = Excellent)",
    min_value=1,
    max_value=5,
    value=4,
    key="feedback_score",
)
feedback_text = st.text_area("Comments (optional)", key="feedback_text")

# ---------- BUTTON 2: SUBMIT FEEDBACK ----------
if st.button("Submit Feedback"):
    if not st.session_state["pred_ready"]:
        st.warning("Please run the prediction first, then submit your feedback.")
    else:
        # Log both models using saved predictions and input summary
        log_prediction(
            model_version="v1_old",
            model_type="baseline",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["v1_pred"],
            latency_ms=st.session_state["latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )

        log_prediction(
            model_version="v2_new",
            model_type="improved",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["v2_pred"],
            latency_ms=st.session_state["latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )

        st.success(
            "Feedback and predictions have been saved to monitoring_logs.csv. "
            "You can now view them in the monitoring dashboard."
        )
