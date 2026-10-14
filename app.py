"""app.py - Streamlit app: Sleep Disorder Classification (loads saved model; never retrains)."""
import os
import streamlit as st
from predict_utils import DISPLAY_NAMES, load_artifacts, predict, validate_input

st.set_page_config(page_title="Sleep Disorder Classification", page_icon="😴", layout="wide")

DISCLAIMER = ("This application is developed for educational and machine-learning demonstration "
              "purposes only. It is not a medical diagnostic tool. Please consult a qualified "
              "healthcare professional for medical concerns.")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    model, preprocessor, meta = get_artifacts()
except FileNotFoundError as e:
    st.title("Sleep Disorder Classification")
    st.error("Model files are missing, so the app cannot make predictions.")
    st.info(f"Missing: {e}. Run `python train_model.py` first and make sure the `model/` folder is in the repository.")
    st.stop()
except Exception:
    st.title("Sleep Disorder Classification")
    st.error("The saved model could not be loaded. It may have been created with a different "
             "scikit-learn version. Re-run `python train_model.py` with the versions in requirements.txt.")
    st.stop()

m = meta["metrics"]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📘 About Project")
    st.write("A final-year mini project that classifies sleep disorder categories "
             "from sleep, lifestyle and health attributes using machine learning.")
    st.header("🤖 Model Information")
    st.write(f"**Model:** {meta['best_model']}")
    st.write(f"**Classes:** No Sleep Disorder, Insomnia, Sleep Apnea")
    st.write(f"**Training records:** {meta['n_train']} (after removing duplicates)")
    st.header("⚠️ Disclaimer")
    st.warning(DISCLAIMER)

# ---------------- Header ----------------
st.title("😴 Sleep Disorder Classification")
st.caption("Machine Learning Based Sleep Health Classification")
st.warning("Educational project only - not a medical diagnosis.")

# ---------------- Input form ----------------
st.subheader("Enter Health & Lifestyle Details")
r = meta["ranges"]
bp = meta["bp_ranges"]
st.caption("Input limits match the value ranges found in the training dataset, "
           "because the model cannot be trusted outside them.")
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.radio("Gender", meta["options"]["Gender"], horizontal=True)
        age = st.number_input("Age (years)", int(r["Age"][0]), int(r["Age"][1]), 40)
        occupation = st.selectbox("Occupation", meta["options"]["Occupation"])
        bmi = st.selectbox("BMI Category", meta["options"]["BMI Category"])
    with c2:
        sleep_dur = st.slider("Sleep Duration (hours/night)", r["Sleep Duration"][0], r["Sleep Duration"][1], 7.0, 0.1)
        quality = st.slider("Quality of Sleep", int(r["Quality of Sleep"][0]), int(r["Quality of Sleep"][1]), 7,
                            help="Your own rating of sleep quality")
        stress = st.slider("Stress Level", int(r["Stress Level"][0]), int(r["Stress Level"][1]), 5)
        activity = st.slider("Physical Activity (minutes/day)", int(r["Physical Activity Level"][0]),
                             int(r["Physical Activity Level"][1]), 60)
    with c3:
        sys_bp = st.number_input("Systolic BP (mmHg)", int(bp["Systolic BP"][0]), int(bp["Systolic BP"][1]), 120)
        dia_bp = st.number_input("Diastolic BP (mmHg)", int(bp["Diastolic BP"][0]), int(bp["Diastolic BP"][1]), 80)
        hr = st.number_input("Resting Heart Rate (bpm)", int(r["Heart Rate"][0]), int(r["Heart Rate"][1]), 72)
        steps = st.number_input("Daily Steps", int(r["Daily Steps"][0]), int(r["Daily Steps"][1]), 7000, step=500)
    clicked = st.button("🔍 Predict Sleep Disorder", type="primary", use_container_width=True)

if clicked:
    user_data = {
        "Gender": gender, "Age": age, "Occupation": occupation, "Sleep Duration": sleep_dur,
        "Quality of Sleep": quality, "Physical Activity Level": activity, "Stress Level": stress,
        "BMI Category": bmi, "Blood Pressure": f"{int(sys_bp)}/{int(dia_bp)}",
        "Heart Rate": hr, "Daily Steps": steps,
    }
    problems = validate_input(user_data, meta)
    if problems:
        st.error("Please fix the following input problems:")
        for p in problems:
            st.write(f"- {p}")
    else:
        try:
            label, probs = predict(user_data, model, preprocessor)
            st.success(f"Prediction: {DISPLAY_NAMES.get(label, label)}")
            if probs:
                st.write("**Model confidence (estimated probability per class):**")
                for cls, p in sorted(probs.items(), key=lambda kv: -kv[1]):
                    st.progress(p, text=f"{DISPLAY_NAMES.get(cls, cls)}: {p:.0%}")
                st.info("These are model scores learned from a small dataset, not medical certainty.")
        except Exception:
            st.error("Sorry, the prediction could not be completed. Please check your inputs and try again.")

st.divider()

# ---------------- Info sections ----------------
a, b = st.columns(2)
with a:
    st.subheader("About the Project")
    st.write("This application uses machine learning to classify sleep disorder categories "
             "based on sleep, lifestyle, and health-related attributes.")
with b:
    st.subheader("How It Works")
    st.markdown("User Input → Data Preprocessing → Machine Learning Model → Prediction → Sleep Disorder Category")

st.subheader("Model Performance")
st.caption(f"Final model: {meta['best_model']} - evaluated on {meta['n_test']} held-out test records.")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Accuracy", f"{m['Accuracy']:.2%}")
k2.metric("Precision (macro)", f"{m['Precision (macro)']:.2%}")
k3.metric("Recall (macro)", f"{m['Recall (macro)']:.2%}")
k4.metric("F1-score (macro)", f"{m['F1 (macro)']:.2%}")
k5.metric("CV F1 (train)", f"{m['CV F1 mean']:.2%}")
st.caption("The test set is small, so these numbers are only a rough estimate of real-world performance.")
if os.path.exists("images/confusion_matrix_logistic_regression.png") and meta["best_model"] == "Logistic Regression":
    st.image("images/confusion_matrix_logistic_regression.png", caption="Confusion matrix (test set)", width=450)

st.subheader("Disclaimer")
st.warning(DISCLAIMER)
