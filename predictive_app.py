import time
import joblib
import pandas as pd
import streamlit as st
from log_utils import log_prediction

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
st.title("Titanic Survival Prediction App with Monitoring")

@st.cache_resource
def load_models():
    return joblib.load("titanic_model_v1.pkl"), joblib.load("titanic_model_v2.pkl")

model_v1, model_v2 = load_models()

# Input section
st.sidebar.header("Passenger Features")
pclass = st.sidebar.selectbox("Passenger Class", [1,2,3])
sex = st.sidebar.selectbox("Sex", ["male","female"])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0.0, 500.0, 32.0)

sex_encoded = 0 if sex=="male" else 1
input_df = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare]],
                        columns=["Pclass","Sex","Age","SibSp","Parch","Fare"])

st.subheader("Input Summary")
st.write(input_df)

if st.button("Run Prediction"):
    start = time.time()
    pred_v1 = model_v1.predict(input_df[["Pclass","Sex","Age"]])[0]
    pred_v2 = model_v2.predict(input_df)[0]
    latency = (time.time() - start) * 1000

    st.write(f"Model v1 Prediction: {'Survived' if pred_v1==1 else 'Did not survive'}")
    st.write(f"Model v2 Prediction: {'Survived' if pred_v2==1 else 'Did not survive'}")
    st.write(f"Latency: {latency:.1f} ms")

    feedback_score = st.slider("Feedback Score (1-5)", 1, 5, 4)
    feedback_text = st.text_area("Comments")

    if st.button("Submit Feedback"):
        log_prediction("v1", "baseline", str(input_df.to_dict()), pred_v1, latency, feedback_score, feedback_text)
        log_prediction("v2", "improved", str(input_df.to_dict()), pred_v2, latency, feedback_score, feedback_text)
        st.success("Feedback logged successfully!")