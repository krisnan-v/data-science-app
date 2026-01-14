import os
import pandas as pd
import streamlit as st
from log_utils import LOG_PATH

st.set_page_config(page_title="Titanic Model Monitoring", layout="wide")
st.title("Titanic Model Monitoring Dashboard")

if not os.path.exists(LOG_PATH):
    st.warning("No logs yet. Run predictions first.")
    st.stop()

logs = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])

st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Predictions", len(logs))
col2.metric("Avg Feedback Score", f"{logs['feedback_score'].mean():.2f}")
col3.metric("Avg Latency (ms)", f"{logs['latency_ms'].mean():.1f}")

st.markdown("---")
st.subheader("Model Comparison")
summary = logs.groupby("model_version").agg({"feedback_score":"mean","latency_ms":"mean"})
st.dataframe(summary)

st.subheader("Recent Comments")
comments = logs[logs["feedback_text"].astype(str).str.strip()!=""].tail(10)
for _, row in comments.iterrows():
    st.write(f"[{row['timestamp']}] {row['model_version']} – {row['feedback_text']}")