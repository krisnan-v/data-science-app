# Titanic Model Monitoring Dashboard

## Overview
This project builds an end-to-end **Agile Data Science application** using the Titanic dataset:
- Two models (baseline v1, improved v2)
- Streamlit prediction app
- Monitoring dashboard
- Logging utility
- Agile iteration evidence

## Features
- Compare predictions from v1 and v2
- Collect latency + user feedback
- Monitor model behaviour with dashboard
- Logs stored in `monitoring_logs.csv`

## Setup
```bash
git clone https://github.com/<your-username>/data-science-app.git
cd data-science-app
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt