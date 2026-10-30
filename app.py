from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------- Configuration -----------------------------
st.set_page_config(
    page_title="CREDVION | AI Fairness Audit",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: 'Manrope', sans-serif;}
.stApp {background: radial-gradient(ellipse at 80% 0%, #102016 0%, transparent 35%), #050805; color:#F5F7F2;}
[data-testid="stSidebar"] {background:#080D09; border-right:1px solid #263529;}
[data-testid="stSidebar"] * {color:#EAF1E8;}
h1,h2,h3 {color:#F5F7F2 !important; letter-spacing:-.5px;}
.cred-hero {background:linear-gradient(110deg,#111C13,#0A110B 65%,#142218); border:1px solid #344735; border-radius:18px; padding:30px 34px; margin:5px 0 24px;}
.cred-kicker {font:10px 'DM Mono'; letter-spacing:2px; color:#A9C19F;}
.cred-hero h1 {font-size:38px; line-height:1.1; margin:12px 0 8px;}
.cred-hero h1 span {color:#B7FF35;}
.cred-hero p {color:#AAB8AA; font-size:13px; max-width:780px; line-height:1.7;}
.cred-card {background:linear-gradient(145deg,#101810,#0A100B); border:1px solid #29392C; border-radius:12px; padding:17px 19px; min-height:112px;}
.cred-label {font:10px 'DM Mono'; color:#91A291; letter-spacing:1px; text-transform:uppercase;}
.cred-value {font-size:27px; font-weight:800; color:#F4F8F2; margin-top:9px;}
.cred-sub {font-size:10px; color:#718273; margin-top:3px;}
div.stButton > button, div.stDownloadButton > button {background:#B7FF35; color:#071007; border:0; border-radius:8px; font-weight:800; min-height:42px;}
div.stButton > button:hover, div.stDownloadButton > button:hover {background:#D0FF80; color:#071007; border:0;}
.stTabs [data-baseweb="tab"] {color:#A9B7AA;}
.stTabs [aria-selected="true"] {color:#B7FF35 !important;}
[data-testid="stMetric"] {background:#101810; border:1px solid #29392C; padding:14px; border-radius:10px;}
[data-testid="stMetricLabel"] {color:#9CAC9D;}
[data-testid="stMetricValue"] {color:#F5F7F2;}
div[data-testid="stDataFrame"] {border:1px solid #29392C; border-radius:9px;}
.stAlert {border-radius:9px;}
.small-note {color:#8FA08F;font-size:11px;}
</style>
"""
st.markdown(BG_CSS, unsafe_allow_html=True)

# ----------------------------- Session state -----------------------------
for key, default in {
    "dataset": None, "dataset_name": None, "train_result": None,
    "external_result": None, "external_df": None, "selected_run": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------------------- Utility functions -----------------------------
def read_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Unable to read this file: {exc}") from exc
    raise ValueError("Please upload a CSV, XLS or XLSX file.")

def normalize_target(series: pd.Series, positive_label: str | None):
    values = list(series.dropna().unique())
    if len(values) != 2:
        raise ValueError(f"Target must contain exactly two classes. Found {len(values)}.")
    if positive_label is not None and str(positive_label) != "":
        found = [v for v in values if str(v) == str(positive_label)]
        if not found:
            raise ValueError("The selected positive/default label is not present in the target column.")
        positive = found[0]
    else:
        positive = sorted(values, key=lambda x: str(x))[-1]
    return (series == positive).astype(int), positive, values

def build_pipeline(X: pd.DataFrame, estimator):
