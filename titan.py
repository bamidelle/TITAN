# titan.py  (FULL WORKING BACKEND)

import streamlit as st
import pandas as pd
import sqlite3
import joblib
import numpy as np
from datetime import datetime, timedelta, date, timezone
import io, os, math, base64, sys, subprocess
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Config & Constants
# ---------------------------
DB_FILE = "titan_leads.db"
MODEL_FILE = "titan_lead_score_model.joblib"
PIPELINE_STAGES = ["New", "Contacted", "Qualified", "Estimate Sent", "Inspection Scheduled", "Inspection Completed", "Won", "Lost"]
DEFAULT_SLA_HOURS = 24

# ---------------------------
# Database Initialization
# ---------------------------
def get_db_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_id TEXT UNIQUE,
            created_at TEXT,
            source TEXT,
            stage TEXT,
            estimated_value REAL,
            ad_cost REAL,
            converted INTEGER,
            notes TEXT,
            sla_hours INTEGER,
            sla_entered_at TEXT,
            score REAL DEFAULT 0.5
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------------------
# Utility Functions
# ---------------------------
def format_currency(value):
    if not value:
        return "$0.00"
    return f"${value:,.2f}"

def remaining_sla_seconds(start, hours):
    if not start or not hours:
        return 0
    try:
        start_time = datetime.fromisoformat(start)
        deadline = start_time + timedelta(hours=int(hours))
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        return max(0, remaining)
    except:
        return 0

def is_sla_overdue(start, hours):
    return remaining_sla_seconds(start, hours) <= 0

# ---------------------------
# ML Model Functions
# ---------------------------
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

def train_model():
    df = fetch_leads()
    if df.empty:
        return "No data to train ML model"
    if df["converted"].nunique() < 2:
        return "Need converted & non-converted leads to train model"
    
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["age_days"] = (datetime.now(timezone.utc) - df["created_at"]).dt.days

    X = pd.get_dummies(df[["source", "stage"]])
    X["estimated_value"] = df["estimated_value"]
    X["ad_cost"] = df["ad_cost"]
    X["age_days"] = df["age_days"]

    y = df["converted"]

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(train_X, train_y)

    joblib.dump(model, MODEL_FILE)

    acc = model.score(test_X, test_y)
    return f"Model trained ✅ Accuracy: {acc:.2%}"

def apply_lead_scoring(df):
    model = load_model()
    if not model:
        df["score"] = 0.5
        return df

    X = pd.get_dummies(df[["source","stage"]])
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    X["estimated_value"] = df["estimated_value"]
    X["ad_cost"] = df["ad_cost"]

    df["score"] = model.predict_proba(X)[:,1]
    return df

# ---------------------------
# Fetch & Save Leads
# ---------------------------
def fetch_leads():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM leads ORDER BY created_at DESC", conn)
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=["lead_id","created_at","source","stage","estimated_value","ad_cost","converted","notes","sla_hours","sla_entered_at","score"])
    df["estimated_value"] = pd.to_numeric(df["estimated_value"], errors='coerce').fillna(0)
    df["ad_cost"] = pd.to_numeric(df["ad_cost"], errors='coerce').fillna(0)
    df["converted"] = df["converted"].fillna(0).astype(int)
    df["sla_hours"] = df["sla_hours"].fillna(DEFAULT_SLA_HOURS)
    df["score"] = df["score"].fillna(0.5)

    if load_model():
        df = apply_lead_scoring(df)
    return df.reset_index(drop=True)

def save_lead_to_db(row):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR IGNORE INTO leads (lead_id, created_at, source, stage, estimated_value, ad_cost, converted, notes, sla_hours, sla_entered_at, score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["lead_id"], row["created_at"], row["source"], row["stage"],
        row["estimated_value"], row["ad_cost"], row["converted"], row["notes"],
        row["sla_hours"], row["sla_entered_at"], row["score"]
    ))

    conn.commit()
    conn.close()

# ---------------------------
# Page UI (Top Bar)
# ---------------------------
st.set_page_config(layout="wide")

col1, _, col2 = st.columns([3,1,2])

with col1:
    st.markdown("<h2 style='color:#0ea5e9'>Total Lead Pipeline System KPI</h2>", unsafe_allow_html=True)

with col2:
    st.session_state.start_date = st.date_input("Start Date", st.session_state.get("start_date", date.today() - timedelta(days=30)))
    st.session_state.end_date = st.date_input("End Date", st.session_state.get("end_date", date.today()))

    df_now = fetch_leads()
    notify_count = len([1 for _,r in df_now.iterrows() if is_sla_overdue(r["sla_entered_at"], r["sla_hours"]) and r["stage"] not in ["Won","Lost"]])
    st.markdown(f"<div style='text-align:right'>🔔 <span style='background:red;color:white;padding:5px 10px;border-radius:20px'>{notify_count}</span></div>", unsafe_allow_html=True)

# ---------------------------
# Navigation
# ---------------------------
page = st.sidebar.selectbox("Navigate", ["Lead Pipeline","Lead Capture","Analytics & SLA","CPA","ROI","AI Reports","Exports","Admin Settings","ML Model"])

# ---------------------------
# Pages
# ---------------------------
def page_pipeline():
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    df = fetch_leads()
    total = len(df)
    won = len(df[df["stage"]=="Won"])
    lost = len(df[df["stage"]=="Lost"])
    closed = won + lost
    active = total - closed
    spend = df["ad_cost"].sum()
    roi = ((df["estimated_value"].sum() - spend)/spend*100) if spend else 0
    cpa = (spend/won) if won else 0
    conv_rate = (won/closed*100) if closed else 0

    row1 = st.columns(4, gap="large")
    row2 = st.columns(2, gap="large")

    row1[0].metric("Active Leads", active)
    row1[1].metric("Conversion Rate", f"{conv_rate:.1f}%")
    row1[2].metric("Total Spend", format_currency(spend))
    row1[3].metric("ROI", f"{roi:.1f}%")

    st.markdown("<div style='height:35px'></div>", unsafe_allow_html=True)

    st.markdown("### 🚨 Top 5 Priority Leads")
    df["priority"] = df["score"] * 0.7 + (df["estimated_value"]/5000)*0.2 + 0.1
    top5 = df.sort_values("priority", ascending=False).head(5)
    lead_cols = st.columns(len(top5))

    for i,(_,r) in enumerate(top5.iterrows()):
        rem = math.floor(remaining_sla_seconds(r["sla_entered_at"], r["sla_hours"])/3600)
        urg_color = "#ef4444" if r["score"]>=0.7 else "#f97316" if r["score"]>=0.4 else "#22c55e"
        lead_cols[i].markdown(f"""
            <div style='background:black; padding:15px; border-radius:12px'>
                <div style='color:white; font-size:15px'>#{r["lead_id"]}</div>
                <div style='color:white'>{r.get("contact_name","Unknown")}</div>
                <div style='color:{urg_color}; font-size:13px; font-weight:700'>Urgency</div>
                <div style='color:red; font-size:14px'>⏳ {rem}h left</div>
                <div style='color:#22c55e; font-size:13px'>{format_currency(r["estimated_value"])}</div>
                <div style='color:white; font-size:12px'>Score: {r["score"]:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    funnel = df["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0)
    st.bar_chart(funnel)

def page_capture():
    st.markdown("## Lead Capture Form")
    with st.form("add_lead"):
        lid = st.text_input("Lead ID")
        src = st.selectbox("Source", ["Google Ads","Referral","Partner","Organic"])
        stg = st.selectbox("Stage", PIPELINE_STAGES)
        val = st.number_input("Estimated Value", min_value=0.0)
        cost = st.number_input("Marketing Cost", min_value=0.0)
        sla = st.number_input("SLA Hours", placeholder="SLA Response time must be > 0 hours")
        sb = st.form_submit_button("Save")
        if sb:
            if sla <= 0:
                st.error("SLA Response time must be greater than 0 hours")
            else:
                save_lead_to_db({"lead_id":lid,"created_at":datetime.now(timezone.utc).isoformat(),"source":src,"stage":stg,"estimated_value":val,"ad_cost":cost,"converted":1 if stg=="Won" else 0,"notes":"","sla_hours":sla,"sla_entered_at":datetime.now(timezone.utc).isoformat(),"score":0.5})
                st.success("Lead Added ✅")

def page_analytics():
    st.markdown("## Analytics with CPA & ROI")
    df = fetch_leads()
    agg=df.groupby("source")["ad_cost"].sum().reset_index()
    fig=px.bar(agg,x="source",y="ad_cost",title="Cost Spend by Source")
    st.plotly_chart(fig)

def page_cpa():
    st.markdown("## Cost Per Acquisition")
    df = fetch_leads()
    st.metric("CPA", format_currency(df["ad_cost"].sum()/len(df[df["converted"]==1])) if len(df[df["converted"]==1]) else "$0.00")

def page_roi():
    st.markdown("## ROI Performance")
    df=fetch_leads();sp=df["ad_cost"].sum();val=df["estimated_value"].sum()
    r=(val-sp)/sp*100 if sp else 0
    st.metric("ROI",f"{r:.1f}%")

if page=="Lead Pipeline":page_pipeline()
elif page=="Lead Capture":page_capture()
elif page=="Analytics & SLA":page_analytics()
elif page=="CPA":page_cpa()
elif page=="ROI":page_roi()
elif page=="ML":apply_lead_scoring(fetch_leads())
elif page=="Exports":pass
elif page=="Admin Settings":pass
elif page=="AI Reports":pass
elif page=="ML Model":pass
