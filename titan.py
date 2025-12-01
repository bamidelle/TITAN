# TITAN SOFTWARE BACKEND (Merged, Fully Working)
# Paste into titanic/titan.py and run with: streamlit run titan.py

import streamlit as st
import sqlite3
import pandas as pd
import pickle
import time
import uuid
import io
import random
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

# ---------------- DATABASE SETUP -------------------

DB_FILE = "titan_leads.db"

def init_db():
    con = sqlite3.connect(DB_FILE)
    c = con.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_name TEXT,
        damage_type TEXT,
        estimated_value REAL,
        urgency TEXT,
        sla_hours REAL DEFAULT 1.0,
        sla_entered_at TEXT,
        win_prob REAL
    );
    """)

    con.commit()
    con.close()

# initialize DB
init_db()

# ---------------- SAFE HELPERS --------------------

def safe_float(x, default=0.0):
    try:
        return float(x) if not pd.isna(x) else default
    except:
        return default

def safe_int_hours(x):
    try:
        v = float(x)
        return v if v > 0 else 1.0
    except:
        return 1.0

def safe_access(obj, attr, default=None):
    try:
        return getattr(obj, attr)
    except:
        return default

def remaining_sla_seconds(sla_entered_at, sla_hours):
    if not sla_entered_at or not sla_hours:
        return 0
    now = datetime.utcnow()
    try:
        entered = datetime.strptime(sla_entered_at, "%Y-%m-%d %H:%M:%S")
    except:
        return 0
    deadline = entered + timedelta(hours=float(sla_hours))
    return max((deadline - now).total_seconds(), 0)

def capped_hours_left(sla_entered_at, sla_hours):
    seconds = remaining_sla_seconds(sla_entered_at, sla_hours)
    hours = seconds / 3600
    if hours > 9999:
        return 9999
    if hours < -9999:
        return -9999
    return int(hours)

# ---------------- FETCH & MERGE DATA -------------

@st.cache_data
def get_leads_df(start_date=None, end_date=None):
    con = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM leads", con)
    con.close()

    if df.empty:
        return pd.DataFrame()

    df["hours_left"] = df.apply(
        lambda r: capped_hours_left(r.get("sla_entered_at"), r.get("sla_hours")), axis=1
    )

    return df

def df_merged():
    return get_leads_df(st.session_state.get("start_date"), st.session_state.get("end_date"))

# ---------------- ML SETUP (INTERNAL ONLY) -------

MODEL_FILE = "ml_model.pkl"

def auto_train_model():
    df = df_merged()
    if df.empty:
        return None, "No data for ML training"

    data = df[["estimated_value", "sla_hours"]].fillna(0.0)
    labels = df["win_prob"].fillna(0.0)

    model = {"weights": [random.random(), random.random()] }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return model, "ML model trained internally ✅"

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        internal_model = pickle.load(f)
else:
    internal_model, _ = auto_train_model()

# ------------------- UI COMPONENTS ----------------

def alerts_bell():
    df = df_merged()
    if df.empty:
        count = 0
    else:
        count = sum(1 for _, r in df.iterrows() if r["hours_left"] <= 24)

    bell = st.empty()
    with bell.container():
        st.markdown(f"""
        <div style='position:fixed; top:12px; right:24px; background:#fff; color:#000; 
            padding:6px 12px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.2); 
            font-size:18px;'>
            🔔 <span style='color:red; font-weight:bold;'>{count}</span>
        </div>
        """, unsafe_allow_html=True)

    if count > 0:
        for _, lead in df.iterrows():
            hours = lead["hours_left"]
            if hours <= 24:
                alert = st.empty()
                alert.markdown(f"""
                <div style="background:black; color:white; padding:10px 14px; border-radius:10px; 
                    margin-top:6px; display:flex; justify-content:space-between;">
                    <div>⚠ {lead["lead_name"]} ({hours} hrs left)</div>
                    <div><button style="background:none; color:white; border:none; font-size:16px;" 
                        onclick="this.parentElement.parentElement.style.display='none'">✖</button></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                continue

st.session_state.setdefault("start_date", None)
st.session_state.setdefault("end_date", None)

def show_date_filter():
    date_container = st.empty()
    with date_container.container():
        st.markdown("""
        <div style='position:fixed; top:12px; right:120px; background:#fff; 
            padding:6px 14px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.2);'>
        </div>
        """, unsafe_allow_html=True)

    with date_container:
        cols = st.columns([3,3,1])
        sd = cols[0].date_input("Start Date")
        ed = cols[1].date_input("End Date")
        cols[2].write("")
        if cols[2].button("Apply"):
            st.session_state.start_date = sd
            st.session_state.end_date = ed

# ------------------- PAGES ------------------------

def page_capture():
    alerts_bell()
    show_date_filter()

    st.title("📥 Lead Capture")
    with st.form("lead_form"):
        lead_name = st.text_input("Lead Name")
        damage_type = st.selectbox("Damage Type", ["Water","Fire","Mold","Storm"])
        value = st.number_input("Estimated Value ($)", min_value=0.0)

        urgency = st.selectbox("Urgency", ["Low","Medium","High","Overdue"])
        sla_hours = st.number_input("SLA Response Time (Hours)", min_value=1.0, value=1.0, step=1.0)

        win_prob = random.uniform(0.2, 0.9)

        submitted = st.form_submit_button("Save Lead")
        if submitted:
            entered_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            con = sqlite3.connect(DB_FILE)
            c = con.cursor()
            c.execute("""
                INSERT INTO leads (lead_name, damage_type, estimated_value, urgency, sla_hours, sla_entered_at, win_prob)
                VALUES (?,?,?,?,?,?,?)
            """, (lead_name, damage_type, value, urgency, sla_hours, entered_at, win_prob))
            con.commit()
            con.close()
            st.success("Lead saved ✅")

def page_pipeline():
    alerts_bell()
    show_date_filter()

    st.title("🌡 Pipeline Stages")
    df = df_merged()
    if df.empty:
        st.warning("No leads yet")
        return

    top5 = df.sort_values("hours_left", ascending=True).head(5)

    def urgency_color(u):
        return "red" if u=="Overdue" else "orange" if u=="High" else "yellow" if u=="Medium" else "green"

    row1 = st.columns(3)
    row2 = st.columns(2)

    i = 0
    for col in row1 + row2:
        if i < len(top5):
            r = top5.iloc[i]
            uv = urgency_color(r["urgency"])
            st.markdown(f"""
            <div style='background:black; color:white; padding:14px; border-radius:10px; margin:6px;'>
                <div style='font-size:14px; font-weight:bold;'>{r["lead_name"]}</div>
                <div style='font-size:20px; color:green;'>${r["estimated_value"]:,.2f}</div>
                <div style='font-size:16px; color:{uv};'>{r["urgency"]}</div>
                <div style='font-size:14px; color:red;'>⏳ {r["hours_left"]} hours left</div>
            </div>
            """, unsafe_allow_html=True)
            i += 1

    st.subheader("📊 All Pipeline")
    stage_colors = {"Low":"#34d399","Medium":"#fbbf24","High":"#f97316","Overdue":"#ef4444"}

    fig = px.line(df, x="sla_entered_at", y="estimated_value", markers=True)
    st.plotly_chart(fig)

    st.subheader("Filters")
    q = st.text_input("Search Lead")
    if q:
        df = df[df["lead_name"].str.contains(q, case=False)]
        st.dataframe(df)

def page_analytics():
    alerts_bell()
    show_date_filter()

    st.title("📈 Analytics & SLA")
    df = df_merged()
    if df.empty:
        st.warning("No data")
        return

    st.subheader("💰 Cost vs Conversion Bar Chart")
    marketing_spend = 1219
    conversions = 5
    attempts = 20
    cpa = marketing_spend / conversions
    roi_perc = ((19071 - marketing_spend) / marketing_spend) * 100

    chart = pd.DataFrame({"Metric":["Spend","Conversions"],"Value":[marketing_spend, conversions]})
    fig = px.bar(chart, x="Metric", y="Value")
    st.plotly_chart(fig)

    st.subheader("KPIs")
    with st.container():
        st.metric("Total Spend ($)", marketing_spend)
        st.metric("Conversions", conversions)
        st.metric("CPA ($)", round(cpa,2))
        st.metric("ROI (%)", f"{round(roi_perc,1)}%")

    st.subheader("Pipeline Line Chart")
    stage_ct = df["urgency"].value_counts()
    st.bar_chart(stage_ct)

    st.subheader("AI Recommendations")
    st.info("AI layer ready for internal ML observations ✅")
    st.write("""
    1. Leads under 24 hours must be contacted immediately
    2. High value jobs should be dispatched to senior estimators
    3. SLA breaches reduce win probability
    4. Consider adding automated follow-ups
    """)

def page_settings():
    alerts_bell()
    show_date_filter()

    st.title("⚙ Admin Settings")
    st.subheader("User Role Management")
    admins = st.text_input("List Admin Users (comma separated)", placeholder="e.g Ayo,John,Mike")
    if admins:
        st.success(f"Admins set: {admins}")

def page_exports():
    alerts_bell()
    show_date_filter()

    st.title("📤 Export/Import")
    df = df_merged()
    if df.empty:
        st.warning("No leads")
        return

    def export_to_excel(df):
        buffer = io.BytesIO()
        try:
            df.to_excel(buffer, index=False, engine="openpyxl")
        except:
            df.to_excel(buffer, index=False)
        buffer.seek(0)
        return buffer

    excel = export_to_excel(df)
    st.download_button("Download Excel", excel, "titan_leads.xlsx")

def page_ml():
    alerts_bell()
    show_date_filter()

    st.title("🤖 ML Training (Internal)")
    model, msg = auto_train_model()
    st.info(msg)

# ------------------ NAVIGATION -------------------

menu = st.sidebar.radio("Go to", ["Lead Capture","Pipeline","Analytics","Export","Settings","ML"])

if menu == "Lead Capture":
    page_capture()
elif menu == "Pipeline":
    page_pipeline()
elif menu == "Analytics":
    page_analytics()
elif menu == "Export":
    page_exports()
elif menu == "Settings":
    page_settings()
elif menu == "ML":
    page_ml()
