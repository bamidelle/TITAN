import os
from datetime import datetime, timedelta, date
import io, base64, traceback
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------
# CONFIG
# ----------------------
APP_TITLE = "TITAN — Backend Admin"
DB_FILE = "titan_backend.db"
MODEL_FILE = "titan_model.joblib"
PIPELINE_STAGES = [
    "New", "Contacted", "Inspection Scheduled", "Inspection Completed",
    "Estimate Sent", "Qualified", "Won", "Lost"
]
DEFAULT_SLA_HOURS = 72

# ----------------------
# DATABASE
# ----------------------
DB_PATH = os.path.join(os.getcwd(), DB_FILE)
ENGINE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(ENGINE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    full_name = Column(String, default="")
    role = Column(String, default="Admin")
    created_at = Column(DateTime, default=datetime.utcnow)

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String, default="Other")
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)
    assigned_to = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    estimated_value = Column(Float, default=0.0)
    stage = Column(String, default="New")
    sla_hours = Column(Integer, default=DEFAULT_SLA_HOURS)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)
    ad_cost = Column(Float, default=0.0)
    converted = Column(Boolean, default=False)
    score = Column(Float, nullable=True)

class LeadHistory(Base):
    __tablename__ = "lead_history"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, nullable=False)
    changed_by = Column(String, nullable=True)
    field = Column(String, nullable=True)
    old_value = Column(String, nullable=True)
    new_value = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_session():
    return SessionLocal()

def leads_to_df(start_date=None, end_date=None):
    s = get_session()
    try:
        rows = s.query(Lead).order_by(Lead.created_at.desc()).all()
        data = []
        for r in rows:
            data.append({
                "lead_id": r.lead_id,
                "created_at": r.created_at,
                "source": r.source,
                "contact_name": r.contact_name,
                "property_address": r.property_address,
                "damage_type": r.damage_type,
                "assigned_to": r.assigned_to,
                "notes": r.notes,
                "estimated_value": r.estimated_value,
                "stage": r.stage,
                "sla_hours": r.sla_hours,
                "sla_entered_at": r.sla_entered_at,
                "ad_cost": r.ad_cost,
                "converted": r.converted,
                "score": r.score
            })
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame(columns=data)

        if start_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            df = df[df["created_at"] >= start_dt]
        if end_date:
            end_dt = datetime.combine(end_date, datetime.max.time())
            df = df[df["created_at"] <= end_dt]
        return df.reset_index(drop=True)
    except Exception:
        s.rollback()
        st.error("Database read error")
        return pd.DataFrame()
    finally:
        s.close()

# ----------------------
# ROUTER SAFETY INIT
# ----------------------
if "filter_src" not in st.session_state:
    st.session_state.filter_src = "All"
if "filter_stage" not in st.session_state:
    st.session_state.filter_stage = "All"
if "search_q" not in st.session_state:
    st.session_state.search_q = ""

# ----------------------
# UI SETUP
# ----------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown("""
<style>
body, .stApp { background: white; color: black; font-family: sans-serif; }
.header { font-weight:800; font-size:20px; margin-bottom:6px; }
.topbar-right { text-align:right; font-size:14px; }
.bell-btn { background:black; color:white; border-radius:10px; padding:10px; }
.kpi-row-gap { height:16px; }
.priority-card { background:black; color:white; border-radius:12px; padding:12px; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

# ----------------------
# SIDEBAR
# ----------------------
with st.sidebar:
    st.header("TITAN Backend (Admin)")
    page = st.radio(
        "Navigate", 
        ["Pipeline Board","Lead Capture","Analytics","CPA & ROI","ML (internal)","Settings","Exports"], 
        index=0
    )
    st.markdown("---")
    st.markdown("Date range")
    quick = st.selectbox("Range", ["Today","Last 7 days","Last 30 days","90 days","All","Custom"], index=4)
    
    if quick == "Custom":
        sd, ed = st.date_input("Start / End", [date.today() - timedelta(days=29), date.today()])
        st.session_state.start_date = sd
        st.session_state.end_date = ed
    elif quick == "Today":
        st.session_state.start_date = date.today()
        st.session_state.end_date = date.today()
    elif quick == "All":
        st.session_state.start_date = None
        st.session_state.end_date = None

# ----------------------
# LOAD LEADS SAFELY
# ----------------------
start_dt = st.session_state.get("start_date", None)
end_dt = st.session_state.get("end_date", None)
leads_df = leads_to_df(start_dt, end_dt)

# ----------------------
# BELL & DATE TOP RIGHT
# ----------------------
def render_topbar():
    overdue_count = 0
    for _, r in leads_df.iterrows():
        try:
            sla_entered = r.get("sla_entered_at") or r.get("created_at")
            sla_h = r.get("sla_hours") or DEFAULT_SLA_HOURS
            if isinstance(sla_entered, str):
                sla_entered = datetime.fromisoformat(sla_entered)
            deadline = sla_entered + timedelta(hours=int(sla_h))
            if deadline < datetime.utcnow() and r.get("stage") not in ("Won","Lost"):
                overdue_count += 1
        except Exception:
            continue

    __left, __right = st.columns([8,2])  # guaranteed to be defined
    with __right:
        today_label = datetime.utcnow().strftime("%Y-%m-%d")
        st.markdown(f"""
            <div class='topbar-right'>
              {today_label} &nbsp;&nbsp;
              <button class='bell-btn'>🔔 <span class='badge'>{overdue_count}</span></button>
            </div>
        """, unsafe_allow_html=True)

# ----------------------
# PAGE DEFINITIONS
# ----------------------
def page_lead_capture():
    render_topbar()
    st.markdown("<div class='header'>📇 Lead Capture</div>", unsafe_allow_html=True)
    with st.form("lead_capture_form"):
        lead_id = st.text_input("Lead ID")
        source = st.selectbox("Lead Source", ["Google Ads","Organic Search","Referral","Other"])
        estimated_value = st.number_input("Estimated value", min_value=0.0)
        ad_cost = st.number_input("Acquisition Cost", min_value=0.0)
        sla_hours = st.number_input(
            "SLA Response Hours", 
            min_value=1, 
            value=DEFAULT_SLA_HOURS, 
            placeholder="SLA Response time must be greater than 0 hours."
        )
        stage = st.selectbox("Stage", PIPELINE_STAGES)
        notes = st.text_area("Notes")
        if st.form_submit_button("Save"):
            if not lead_id:
                st.error("Lead ID required")
                return
            try:
                upsert_lead_record({
                    "lead_id": lead_id.strip(),
                    "source": source,
                    "estimated_value": float(estimated_value or 0.0),
                    "ad_cost": float(ad_cost or 0.0),
                    "sla_hours": int(sla_hours),
                    "sla_entered_at": datetime.utcnow(),
                    "stage": stage,
                    "notes": notes
                }, actor="admin")
                st.success("Lead saved")
                st.experimental_rerun()
            except Exception as e:
                st.error(str(e))
                st.write(traceback.format_exc())

def page_pipeline_board():
    render_topbar()
    st.markdown("<div class='header'>Pipeline Board</div>", unsafe_allow_html=True)
    df = leads_df.copy()

    if df.empty:
        st.info("No leads yet.")
        return

    total = len(df)
    qualified = df[df["stage"]=="Qualified"].shape[0]
    won = df[df["stage"]=="Won"].shape[0]
    lost = df[df["stage"]=="Lost"].shape[0]
    closed = won + lost
    conversion_rate = (won / closed * 100) if closed else 0.0

    KPI_ITEMS = [
        ("Total Leads", str(total)),
        ("Qualified", str(qualified)),
        ("Won", str(won)),
        ("Lost", str(lost)),
        ("Conversion Rate", f"{conversion_rate:.1f}%")
    ]

    r1 = st.columns(4)
    for col, (title, val) in zip(r1, KPI_ITEMS[:4]):
        col.markdown(f"<div class='kpi-card'><div class='kpi-title'>{title}</div><div class='kpi-number'>{val}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='kpi-row-gap'></div>", unsafe_allow_html=True)

    r2 = st.columns(1)
    r2[0].markdown(f"<div class='kpi-card'><div class='kpi-title'>Conversion Rate</div><div class='kpi-number'>{conversion_rate:.1f}%</div></div>", unsafe_allow_html=True)

    # PRIORITY SAFE (same scoring)
    st.markdown("---")
    st.subheader("TOP 5 PRIORITY LEADS")
    df["priority_score"] = df.apply(lambda r: compute_priority_for_row(r), axis=1)
    pr_df = df.sort_values("priority_score", ascending=False).head(5)

    for _, r in pr_df.iterrows():
        st.markdown(f"""
        <div class='priority-card'>
            #{r['lead_id']} — Priority Score: {r['priority_score']:.2f}
        </div>
        """, unsafe_allow_html=True)

def page_analytics():
    render_topbar()
    st.markdown("<div class='header'>📈 Analytics & SLA</div>", unsafe_allow_html=True)
    df = leads_df.copy()
    if df.empty:
        st.info("No data")
        return
    
    st.subheader("Cost vs Conversion")
    agg = df.groupby("source").agg(
        total_spend=("ad_cost","sum"),
        conversions=("stage", lambda s: (s=="Won").sum())
    ).reset_index()

    if not agg.empty:
        fig = px.bar(agg, x="source", y=["total_spend","conversions"], barmode="group")
        st.plotly_chart(fig, use_container_width=True)

def page_cpa_roi():
    render_topbar()
    df = leads_df.copy()
    if df.empty:
        st.info("No data"); return
    spend = df["ad_cost"].sum()
    won = df[df["stage"]=="Won"]
    conv = len(won)
    cpa = spend / conv if conv else 0
    rev = won["estimated_value"].sum()
    roi = rev - spend
    roi_pct = (roi/spend*100) if spend else 0

    a,b,c = st.columns(3)
    a.metric("Spend", f"${spend:,.0f}")
    b.metric("CPA", f"${cpa:,.2f}")
    c.metric("ROI", f"${roi:,.0f} ({roi_pct:.1f}%)")

def page_ml_internal():
    render_topbar()
    st.markdown("<div class='header'>🧠 Internal ML</div>", unsafe_allow_html=True)
    if st.button("Train model"):
        acc, msg = train_internal_model()
        st.success(f"Trained ({acc})")
    model, cols = load_internal_model()
    if model:
        st.success("Model available")
        if st.button("Score & persist"):
            df = leads_to_df()
            scored = score_dataframe(df.copy(), model, cols)
            s = get_session()
            try:
                for _, r in scored.iterrows():
                    lead = s.query(Lead).filter(Lead.lead_id == r["lead_id"]).first()
                    if lead:
                        lead.score = float(r["score"])
                        s.add(lead)
                s.commit()
                st.success("Scores saved")
            except Exception as e:
                s.rollback(); st.error(str(e))
            finally:
                s.close()

def page_settings():
    render_topbar()
    st.subheader("Add user role")
    with st.form("userform"):
        u = st.text_input("Username")
        f = st.text_input("Full name")
        r = st.selectbox("Role", ["Admin","Estimator","Adjuster","Tech","Viewer"])
        if st.form_submit_button("Save user"):
            if u:
                add_user(u.strip(), f.strip(), r)
                st.success("User saved")
                st.experimental_rerun()

def page_exports():
    render_topbar()
    st.subheader("Export Leads")
    df = leads_to_df(None,None)
    if not df.empty:
        buff=io.BytesIO()
        df.to_excel(buff,index=False,engine="openpyxl")
        buff.seek(0)
        b64=base64.b64encode(buff.read()).decode()
        st.download_button("Download XLSX", buff, "leads.xlsx")

# ----------------------
# GUARANTEED FALLBACK ROUTER
# ----------------------
if page == "Pipeline Board":
    page_pipeline_board()
elif page == "Lead Capture":
    page_lead_capture()
elif page == "Analytics":
    page_analytics()
elif page == "CPA & ROI":
    page_cpa_roi()
elif page == "ML (internal)":
    page_ml_internal()
elif page == "Settings":
    page_settings()
elif page == "Exports":
    page_exports()
