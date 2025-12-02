# project_x_restoration_app.py
# Single-file Streamlit app implementing pipeline, analytics, SLA, CPA/ROI, internal ML (silent), alerts, export.
import os
from datetime import datetime, timedelta, date
import time
import random
import traceback

import streamlit as st
import pandas as pd

# Optional libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

try:
    import joblib
except Exception:
    joblib = None

# scikit-learn detection & safe imports
SKL_AVAILABLE = False
try:
    import sklearn
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKL_AVAILABLE = True
except Exception:
    SKL_AVAILABLE = False

# SQLAlchemy setup (expire_on_commit=False to reduce DetachedInstanceError risk)
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, func
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

BASE_DIR = os.getcwd()
DB_FILE = os.path.join(BASE_DIR, "project_x_restoration.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
Base = declarative_base()

# --------------------------
# ORM (Lead + Estimate)
# --------------------------
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    source = Column(String, default="Other")
    source_details = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)
    assigned_to = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    estimated_value = Column(Float, nullable=True)
    status = Column(String, default="New")
    created_at = Column(DateTime, default=datetime.utcnow)
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime, nullable=True)
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    estimate_submitted = Column(Boolean, default=False)
    estimate_submitted_at = Column(DateTime, nullable=True)
    awarded_date = Column(DateTime, nullable=True)
    awarded_comment = Column(Text, nullable=True)
    awarded_invoice = Column(String, nullable=True)
    lost_date = Column(DateTime, nullable=True)
    lost_comment = Column(Text, nullable=True)
    qualified = Column(Boolean, default=False)
    cost_to_acquire = Column(Float, default=0.0)
    predicted_prob = Column(Float, nullable=True)

Base.metadata.create_all(bind=engine)

# --------------------------
# Helpers
# --------------------------
def get_session():
    return SessionLocal()

def save_uploaded_file(uploaded_file, prefix="file"):
    if uploaded_file is None:
        return None
    fname = f"{prefix}_{int(datetime.utcnow().timestamp())}_{uploaded_file.name}"
    path = os.path.join(UPLOAD_FOLDER, fname)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def leads_to_df(session, start_date=None, end_date=None):
    rows = session.query(Lead).order_by(Lead.created_at.desc()).all()
    data = []
    for r in rows:
        data.append({
            "id": r.id,
            "source": r.source,
            "source_details": r.source_details,
            "contact_name": r.contact_name,
            "contact_phone": r.contact_phone,
            "contact_email": r.contact_email,
            "property_address": r.property_address,
            "damage_type": r.damage_type,
            "assigned_to": r.assigned_to,
            "notes": r.notes,
            "estimated_value": float(r.estimated_value or 0.0),
            "status": r.status,
            "created_at": r.created_at,
            "sla_hours": r.sla_hours,
            "sla_entered_at": r.sla_entered_at or r.created_at,
            "contacted": bool(r.contacted),
            "inspection_scheduled": bool(r.inspection_scheduled),
            "inspection_scheduled_at": r.inspection_scheduled_at,
            "inspection_completed": bool(r.inspection_completed),
            "estimate_submitted": bool(r.estimate_submitted),
            "awarded_date": r.awarded_date,
            "awarded_invoice": r.awarded_invoice,
            "lost_date": r.lost_date,
            "qualified": bool(r.qualified),
            "cost_to_acquire": float(r.cost_to_acquire or 0.0),
            "predicted_prob": float(r.predicted_prob) if r.predicted_prob is not None else None
        })
    df = pd.DataFrame(data)
    if df.empty:
        return df
    # optional date filter
    if start_date is not None and end_date is not None:
        sdt = datetime.combine(start_date, datetime.min.time())
        edt = datetime.combine(end_date, datetime.max.time())
        df = df[(df["created_at"] >= sdt) & (df["created_at"] <= edt)].copy()
    return df

def calculate_remaining_sla(sla_entered_at, sla_hours):
    try:
        if sla_entered_at is None:
            sla_entered_at = datetime.utcnow()
        if isinstance(sla_entered_at, str):
            sla_entered_at = datetime.fromisoformat(sla_entered_at)
        deadline = sla_entered_at + timedelta(hours=int(sla_hours or 24))
        remain = deadline - datetime.utcnow()
        return max(remain.total_seconds(), 0.0), (remain.total_seconds() <= 0)
    except Exception:
        return float("inf"), False

# Priority scoring function
def compute_priority_for_lead_row(lead_row, weights):
    try:
        val = float(lead_row.get("estimated_value") or 0.0)
        baseline = float(weights.get("value_baseline", 5000.0))
        value_score = min(1.0, val / max(1.0, baseline))
    except Exception:
        value_score = 0.0
    try:
        sla_entered = lead_row.get("sla_entered_at") or lead_row.get("created_at")
        if isinstance(sla_entered, str):
            sla_entered = datetime.fromisoformat(sla_entered)
        deadline = sla_entered + timedelta(hours=int(lead_row.get("sla_hours") or 24))
        time_left_h = max((deadline - datetime.utcnow()).total_seconds() / 3600.0, 0.0)
    except Exception:
        time_left_h = 9999.0
    sla_score = max(0.0, (72.0 - min(time_left_h, 72.0)) / 72.0)
    contacted_flag = 0.0 if bool(lead_row.get("contacted")) else 1.0
    inspection_flag = 0.0 if bool(lead_row.get("inspection_scheduled")) else 1.0
    estimate_flag = 0.0 if bool(lead_row.get("estimate_submitted")) else 1.0
    urgency_component = (contacted_flag * weights.get("contacted_w", 0.6) +
                        inspection_flag * weights.get("inspection_w", 0.5) +
                        estimate_flag * weights.get("estimate_w", 0.5))
    total_weight = (weights.get("value_weight", 0.5) +
                    weights.get("sla_weight", 0.35) +
                    weights.get("urgency_weight", 0.15))
    if total_weight <= 0:
        total_weight = 1.0
    score = (value_score * weights.get("value_weight", 0.5) +
             sla_score * weights.get("sla_weight", 0.35) +
             urgency_component * weights.get("urgency_weight", 0.15)) / total_weight
    score = max(0.0, min(score, 1.0))
    return score

# Safe OneHotEncoder kwargs depending on sklearn version
def onehot_sparse_kw():
    # prefer sparse_output if available; else fallback to sparse=False
    try:
        return {"sparse_output": False}
    except Exception:
        return {"sparse": False}

# --------------------------
# ML: internal auto-train (silent)
# --------------------------
MODEL_PATH = os.path.join(BASE_DIR, "lead_model_internal.joblib")

def build_pipeline():
    if not SKL_AVAILABLE:
        return None
    numeric_cols = ["estimated_value", "sla_hours", "cost_to_acquire"]
    categorical_cols = ["damage_type", "source", "assigned_to"]
    ohe_kwargs = {}
    # pick compatibility
    try:
        ohe_kwargs["sparse_output"] = False
        _ = OneHotEncoder(**ohe_kwargs)
    except Exception:
        ohe_kwargs = {"sparse": False}
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", **ohe_kwargs), categorical_cols)
    ], remainder="drop")
    pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=80, random_state=42))])
    return pipe, numeric_cols, categorical_cols

def auto_train_model_if_enough(session):
    if not SKL_AVAILABLE or joblib is None:
        return None
    try:
        df = leads_to_df(session)
        if df.empty:
            return None
        # label: awarded -> 1 others -> 0 (only when enough labels)
        df_l = df[df["status"].str.lower().isin(["awarded", "lost", "award", "won"])].copy()
        if len(df_l) < 12:
            return None
        pipe_tuple = build_pipeline()
        if pipe_tuple is None:
            return None
        pipe, num_cols, cat_cols = pipe_tuple
        X = df_l[num_cols + cat_cols].fillna(0)
        X[cat_cols] = X[cat_cols].astype(str).fillna("unknown")
        y = (df_l["status"].str.lower() == "awarded").astype(int)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
        except Exception:
            X_train, X_test, y_train, y_test = X, X, y, y
        pipe.fit(X_train, y_train)
        try:
            joblib.dump(pipe, MODEL_PATH)
        except Exception:
            pass
        # write back probabilities for labeled set
        try:
            probs = pipe.predict_proba(X)[:, 1]
            for lid, p in zip(df_l["id"].tolist(), probs.tolist()):
                lead = session.query(Lead).filter(Lead.id == int(lid)).first()
                if lead:
                    lead.predicted_prob = float(p)
                    session.add(lead)
            session.commit()
        except Exception:
            session.rollback()
        return pipe
    except Exception:
        return None

# Background ML thread (silent)
def start_ml_background():
    if not SKL_AVAILABLE or joblib is None:
        return
    def loop():
        while True:
            s = get_session()
            try:
                auto_train_model_if_enough(s)
            except Exception:
                pass
            finally:
                s.close()
            time.sleep(60 * 30)
    import threading
    threading.Thread(target=loop, daemon=True).start()

start_ml_background()

# --------------------------
# UI CSS and page config
# --------------------------
st.set_page_config(page_title="Project X — Pipeline", layout="wide", initial_sidebar_state="expanded")
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;600;700&display=swap');
:root{
  --bg: #ffffff;
  --text: #0b1220;
  --muted: #6b7280;
  --primary-blue: #2563eb;
  --money-green: #22c55e;
  --accent-orange: #f97316;
  --danger: #ef4444;
}
body, .stApp { background: var(--bg); color: var(--text); font-family: 'Comfortaa', sans-serif; }
.header { font-size:20px; font-weight:700; color:var(--text); padding:8px 0; }
.metric-card { border-radius: 12px; padding:12px; margin:6px; color:#fff; background:#000; box-shadow: 0 6px 16px rgba(16,24,40,0.06); }
.kpi-title { color: #ffffff; font-weight:700; font-size:13px; margin-bottom:6px; }
.kpi-value { font-weight:900; font-size:24px; }
.kpi-note { font-size:12px; color:rgba(255,255,255,0.9); margin-top:6px; }
.progress-wrap { width:100%; background:#111; height:8px; border-radius:8px; margin-top:8px; overflow:hidden; }
.progress-fill { height:100%; border-radius:8px; transition: width .35s ease; }
.priority-card { background:#000; color:#fff; padding:12px; border-radius:12px; margin-bottom:10px; }
.small-muted { color: #6b7280; font-size:12px; }
.bell { display:inline-block; padding:8px 12px; background:#000; border-radius:10px; color:#fff; cursor:pointer; }
.badge { background:var(--danger); color:#fff; border-radius:999px; padding:2px 8px; font-size:12px; margin-left:8px; }
"""
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

# --------------------------
# Sidebar controls
# --------------------------
with st.sidebar:
    st.header("Controls")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "CPA & ROI", "ML (Internal)", "Exports"], index=1)
    st.markdown("---")
    if "weights" not in st.session_state:
        st.session_state.weights = {
            "value_weight": 0.5, "sla_weight": 0.35, "urgency_weight": 0.15,
            "contacted_w": 0.6, "inspection_w": 0.5, "estimate_w": 0.5, "value_baseline": 5000.0
        }
    st.markdown("### Priority weights (admin)")
    st.session_state.weights["value_weight"] = st.slider("Value weight", 0.0, 1.0, float(st.session_state.weights["value_weight"]), step=0.05)
    st.session_state.weights["sla_weight"] = st.slider("SLA weight", 0.0, 1.0, float(st.session_state.weights["sla_weight"]), step=0.05)
    st.session_state.weights["urgency_weight"] = st.slider("Urgency weight", 0.0, 1.0, float(st.session_state.weights["urgency_weight"]), step=0.05)
    st.markdown("---")
    st.write("ML: internal only. Runs when enough labels exist.")
    if st.button("Force internal retrain"):
        s = get_session()
        try:
            res = auto_train_model_if_enough(s)
            if res:
                st.success("Internal retrain finished.")
            else:
                st.info("Retrain did not run (insufficient labeled data or error).")
        except Exception as e:
            st.error(f"Retrain error: {e}")
        finally:
            s.close()
    st.markdown("---")
    if st.button("Add Demo Lead"):
        s = get_session()
        try:
            demo = Lead(
                source="Google Ads", source_details="demo",
                contact_name="Demo User", contact_phone="+1555000", contact_email="demo@example.com",
                property_address="100 Demo St", damage_type="water",
                assigned_to="Alex", notes="Demo", estimated_value=4200.0,
                sla_hours=24, sla_entered_at=datetime.utcnow(), qualified=True, cost_to_acquire=50.0
            )
            s.add(demo); s.commit()
            st.success("Demo lead added.")
        except Exception as e:
            s.rollback(); st.error(f"Failed to add demo: {e}")
        finally:
            s.close()

# Alerts (overdue) bell in header
st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'>", unsafe_allow_html=True)
st.markdown("<div class='header'>Project X — Sales & Conversion Tracker</div>", unsafe_allow_html=True)

# compute overdue count
s = get_session()
try:
    all_leads = s.query(Lead).all()
    overdue_count = 0
    overdue_list = []
    for L in all_leads:
        sla_sec, overdue_flag = calculate_remaining_sla(L.sla_entered_at or L.created_at, L.sla_hours)
        if overdue_flag and L.status.lower() not in ("awarded", "lost"):
            overdue_count += 1
            overdue_list.append({"id": L.id, "name": L.contact_name or "No name", "phone": L.contact_phone or "", "created_at": L.created_at})
finally:
    s.close()

st.markdown(f"<div style='display:flex;align-items:center;gap:8px;'><div class='bell'>🔔 Alerts <span class='badge'>{overdue_count}</span></div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# small interactive overdue dropdown
if "show_overdue" not in st.session_state:
    st.session_state.show_overdue = False

if st.button("View overdue leads"):
    st.session_state.show_overdue = not st.session_state.show_overdue

if st.session_state.show_overdue:
    st.markdown("<div style='background:#111;color:#fff;padding:12px;border-radius:10px;max-width:420px;'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;justify-content:space-between;align-items:center;'><b>Overdue leads</b> <button onclick=\"(function(){document.querySelector('#__st_container').style.display='none'})()\">X</button></div>", unsafe_allow_html=True)
    if overdue_list:
        for od in overdue_list:
            st.markdown(f"- #{od['id']} — {od['name']} — {od['phone']}")
    else:
        st.markdown("No overdue leads right now ✅")
    st.markdown("</div>", unsafe_allow_html=True)

# tiny toast system using session_state
if "toasts" not in st.session_state:
    st.session_state.toasts = []

def push_toast(msg, kind="info"):
    st.session_state.toasts.insert(0, {"msg": msg, "kind": kind, "ts": time.time()})
    st.session_state.toasts = st.session_state.toasts[:6]

if st.session_state.toasts:
    for t in st.session_state.toasts:
        if t["kind"] == "success":
            st.success(t["msg"])
        elif t["kind"] == "warning":
            st.warning(t["msg"])
        elif t["kind"] == "error":
            st.error(t["msg"])
        else:
            st.info(t["msg"])

# --------------------------
# Page: Leads / Capture
# --------------------------
if page == "Leads / Capture":
    st.header("📇 Lead Capture")
    st.markdown("<em>All fields persist. Cost to acquire lead is saved for CPA/ROI.</em>", unsafe_allow_html=True)
    with st.form("lead_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            source = st.selectbox("Lead Source", ["Google Ads", "Website Form", "Referral", "Facebook", "Instagram", "TikTok", "LinkedIn", "Twitter/X", "YouTube", "Yelp", "Nextdoor", "Phone", "Insurance", "Other"])
            source_details = st.text_input("Source details (UTM / notes)")
            contact_name = st.text_input("Contact name")
            contact_phone = st.text_input("Contact phone")
            contact_email = st.text_input("Contact email")
        with c2:
            property_address = st.text_input("Property address")
            damage_type = st.selectbox("Damage type", ["water", "fire", "mold", "contents", "reconstruction", "other"])
            assigned_to = st.text_input("Assigned to")
            qualified_choice = st.selectbox("Is the Lead Qualified?", ["No", "Yes"])
            sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=24)
        notes = st.text_area("Notes")
        estimated_value = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=100.0)
        cost_to_acquire = st.number_input("Cost to acquire lead ($)", min_value=0.0, value=0.0, step=1.0)
        submitted = st.form_submit_button("Create Lead")
        if submitted:
            s = get_session()
            try:
                new = Lead(
                    source=source, source_details=source_details,
                    contact_name=contact_name, contact_phone=contact_phone, contact_email=contact_email,
                    property_address=property_address, damage_type=damage_type, assigned_to=assigned_to,
                    notes=notes, estimated_value=float(estimated_value or 0.0), sla_hours=int(sla_hours),
                    sla_entered_at=datetime.utcnow(), qualified=(qualified_choice == "Yes"),
                    cost_to_acquire=float(cost_to_acquire or 0.0), created_at=datetime.utcnow(), status="New"
                )
                s.add(new); s.commit()
                push_toast(f"Lead created (ID: {new.id})", "success")
            except Exception as e:
                s.rollback(); st.error(f"Failed to create lead: {e}")
            finally:
                s.close()

    st.markdown("---")
    s = get_session()
    df = leads_to_df(s)
    s.close()
    if df.empty:
        st.info("No leads yet. Create one above.")
    else:
        qcol1, qcol2, qcol3 = st.columns([2,2,4])
        with qcol1:
            q_status = st.selectbox("Filter status", ["All"] + sorted(df["status"].dropna().unique().tolist()))
        with qcol2:
            q_source = st.selectbox("Filter source", ["All"] + sorted(df["source"].dropna().unique().tolist()))
        with qcol3:
            q_text = st.text_input("Quick search (name, phone, email, address)")
        df_view = df.copy()
        if q_status and q_status != "All":
            df_view = df_view[df_view["status"] == q_status]
        if q_source and q_source != "All":
            df_view = df_view[df_view["source"] == q_source]
        if q_text:
            q2 = q_text.lower()
            df_view = df_view[
                df_view["contact_name"].fillna("").str.lower().str.contains(q2) |
                df_view["contact_phone"].fillna("").str.lower().str.contains(q2) |
                df_view["contact_email"].fillna("").str.lower().str.contains(q2) |
                df_view["property_address"].fillna("").str.lower().str.contains(q2)
            ]
        st.dataframe(df_view.sort_values("created_at", ascending=False).head(200))
# -------------------------------------------
# SIDEBAR NAVIGATION (REPLACES DASHBOARD, RETAINS PIPELINE)
# -------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Pipeline", "Lead Capture", "Analytics & SLA", "CPA", "ROI", "Exports", "Settings", "ML", "Reports"]
)

# 2 ROWS KPI SPACING FIX + PRIORITY CARD STYLES ALREADY IN GLOBAL CSS ✅

# -------------------------------------------
# PAGE RENDERING LOGIC
# -------------------------------------------
if page == "Pipeline":
    page_pipeline()
elif page == "Lead Capture":
    page_capture()
elif page == "Analytics & SLA":
    page_analytics()
elif page == "CPA":
    page_cpa()
elif page == "ROI":
    page_roi()
elif page == "Exports":
    page_exports()
elif page == "Settings":
    page_settings()
elif page == "ML":
    page_ml()
elif page == "Reports":
    page_reports()

# --------------------------
# Page: Pipeline Board
# --------------------------
elif page == "Pipeline Board":
    st.header("TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR")
    st.markdown("<em>High-level pipeline performance at a glance. Use date selectors top-right to filter.</em>", unsafe_allow_html=True)

    # top-right date selection (Google Ads style)
    colm, colr = st.columns([3,1])
    with colr:
        quick_range = st.selectbox("Date range", ["Today", "Yesterday", "Last 7 days", "Last 30 days", "All", "Custom"], index=0)
        if quick_range == "Today":
            sdt = date.today(); edt = date.today()
        elif quick_range == "Yesterday":
            sdt = date.today() - timedelta(days=1); edt = sdt
        elif quick_range == "Last 7 days":
            sdt = date.today() - timedelta(days=7); edt = date.today()
        elif quick_range == "Last 30 days":
            sdt = date.today() - timedelta(days=30); edt = date.today()
        elif quick_range == "All":
            ssession = get_session()
            td = leads_to_df(ssession)
            ssession.close()
            if td.empty:
                sdt = date.today(); edt = date.today()
            else:
                sdt = td["created_at"].min().date(); edt = td["created_at"].max().date()
        else:
            custom = st.date_input("Start/End", [date.today(), date.today()])
            if isinstance(custom, (list, tuple)) and len(custom) == 2:
                sdt, edt = custom[0], custom[1]
            else:
                sdt = date.today(); edt = date.today()

    s = get_session()
    df = leads_to_df(s, sdt, edt)
    s.close()

    total_leads = len(df)
    qualified_leads = int(df[df["qualified"] == True].shape[0]) if not df.empty else 0
    sla_success_count = int(df[df["contacted"] == True].shape[0]) if not df.empty else 0
    sla_success_pct = (sla_success_count / total_leads * 100) if total_leads else 0.0
    qualification_pct = (qualified_leads / total_leads * 100) if total_leads else 0.0
    awarded_count = int(df[df["status"].str.lower() == "awarded"].shape[0]) if not df.empty else 0
    lost_count = int(df[df["status"].str.lower() == "lost"].shape[0]) if not df.empty else 0
    closed = awarded_count + lost_count
    conversion_rate = (awarded_count / closed * 100) if closed else 0.0
    inspection_scheduled_count = int(df[df["inspection_scheduled"] == True].shape[0]) if not df.empty else 0
    inspection_pct = (inspection_scheduled_count / qualified_leads * 100) if qualified_leads else 0.0
    estimate_sent_count = int(df[df["estimate_submitted"] == True].shape[0]) if not df.empty else 0
    pipeline_job_value = float(df["estimated_value"].sum()) if not df.empty else 0.0
    active_leads = total_leads - (awarded_count + lost_count) if not df.empty else 0

    KPI_ITEMS = [
        ("#2563eb", "Active Leads", f"{active_leads}", "Leads currently in pipeline"),
        ("#0ea5a4", "SLA Success", f"{sla_success_pct:.1f}%", "Leads contacted within SLA"),
        ("#a855f7", "Qualification Rate", f"{qualification_pct:.1f}%", "Leads marked qualified"),
        ("#f97316", "Conversion Rate", f"{conversion_rate:.1f}%", "Won / Closed"),
        ("#ef4444", "Inspections Booked", f"{inspection_pct:.1f}%", "Qualified → Scheduled"),
        ("#6d28d9", "Estimates Sent", f"{estimate_sent_count}", "Estimates submitted"),
        ("#22c55e", "Pipeline Job Value", f"${pipeline_job_value:,.0f}", "Total pipeline job value")
    ]

    # Render two rows: first 4, then 3
    row1 = st.columns(4)
    row2 = st.columns(3)

    for col, (color, title, value, note) in zip(row1, KPI_ITEMS[:4]):
        with col:
            # simple percent for visual progress
            if "Active" in title:
                pct = (active_leads / max(1, total_leads) * 100) if total_leads else 0
            elif "SLA" in title:
                pct = sla_success_pct
            elif "Qualification" in title:
                pct = qualification_pct
            elif "Conversion" in title:
                pct = conversion_rate
            else:
                pct = random.randint(30, 90)
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='kpi-title'>{title}</div>
                    <div class='kpi-value' style='color:{color};'>{value}</div>
                    <div class='kpi-note'>{note}</div>
                    <div class='progress-wrap'><div class='progress-fill' style='width:{pct:.1f}%; background:{color};'></div></div>
                </div>
            """, unsafe_allow_html=True)

    for col, (color, title, value, note) in zip(row2, KPI_ITEMS[4:]):
        with col:
            if "Inspections" in title:
                pct = inspection_pct
            elif "Estimates" in title:
                pct = (estimate_sent_count / max(1, total_leads)) * 100 if total_leads else 0
            elif "Pipeline Job Value" in title:
                baseline = 5000.0 * max(1, total_leads)
                pct = min(100.0, (pipeline_job_value / baseline) * 100.0) if baseline else 0
            else:
                pct = random.randint(30, 90)
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='kpi-title'>{title}</div>
                    <div class='kpi-value' style='color:{color};'>{value}</div>
                    <div class='kpi-note'>{note}</div>
                    <div class='progress-wrap'><div class='progress-fill' style='width:{pct:.1f}%; background:{color};'></div></div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    # TOP 5 Priority
    st.markdown("### TOP 5 PRIORITY LEADS")
    st.markdown("<em>Highest urgency leads by priority score (0–1). Address these first.</em>", unsafe_allow_html=True)
    priority_list = []
    for _, row in df.iterrows():
        try:
            score = compute_priority_for_lead_row(row, st.session_state.weights)
        except Exception:
            score = 0.0
        sla_sec, overdue = calculate_remaining_sla(row.get("sla_entered_at") or row.get("created_at"), row.get("sla_hours"))
        time_left_h = sla_sec / 3600.0 if sla_sec not in (None, float("inf")) else 9999.0
        priority_list.append({
            "id": int(row["id"]),
            "contact_name": row.get("contact_name") or "No name",
            "estimated_value": float(row.get("estimated_value") or 0.0),
            "time_left_hours": time_left_h,
            "priority_score": score,
            "status": row.get("status"),
            "sla_overdue": overdue
        })
    pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

    if pr_df.empty:
        st.info("No priority leads to display.")
    else:
        for _, r in pr_df.head(5).iterrows():
            score = r["priority_score"]
            if score >= 0.7:
                priority_color = "#ef4444"; priority_label = "🔴 CRITICAL"
            elif score >= 0.45:
                priority_color = "#f97316"; priority_label = "🟠 HIGH"
            else:
                priority_color = "#22c55e"; priority_label = "🟢 NORMAL"
            if r["sla_overdue"]:
                sla_html = f"<span style='color:#ef4444;font-weight:700;'>❗ OVERDUE</span>"
            else:
                hrs = int(r['time_left_hours'])
                mins = int((r['time_left_hours'] * 60) % 60)
                sla_html = f"<span style='color:#ef4444;font-weight:700;'>⏳ {hrs}h {mins}m left</span>"
            money_html = f"<span style='color:#22c55e;font-weight:800;'>${r['estimated_value']:,.0f}</span>"
            st.markdown(f"""
                <div class='priority-card'>
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="flex:1;">
                      <div style="margin-bottom:6px;">
                        <span style="color:{priority_color}; font-weight:800;">{priority_label}</span>
                        <span style="display:inline-block; padding:6px 12px; border-radius:18px; font-size:12px; font-weight:600; margin-left:8px; background:#111; color:#fff;">{r['status']}</span>
                      </div>
                      <div style="font-size:16px; font-weight:800;">#{int(r['id'])} — {r['contact_name']}</div>
                      <div style="font-size:13px; color:#6b7280; margin-top:6px;">Est: {money_html}</div>
                      <div style="font-size:13px; margin-top:8px; color:#6b7280;">{sla_html}</div>
                    </div>
                    <div style="text-align:right; padding-left:18px;">
                      <div style="font-size:28px; font-weight:900; color:{priority_color};">{r['priority_score']:.2f}</div>
                      <div style="font-size:11px; color:#6b7280; text-transform:uppercase;">Priority</div>
                    </div>
                  </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 All Leads (expand a card to edit / change status)")
    st.markdown("<em>Expand a lead to edit details, change status, upload invoice when awarded, and create estimates.</em>", unsafe_allow_html=True)

    # Display leads as expandable cards and allow safe updates
    s = get_session()
    lead_objs = s.query(Lead).order_by(Lead.created_at.desc()).all()
    # Convert to safe structures to avoid DetachedInstanceError
    lead_list = []
    for L in lead_objs:
        lead_list.append({
            "id": L.id,
            "source": L.source,
            "contact_name": L.contact_name,
            "contact_phone": L.contact_phone,
            "contact_email": L.contact_email,
            "property_address": L.property_address,
            "damage_type": L.damage_type,
            "assigned_to": L.assigned_to,
            "notes": L.notes,
            "estimated_value": float(L.estimated_value or 0.0),
            "status": L.status,
            "created_at": L.created_at,
            "sla_hours": L.sla_hours,
            "sla_entered_at": L.sla_entered_at,
            "contacted": bool(L.contacted),
            "inspection_scheduled": bool(L.inspection_scheduled),
            "inspection_completed": bool(L.inspection_completed),
            "estimate_submitted": bool(L.estimate_submitted),
            "awarded_date": L.awarded_date,
            "awarded_invoice": L.awarded_invoice,
            "lost_date": L.lost_date,
            "qualified": bool(L.qualified),
            "cost_to_acquire": float(L.cost_to_acquire or 0.0)
        })
    s.close()

    for lead in lead_list:
        est_val_display = f"${lead['estimated_value']:,.0f}"
        card_title = f"#{lead['id']} — {lead['contact_name'] or 'No name'} — {lead['damage_type'] or 'Unknown'} — {est_val_display}"
        with st.expander(card_title, expanded=False):
            colA, colB = st.columns([3,1])
            with colA:
                st.write(f"**Source:** {lead['source'] or '—'}   |   **Assigned:** {lead['assigned_to'] or '—'}")
                st.write(f"**Address:** {lead['property_address'] or '—'}")
                st.write(f"**Notes:** {lead['notes'] or '—'}")
                st.write(f"**Created:** {lead['created_at'].strftime('%Y-%m-%d %H:%M') if lead['created_at'] else '—'}")
            with colB:
                entered = lead['sla_entered_at'] or lead['created_at']
                if isinstance(entered, str):
                    try:
                        entered = datetime.fromisoformat(entered)
                    except:
                        entered = datetime.utcnow()
                if entered is None:
                    entered = datetime.utcnow()
                deadline = entered + timedelta(hours=(lead['sla_hours'] or 24))
                remaining = deadline - datetime.utcnow()
                if remaining.total_seconds() <= 0:
                    sla_status_html = "<div style='color:#ef4444;font-weight:700;'>❗ OVERDUE</div>"
                else:
                    hours = int(remaining.total_seconds() // 3600)
                    mins = int((remaining.total_seconds() % 3600) // 60)
                    sla_status_html = f"<div style='color:#ef4444;font-weight:700;'>⏳ {hours}h {mins}m</div>"
                st.markdown(f"<div style='text-align:right;'>{sla_status_html}</div>", unsafe_allow_html=True)

            st.markdown("---")
            qc1, qc2, qc3, qc4 = st.columns([1,1,1,4])
            phone = (lead.get('contact_phone') or "").strip()
            email = (lead.get('contact_email') or "").strip()
            if phone:
                with qc1:
                    st.markdown(f"<a href='tel:{phone}'><button style='padding:8px 12px;border-radius:8px;background:#2563eb;color:#fff;border:none;'>📞 Call</button></a>", unsafe_allow_html=True)
                with qc2:
                    wa_number = phone.lstrip("+").replace(" ", "").replace("-", "")
                    wa_link = f"https://wa.me/{wa_number}?text=Hi%2C%20following%20up%20on%20your%20restoration%20request."
                    st.markdown(f"<a href='{wa_link}' target='_blank'><button style='padding:8px 12px;border-radius:8px;background:#22c55e;color:#000;border:none;'>💬 WhatsApp</button></a>", unsafe_allow_html=True)
            else:
                qc1.write(" "); qc2.write(" ")
            if email:
                with qc3:
                    st.markdown(f"<a href='mailto:{email}?subject=Follow%20up'><button style='padding:8px 12px;border-radius:8px;background:transparent;color:#0b1220;border:1px solid #e5e7eb;'>✉️ Email</button></a>", unsafe_allow_html=True)
            else:
                qc3.write(" ")
            qc4.write("")

            st.markdown("---")
            with st.form(f"update_lead_{lead['id']}"):
                new_status = st.selectbox("Status", ["New", "Contacted", "Inspection Scheduled", "Inspection Completed", "Estimate Submitted", "Awarded", "Lost"], index= ["New", "Contacted", "Inspection Scheduled", "Inspection Completed", "Estimate Submitted", "Awarded", "Lost"].index(lead['status']) if lead['status'] in ["New", "Contacted", "Inspection Scheduled", "Inspection Completed", "Estimate Submitted", "Awarded", "Lost"] else 0, key=f"status_{lead['id']}")
                new_assigned = st.text_input("Assigned to", value=lead['assigned_to'] or "", key=f"assign_{lead['id']}")
                new_contacted = st.checkbox("Contacted", value=bool(lead['contacted']), key=f"contacted_{lead['id']}")
                insp_sched = st.checkbox("Inspection Scheduled", value=bool(lead['inspection_scheduled']), key=f"insp_sched_{lead['id']}")
                insp_comp = st.checkbox("Inspection Completed", value=bool(lead['inspection_completed']), key=f"insp_comp_{lead['id']}")
                est_sub = st.checkbox("Estimate Submitted", value=bool(lead['estimate_submitted']), key=f"est_sub_{lead['id']}")
                new_notes = st.text_area("Notes", value=lead['notes'] or "", key=f"notes_{lead['id']}")
                new_est_val = st.number_input("Job Value Estimate (USD)", value=float(lead['estimated_value'] or 0.0), min_value=0.0, step=100.0, key=f"estval_{lead['id']}")
                awarded_invoice_file = None
                award_comment = None
                lost_comment = None
                if new_status == "Awarded":
                    st.markdown("**Award details**")
                    award_comment = st.text_area("Award comment", key=f"award_comment_{lead['id']}")
                    awarded_invoice_file = st.file_uploader("Upload Invoice File (optional) — only for Awarded", type=["pdf","jpg","jpeg","png","xlsx","csv"], key=f"award_inv_{lead['id']}")
                elif new_status == "Lost":
                    st.markdown("**Lost details**")
                    lost_comment = st.text_area("Lost comment", key=f"lost_comment_{lead['id']}")

                if st.form_submit_button("💾 Update Lead"):
                    try:
                        db = get_session()
                        db_lead = db.query(Lead).filter(Lead.id == int(lead['id'])).first()
                        if db_lead:
                            db_lead.status = new_status
                            db_lead.assigned_to = new_assigned
                            db_lead.contacted = bool(new_contacted)
                            db_lead.inspection_scheduled = bool(insp_sched)
                            db_lead.inspection_completed = bool(insp_comp)
                            db_lead.estimate_submitted = bool(est_sub)
                            db_lead.notes = new_notes
                            db_lead.estimated_value = float(new_est_val or 0.0)
                            if db_lead.sla_entered_at is None:
                                db_lead.sla_entered_at = datetime.utcnow()
                            if new_status == "Awarded":
                                db_lead.awarded_date = datetime.utcnow()
                                db_lead.awarded_comment = award_comment
                                if awarded_invoice_file:
                                    path = save_uploaded_file(awarded_invoice_file, prefix=f"lead_{db_lead.id}_inv")
                                    db_lead.awarded_invoice = path
                            if new_status == "Lost":
                                db_lead.lost_date = datetime.utcnow()
                                db_lead.lost_comment = lost_comment
                            db.add(db_lead); db.commit()
                            db.close()
                            push_toast(f"Lead #{db_lead.id} updated.", "success")
                        else:
                            st.error("Lead not found.")
                    except Exception as e:
                        st.error(f"Failed to update lead: {e}")
                        st.write(traceback.format_exc())

# --------------------------
# Page: Analytics & SLA (contains pipeline donut)
# --------------------------
elif page == "Analytics & SLA":
    st.header("📈 Analytics — Pipeline & SLA")
    st.markdown("<em>Visual analytics: pipeline donut, CPA, SLA trends. Use date range to filter (defaults to Today).</em>", unsafe_allow_html=True)
    s = get_session()
    df_all = leads_to_df(s)
    s.close()
    if df_all.empty:
        st.info("No leads to analyze.")
    else:
        min_date = df_all["created_at"].min().date()
        max_date = df_all["created_at"].max().date()
        col_start, col_end = st.columns(2)
        start_date = col_start.date_input("Start date", min_value=min_date, value=min_date)
        end_date = col_end.date_input("End date", min_value=start_date, value=max_date)
        df_range = leads_to_df(get_session(), start_date, end_date)

        # pipeline donut (only on analytics page)
        st.markdown("#### Lead Pipeline Stages (donut)")
        stage_order = ["New", "Contacted", "Inspection Scheduled", "Inspection Completed", "Estimate Submitted", "Awarded", "Lost"]
        if df_range.empty:
            st.info("No leads in selected range.")
        else:
            stage_counts = df_range["status"].value_counts().reindex(stage_order, fill_value=0)
            pie_df = pd.DataFrame({"status": stage_counts.index, "count": stage_counts.values})
            if px:
                fig = px.pie(pie_df, names="status", values="count", hole=0.45, color="status",
                             color_discrete_map={
                                "New": "#2563eb", "Contacted": "#eab308", "Inspection Scheduled": "#f97316",
                                "Inspection Completed": "#14b8a6", "Estimate Submitted": "#a855f7", "Awarded": "#22c55e", "Lost": "#ef4444"
                             })
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.table(pie_df)

        st.markdown("---")
        # SLA Overdue line chart above table
        st.markdown("#### SLA / Overdue Leads (last 30 days)")
        today = datetime.utcnow().date()
        ts_rows = []
        for d in range(30, -1, -1):
            day = today - timedelta(days=d)
            day_start = datetime.combine(day, datetime.min.time())
            day_end = datetime.combine(day, datetime.max.time())
            overdue_count = 0
            for _, row in df_range.iterrows():
                sla_entered = row.get("sla_entered_at") or row.get("created_at")
                if isinstance(sla_entered, str):
                    try:
                        sla_entered = datetime.fromisoformat(sla_entered)
                    except:
                        sla_entered = row.get("created_at") or datetime.utcnow()
                deadline = sla_entered + timedelta(hours=int(row.get("sla_hours") or 24))
                if deadline <= day_end and row.get("status") not in ("Awarded", "Lost", "AWARDED", "LOST"):
                    overdue_count += 1
            ts_rows.append({"date": day, "overdue_count": overdue_count})
        ts_df = pd.DataFrame(ts_rows)
        if px:
            fig = px.line(ts_df, x="date", y="overdue_count", markers=True, labels={"overdue_count": "Overdue leads"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(ts_df)

        st.markdown("#### Overdue leads (table)")
        overdue_rows = []
        for _, row in df_range.iterrows():
            sla_entered = row.get("sla_entered_at") or row.get("created_at")
            if isinstance(sla_entered, str):
                try:
                    sla_entered = datetime.fromisoformat(sla_entered)
                except:
                    sla_entered = datetime.utcnow()
            sla_hours = int(row.get("sla_hours") or 24)
            deadline = sla_entered + timedelta(hours=sla_hours)
            overdue = deadline < datetime.utcnow() and row.get("status") not in ("Awarded", "Lost", "AWARDED", "LOST")
            overdue_rows.append({
                "id": row.get("id"),
                "contact": row.get("contact_name"),
                "status": row.get("status"),
                "deadline": deadline,
                "overdue": overdue
            })
        df_overdue = pd.DataFrame(overdue_rows)
        if not df_overdue.empty:
            st.dataframe(df_overdue[df_overdue["overdue"] == True].sort_values("deadline"))
        else:
            st.info("No SLA overdue leads in this range.")

# --------------------------
# Page: CPA & ROI
# --------------------------
elif page == "CPA & ROI":
    st.header("💰 CPA & ROI")
    st.markdown("<em>Total Marketing Spend vs Conversions (Won). Default date range: Today.</em>", unsafe_allow_html=True)
    s = get_session()
    df_all = leads_to_df(s)
    s.close()
    if df_all.empty:
        st.info("No leads yet.")
    else:
        col1, col2 = st.columns(2)
        start = col1.date_input("Start date", value=date.today())
        end = col2.date_input("End date", value=date.today())
        df_view = leads_to_df(get_session(), start, end)
        total_spend = float(df_view["cost_to_acquire"].fillna(0).sum()) if not df_view.empty else 0.0
        conversions = int(df_view[df_view["status"].str.lower() == "awarded"].shape[0]) if not df_view.empty else 0
        cpa = (total_spend / conversions) if conversions else 0.0
        revenue = float(df_view[df_view["status"].str.lower() == "awarded"]["estimated_value"].fillna(0).sum()) if not df_view.empty else 0.0
        roi_value = revenue - total_spend
        roi_pct = (roi_value / total_spend * 100) if total_spend else 0.0

        # Colored font output (no stylized container)
        st.markdown(f"💰 **Total Marketing Spend:** <span style='color:#ef4444;font-weight:700;'>${total_spend:,.2f}</span>", unsafe_allow_html=True)
        st.markdown(f"✅ **Conversions (Won):** <span style='color:#2563eb;font-weight:700;'>{conversions}</span>", unsafe_allow_html=True)
        st.markdown(f"🎯 **CPA:** <span style='color:#f97316;font-weight:700;'>${cpa:,.2f}</span>", unsafe_allow_html=True)
        st.markdown(f"📈 **ROI:** <span style='color:#22c55e;font-weight:700;'>${roi_value:,.2f} ({roi_pct:.1f}%)</span>", unsafe_allow_html=True)

        # chart
        if not df_view.empty:
            agg = df_view.copy()
            agg["date"] = agg["created_at"].dt.date
            agg = agg.groupby("date").agg({"cost_to_acquire": "sum", "id": lambda s: df_view.loc[s.index, "status"].str.lower().eq("awarded").sum()}).reset_index()
            agg = agg.rename(columns={"id": "conversions"})
            if px:
                fig = px.line(agg, x="date", y=["cost_to_acquire", "conversions"], markers=True)
                fig.update_layout(yaxis_title="Value", xaxis_title="Date", legend_title="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(agg)

# --------------------------
# Page: ML (Internal)
# --------------------------
elif page == "ML (Internal)":
    st.header("🧠 ML — Internal (no user tuning)")
    st.markdown("<em>Model runs internally; no parameters exposed.</em>", unsafe_allow_html=True)
    if not SKL_AVAILABLE or joblib is None:
        st.info("scikit-learn or joblib not available — ML disabled.")
    else:
        s = get_session()
        df = leads_to_df(s)
        s.close()
        labeled = df[df["status"].str.lower().isin(["awarded", "lost"])]
        st.write(f"Labeled leads (awarded/lost): {len(labeled)}")
        st.write("Model persisted at:", MODEL_PATH if os.path.exists(MODEL_PATH) else "No persisted model")
        if st.button("Force one-off internal train now"):
            s = get_session()
            try:
                res = auto_train_model_if_enough(s)
                if res:
                    st.success("Internal training completed and model saved.")
                else:
                    st.info("Training did not run (insufficient labeled data or other).")
            except Exception as e:
                st.error(f"Training failed: {e}")
            finally:
                s.close()
        if not df.empty:
            dfp = df.copy()
            dfp["win_prob"] = dfp["predicted_prob"].fillna(0) * 100
            st.dataframe(dfp.sort_values("win_prob", ascending=False)[["id", "contact_name", "status", "win_prob"]].head(200))

# --------------------------
# Page: Exports
# --------------------------
elif page == "Exports":
    st.header("📤 Export data")
    s = get_session()
    df_leads = leads_to_df(s)
    s.close()
    if df_leads.empty:
        st.info("No leads yet to export.")
    else:
        csv = df_leads.to_csv(index=False).encode("utf-8")
        st.download_button("Download leads.csv", csv, file_name="leads.csv", mime="text/csv")

# --------------------------
# End of app
# --------------------------
st.markdown("---")
st.markdown("Project X — Restoration Pipeline — Ready")

# End of file
