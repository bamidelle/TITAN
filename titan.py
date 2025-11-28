# project_x_restoration_full_v2.py
# Single-file Restoration Lead Pipeline + Analytics + Internal ML + Phase 2 UX
# Run: streamlit run project_x_restoration_full_v2.py

import os
import time
import threading
import traceback
from datetime import datetime, timedelta, date

import streamlit as st
import pandas as pd

# optional plotting
try:
    import plotly.express as px
except Exception:
    px = None

# joblib for ML persistence
try:
    import joblib
except Exception:
    joblib = None

# sklearn (defensive import)
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ---------------------------
# CONFIG
# ---------------------------
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_restoration_v2.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
MODEL_FILE = os.path.join(os.getcwd(), "lead_model_v2.joblib")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads_v2")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# ---------------------------
# Lead statuses and colors
# ---------------------------
class LeadStatus:
    NEW = "New"
    CONTACTED = "Contacted"
    INSPECTION_SCHEDULED = "Inspection Scheduled"
    INSPECTION_COMPLETED = "Inspection Completed"
    ESTIMATE_SUBMITTED = "Estimate Submitted"
    AWARDED = "Awarded"
    LOST = "Lost"
    ALL = [NEW, CONTACTED, INSPECTION_SCHEDULED, INSPECTION_COMPLETED, ESTIMATE_SUBMITTED, AWARDED, LOST]

STAGE_COLORS = {
    LeadStatus.NEW: "#2563EB",
    LeadStatus.CONTACTED: "#EAB308",
    LeadStatus.INSPECTION_SCHEDULED: "#F97316",
    LeadStatus.INSPECTION_COMPLETED: "#14B8A6",
    LeadStatus.ESTIMATE_SUBMITTED: "#A855F7",
    LeadStatus.AWARDED: "#22C55E",
    LeadStatus.LOST: "#EF4444",
}

# ---------------------------
# ORM Model
# ---------------------------
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    source = Column(String, default="Unknown")
    source_details = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)
    assigned_to = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    estimated_value = Column(Float, nullable=True)
    status = Column(String, default=LeadStatus.NEW)
    created_at = Column(DateTime, default=datetime.utcnow)
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime, nullable=True)
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_completed = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime, nullable=True)
    estimate_submitted = Column(Boolean, default=False)
    estimate_approved = Column(Boolean, default=False)
    awarded_comment = Column(Text, nullable=True)
    awarded_date = Column(DateTime, nullable=True)
    awarded_invoice = Column(String, nullable=True)
    lost_comment = Column(Text, nullable=True)
    lost_date = Column(DateTime, nullable=True)
    qualified = Column(Boolean, default=False)
    cost_to_acquire = Column(Float, default=0.0)
    predicted_prob = Column(Float, nullable=True)

# ---------------------------
# DB init + migration safety
# ---------------------------
def init_db():
    Base.metadata.create_all(bind=engine)
    insp = inspect(engine)
    cols = [c["name"] for c in insp.get_columns("leads")]
    with engine.connect() as conn:
        def try_add(col_sql):
            try:
                conn.execute(text(col_sql))
            except Exception:
                pass
        # best-effort column additions for older DBs
        additions = {
            "cost_to_acquire": "ALTER TABLE leads ADD COLUMN cost_to_acquire FLOAT;",
            "predicted_prob": "ALTER TABLE leads ADD COLUMN predicted_prob FLOAT;",
            "awarded_invoice": "ALTER TABLE leads ADD COLUMN awarded_invoice TEXT;",
            "estimate_approved": "ALTER TABLE leads ADD COLUMN estimate_approved FLOAT;",
            "sla_entered_at": "ALTER TABLE leads ADD COLUMN sla_entered_at TEXT;"
        }
        for col, sql in additions.items():
            if col not in cols:
                try_add(sql)

init_db()

# ---------------------------
# Utilities (DB access, file save)
# ---------------------------
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

# ---------------------------
# SLA calculation
# ---------------------------
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
        return 0.0, False

# ---------------------------
# Build pandas DataFrame from DB, with date filtering default TODAY
# ---------------------------
def leads_df(session, start_date: date = None, end_date: date = None):
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
            "inspection_completed": bool(r.inspection_completed),
            "inspection_scheduled_at": r.inspection_scheduled_at,
            "estimate_submitted": bool(r.estimate_submitted),
            "estimate_approved": bool(getattr(r, "estimate_approved", False)),
            "awarded_date": r.awarded_date,
            "awarded_invoice": getattr(r, "awarded_invoice", None),
            "lost_date": r.lost_date,
            "qualified": bool(r.qualified),
            "cost_to_acquire": float(r.cost_to_acquire or 0.0),
            "predicted_prob": float(r.predicted_prob) if r.predicted_prob is not None else None
        })
    df = pd.DataFrame(data)
    if df.empty:
        # return an empty df with columns
        df = pd.DataFrame(columns=[
            "id","source","source_details","contact_name","contact_phone","contact_email",
            "property_address","damage_type","assigned_to","notes","estimated_value","status",
            "created_at","sla_hours","sla_entered_at","contacted","inspection_scheduled","inspection_scheduled_at",
            "inspection_completed","estimate_submitted","estimate_approved","awarded_date","awarded_invoice","lost_date","qualified",
            "cost_to_acquire","predicted_prob"
        ])
        return df
    # default to TODAY if no range specified
    if start_date is None and end_date is None:
        today = datetime.utcnow().date()
        start_date = today
        end_date = today
    if start_date is not None and end_date is not None:
        sdt = datetime.combine(start_date, datetime.min.time())
        edt = datetime.combine(end_date, datetime.max.time())
        df = df[(df["created_at"] >= sdt) & (df["created_at"] <= edt)].copy()
    return df

# ---------------------------
# Priority scoring function (used for Top 5)
# ---------------------------
def compute_priority_for_lead_row(lead_row, weights=None):
    if weights is None:
        weights = {
            "value_weight": 0.5, "sla_weight": 0.35, "urgency_weight": 0.15,
            "contacted_w": 0.6, "inspection_w": 0.5, "estimate_w": 0.5, "value_baseline": 5000.0
        }
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
    return max(0.0, min(score, 1.0))

# ---------------------------
# ML pipeline & internal autorun
# ---------------------------
ML_MIN_LABELS = 12  # minimum labeled examples to train
SKLEARN_OK = SKLEARN_AVAILABLE and joblib is not None

def build_ml_pipeline():
    if not SKLEARN_OK:
        return None
    numeric_cols = ["estimated_value", "sla_hours", "cost_to_acquire"]
    categorical_cols = ["damage_type", "source", "assigned_to"]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ], remainder="drop")
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=120, random_state=42))
    ])
    return pipeline, numeric_cols, categorical_cols

def auto_train_model(session):
    if not SKLEARN_OK:
        return None
    try:
        df = leads_df(session)
        if df.empty:
            return None
        labeled = df[df["status"].isin([LeadStatus.AWARDED, LeadStatus.LOST])]
        if len(labeled) < ML_MIN_LABELS:
            return None
        pipeline_tuple = build_ml_pipeline()
        if pipeline_tuple is None:
            return None
        pipeline, numeric_cols, categorical_cols = pipeline_tuple
        X = df[numeric_cols + categorical_cols].copy()
        X[numeric_cols] = X[numeric_cols].fillna(0.0)
        X[categorical_cols] = X[categorical_cols].fillna("unknown").astype(str)
        y = (df["status"] == LeadStatus.AWARDED).astype(int)
        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
        pipeline.fit(X_train, y_train)
        try:
            joblib.dump(pipeline, MODEL_FILE)
        except Exception:
            pass
        # update preds
        try:
            probs = pipeline.predict_proba(X)[:, 1]
            for lid, p in zip(df["id"], probs):
                lead = session.query(Lead).filter(Lead.id == int(lid)).first()
                if lead:
                    lead.predicted_prob = float(p)
                    session.add(lead)
            session.commit()
        except Exception:
            pass
        return pipeline
    except Exception:
        return None

def ml_retrain_daemon(interval_min=30):
    if not SKLEARN_OK:
        return
    def loop():
        while True:
            s = get_session()
            try:
                trained = auto_train_model(s)
                if trained:
                    # silent log
                    print("[ML] internal retrain complete")
            except Exception as e:
                print("[ML] retrain error:", e)
            finally:
                s.close()
            time.sleep(interval_min * 60)
    t = threading.Thread(target=loop, daemon=True)
    t.start()

# start ML daemon
try:
    ml_retrain_daemon(interval_min=30)
except Exception:
    pass

# ---------------------------
# SLA alert system (UI placeholders + badge + modal)
# ---------------------------
def count_overdue_leads(session, start_date=None, end_date=None):
    df = leads_df(session, start_date, end_date)
    cnt = 0
    for _, row in df.iterrows():
        _, overdue = calculate_remaining_sla(row.get("sla_entered_at"), row.get("sla_hours"))
        if overdue and row.get("status") not in (LeadStatus.AWARDED, LeadStatus.LOST):
            cnt += 1
    return cnt

# background SLA notifier (prints / placeholder)
def sla_background_worker(interval_sec=300):
    def loop():
        while True:
            s = get_session()
            try:
                cnt = count_overdue_leads(s)
                if cnt > 0:
                    print(f"[SLA Monitor] {cnt} overdue leads")
            except Exception:
                pass
            finally:
                s.close()
            time.sleep(interval_sec)
    t = threading.Thread(target=loop, daemon=True)
    t.start()

# start SLA daemon
try:
    sla_background_worker(interval_sec=300)
except Exception:
    pass

# ---------------------------
# UI CSS and page setup
# ---------------------------
st.set_page_config(page_title="Project X — Restoration Pipeline", layout="wide", initial_sidebar_state="expanded")
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');
body, .stApp { font-family: 'Comfortaa', cursive; background:#0b1220; color:#fff; }
.header { font-size:20px; font-weight:700; padding:8px 0; color:#fff; }
.metric-card { border-radius:12px; padding:14px; margin:8px; color:#fff; background:#000; box-shadow:0 10px 20px rgba(0,0,0,0.4); }
.kpi-title { color:#fff; font-weight:700; font-size:13px; }
.kpi-value { font-weight:900; font-size:26px; }
.progress-wrap { width:100%; background:#111; height:8px; border-radius:8px; margin-top:8px; overflow:hidden; }
.progress-fill { height:100%; border-radius:8px; transition: width .35s ease; }
.priority-card { background:#000; color:#fff; padding:12px; border-radius:12px; margin-bottom:10px; }
.small-muted { color:#9ca3af; font-size:12px; }
.bell { position:relative; display:inline-block; padding:6px 10px; background:#111; border-radius:8px; color:#fff; }
.bell-badge { position:absolute; top:-6px; right:-6px; background:#ef4444; color:#fff; border-radius:10px; padding:2px 6px; font-size:12px; }
@media (max-width:700px) {
  .metric-card { width:100% !important; }
}
"""
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

# ---------------------------
# Sidebar controls (pages, quick add, retrain)
# ---------------------------
with st.sidebar:
    st.title("Controls")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "CPA & ROI", "ML (Internal)", "Exports"], index=1)
    st.markdown("---")
    st.markdown("Quick Add Demo Lead")
    if st.button("Add Demo Lead"):
        s = get_session()
        demo = Lead(
            source="Google Ads",
            source_details="gclid=demo",
            contact_name="Demo User",
            contact_phone="+15550000",
            contact_email="demo@example.com",
            property_address="100 Demo Ave",
            damage_type="water",
            assigned_to="Alex",
            notes="Demo lead",
            estimated_value=4500,
            sla_hours=24,
            cost_to_acquire=45.0,
            qualified=True,
            created_at=datetime.utcnow()
        )
        s.add(demo)
        s.commit()
        s.close()
        st.success("Demo lead added.")
    st.markdown("---")
    st.markdown("Model (internal-only)")
    st.write("ML auto-trains in the background when enough labeled data exists.")
    if st.button("Force internal train now"):
        s = get_session()
        try:
            res = auto_train_model(s)
            if res:
                st.success("Internal train complete.")
            else:
                st.warning("Training not completed (not enough labeled data?)")
        except Exception as e:
            st.error(f"Error training: {e}")
        finally:
            s.close()

# ---------------------------
# Top header with bell badge and modal trigger
# ---------------------------
st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'>", unsafe_allow_html=True)
st.markdown("<div class='header'>Project X — Sales & Conversion Tracker</div>", unsafe_allow_html=True)

# compute badge count
sess = get_session()
today = date.today()
# top-right date filter defaults to Today
# We'll put UI later — compute overdue globally for badge
badge_overdue = count_overdue_leads(sess)
sess.close()

# Notification bell element (click toggles modal)
if "show_sla_modal" not in st.session_state:
    st.session_state.show_sla_modal = False
if "toasts" not in st.session_state:
    st.session_state.toasts = []  # list of messages to display

def toggle_sla_modal():
    st.session_state.show_sla_modal = not st.session_state.show_sla_modal

# render bell
bell_html = f"""
<div style='display:flex; align-items:center;'>
  <div class='bell' onclick="window.scrollTo(0,0)">
    🔔 Alerts
    <span style='margin-left:8px; font-size:13px; color:#9ca3af;'>{badge_overdue} overdue</span>
  </div>
</div>
"""
st.markdown(bell_html, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Pages
# ---------------------------
# Helper to show toasts (simple)
def show_toast(msg, kind="info"):
    # append to session to show at top of page
    st.session_state.toasts.insert(0, (msg, kind))
    # keep only last 5
    st.session_state.toasts = st.session_state.toasts[:5]

# display toasts area
if st.session_state.toasts:
    for msg, kind in st.session_state.toasts[:4]:
        if kind == "success":
            st.success(msg)
        elif kind == "warning":
            st.warning(msg)
        elif kind == "error":
            st.error(msg)
        else:
            st.info(msg)

# ---------------------------
# Page: Leads / Capture
# ---------------------------
if page == "Leads / Capture":
    st.header("📇 Lead Capture")
    st.markdown("<em>All fields persist. Cost to Acquire is stored and used for CPA/ROI.</em>", unsafe_allow_html=True)

    with st.form("lead_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            source = st.selectbox("Lead Source", [
                "Google Ads", "Website Form", "Referral", "Facebook", "Instagram", "TikTok",
                "LinkedIn", "X/Twitter", "YouTube", "Yelp", "Nextdoor", "Phone", "Insurance", "Other"
            ])
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
                lead = Lead(
                    source=source, source_details=source_details,
                    contact_name=contact_name, contact_phone=contact_phone, contact_email=contact_email,
                    property_address=property_address, damage_type=damage_type, assigned_to=assigned_to,
                    notes=notes, estimated_value=float(estimated_value or 0.0),
                    status=LeadStatus.NEW, created_at=datetime.utcnow(),
                    sla_hours=int(sla_hours), sla_entered_at=datetime.utcnow(),
                    qualified=True if qualified_choice == "Yes" else False,
                    cost_to_acquire=float(cost_to_acquire or 0.0)
                )
                s.add(lead); s.commit(); s.refresh(lead)
                show_toast(f"Lead created (ID: {lead.id})", "success")
            except Exception as e:
                st.error(f"Failed to create lead: {e}")
                st.write(traceback.format_exc())
            finally:
                s.close()

    st.markdown("---")
    # quick filters + search
    s = get_session()
    df = leads_df(s)
    s.close()
    if df.empty:
        st.info("No leads yet. Create one above.")
    else:
        qcol1, qcol2, qcol3 = st.columns([2,2,4])
        with qcol1:
            q_status = st.selectbox("Filter status", ["All"] + LeadStatus.ALL)
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

# ---------------------------
# Page: Pipeline Board
# ---------------------------
elif page == "Pipeline Board":
    st.header("TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR")
    st.markdown("<em>High-level pipeline performance at a glance. Use the date selectors to view different windows.</em>", unsafe_allow_html=True)

    # Date selectors top-right (Today default)
    col_left, col_right = st.columns([4,2])
    with col_right:
        quick_range = st.selectbox("Quick range", ["Today", "Yesterday", "Last 7 days", "Last 30 days", "All", "Custom"], index=0)
        if quick_range == "Today":
            sdt = date.today(); edt = date.today()
        elif quick_range == "Yesterday":
            sdt = date.today() - timedelta(days=1); edt = sdt
        elif quick_range == "Last 7 days":
            sdt = date.today() - timedelta(days=7); edt = date.today()
        elif quick_range == "Last 30 days":
            sdt = date.today() - timedelta(days=30); edt = date.today()
        elif quick_range == "All":
            s2 = get_session(); df_all = leads_df(s2, None, None); s2.close()
            if df_all.empty:
                sdt = date.today(); edt = date.today()
            else:
                sdt = df_all["created_at"].min().date(); edt = df_all["created_at"].max().date()
        else:
            custom = st.date_input("Start, End", [date.today(), date.today()])
            if isinstance(custom, (list, tuple)) and len(custom) == 2:
                sdt, edt = custom[0], custom[1]
            else:
                sdt = date.today(); edt = date.today()

    s = get_session()
    df = leads_df(s, sdt, edt)
    s.close()

    total_leads = len(df)
    qualified_leads = int(df[df["qualified"] == True].shape[0]) if not df.empty else 0
    sla_success_count = int(df[df["contacted"] == True].shape[0]) if not df.empty else 0
    sla_success_pct = (sla_success_count / total_leads * 100) if total_leads else 0.0
    qualification_pct = (qualified_leads / total_leads * 100) if total_leads else 0.0

    # conversion count counts both awarded and estimate_submitted/approved
    awarded_count = int(df[df["status"] == LeadStatus.AWARDED].shape[0]) if not df.empty else 0
    estimate_submitted_count = int(df[df["estimate_submitted"] == True].shape[0]) if not df.empty else 0
    estimate_approved_count = int(df[df["estimate_approved"] == True].shape[0]) if not df.empty else 0
    conversion_set = set()
    if not df.empty:
        if "id" in df.columns:
            conversion_set.update(df[df["status"] == LeadStatus.AWARDED]["id"].tolist())
            conversion_set.update(df[df["estimate_submitted"] == True]["id"].tolist())
            conversion_set.update(df[df["estimate_approved"] == True]["id"].tolist())
    conversion_count = len(conversion_set)
    closed = awarded_count + int(df[df["status"] == LeadStatus.LOST].shape[0]) if not df.empty else 0
    conversion_rate = (awarded_count / closed * 100) if closed else (conversion_count / total_leads * 100 if total_leads else 0.0)
    inspection_scheduled_count = int(df[df["inspection_scheduled"] == True].shape[0]) if not df.empty else 0
    inspection_pct = (inspection_scheduled_count / qualified_leads * 100) if qualified_leads else 0.0
    estimate_sent_count = int(df[df["estimate_submitted"] == True].shape[0]) if not df.empty else 0
    pipeline_job_value = float(df["estimated_value"].sum()) if not df.empty else 0.0
    active_leads = total_leads - (awarded_count + int(df[df["status"] == LeadStatus.LOST].shape[0])) if not df.empty else 0

    KPI_ITEMS = [
        ("Active Leads", f"{active_leads}", "Leads currently in pipeline"),
        ("SLA Success", f"{sla_success_pct:.1f}%", "Leads contacted within SLA"),
        ("Qualification Rate", f"{qualification_pct:.1f}%", "Leads marked qualified"),
        ("Conversion Rate", f"{conversion_rate:.1f}%", "Won / Closed or Est. Approved"),
        ("Inspections Booked", f"{inspection_pct:.1f}%", "Qualified → Scheduled"),
        ("Estimates Sent", f"{estimate_sent_count}", "Estimates submitted"),
        ("Pipeline Job Value", f"${pipeline_job_value:,.0f}", "Total pipeline job value")
    ]

    num_colors = ["#3B82F6", "#ef4444", "#A78BFA", "#F97316", "#06B6D4", "#7C3AED", "#10B981"]
    bar_colors = ["#60A5FA", "#F87171", "#C4B5FD", "#FB923C", "#67E8F9", "#A78BFA", "#34D399"]

    # render row 1 (4 cards)
    st.markdown("<div style='display:flex; flex-wrap:wrap; gap:8px; align-items:stretch;'>", unsafe_allow_html=True)
    for i in range(4):
        title, value, note = KPI_ITEMS[i]
        color = num_colors[i]
        bar = bar_colors[i]
        # pct calculation
        if title == "Active Leads":
            pct = (active_leads / max(1, total_leads) * 100) if total_leads else 0
        elif title == "SLA Success":
            pct = sla_success_pct
        elif title == "Qualification Rate":
            pct = qualification_pct
        elif title == "Conversion Rate":
            pct = conversion_rate
        else:
            pct = 0
        st.markdown(f"""
            <div class='metric-card' style='width:24%; min-width:200px;'>
                <div class='kpi-title'>{title}</div>
                <div class='kpi-value' style='color:{color};'>{value}</div>
                <div class='small-muted'>{note}</div>
                <div class='progress-wrap'><div class='progress-fill' style='width:{pct:.1f}%; background:{bar};'></div></div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # row 2 (3 cards)
    st.markdown("<div style='display:flex; flex-wrap:wrap; gap:8px; margin-top:6px;'>", unsafe_allow_html=True)
    for i in range(4, 7):
        title, value, note = KPI_ITEMS[i]
        color = num_colors[i]
        bar = bar_colors[i]
        if title == "Estimates Sent":
            pct = (estimate_sent_count / max(1, total_leads)) * 100 if total_leads else 0
        elif title == "Pipeline Job Value":
            baseline = 5000.0 * max(1, total_leads)
            pct = min(100, (pipeline_job_value / max(1.0, baseline)) * 100)
        else:
            pct = 0
        st.markdown(f"""
            <div class='metric-card' style='width:31%; min-width:200px;'>
                <div class='kpi-title'>{title}</div>
                <div class='kpi-value' style='color:{color};'>{value}</div>
                <div class='small-muted'>{note}</div>
                <div class='progress-wrap'><div class='progress-fill' style='width:{pct:.1f}%; background:{bar};'></div></div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Pipeline Stages donut
    st.markdown("### Lead Pipeline Stages")
    st.markdown("<em>Distribution of leads across pipeline stages. Use the date selector above to narrow the period.</em>", unsafe_allow_html=True)
    stage_counts = df["status"].value_counts().reindex(LeadStatus.ALL, fill_value=0)
    pie_df = pd.DataFrame({"status": stage_counts.index, "count": stage_counts.values})
    if pie_df["count"].sum() == 0:
        st.info("No leads in selected range.")
    else:
        if px:
            fig = px.pie(pie_df, names="status", values="count", hole=0.45, color="status", color_discrete_map=STAGE_COLORS)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.table(pie_df)

    st.markdown("---")
    # TOP 5 PRIORITY LEADS
    st.markdown("### TOP 5 PRIORITY LEADS")
    st.markdown("<em>Highest urgency leads by priority score (0–1). Address these first.</em>", unsafe_allow_html=True)
    pr_list = []
    w = st.session_state.get("weights", {"value_weight":0.5,"sla_weight":0.35,"urgency_weight":0.15,"contacted_w":0.6,"inspection_w":0.5,"estimate_w":0.5,"value_baseline":5000.0})
    for _, row in df.iterrows():
        try:
            score = compute_priority_for_lead_row(row, w)
        except Exception:
            score = 0.0
        sla_sec, overdue = calculate_remaining_sla(row.get("sla_entered_at") or row.get("created_at"), row.get("sla_hours"))
        pr_list.append({
            "id": int(row["id"]),
            "name": row.get("contact_name") or "No name",
            "value": float(row.get("estimated_value") or 0.0),
            "score": score,
            "status": row.get("status"),
            "overdue": overdue,
            "prob": row.get("predicted_prob")
        })
    pr_df = pd.DataFrame(pr_list).sort_values("score", ascending=False)
    if pr_df.empty:
        st.info("No priority leads")
    else:
        for _, r in pr_df.head(5).iterrows():
            label = "🔴 CRITICAL" if r["score"] >= 0.7 else ("🟠 HIGH" if r["score"] >= 0.45 else "🟢 NORMAL")
            prob_html = ""
            if r.get("prob") is not None:
                p = r["prob"] * 100
                prob_color = "#22c55e" if p > 70 else ("#f97316" if p > 40 else "#ef4444")
                prob_html = f"<span style='color:{prob_color}; font-weight:700; margin-left:8px;'>📊 {p:.0f}%</span>"
            overdue_html = "<span style='color:#ef4444;'> ❗OVERDUE</span>" if r["overdue"] else ""
            st.markdown(f"<div style='background:#000; padding:10px; border-radius:10px; margin-bottom:8px;'><b>{label}</b> #{r['id']} — {r['name']} — ${r['value']:,.0f}{prob_html}{overdue_html}</div>", unsafe_allow_html=True)

    st.markdown("---")
    # All leads expandable with edit
    st.markdown("### 📋 All Leads (expand a card to edit / change status)")
    st.markdown("<em>Expand a lead to edit details, change status, upload invoice when awarded, and create estimates.</em>", unsafe_allow_html=True)
    s2 = get_session()
    all_leads = s2.query(Lead).order_by(Lead.created_at.desc()).all()
    for lead in all_leads:
        title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'Unknown'} — ${lead.estimated_value or 0:.0f}"
        with st.expander(title):
            colA, colB = st.columns([3,1])
            with colA:
                st.write(f"**Source:** {lead.source}  |  **Assigned:** {lead.assigned_to or '—'}")
                st.write(f"**Address:** {lead.property_address or '—'}")
                st.write(f"**Notes:** {lead.notes or '—'}")
                st.write(f"**Created:** {lead.created_at.strftime('%Y-%m-%d %H:%M') if lead.created_at else '—'}")
            with colB:
                entered = lead.sla_entered_at or lead.created_at
                if isinstance(entered, str):
                    try:
                        entered = datetime.fromisoformat(entered)
                    except:
                        entered = datetime.utcnow()
                if entered is None:
                    entered = datetime.utcnow()
                deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                remaining = deadline - datetime.utcnow()
                if remaining.total_seconds() <= 0:
                    sla_html = "<div style='color:#ef4444; font-weight:700;'>❗ OVERDUE</div>"
                else:
                    hrs = int(remaining.total_seconds() // 3600)
                    mins = int((remaining.total_seconds() % 3600) // 60)
                    sla_html = f"<div style='color:#ef4444; font-weight:700;'>⏳ {hrs}h {mins}m</div>"
                st.markdown(sla_html, unsafe_allow_html=True)
            st.markdown("---")
            with st.form(f"update_lead_{lead.id}"):
                ns = st.selectbox("Status", LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status) if lead.status in LeadStatus.ALL else 0, key=f"status_{lead.id}")
                na = st.text_input("Assigned to", value=lead.assigned_to or "", key=f"assign_{lead.id}")
                nc = st.checkbox("Contacted", value=bool(lead.contacted), key=f"contacted_{lead.id}")
                nsched = st.checkbox("Inspection Scheduled", value=bool(lead.inspection_scheduled), key=f"insp_sched_{lead.id}")
                ncomp = st.checkbox("Inspection Completed", value=bool(lead.inspection_completed), key=f"insp_comp_{lead.id}")
                nsub = st.checkbox("Estimate Submitted", value=bool(lead.estimate_submitted), key=f"est_sub_{lead.id}")
                napp = st.checkbox("Estimate Approved", value=bool(getattr(lead,"estimate_approved",False)), key=f"est_app_{lead.id}")
                nnotes = st.text_area("Notes", value=lead.notes or "", key=f"notes_{lead.id}")
                new_val = st.number_input("Estimated value", value=float(lead.estimated_value or 0.0), min_value=0.0, step=100.0, key=f"estval_{lead.id}")
                awarded_invoice_file = None
                award_comment = None
                lost_comment = None
                if ns == LeadStatus.AWARDED:
                    award_comment = st.text_area("Award Comment", key=f"award_comment_{lead.id}")
                    awarded_invoice_file = st.file_uploader("Invoice file (optional)", type=["pdf","jpg","png","xlsx","csv"], key=f"award_inv_{lead.id}")
                elif ns == LeadStatus.LOST:
                    lost_comment = st.text_area("Lost Comment", key=f"lost_comment_{lead.id}")
                if st.form_submit_button("Save"):
                    dbs = get_session()
                    dblead = dbs.query(Lead).filter(Lead.id == lead.id).first()
                    if dblead:
                        dblead.status = ns
                        dblead.assigned_to = na
                        dblead.contacted = bool(nc)
                        dblead.inspection_scheduled = bool(nsched)
                        dblead.inspection_completed = bool(ncomp)
                        dblead.estimate_submitted = bool(nsub)
                        dblead.estimate_approved = bool(napp)
                        dblead.notes = nnotes
                        dblead.estimated_value = float(new_val or 0.0)
                        if dblead.sla_entered_at is None:
                            dblead.sla_entered_at = datetime.utcnow()
                        if ns == LeadStatus.AWARDED:
                            dblead.awarded_date = datetime.utcnow()
                            dblead.awarded_comment = award_comment
                            if awarded_invoice_file is not None:
                                path = save_uploaded_file(awarded_invoice_file, prefix=f"lead_{dblead.id}_inv")
                                dblead.awarded_invoice = path
                        if ns == LeadStatus.LOST:
                            dblead.lost_date = datetime.utcnow()
                            dblead.lost_comment = lost_comment
                        dbs.add(dblead)
                        dbs.commit()
                        dbs.close()
                        show_toast(f"Lead #{dblead.id} updated", "success")
                    else:
                        st.error("Lead not found")
    s2.close()

# ---------------------------
# Page: Analytics & SLA
# ---------------------------
elif page == "Analytics & SLA":
    st.header("📈 Analytics — SLA & Trends")
    st.markdown("<em>Select a date range and see trends for pipeline & SLA.</em>", unsafe_allow_html=True)
    s = get_session()
    df_all = leads_df(s, None, None)
    if df_all.empty:
        st.info("No leads recorded yet.")
        s.close()
    else:
        min_date = df_all["created_at"].min().date()
        max_date = df_all["created_at"].max().date()
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start date", min_value=min_date, value=min_date)
        end_date = col2.date_input("End date", min_value=start_date, value=max_date)

        df_range = leads_df(s, start_date, end_date)
        st.markdown("#### Pipeline Stages (donut)")
        stage_counts = df_range["status"].value_counts().reindex(LeadStatus.ALL, fill_value=0)
        pie_df = pd.DataFrame({"status": stage_counts.index, "count": stage_counts.values})
        if pie_df["count"].sum() == 0:
            st.info("No leads in selected range.")
        else:
            if px:
                fig = px.pie(pie_df, names="status", values="count", hole=0.45, color="status", color_discrete_map=STAGE_COLORS)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.table(pie_df)

        st.markdown("---")
        st.subheader("SLA / Overdue Leads")
        st.markdown("<em>Trend of SLA overdue counts (last 30 days) and current overdue leads table.</em>", unsafe_allow_html=True)

        today_dt = datetime.utcnow().date()
        days_back = 30
        ts_rows = []
        for d in range(days_back, -1, -1):
            day = today_dt - timedelta(days=d)
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
                if deadline <= day_end and row.get("status") not in (LeadStatus.AWARDED, LeadStatus.LOST):
                    overdue_count += 1
            ts_rows.append({"date": day, "overdue_count": overdue_count})
        ts_df = pd.DataFrame(ts_rows)
        if px:
            fig = px.line(ts_df, x="date", y="overdue_count", markers=True, labels={"overdue_count": "Overdue leads"})
            fig.update_layout(margin=dict(t=6,b=6))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(ts_df)

        # overdue table current
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
            overdue = deadline < datetime.utcnow() and row.get("status") not in (LeadStatus.AWARDED, LeadStatus.LOST)
            overdue_rows.append({"id": row.get("id"), "contact": row.get("contact_name"), "status": row.get("status"), "deadline": deadline, "overdue": overdue})
        df_overdue = pd.DataFrame(overdue_rows)
        if not df_overdue.empty:
            st.dataframe(df_overdue[df_overdue["overdue"] == True].sort_values("deadline"))
        else:
            st.success("No SLA overdue leads in this range 🎉")
    s.close()

# ---------------------------
# Page: CPA & ROI
# ---------------------------
elif page == "CPA & ROI":
    st.header("💰 CPA & ROI")
    st.markdown("<em>Total Marketing Spend vs Conversions. Default date range is Today.</em>", unsafe_allow_html=True)
    s = get_session()
    df_all = leads_df(s, None, None)
    if df_all.empty:
        st.info("No leads yet.")
        s.close()
    else:
        col1, col2 = st.columns(2)
        start = col1.date_input("Start date", value=date.today())
        end = col2.date_input("End date", value=date.today())
        df_view = leads_df(s, start, end)
        total_spend = float(df_view["cost_to_acquire"].fillna(0).sum()) if not df_view.empty else 0.0
        conv_ids = set()
        if not df_view.empty:
            conv_ids.update(df_view[df_view["status"] == LeadStatus.AWARDED]["id"].tolist())
            conv_ids.update(df_view[df_view["estimate_submitted"] == True]["id"].tolist())
            if "estimate_approved" in df_view.columns:
                conv_ids.update(df_view[df_view["estimate_approved"] == True]["id"].tolist())
        conversions = len(conv_ids)
        cpa = (total_spend / conversions) if conversions else 0.0
        revenue = float(df_view[df_view["status"] == LeadStatus.AWARDED]["estimated_value"].fillna(0).sum()) if not df_view.empty else 0.0
        roi_value = revenue - total_spend
        roi_pct = (roi_value / total_spend * 100) if total_spend else 0.0

        st.markdown(f"<div style='font-family:Comfortaa; font-size:16px;'>💰 Total Marketing Spend: <span style='color:#ef4444; font-weight:800;'>${total_spend:,.2f}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-family:Comfortaa; font-size:16px;'>✅ Conversions (Won or Est Sent): <span style='color:#2563eb; font-weight:800;'>{conversions}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-family:Comfortaa; font-size:16px;'>🎯 CPA: <span style='color:#f97316; font-weight:800;'>${cpa:,.2f}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-family:Comfortaa; font-size:16px;'>📈 ROI: <span style='color:#22c55e; font-weight:800;'>${roi_value:,.2f} ({roi_pct:.1f}%)</span></div>", unsafe_allow_html=True)

        if not df_view.empty and "created_at" in df_view.columns:
            chart_df = pd.DataFrame({"date": df_view["created_at"].dt.date, "spend": df_view["cost_to_acquire"], "won": df_view["status"].apply(lambda s: 1 if s == LeadStatus.AWARDED else 0), "est_sent": df_view["estimate_submitted"].apply(lambda b: 1 if b else 0)})
            agg = chart_df.groupby("date").agg({"spend": "sum", "won": "sum", "est_sent": "sum"}).reset_index()
            agg["conversions"] = agg["won"] + agg["est_sent"]
            if px:
                fig = px.line(agg, x="date", y=["spend", "conversions"], markers=True)
                fig.update_layout(yaxis_title="Value / Conversions")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(agg)
    s.close()

# ---------------------------
# Page: ML (internal)
# ---------------------------
elif page == "ML (Internal)":
    st.header("🧠 ML — Internal (no user tuning)")
    st.markdown("<em>Model runs internally — only ops/admins can trigger a forced retrain here.</em>", unsafe_allow_html=True)
    if not SKLEARN_OK:
        st.error("scikit-learn or joblib not available — ML disabled.")
    else:
        s = get_session()
        df = leads_df(s, None, None)
        labeled = df[df["status"].isin([LeadStatus.AWARDED, LeadStatus.LOST])]
        st.write(f"Labeled leads (awarded/lost): {len(labeled)}")
        st.write("Model file:", MODEL_FILE if os.path.exists(MODEL_FILE) else "No model persisted")
        if st.button("Force one-off internal train now"):
            try:
                trained = auto_train_model(s)
                if trained:
                    st.success("Internal training completed and model saved.")
                else:
                    st.warning("Training not completed — not enough labeled data or training failed.")
            except Exception as e:
                st.error(f"Training failed: {e}")
        # show sample predictions
        dfp = df.copy()
        dfp["win_prob"] = dfp["predicted_prob"].fillna(0) * 100
        if not dfp.empty:
            st.dataframe(dfp.sort_values("win_prob", ascending=False)[["id", "contact_name", "status", "win_prob"]].head(200))
        s.close()

# ---------------------------
# Page: Exports
# ---------------------------
elif page == "Exports":
    st.header("📤 Export data")
    s = get_session()
    df_leads = leads_df(s, None, None)
    if df_leads.empty:
        st.info("No leads to export.")
    else:
        csv = df_leads.to_csv(index=False).encode("utf-8")
        st.download_button("Download leads.csv", csv, file_name="leads.csv", mime="text/csv")
    s.close()

# ---------------------------
# End of app
# ---------------------------
