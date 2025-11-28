# project_x_complete_full.py
# Single-file Streamlit app for Restoration Lead Pipeline + Analytics + ML
# Save as project_x_complete_full.py and run: streamlit run project_x_complete_full.py

import os
import time
import traceback
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List

import streamlit as st
import pandas as pd

# Defensive optional imports
try:
    import plotly.express as px
except Exception:
    px = None

try:
    import joblib
except Exception:
    joblib = None

try:
    from docx import Document
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

try:
    import shap
except Exception:
    shap = None

# sklearn defensive imports
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
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
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_complete_full.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "lead_conversion_model.joblib")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploaded_files")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# ---------------------------
# Lead status enum
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

# ---------------------------
# ORM models
# ---------------------------
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    source = Column(String, default="Unknown")
    source_details = Column(Text, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(Text, nullable=True)
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
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    estimate_submitted = Column(Boolean, default=False)
    estimate_submitted_at = Column(DateTime, nullable=True)

    awarded_comment = Column(Text, nullable=True)
    awarded_date = Column(DateTime, nullable=True)
    awarded_invoice = Column(String, nullable=True)

    lost_comment = Column(Text, nullable=True)
    lost_date = Column(DateTime, nullable=True)

    qualified = Column(Boolean, default=False)

    # New fields
    cost_to_acquire = Column(Float, default=0.0)  # cost to acquire this lead (CPA input)
    predicted_prob = Column(Float, nullable=True)  # model-predicted win probability

# estimates table
class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, nullable=False)
    amount = Column(Float, default=0.0)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    approved = Column(Boolean, default=False)
    lost = Column(Boolean, default=False)

# ---------------------------
# DB init & migrations
# ---------------------------
def create_tables_and_migrate():
    Base.metadata.create_all(bind=engine)
    # Try to add missing columns via ALTER TABLE (SQLite allows adding columns)
    insp = inspect(engine)
    cols = [c["name"] for c in insp.get_columns("leads")]
    with engine.connect() as conn:
        # cost_to_acquire
        if "cost_to_acquire" not in cols:
            try:
                conn.execute(text("ALTER TABLE leads ADD COLUMN cost_to_acquire FLOAT DEFAULT 0;"))
            except Exception:
                pass
        # predicted_prob
        if "predicted_prob" not in cols:
            try:
                conn.execute(text("ALTER TABLE leads ADD COLUMN predicted_prob FLOAT;"))
            except Exception:
                pass
        # Add boolean columns for common platforms for easier queries (optional)
        platform_cols = [
            "facebook","instagram","tiktok","linkedin","x_twitter","youtube",
            "yelp","nextdoor","angi","yellowpages","website_form","email_campaign","referral_platform"
        ]
        for p in platform_cols:
            if p not in cols:
                try:
                    conn.execute(text(f"ALTER TABLE leads ADD COLUMN {p} INTEGER DEFAULT 0;"))
                except Exception:
                    # ignore if cannot add
                    pass

create_tables_and_migrate()

# ---------------------------
# DB helpers
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

# load DataFrame of leads
def leads_df(session):
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
        # make sure types exist
        df = pd.DataFrame(columns=[
            "id","source","source_details","contact_name","contact_phone","contact_email",
            "property_address","damage_type","assigned_to","notes","estimated_value","status",
            "created_at","sla_hours","sla_entered_at","contacted","inspection_scheduled","inspection_scheduled_at",
            "inspection_completed","estimate_submitted","awarded_date","awarded_invoice","lost_date","qualified",
            "cost_to_acquire","predicted_prob"
        ])
    return df

def estimates_df(session):
    rows = session.query(Estimate).all()
    data = []
    for r in rows:
        data.append({
            "id": r.id,
            "lead_id": r.lead_id,
            "amount": r.amount,
            "details": r.details,
            "created_at": r.created_at,
            "approved": bool(r.approved),
            "lost": bool(r.lost)
        })
    return pd.DataFrame(data)

# ---------------------------
# UI style: Comfortaa & card CSS
# ---------------------------
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;600;700&display=swap');
:root{
  --bg: #0b0b0b;
  --text: #ffffff;
  --muted: #9ca3af;
  --card-radius: 14px;
  --danger: #ef4444;
  --accent: #f97316;
  --green: #22c55e;
}
body, .stApp { background: #ffffff; color: #0b1220; font-family: 'Comfortaa', cursive; }
.header { font-family: 'Comfortaa', cursive; font-size:20px; font-weight:700; color:#0b1220; padding:8px 0; }
.metric-card { border-radius: var(--card-radius); padding:16px; margin:8px; color:#fff; display:inline-block; vertical-align:top; box-shadow: 0 6px 16px rgba(0,0,0,0.08); transition: transform .18s ease, box-shadow .18s ease; background: #000; }
.metric-card:hover { transform: translateY(-6px); box-shadow: 0 12px 28px rgba(0,0,0,0.18); }
.kpi-title { color: #ffffff; font-weight:700; font-size:13px; margin-bottom:6px; opacity:0.95; font-family: 'Comfortaa', cursive; }
.kpi-value { font-weight:800; font-size:28px; color:#ffffff; font-family: 'Comfortaa', cursive; }
.kpi-note { font-size:12px; color:rgba(255,255,255,0.9); margin-top:6px; font-family: 'Comfortaa', cursive; }
.progress-bar-wrap { width:100%; background:#111; border-radius:8px; height:10px; margin-top:10px; }
.progress-bar-fill { height:100%; border-radius:8px; background:linear-gradient(90deg, #fff, #fff); transition: width .4s ease; }
.priority-card { background:#000; color:#fff; padding:12px; border-radius:12px; margin-bottom:12px; font-family: 'Comfortaa', cursive; }
.small-muted { font-size:12px; color:var(--muted); font-family: 'Comfortaa', cursive; }
"""

# ---------------------------
# Utilities: SLA remaining, priority score, model helpers
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
        return float("inf"), False

def compute_priority_for_lead_row(lead_row, weights, ml_prob=None):
    # value-based
    try:
        val = float(lead_row.get("estimated_value") or 0.0)
        baseline = float(weights.get("value_baseline", 5000.0))
        value_score = min(1.0, val / max(1.0, baseline))
    except Exception:
        value_score = 0.0
    # SLA urgency
    try:
        sla_entered = lead_row.get("sla_entered_at") or lead_row.get("created_at")
        if sla_entered is None:
            time_left_h = 9999.0
        else:
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
    if ml_prob is not None:
        score = max(0.0, min(1.0, 0.75 * score + 0.25 * ml_prob))
    return score

# ---------------------------
# ML: feature builder, train, save, load, predict
# ---------------------------
def build_feature_df_for_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if df is None or df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    d = df.copy()
    d["label_awarded"] = (d["status"] == LeadStatus.AWARDED).astype(int)
    # ensure columns
    for c in ["estimated_value", "qualified", "sla_hours", "inspection_scheduled", "estimate_submitted", "damage_type", "source"]:
        if c not in d.columns:
            d[c] = 0
    X = d[["estimated_value", "qualified", "sla_hours", "inspection_scheduled", "estimate_submitted", "damage_type", "source"]].copy()
    X["estimated_value"] = X["estimated_value"].fillna(0).astype(float)
    X["qualified"] = X["qualified"].astype(int)
    X["sla_hours"] = X["sla_hours"].fillna(24).astype(int)
    X["inspection_scheduled"] = X["inspection_scheduled"].astype(int)
    X["estimate_submitted"] = X["estimate_submitted"].astype(int)
    X["damage_type"] = X["damage_type"].fillna("unknown").astype(str)
    X["source"] = X["source"].fillna("unknown").astype(str)
    y = d["label_awarded"]
    return X, y

def create_sklearn_pipeline():
    if not SKLEARN_AVAILABLE:
        return None
    numeric_cols = ["estimated_value", "qualified", "sla_hours", "inspection_scheduled", "estimate_submitted"]
    categorical_cols = ["damage_type", "source"]
    preproc = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols)
    ], remainder="drop")
    model = Pipeline(steps=[
        ("pre", preproc),
        ("clf", RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42))
    ])
    return model

def train_and_save_model(safe_save=True):
    if not SKLEARN_AVAILABLE:
        return None, "scikit-learn not installed"
    s = get_session()
    df = leads_df(s)
    if df.empty:
        return None, "no data"
    X, y = build_feature_df_for_model(df)
    if X.empty:
        return None, "empty features"
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)
        model = create_sklearn_pipeline()
        model.fit(X_train, y_train)
        # save
        if safe_save and joblib is not None:
            try:
                joblib.dump(model, MODEL_FILE)
            except Exception:
                pass
        # metrics
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None and len(set(y_test)) > 1 else None,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        # store model in session
        st.session_state.lead_model = model
        st.session_state.model_metrics = metrics
        st.session_state.last_train_time = time.time()
        return model, metrics
    except Exception as e:
        return None, str(e)

def load_model_safe(path=MODEL_FILE):
    if joblib is None or not os.path.exists(path):
        return None
    try:
        m = joblib.load(path)
        st.session_state.lead_model = m
        return m
    except Exception:
        return None

def predict_and_persist_all():
    if not SKLEARN_AVAILABLE:
        return False, "scikit-learn missing"
    model = st.session_state.get("lead_model")
    if model is None:
        m = load_model_safe()  # try load from disk
        model = m or st.session_state.get("lead_model")
    if model is None:
        return False, "no model loaded"
    s = get_session()
    df = leads_df(s)
    if df.empty:
        return False, "no leads"
    X, y = build_feature_df_for_model(df)
    if X.empty:
        return False, "no features"
    try:
        probs = model.predict_proba(X)[:, 1]
        for lid, p in zip(df["id"], probs):
            ld = s.query(Lead).filter(Lead.id == int(lid)).first()
            if ld:
                ld.predicted_prob = float(p)
                s.add(ld)
        s.commit()
        return True, "predictions saved"
    except Exception as e:
        return False, str(e)

# ---------------------------
# CPA / ROI helpers
# ---------------------------
def marketing_metrics(df: pd.DataFrame) -> Tuple[float, int, float, float, float]:
    # returns (total_spend, conversions, cpa, revenue, roi_pct)
    if df is None or df.empty:
        return 0.0, 0, 0.0, 0.0, 0.0
    spend = df["cost_to_acquire"].fillna(0.0).sum()
    conversions = int(df[df["status"] == LeadStatus.AWARDED].shape[0])
    cpa = (spend / conversions) if conversions else 0.0
    revenue = float(df[df["status"] == LeadStatus.AWARDED]["estimated_value"].fillna(0.0).sum())
    roi = revenue - spend
    roi_pct = (roi / spend * 100.0) if spend else 0.0
    return spend, conversions, cpa, roi, roi_pct

# ---------------------------
# Application state defaults
# ---------------------------
if "weights" not in st.session_state:
    st.session_state.weights = {
        "value_weight": 0.5, "sla_weight": 0.35, "urgency_weight": 0.15,
        "contacted_w": 0.6, "inspection_w": 0.5, "estimate_w": 0.5, "value_baseline": 5000.0
    }
if "lead_model" not in st.session_state:
    st.session_state.lead_model = None
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = None
if "last_train_time" not in st.session_state:
    st.session_state.last_train_time = None
if "user_role" not in st.session_state:
    st.session_state.user_role = "free"

# ---------------------------
# APP START & Layout
# ---------------------------
st.set_page_config(page_title="Project X — Pipeline", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
st.markdown("<div class='header'>Project X — Sales & Conversion Tracker</div>", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Controls")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "CPA & ROI Dashboard", "ML Model", "Exports"], index=1)
    st.markdown("---")
    st.markdown("### Priority weight tuning")
    st.session_state.weights["value_weight"] = st.slider("Estimate value weight", 0.0, 1.0, float(st.session_state.weights["value_weight"]), step=0.05)
    st.session_state.weights["sla_weight"] = st.slider("SLA urgency weight", 0.0, 1.0, float(st.session_state.weights["sla_weight"]), step=0.05)
    st.session_state.weights["urgency_weight"] = st.slider("Flags urgency weight", 0.0, 1.0, float(st.session_state.weights["urgency_weight"]), step=0.05)
    st.markdown("---")
    st.markdown("### Model (optional)")
    st.write("Joblib model path (optional):")
    model_path_in = st.text_input("Model path", value=MODEL_FILE)
    if st.button("Load model"):
        if joblib and os.path.exists(model_path_in):
            m = load_model_safe(model_path_in)
            if m:
                st.success("Model loaded.")
            else:
                st.error("Failed to load model.")
        else:
            st.info("Path not found or joblib missing.")
    if st.button("Train & save model"):
        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn is required to train.")
        else:
            m, metrics = train_and_save_model()
            if m is None:
                st.error(f"Train failed: {metrics}")
            else:
                st.success("Model trained and saved (if joblib available).")
                st.session_state.model_metrics = metrics
    if st.button("Predict & persist for all leads"):
        ok, msg = predict_and_persist_all()
        if ok:
            st.success(msg)
        else:
            st.error(msg)
    st.markdown("---")
    if st.button("Add Demo Lead"):
        s = get_session()
        add = Lead(
            source="Google Ads",
            source_details="gclid=demo",
            contact_name="Demo Customer",
            contact_phone="+15550000",
            contact_email="demo@example.com",
            property_address="100 Demo Ave",
            damage_type="water",
            assigned_to="Alex",
            estimated_value=4500,
            notes="Demo lead",
            sla_hours=24,
            cost_to_acquire=10.0,
            qualified=True
        )
        s.add(add); s.commit()
        st.success("Demo lead added")

# ---------------------------
# Page: Leads / Capture
# ---------------------------
if page == "Leads / Capture":
    st.header("📇 Lead Capture")
    st.markdown("<em>Add new leads manually or via integrations. Fill cost to acquire for CPA calculation (defaults to $0).</em>", unsafe_allow_html=True)

    with st.form("lead_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            source = st.selectbox("Lead Source", [
                "Google Ads","Website Form","Referral","Email Campaign",
                "Facebook","Instagram","TikTok","LinkedIn","X/Twitter","YouTube",
                "Yelp","Nextdoor","Angi Leads","Yellow Pages","Other"
            ])
            source_details = st.text_input("Source details (UTM / notes)", placeholder="utm_source=google...")
            contact_name = st.text_input("Contact name", placeholder="John Doe")
            contact_phone = st.text_input("Contact phone", placeholder="+1-555-0123")
            contact_email = st.text_input("Contact email", placeholder="name@example.com")
        with c2:
            property_address = st.text_input("Property address", placeholder="123 Main St, City, State")
            damage_type = st.selectbox("Damage type", ["water", "fire", "mold", "contents", "reconstruction", "other"])
            assigned_to = st.text_input("Assigned to", placeholder="Estimator name")
            qualified_choice = st.selectbox("Is the Lead Qualified?", ["No", "Yes"], index=0)
            sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=24, step=1)
        notes = st.text_area("Notes", placeholder="Additional context...")
        estimated_value = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=100.0)
        cost_to_acquire = st.number_input("Cost to Acquire Lead ($)", min_value=0.0, value=0.0, step=1.0)
        submitted = st.form_submit_button("Create Lead", help="Create a new lead")
        if submitted:
            s = get_session()
            lead = Lead(
                source=source,
                source_details=source_details,
                contact_name=contact_name,
                contact_phone=contact_phone,
                contact_email=contact_email,
                property_address=property_address,
                damage_type=damage_type,
                assigned_to=assigned_to,
                notes=notes,
                estimated_value=float(estimated_value or 0.0),
                sla_hours=int(sla_hours),
                sla_entered_at=datetime.utcnow(),
                qualified=True if qualified_choice == "Yes" else False,
                cost_to_acquire=float(cost_to_acquire or 0.0)
            )
            s.add(lead); s.commit()
            st.success(f"Lead created (ID: {lead.id})")

    st.markdown("---")
    st.subheader("Recent leads")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads yet.")
    else:
        st.dataframe(df.sort_values("created_at", ascending=False).head(50))

# ---------------------------
# Page: Pipeline Board
# ---------------------------
elif page == "Pipeline Board":
    st.header("TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR")
    st.markdown("<em>High-level pipeline performance at a glance. Use filters and cards to drill into details.</em>", unsafe_allow_html=True)

    s = get_session()
    df = leads_df(s)

    # Top-right date selection (Quick select like Google Ads)
    top_left, top_right = st.columns([3, 1])
    with top_right:
        quick = st.selectbox("Range", ["Today", "Yesterday", "Last 7 days", "Last 30 days", "All", "Custom"], index=0)
        today = datetime.utcnow().date()
        if quick == "Today":
            start_dt = datetime.combine(today, datetime.min.time())
            end_dt = datetime.combine(today, datetime.max.time())
        elif quick == "Yesterday":
            d = today - timedelta(days=1)
            start_dt = datetime.combine(d, datetime.min.time()); end_dt = datetime.combine(d, datetime.max.time())
        elif quick == "Last 7 days":
            sday = today - timedelta(days=7)
            start_dt = datetime.combine(sday, datetime.min.time()); end_dt = datetime.combine(today, datetime.max.time())
        elif quick == "Last 30 days":
            sday = today - timedelta(days=30)
            start_dt = datetime.combine(sday, datetime.min.time()); end_dt = datetime.combine(today, datetime.max.time())
        elif quick == "All":
            if df.empty:
                start_dt = datetime.combine(today, datetime.min.time())
                end_dt = datetime.combine(today, datetime.max.time())
            else:
                start_dt = df["created_at"].min()
                end_dt = df["created_at"].max()
        else:  # Custom
            sd, ed = st.date_input("Custom range", [today, today])
            start_dt = datetime.combine(sd, datetime.min.time())
            end_dt = datetime.combine(ed, datetime.max.time())

    # filter view
    if not df.empty:
        df_view = df[(df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)].copy()
    else:
        df_view = df.copy()

    # compute KPI values
    total_leads = len(df_view)
    qualified_leads = int(df_view[df_view["qualified"] == True].shape[0]) if not df_view.empty else 0
    sla_success_count = df_view.apply(lambda r: bool(r.get("contacted")), axis=1).sum() if not df_view.empty else 0
    sla_success_pct = (sla_success_count / total_leads * 100) if total_leads else 0.0
    qualification_pct = (qualified_leads / total_leads * 100) if total_leads else 0.0
    awarded_count = int(df_view[df_view["status"] == LeadStatus.AWARDED].shape[0]) if not df_view.empty else 0
    lost_count = int(df_view[df_view["status"] == LeadStatus.LOST].shape[0]) if not df_view.empty else 0
    closed = awarded_count + lost_count
    conversion_rate = (awarded_count / closed * 100) if closed else 0.0
    inspection_scheduled_count = int(df_view[df_view["inspection_scheduled"] == True].shape[0]) if not df_view.empty else 0
    inspection_pct = (inspection_scheduled_count / qualified_leads * 100) if qualified_leads else 0.0
    estimate_sent_count = int(df_view[df_view["estimate_submitted"] == True].shape[0]) if not df_view.empty else 0
    pipeline_job_value = float(df_view["estimated_value"].sum()) if not df_view.empty else 0.0
    active_leads = total_leads - (awarded_count + lost_count)

    # If model present, attach win_prob column for display
    model = st.session_state.get("lead_model")
    if model is not None and SKLEARN_AVAILABLE and not df_view.empty:
        X_all, _ = build_feature_df_for_model(df_view)
        try:
            proba = model.predict_proba(X_all)[:, 1]
            df_view["win_prob"] = proba
        except Exception:
            df_view["win_prob"] = df_view.get("predicted_prob", None)
    else:
        df_view["win_prob"] = df_view.get("predicted_prob", None)

    # KPI cards (7 in requested order) - black background and progress bars
    KPI_ITEMS = [
        ("Active Leads", f"{active_leads}", "Leads currently in pipeline"),
        ("SLA Success", f"{sla_success_pct:.1f}%", "Leads contacted within SLA"),
        ("Qualification Rate", f"{qualification_pct:.1f}%", "Leads marked qualified"),
        ("Conversion Rate", f"{conversion_rate:.1f}%", "Won / Closed"),
        ("Inspections Booked", f"{inspection_pct:.1f}%", "Qualified → Scheduled"),
        ("Estimates Sent", f"{estimate_sent_count}", "Estimates submitted"),
        ("Pipeline Job Value", f"${pipeline_job_value:,.0f}", "Total pipeline job value")
    ]

    st.markdown("<div style='display:flex; flex-wrap:wrap; gap:8px; align-items:stretch;'>", unsafe_allow_html=True)
    for title, value, note in KPI_ITEMS:
        # compute a fake progress as simple function (or real percent for some)
        if title == "Active Leads":
            pct = (active_leads / max(total_leads, 1)) * 100 if total_leads else 0
        elif title == "SLA Success":
            pct = sla_success_pct
        elif title == "Qualification Rate":
            pct = qualification_pct
        elif title == "Conversion Rate":
            pct = conversion_rate
        elif title == "Inspections Booked":
            pct = inspection_pct
        elif title == "Estimates Sent":
            # relative to total leads
            pct = (estimate_sent_count / max(total_leads, 1)) * 100 if total_leads else 0
        elif title == "Pipeline Job Value":
            # relative to baseline
            baseline = st.session_state.weights.get("value_baseline", 5000.0) * max(1, total_leads)
            pct = min(100, (pipeline_job_value / max(1.0, baseline)) * 100)
        else:
            pct = 0
        pct = max(0.0, min(100.0, pct))
        st.markdown(f"""
            <div class="metric-card">
              <div class="kpi-title">{title}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-note">{note}</div>
              <div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:{pct:.1f}%"></div></div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Pipeline stages donut
    st.markdown("### Lead Pipeline Stages")
    st.markdown("<em>Distribution of leads across pipeline stages. Use this to spot stage drop-offs quickly.</em>", unsafe_allow_html=True)
    stage_colors = {
        LeadStatus.NEW: "#2563eb",
        LeadStatus.CONTACTED: "#eab308",
        LeadStatus.INSPECTION_SCHEDULED: "#f97316",
        LeadStatus.INSPECTION_COMPLETED: "#14b8a6",
        LeadStatus.ESTIMATE_SUBMITTED: "#a855f7",
        LeadStatus.AWARDED: "#22c55e",
        LeadStatus.LOST: "#ef4444"
    }
    stage_counts = df_view["status"].value_counts().reindex(LeadStatus.ALL, fill_value=0)
    pie_df = pd.DataFrame({"status": stage_counts.index, "count": stage_counts.values})
    if pie_df["count"].sum() == 0:
        st.info("No leads available to show pipeline stages.")
    else:
        if px:
            fig = px.pie(pie_df, names="status", values="count", hole=0.45, color="status", color_discrete_map=stage_colors)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(t=10, b=10), legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.table(pie_df)

    st.markdown("---")

    # Top 5 Priority Leads
    st.markdown("### TOP 5 PRIORITY LEADS")
    st.markdown("<em>Highest urgency leads by priority score (0–1). Address these first.</em>", unsafe_allow_html=True)
    priority_list = []
    for _, row in df_view.iterrows():
        try:
            ml_prob = float(row.get("win_prob")) if row.get("win_prob") is not None else None
        except Exception:
            ml_prob = None
        try:
            score = compute_priority_for_lead_row(row, st.session_state.weights, ml_prob=ml_prob)
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
            "sla_overdue": overdue,
            "conversion_prob": ml_prob,
            "damage_type": row.get("damage_type", "Unknown")
        })
    pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

    if pr_df.empty:
        st.info("No priority leads to display.")
    else:
        for _, r in pr_df.head(5).iterrows():
            score = r["priority_score"]
            status = r["status"]
            status_color = stage_colors.get(status, "#000000")
            if score >= 0.7:
                priority_color = "#ef4444"
                priority_label = "🔴 CRITICAL"
            elif score >= 0.45:
                priority_color = "#f97316"
                priority_label = "🟠 HIGH"
            else:
                priority_color = "#22c55e"
                priority_label = "🟢 NORMAL"
            if r["sla_overdue"]:
                sla_html = f"<span style='color:#ef4444;font-weight:700;'>❗ OVERDUE</span>"
            else:
                hours_left = int(r['time_left_hours'])
                mins_left = int((r['time_left_hours'] * 60) % 60)
                sla_html = f"<span style='color:#ef4444;font-weight:700;'>⏳ {hours_left}h {mins_left}m left</span>"
            conv_html = ""
            if r["conversion_prob"] is not None:
                conv_pct = r["conversion_prob"] * 100
                conv_color = "#22c55e" if conv_pct > 70 else ("#f97316" if conv_pct > 40 else "#ef4444")
                conv_html = f"<span style='color:{conv_color};font-weight:600;margin-left:12px;'>📊 {conv_pct:.0f}% Win Prob</span>"
            st.markdown(f"""
            <div class="priority-card">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="flex:1;">
                  <div style="margin-bottom:6px;">
                    <span style="color:{priority_color}; font-weight:800;">{priority_label}</span>
                    <span style="display:inline-block; padding:6px 12px; border-radius:18px; font-size:12px; font-weight:600; margin-left:8px; background:{status_color}22; color:{status_color};">{status}</span>
                  </div>
                  <div style="font-size:16px; font-weight:800; color:var(--text);">#{int(r['id'])} — {r['contact_name']}</div>
                  <div style="font-size:13px; color:var(--muted); margin-top:6px;">{r['damage_type'].title()} | Est: <strong style='color:var(--green);'>${r['estimated_value']:,.0f}</strong></div>
                  <div style="font-size:13px; margin-top:8px; color:var(--muted);">{sla_html} {conv_html}</div>
                </div>
                <div style="text-align:right; padding-left:18px;">
                  <div style="font-size:28px; font-weight:900; color:{priority_color};">{r['priority_score']:.2f}</div>
                  <div style="font-size:11px; color:var(--muted); text-transform:uppercase;">Priority</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 All Leads (expand a card to edit / change status)")
    st.markdown("<em>Expand a lead to edit details, change status, upload invoice when awarded, and create estimates.</em>", unsafe_allow_html=True)
    for lead_row in (s.query(Lead).order_by(Lead.created_at.desc()).all() if s else []):
        est_val_display = f"${lead_row.estimated_value:,.0f}" if lead_row.estimated_value else "$0"
        card_title = f"#{lead_row.id} — {lead_row.contact_name or 'No name'} — {lead_row.damage_type or 'Unknown'} — {est_val_display}"
        with st.expander(card_title, expanded=False):
            colA, colB = st.columns([3, 1])
            with colA:
                st.write(f"**Source:** {lead_row.source or '—'}   |   **Assigned:** {lead_row.assigned_to or '—'}")
                st.write(f"**Address:** {lead_row.property_address or '—'}")
                st.write(f"**Notes:** {lead_row.notes or '—'}")
                st.write(f"**Created:** {lead_row.created_at.strftime('%Y-%m-%d %H:%M') if lead_row.created_at else '—'}")
                st.write(f"**Cost to Acquire:** ${float(lead_row.cost_to_acquire or 0.0):,.2f}")
                try:
                    if lead_row.predicted_prob is not None:
                        st.write(f"**Predicted Win Prob:** {lead_row.predicted_prob*100:.1f}%")
                except Exception:
                    pass
            with colB:
                entered = lead_row.sla_entered_at or lead_row.created_at
                if isinstance(entered, str):
                    try:
                        entered = datetime.fromisoformat(entered)
                    except:
                        entered = datetime.utcnow()
                if entered is None:
                    entered = datetime.utcnow()
                deadline = entered + timedelta(hours=(lead_row.sla_hours or 24))
                remaining = deadline - datetime.utcnow()
                if remaining.total_seconds() <= 0:
                    sla_status_html = "<div style='color:var(--danger);font-weight:700;'>❗ OVERDUE</div>"
                else:
                    hours = int(remaining.total_seconds() // 3600)
                    mins = int((remaining.total_seconds() % 3600) // 60)
                    sla_status_html = f"<div style='color:var(--danger);font-weight:700;'>⏳ {hours}h {mins}m</div>"
                st.markdown(f"<div style='text-align:right;'><div style='display:inline-block; padding:6px 12px; border-radius:20px; background:{stage_colors.get(lead_row.status,'#000')}22; color:{stage_colors.get(lead_row.status,'#000')}; font-weight:700;'>{lead_row.status}</div><div style='margin-top:12px;'>{sla_status_html}</div></div>", unsafe_allow_html=True)
            st.markdown("---")
            qc1, qc2, qc3, qc4 = st.columns([1,1,1,4])
            phone = (lead_row.contact_phone or "").strip()
            email = (lead_row.contact_email or "").strip()
            if phone:
                with qc1:
                    st.markdown(f"<a href='tel:{phone}'><button class='btn-animated' style='background:var(--green); color:#000; border-radius:8px;'>📞 Call</button></a>", unsafe_allow_html=True)
                with qc2:
                    wa_number = phone.lstrip("+").replace(" ", "").replace("-", "")
                    wa_link = f"https://wa.me/{wa_number}?text=Hi%2C%20following%20up%20on%20your%20restoration%20request."
                    st.markdown(f"<a href='{wa_link}' target='_blank'><button class='btn-animated' style='background:var(--green); color:#000; border-radius:8px;'>💬 WhatsApp</button></a>", unsafe_allow_html=True)
            else:
                qc1.write(" "); qc2.write(" ")
            if email:
                with qc3:
                    st.markdown(f"<a href='mailto:{email}?subject=Follow%20up'><button class='btn-animated' style='background:transparent; color:var(--text); border:1px solid #e5e7eb; border-radius:8px;'>✉️ Email</button></a>", unsafe_allow_html=True)
            else:
                qc3.write(" ")
            qc4.write("")
            st.markdown("---")
            with st.form(f"update_lead_{lead_row.id}"):
                st.markdown("#### Update Lead")
                u1, u2 = st.columns(2)
                with u1:
                    new_status = st.selectbox("Status", LeadStatus.ALL, index=LeadStatus.ALL.index(lead_row.status) if lead_row.status in LeadStatus.ALL else 0, key=f"status_{lead_row.id}")
                    new_assigned = st.text_input("Assigned to", value=lead_row.assigned_to or "", key=f"assign_{lead_row.id}")
                    new_contacted = st.checkbox("Contacted", value=bool(lead_row.contacted), key=f"contacted_{lead_row.id}")
                with u2:
                    insp_sched = st.checkbox("Inspection Scheduled", value=bool(lead_row.inspection_scheduled), key=f"insp_sched_{lead_row.id}")
                    insp_comp = st.checkbox("Inspection Completed", value=bool(lead_row.inspection_completed), key=f"insp_comp_{lead_row.id}")
                    est_sub = st.checkbox("Estimate Submitted", value=bool(lead_row.estimate_submitted), key=f"est_sub_{lead_row.id}")
                new_notes = st.text_area("Notes", value=lead_row.notes or "", key=f"notes_{lead_row.id}")
                new_est_val = st.number_input("Job Value Estimate (USD)", value=float(lead_row.estimated_value or 0.0), min_value=0.0, step=100.0, key=f"estval_{lead_row.id}")
                new_cost = st.number_input("Cost to Acquire Lead ($)", value=float(lead_row.cost_to_acquire or 0.0), min_value=0.0, step=1.0, key=f"cost_{lead_row.id}")

                awarded_invoice_file = None
                award_comment = None
                lost_comment = None
                if new_status == LeadStatus.AWARDED:
                    st.markdown("**Award details**")
                    award_comment = st.text_area("Award comment", key=f"award_comment_{lead_row.id}")
                    awarded_invoice_file = st.file_uploader("Upload Invoice File (optional) — only for Awarded", type=["pdf","jpg","jpeg","png","xlsx","csv"], key=f"award_inv_{lead_row.id}")
                elif new_status == LeadStatus.LOST:
                    st.markdown("**Lost details**")
                    lost_comment = st.text_area("Lost comment", key=f"lost_comment_{lead_row.id}")

                if st.form_submit_button("💾 Update Lead"):
                    try:
                        db = get_session()
                        db_lead = db.query(Lead).filter(Lead.id == lead_row.id).first()
                        if db_lead:
                            db_lead.status = new_status
                            db_lead.assigned_to = new_assigned
                            db_lead.contacted = bool(new_contacted)
                            db_lead.inspection_scheduled = bool(insp_sched)
                            db_lead.inspection_completed = bool(insp_comp)
                            db_lead.estimate_submitted = bool(est_sub)
                            db_lead.notes = new_notes
                            db_lead.estimated_value = float(new_est_val or 0.0)
                            db_lead.cost_to_acquire = float(new_cost or 0.0)
                            if db_lead.sla_entered_at is None:
                                db_lead.sla_entered_at = datetime.utcnow()
                            if new_status == LeadStatus.AWARDED:
                                db_lead.awarded_date = datetime.utcnow()
                                db_lead.awarded_comment = award_comment
                                if awarded_invoice_file is not None:
                                    path = save_uploaded_file(awarded_invoice_file, prefix=f"lead_{db_lead.id}_inv")
                                    db_lead.awarded_invoice = path
                            if new_status == LeadStatus.LOST:
                                db_lead.lost_date = datetime.utcnow()
                                db_lead.lost_comment = lost_comment
                            db.add(db_lead); db.commit()
                            st.success(f"Lead #{db_lead.id} updated.")
                            # Recalculate prediction for this lead if model exists
                            try:
                                if st.session_state.lead_model is not None and SKLEARN_AVAILABLE:
                                    ok, msg = predict_and_persist_all()  # cheap but recomputes all
                            except Exception:
                                pass
                        else:
                            st.error("Lead not found.")
                    except Exception as e:
                        st.error(f"Failed to update lead: {e}")
                        st.write(traceback.format_exc())

# ---------------------------
# Page: Analytics & SLA
# ---------------------------
elif page == "Analytics & SLA":
    st.header("📈 Analytics — SLA & Funnel")
    st.markdown("<em>View SLA trends and stage distributions (CPA and Conversion Velocity moved to CPA & ROI Dashboard).</em>", unsafe_allow_html=True)

    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads to analyze.")
    else:
        # date range (basic)
        min_date = df["created_at"].min()
        max_date = df["created_at"].max()
        col_start, col_end = st.columns(2)
        start_date = col_start.date_input("Start date", min_value=min_date.date() if min_date is not None else datetime.utcnow().date(), value=min_date.date() if min_date is not None else datetime.utcnow().date())
        end_date = col_end.date_input("End date", min_value=start_date, value=max_date.date() if max_date is not None else datetime.utcnow().date())
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        df_range = df[(df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)].copy()

        # Pipeline stages donut (kept)
        st.markdown("#### Pipeline Stages")
        st.markdown("<em>Distribution of leads across pipeline stages within selected date range.</em>", unsafe_allow_html=True)
        stage_counts = df_range["status"].value_counts().reindex(LeadStatus.ALL, fill_value=0)
        pie_df = pd.DataFrame({"status": stage_counts.index, "count": stage_counts.values})
        if pie_df["count"].sum() == 0:
            st.info("No leads in selected range.")
        else:
            if px:
                fig = px.pie(pie_df, names="status", values="count", hole=0.45, color="status",
                             color_discrete_map=stage_colors)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.table(pie_df)

        # SLA Overdue trend (last 30 days) with line chart above table
        st.markdown("---")
        st.subheader("SLA / Overdue Leads")
        st.markdown("<em>Trend of SLA overdue counts (last 30 days) and current overdue leads table.</em>", unsafe_allow_html=True)

        today = datetime.utcnow().date()
        days_back = 30
        ts_rows = []
        for d in range(days_back, -1, -1):
            day = today - pd.Timedelta(days=d)
            day_end = datetime.combine(day, datetime.max.time())
            overdue_count = 0
            for _, row in df_range.iterrows():
                sla_entered = row.get("sla_entered_at") or row.get("created_at")
                try:
                    if pd.isna(sla_entered) or sla_entered is None:
                        sla_entered = row.get("created_at") or datetime.utcnow()
                    elif isinstance(sla_entered, str):
                        sla_entered = datetime.fromisoformat(sla_entered)
                except:
                    sla_entered = row.get("created_at") or datetime.utcnow()
                deadline = sla_entered + timedelta(hours=int(row.get("sla_hours") or 24))
                if deadline <= day_end and row.get("status") not in (LeadStatus.AWARDED, LeadStatus.LOST):
                    overdue_count += 1
            ts_rows.append({"date": day, "overdue_count": overdue_count})
        ts_df = pd.DataFrame(ts_rows)
        if not ts_df.empty:
            if px:
                fig = px.line(ts_df, x="date", y="overdue_count", markers=True, labels={"overdue_count": "Overdue leads"})
                fig.update_layout(margin=dict(t=6, b=6))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(ts_df)

        # Overdue table current
        overdue_rows = []
        for _, row in df_range.iterrows():
            sla_entered = row.get("sla_entered_at") or row.get("created_at")
            try:
                if pd.isna(sla_entered) or sla_entered is None:
                    sla_entered = datetime.utcnow()
                elif isinstance(sla_entered, str):
                    sla_entered = datetime.fromisoformat(sla_entered)
            except:
                sla_entered = datetime.utcnow()
            sla_hours = int(row.get("sla_hours") or 24)
            deadline = sla_entered + timedelta(hours=sla_hours)
            overdue = deadline < datetime.utcnow() and row.get("status") not in (LeadStatus.AWARDED, LeadStatus.LOST)
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
            st.info("No SLA overdue leads.")

# ---------------------------
# Page: CPA & ROI Dashboard
# ---------------------------
elif page == "CPA & ROI Dashboard":
    st.header("💰 CPA & ROI Dashboard")
    st.markdown("<em>Track total marketing spend vs conversions, CPA and ROI over selectable date ranges. Date selector starts with Today.</em>", unsafe_allow_html=True)

    s = get_session()
    df = leads_df(s)

    # top-right date selector similar to pipeline board
    top_left, top_right = st.columns([3, 1])
    with top_right:
        quick = st.selectbox("Range", ["Today", "Yesterday", "Last 7 days", "Last 30 days", "All", "Custom"], index=0)
        today = datetime.utcnow().date()
        if quick == "Today":
            start_dt = datetime.combine(today, datetime.min.time())
            end_dt = datetime.combine(today, datetime.max.time())
        elif quick == "Yesterday":
            d = today - timedelta(days=1)
            start_dt = datetime.combine(d, datetime.min.time()); end_dt = datetime.combine(d, datetime.max.time())
        elif quick == "Last 7 days":
            sday = today - timedelta(days=7)
            start_dt = datetime.combine(sday, datetime.min.time()); end_dt = datetime.combine(today, datetime.max.time())
        elif quick == "Last 30 days":
            sday = today - timedelta(days=30)
            start_dt = datetime.combine(sday, datetime.min.time()); end_dt = datetime.combine(today, datetime.max.time())
        elif quick == "All":
            if df.empty:
                start_dt = datetime.combine(today, datetime.min.time()); end_dt = datetime.combine(today, datetime.max.time())
            else:
                start_dt = df["created_at"].min(); end_dt = df["created_at"].max()
        else:  # Custom
            sd, ed = st.date_input("Custom range", [today, today])
            start_dt = datetime.combine(sd, datetime.min.time()); end_dt = datetime.combine(ed, datetime.max.time())

    df_view = df.copy() if not df.empty else df
    if not df_view.empty:
        df_view = df_view[(df_view["created_at"] >= start_dt) & (df_view["created_at"] <= end_dt)]

    spend, conversions, cpa, roi, roi_pct = marketing_metrics(df_view)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Marketing Spend", f"${spend:,.2f}")
    col2.metric("Conversions (Won)", f"{conversions}")
    col3.metric("CPA", f"${cpa:,.2f}")
    col4.metric("ROI", f"${roi:,.2f} ({roi_pct:.1f}%)")

    # chart: total marketing spend vs number of conversions (over time)
    st.markdown("### Total Marketing Spend vs Conversions")
    if df_view.empty:
        st.info("No data for selected range.")
    else:
        # build simple daily aggregates
        df_view["date"] = df_view["created_at"].dt.date
        agg = df_view.groupby("date").agg(total_spend=("cost_to_acquire", "sum"), conversions=("status", lambda s: (s == LeadStatus.AWARDED).sum())).reset_index()
        if px:
            fig = px.line(agg, x="date", y=["total_spend", "conversions"], markers=True)
            fig.update_layout(yaxis_title="Value", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(agg)

    # ROI breakdown by source
    st.markdown("---")
    st.markdown("#### ROI by Source")
    if not df_view.empty:
        src = df_view.groupby("source").agg(spend=("cost_to_acquire","sum"), revenue=("estimated_value", lambda s: df_view.loc[s.index][df_view["status"]==LeadStatus.AWARDED]["estimated_value"].sum() if True else 0)).reset_index()
        # simpler approach: compute win revenue per source
        rows = []
        for src_name, grp in df_view.groupby("source"):
            spend_src = grp["cost_to_acquire"].sum()
            revenue_src = grp[grp["status"] == LeadStatus.AWARDED]["estimated_value"].sum()
            roi_src = revenue_src - spend_src
            roi_pct_src = (roi_src / spend_src * 100.0) if spend_src else 0.0
            rows.append({"source": src_name, "spend": spend_src, "revenue": revenue_src, "roi": roi_src, "roi_pct": roi_pct_src})
        dfsrc = pd.DataFrame(rows).sort_values("spend", ascending=False)
        st.dataframe(dfsrc)

# ---------------------------
# Page: ML Model
# ---------------------------
elif page == "ML Model":
    st.header("🧠 ML — Train & Evaluate")
    st.markdown("<em>Train a RandomForest baseline to predict lead win probability. Optional: save model to disk.</em>", unsafe_allow_html=True)

    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is not installed. Install it to use ML features.")
    else:
        s = get_session()
        df = leads_df(s)
        if df.empty:
            st.info("No leads to train on — collect some awarded leads for labels.")
        else:
            X, y = build_feature_df_for_model(df)
            st.markdown(f"Dataset size: **{len(X)}** leads — awarded count: **{int(y.sum())}**")
            test_size = st.slider("Test set size (%)", 5, 50, 20)
            n_estimators = st.number_input("n_estimators", min_value=10, value=120, step=10)
            max_depth = st.number_input("max_depth", min_value=2, value=6, step=1)
            if st.button("Train model (RandomForest)"):
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42, stratify=y if y.nunique()>1 else None)
                    model = create_sklearn_pipeline()
                    model.set_params(clf__n_estimators=int(n_estimators), clf__max_depth=int(max_depth))
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    roc = roc_auc_score(y_test, y_proba) if y_proba is not None and len(set(y_test))>1 else None
                    cm = confusion_matrix(y_test, y_pred)
                    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc, "confusion_matrix": cm.tolist()}
                    # save
                    if joblib:
                        try:
                            joblib.dump(model, MODEL_FILE)
                            st.success(f"Model trained and saved to {MODEL_FILE}")
                        except Exception:
                            st.warning("Model trained but failed to save (joblib error).")
                    else:
                        st.success("Model trained (not saved; joblib not installed).")
                    st.session_state.lead_model = model
                    st.session_state.model_metrics = metrics
                    st.session_state.last_train_time = time.time()
                    st.write("Metrics:", metrics)
                    if y_proba is not None and px:
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
                        fig = px.line(roc_df, x="fpr", y="tpr", title="ROC Curve (test)")
                        fig.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, line_dash='dash')
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.write(traceback.format_exc())
            st.markdown("---")
            if st.session_state.get("lead_model") is not None:
                if st.button("Predict & persist probabilities for current leads"):
                    ok, msg = predict_and_persist_all()
                    if ok:
                        st.success("Predictions saved to DB.")
                    else:
                        st.error(f"Prediction error: {msg}")
                st.markdown("Model metrics (session)")
                st.write(st.session_state.get("model_metrics"))

# ---------------------------
# Page: Exports
# ---------------------------
elif page == "Exports":
    st.header("📤 Export data")
    s = get_session()
    df_leads = leads_df(s)
    if df_leads.empty:
        st.info("No leads to export.")
    else:
        csv = df_leads.to_csv(index=False).encode("utf-8")
        st.download_button("Download leads.csv", csv, file_name="leads.csv", mime="text/csv")
    df_est = estimates_df(s)
    if not df_est.empty:
        st.download_button("Download estimates.csv", df_est.to_csv(index=False).encode("utf-8"), file_name="estimates.csv", mime="text/csv")

# ---------------------------
# End of app
# ---------------------------

# small convenience function for compute_priority_for_lead_row used earlier (exposed)
def compute_priority_for_lead_row_wrapper(lead_row):
    return compute_priority_for_lead_row(lead_row, st.session_state.weights)

# keep the script stable
if __name__ == "__main__":
    pass
