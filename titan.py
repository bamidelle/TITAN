# project_x_restoration_final.py
# Full single-file Streamlit app — Project X Restoration Pipeline
# Features:
# - Lead capture with many sources and cost_to_acquire
# - Pipeline dashboard: 7 KPI cards arranged in 2 rows with colored numeric fonts and progress bars
# - Lead Pipeline Stages (donut), TOP 5 PRIORITY LEADS, All Leads expandable/editable
# - Analytics (CPA & ROI) with date selectors and colored numeric fonts (no badges)
# - Internal-only ML: auto-train when enough labeled data; OneHotEncoder uses sparse_output=False
# - Search & quick filters, mobile-friendly responsive layout, automated SLA alerts, lead scoring
# - All inputs persisted in SQLite via SQLAlchemy; exports available
# Run: streamlit run project_x_restoration_final.py

import os
import time
import threading
from datetime import datetime, timedelta, date
import traceback

import streamlit as st
import pandas as pd

# Optional plotting
try:
    import plotly.express as px
except Exception:
    px = None

# Optional model persistence
try:
    import joblib
except Exception:
    joblib = None

# sklearn (defensive)
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
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_restoration_final.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
MODEL_PATH = os.path.join(os.getcwd(), "model_lead.joblib")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# ---------------------------
# Lead statuses & colors
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

stage_colors = {
    LeadStatus.NEW: "#2563eb",
    LeadStatus.CONTACTED: "#eab308",
    LeadStatus.INSPECTION_SCHEDULED: "#f97316",
    LeadStatus.INSPECTION_COMPLETED: "#14b8a6",
    LeadStatus.ESTIMATE_SUBMITTED: "#a855f7",
    LeadStatus.AWARDED: "#22c55e",
    LeadStatus.LOST: "#ef4444"
}

# ---------------------------
# ORM models
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
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    estimate_submitted = Column(Boolean, default=False)
    awarded_comment = Column(Text, nullable=True)
    awarded_date = Column(DateTime, nullable=True)
    awarded_invoice = Column(String, nullable=True)
    lost_comment = Column(Text, nullable=True)
    lost_date = Column(DateTime, nullable=True)
    qualified = Column(Boolean, default=False)
    cost_to_acquire = Column(Float, default=0.0)
    predicted_prob = Column(Float, nullable=True)

# ---------------------------
# DB init & migrations
# ---------------------------
def init_db():
    Base.metadata.create_all(bind=engine)
    insp = inspect(engine)
    cols = [c["name"] for c in insp.get_columns("leads")]
    # add predicted_prob column if missing
    with engine.connect() as conn:
        if "predicted_prob" not in cols:
            try:
                conn.execute(text("ALTER TABLE leads ADD COLUMN predicted_prob FLOAT;"))
            except Exception:
                pass

init_db()

# ---------------------------
# Utilities
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
# leads_df (defined early for ML)
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
        # return empty frame with columns
        df = pd.DataFrame(columns=[
            "id","source","source_details","contact_name","contact_phone","contact_email",
            "property_address","damage_type","assigned_to","notes","estimated_value","status",
            "created_at","sla_hours","sla_entered_at","contacted","inspection_scheduled","inspection_scheduled_at",
            "inspection_completed","estimate_submitted","awarded_date","awarded_invoice","lost_date","qualified",
            "cost_to_acquire","predicted_prob"
        ])
    if start_date is not None and end_date is not None and not df.empty:
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        df = df[(df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)].copy()
    return df

# ---------------------------
# ML Internal (auto-train + persist)
# ---------------------------

ML_MIN_LABELS = 10


def build_ml_pipeline():
    if not SKLEARN_AVAILABLE:
        return None
    numeric_cols = ["estimated_value", "sla_hours", "cost_to_acquire"]
    categorical_cols = ["damage_type", "source", "assigned_to"]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ], remainder="drop")
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    return pipeline, numeric_cols, categorical_cols


def internal_ml_autorun(min_labels_required=ML_MIN_LABELS):
    if not SKLEARN_AVAILABLE:
        return None, "sklearn_missing"
    s = get_session()
    try:
        df = leads_df(s)
        if df.empty:
            return None, "no_data"
        labeled = df[df["status"].isin([LeadStatus.AWARDED, LeadStatus.LOST])]
        if len(labeled) < min_labels_required:
            return None, f"need {min_labels_required} labeled leads, have {len(labeled)}"
        pipeline_build = build_ml_pipeline()
        if pipeline_build is None:
            return None, "pipeline_build_failed"
        pipeline, numeric_cols, categorical_cols = pipeline_build
        X = df[numeric_cols + categorical_cols].copy()
        X[numeric_cols] = X[numeric_cols].fillna(0.0)
        X[categorical_cols] = X[categorical_cols].fillna("unknown").astype(str)
        y = (df["status"] == LeadStatus.AWARDED).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        # persist model
        if joblib is not None:
            try:
                joblib.dump(pipeline, MODEL_PATH)
            except Exception:
                pass
        # predict proba and write back to DB
        try:
            probs = pipeline.predict_proba(X)[:, 1]
            for lid, p in zip(df["id"], probs):
                lead = s.query(Lead).filter(Lead.id == int(lid)).first()
                if lead:
                    lead.predicted_prob = float(p)
                    s.add(lead)
            s.commit()
        except Exception:
            pass
        return pipeline, f"trained {len(labeled)} labels"
    except Exception as e:
        return None, str(e)
    finally:
        s.close()

# Attempt autorun at startup (safe)
try:
    ml_model, ml_msg = internal_ml_autorun()
except Exception as e:
    ml_model, ml_msg = None, str(e)

# ---------------------------
# UI CSS
# ---------------------------

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');
:root{--card-radius:12px}
body, .stApp { font-family: 'Comfortaa', cursive; }
.metric-card { border-radius: var(--card-radius); padding:14px; margin:8px; color:#fff; background:#000; }
.kpi-title { color:#fff; font-weight:700; font-size:13px; }
.kpi-value { font-weight:800; font-size:26px; }
.progress-wrap { width:100%; background:#111; height:10px; border-radius:8px; margin-top:8px; }
.progress-fill { height:100%; border-radius:8px; transition: width .3s ease; }
.priority-card { background:#000; color:#fff; padding:12px; border-radius:12px; margin-bottom:10px; }
.small-muted { color:#9ca3af; font-size:12px; }
@media (max-width: 700px) {
  .metric-card { width:100% !important; }
}
"""

st.set_page_config(page_title="Project X — Pipeline", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Controls")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "CPA & ROI", "ML (Internal)", "Exports"], index=1)
    st.markdown("---")
    st.markdown("### Quick Add / Demo")
    if st.button("Add Demo Lead"):
        s = get_session()
        demo = Lead(
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
            cost_to_acquire=50.0,
            qualified=True
        )
        s.add(demo); s.commit(); s.close()
        st.success("Demo lead added")
    st.markdown("---")
    st.markdown("### Priority weights (advanced)")
    if "weights" not in st.session_state:
        st.session_state.weights = {
            "value_weight": 0.5, "sla_weight": 0.35, "urgency_weight": 0.15
        }
    st.session_state.weights["value_weight"] = st.slider("Estimate value weight", 0.0, 1.0, float(st.session_state.weights["value_weight"]), step=0.05)
    st.session_state.weights["sla_weight"] = st.slider("SLA urgency weight", 0.0, 1.0, float(st.session_state.weights["sla_weight"]), step=0.05)
    st.session_state.weights["urgency_weight"] = st.slider("Flags urgency weight", 0.0, 1.0, float(st.session_state.weights["urgency_weight"]), step=0.05)
    st.markdown("---")
    st.markdown("Model: internal-only")
    st.write("Internal ML auto-trains — no user parameters exposed.")

# ---------------------------
# Page: Leads / Capture
# ---------------------------
if page == "Leads / Capture":
    st.header("📇 Lead Capture")
    st.markdown("<em>Add and save leads; every input is persisted—viewable by date in dashboards.</em>", unsafe_allow_html=True)

    with st.form("lead_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            source = st.selectbox("Lead Source", ["Google Ads", "Website Form", "Referral", "Facebook", "Instagram", "TikTok", "LinkedIn", "X/Twitter", "YouTube", "Yelp", "Nextdoor", "Phone", "Insurance", "Other"])
            source_details = st.text_input("Source details (UTM / notes)")
            contact_name = st.text_input("Contact name")
            contact_phone = st.text_input("Contact phone")
            contact_email = st.text_input("Contact email")
        with c2:
            property_address = st.text_input("Property address")
            damage_type = st.selectbox("Damage type", ["water", "fire", "mold", "contents", "reconstruction", "other"])
            assigned_to = st.text_input("Assigned to")
            qualified_choice = st.selectbox("Is the Lead Qualified?", ["No", "Yes"], index=0)
            sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=24, step=1)
        notes = st.text_area("Notes")
        estimated_value = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=100.0)
        cost_to_acquire = st.number_input("Cost to acquire lead ($)", min_value=0.0, value=0.0, step=1.0)
        submitted = st.form_submit_button("Create Lead")
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
                status=LeadStatus.NEW,
                sla_hours=int(sla_hours),
                sla_entered_at=datetime.utcnow(),
                qualified=True if qualified_choice=="Yes" else False,
                cost_to_acquire=float(cost_to_acquire or 0.0)
            )
            s.add(lead); s.commit(); s.refresh(lead); s.close()
            st.success(f"Lead created (ID: {lead.id})")

    st.markdown("---")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads yet. Create one above.")
    else:
        st.dataframe(df.sort_values("created_at", ascending=False).head(50))
    s.close()

# ---------------------------
# Page: Pipeline Board
# ---------------------------
elif page == "Pipeline Board":
    st.header("TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR")
    st.markdown("<em>High-level pipeline performance at a glance. Use date filter on top-right to narrow data.</em>", unsafe_allow_html=True)

    s = get_session()
    df_all = leads_df(s)

    # Date selector top-right
    col_l, col_r = st.columns([3,1])
    with col_r:
        quick = st.selectbox("Range", ["Today","Yesterday","Last 7 days","Last 30 days","All","Custom"], index=0)
        today = datetime.utcnow().date()
        if quick == "Today":
            sdt = today; edt = today
        elif quick == "Yesterday":
            sdt = today - timedelta(days=1); edt = sdt
        elif quick == "Last 7 days":
            sdt = today - timedelta(days=7); edt = today
        elif quick == "Last 30 days":
            sdt = today - timedelta(days=30); edt = today
        elif quick == "All":
            if df_all.empty:
                sdt = today; edt = today
            else:
                sdt = df_all["created_at"].min().date(); edt = df_all["created_at"].max().date()
        else:
            sd, ed = st.date_input("Select start & end", [today, today])
            sdt = sd; edt = ed

    df = leads_df(s, start_date=sdt, end_date=edt)

    total_leads = len(df)
    qualified_leads = int(df[df["qualified"]==True].shape[0]) if not df.empty else 0
    sla_success_count = int(df.apply(lambda r: bool(r.get("contacted")), axis=1).sum()) if not df.empty else 0
    sla_success_pct = (sla_success_count / total_leads * 100) if total_leads else 0.0
    qualification_pct = (qualified_leads / total_leads * 100) if total_leads else 0.0
    awarded_count = int(df[df["status"]==LeadStatus.AWARDED].shape[0]) if not df.empty else 0
    lost_count = int(df[df["status"]==LeadStatus.LOST].shape[0]) if not df.empty else 0
    closed = awarded_count + lost_count
    conversion_rate = (awarded_count / closed * 100) if closed else 0.0
    inspection_scheduled_count = int(df[df["inspection_scheduled"]==True].shape[0]) if not df.empty else 0
    inspection_pct = (inspection_scheduled_count / qualified_leads * 100) if qualified_leads else 0.0
    estimate_sent_count = int(df[df["estimate_submitted"]==True].shape[0]) if not df.empty else 0
    pipeline_job_value = float(df["estimated_value"].sum()) if not df.empty else 0.0
    active_leads = total_leads - (awarded_count + lost_count)

    # KPI definitions (7 cards) in specific order
    KPI_ITEMS = [
        ("Active Leads", f"{active_leads}", "Leads currently in pipeline"),
        ("SLA Success", f"{sla_success_pct:.1f}%", "Leads contacted within SLA"),
        ("Qualification Rate", f"{qualification_pct:.1f}%", "Leads marked qualified"),
        ("Conversion Rate", f"{conversion_rate:.1f}%", "Won / Closed"),
        ("Inspections Booked", f"{inspection_pct:.1f}%", "Qualified → Scheduled"),
        ("Estimates Sent", f"{estimate_sent_count}", "Estimates submitted"),
        ("Pipeline Job Value", f"${pipeline_job_value:,.0f}", "Total pipeline job value")
    ]

    num_colors = ["#10B981","#3B82F6","#F97316","#EF4444","#A78BFA","#06B6D4","#F59E0B"]
    bar_colors = ["#34D399","#60A5FA","#FB923C","#F87171","#C084FC","#67E8F9","#FBBF24"]

    # Render first row (4 cards)
    st.markdown("<div style='display:flex; flex-wrap:wrap; gap:8px;'>", unsafe_allow_html=True)
    for i in range(4):
        color = num_colors[i]
        bar = bar_colors[i]
        title, value, note = KPI_ITEMS[i]
        # determine pct for progress bar (simple heuristic)
        if title == "Active Leads":
            pct = (active_leads / max(1, total_leads)) * 100 if total_leads else 0
        elif title == "SLA Success":
            pct = sla_success_pct
        elif title == "Qualification Rate":
            pct = qualification_pct
        elif title == "Conversion Rate":
            pct = conversion_rate
        else:
            pct = 0
        st.markdown(f"""
            <div class='metric-card' style='width:24%; min-width:200px; background:#000;'>
                <div class='kpi-title' style='color:white'>{title}</div>
                <div class='kpi-value' style='color:{color}; font-weight:900;'>{value}</div>
                <div class='kpi-note' style='color:rgba(255,255,255,0.8);'>{note}</div>
                <div class='progress-wrap'><div class='progress-fill' style='width:{pct:.1f}%; background:{bar};'></div></div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Second row (3 cards)
    st.markdown("<div style='display:flex; flex-wrap:wrap; gap:8px; margin-top:6px;'>", unsafe_allow_html=True)
    for i in range(4,7):
        idx = i
        color = num_colors[idx]
        bar = bar_colors[idx]
        title, value, note = KPI_ITEMS[idx]
        if title == "Estimates Sent":
            pct = (estimate_sent_count / max(1, total_leads)) * 100 if total_leads else 0
        elif title == "Pipeline Job Value":
            baseline = st.session_state.get("value_baseline", 5000.0) * max(1, total_leads)
            pct = min(100, (pipeline_job_value / max(1.0, baseline)) * 100)
        else:
            pct = 0
        st.markdown(f"""
            <div class='metric-card' style='width:31%; min-width:200px; background:#000;'>
                <div class='kpi-title' style='color:white'>{title}</div>
                <div class='kpi-value' style='color:{color}; font-weight:900;'>{value}</div>
                <div class='kpi-note' style='color:rgba(255,255,255,0.8);'>{note}</div>
                <div class='progress-wrap'><div class='progress-fill' style='width:{pct:.1f}%; background:{bar};'></div></div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Lead Pipeline Stages (Donut)
    st.markdown("### Lead Pipeline Stages")
    st.markdown("<em>Distribution of leads across pipeline stages. Use the date selector above to filter.</em>", unsafe_allow_html=True)

    stage_counts = df["status"].value_counts().reindex(LeadStatus.ALL, fill_value=0)
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

    # TOP 5 PRIORITY LEADS
    st.markdown("### TOP 5 PRIORITY LEADS")
    st.markdown("<em>Highest urgency leads by priority score (0–1). Address these first.</em>", unsafe_allow_html=True)

    # build priority list
    priority_list = []
    for _, row in df.iterrows():
        try:
            ml_prob = float(row.get("predicted_prob")) if row.get("predicted_prob") is not None else None
        except Exception:
            ml_prob = None
        # compute score
        try:
            value = float(row.get("estimated_value") or 0.0)
        except Exception:
            value = 0.0
        try:
            sla_entered = row.get("sla_entered_at") or row.get("created_at")
            if isinstance(sla_entered, str):
                sla_entered = datetime.fromisoformat(sla_entered)
            deadline = sla_entered + timedelta(hours=int(row.get("sla_hours") or 24))
            time_left = (deadline - datetime.utcnow()).total_seconds() / 3600.0
        except Exception:
            time_left = 9999.0
        # simple scoring: value + sla urgency + not contacted
        sla_score = max(0.0, (72.0 - min(max(time_left,0.0), 72.0)) / 72.0)
        contacted_flag = 0.0 if bool(row.get("contacted")) else 1.0
        base_score = min(1.0, (value / max(1.0, st.session_state.get("value_baseline", 5000.0))) * 0.5 + sla_score * 0.35 + contacted_flag * 0.15)
        if ml_prob is not None:
            score = max(0.0, min(1.0, 0.7 * base_score + 0.3 * ml_prob))
        else:
            score = base_score
        sla_sec, overdue = calculate_remaining_sla(row.get("sla_entered_at"), row.get("sla_hours"))
        priority_list.append({
            "id": int(row.get("id")),
            "contact_name": row.get("contact_name") or "No name",
            "estimated_value": float(row.get("estimated_value") or 0.0),
            "time_left_hours": sla_sec/3600.0 if sla_sec not in (None, float('inf')) else 9999.0,
            "priority_score": score,
            "status": row.get("status"),
            "sla_overdue": overdue,
            "conversion_prob": ml_prob
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
            <div style="background: linear-gradient(180deg, rgba(0,0,0,0.04), rgba(0,0,0,0.02)); padding:12px; border-radius:12px; margin-bottom:10px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="flex:1;">
                  <div style="margin-bottom:6px;">
                    <span style="color:{priority_color}; font-weight:800;">{priority_label}</span>
                    <span style="display:inline-block; padding:6px 12px; border-radius:18px; font-size:12px; font-weight:600; margin-left:8px; background:{status_color}22; color:{status_color};">{status}</span>
                  </div>
                  <div style="font-size:16px; font-weight:800; color:var(--text);">#{int(r['id'])} — {r['contact_name']}</div>
                  <div style="font-size:13px; color:#9ca3af; margin-top:6px;">Est: <span style='color:#22c55e; font-weight:800;'>${r['estimated_value']:,.0f}</span></div>
                  <div style="font-size:13px; margin-top:8px; color:#9ca3af;">{sla_html} {conv_html}</div>
                </div>
                <div style="text-align:right; padding-left:18px;">
                  <div style="font-size:28px; font-weight:900; color:{priority_color};">{r['priority_score']:.2f}</div>
                  <div style="font-size:11px; color:#9ca3af; text-transform:uppercase;">Priority</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # All Leads section
    st.markdown("### 📋 All Leads (expand a card to edit / change status)")
    st.markdown("<em>Expand a lead to edit details, change status, upload invoice when awarded, and create estimates.</em>", unsafe_allow_html=True)

    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()
    for lead in leads:
        est_val_display = f"${lead.estimated_value:,.0f}" if lead.estimated_value else "$0"
        card_title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'Unknown'} — {est_val_display}"
        with st.expander(card_title, expanded=False):
            colA, colB = st.columns([3,1])
            with colA:
                st.write(f"**Source:** {lead.source or '—'}   |   **Assigned:** {lead.assigned_to or '—'}")
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
                    sla_status_html = "<div style='color:#ef4444;font-weight:700;'>❗ OVERDUE</div>"
                else:
                    hours = int(remaining.total_seconds() // 3600)
                    mins = int((remaining.total_seconds() % 3600) // 60)
                    sla_status_html = f"<div style='color:#ef4444;font-weight:700;'>⏳ {hours}h {mins}m</div>"
                st.markdown(f"<div style='text-align:right;'><div style='display:inline-block; padding:6px 12px; border-radius:20px; background:{stage_colors.get(lead.status,'#000')}22; color:{stage_colors.get(lead.status,'#000')}; font-weight:700;'>{lead.status}</div><div style='margin-top:12px;'>{sla_status_html}</div></div>", unsafe_allow_html=True)

            st.markdown("---")
            qc1, qc2, qc3, qc4 = st.columns([1,1,1,4])
            phone = (lead.contact_phone or "").strip()
            email = (lead.contact_email or "").strip()
            if phone:
                with qc1:
                    st.markdown(f"<a href='tel:{phone}'><button style='padding:8px 12px; border-radius:8px; background:#111; color:#fff;'>📞 Call</button></a>", unsafe_allow_html=True)
                with qc2:
                    wa_number = phone.lstrip("+").replace(" ", "").replace("-", "")
                    wa_link = f"https://wa.me/{wa_number}?text=Hi%2C%20following%20up%20on%20your%20restoration%20request."
                    st.markdown(f"<a href='{wa_link}' target='_blank'><button style='padding:8px 12px; border-radius:8px; background:#111; color:#fff;'>💬 WhatsApp</button></a>", unsafe_allow_html=True)
            else:
                qc1.write(" "); qc2.write(" ")
            if email:
                with qc3:
                    st.markdown(f"<a href='mailto:{email}?subject=Follow%20up'><button style='padding:8px 12px; border-radius:8px; background:#111; color:#fff;'>✉️ Email</button></a>", unsafe_allow_html=True)
            else:
                qc3.write(" ")
            qc4.write("")

            st.markdown("---")

            with st.form(f"update_lead_{lead.id}"):
                st.markdown("#### Update Lead")
                u1, u2 = st.columns(2)
                with u1:
                    new_status = st.selectbox("Status", LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status) if lead.status in LeadStatus.ALL else 0, key=f"status_{lead.id}")
                    new_assigned = st.text_input("Assigned to", value=lead.assigned_to or "", key=f"assign_{lead.id}")
                    new_contacted = st.checkbox("Contacted", value=bool(lead.contacted), key=f"contacted_{lead.id}")
                with u2:
                    insp_sched = st.checkbox("Inspection Scheduled", value=bool(lead.inspection_scheduled), key=f"insp_sched_{lead.id}")
                    insp_comp = st.checkbox("Inspection Completed", value=bool(lead.inspection_completed), key=f"insp_comp_{lead.id}")
                    est_sub = st.checkbox("Estimate Submitted", value=bool(lead.estimate_submitted), key=f"est_sub_{lead.id}")
                new_notes = st.text_area("Notes", value=lead.notes or "", key=f"notes_{lead.id}")
                new_est_val = st.number_input("Job Value Estimate (USD)", value=float(lead.estimated_value or 0.0), min_value=0.0, step=100.0, key=f"estval_{lead.id}")

                awarded_invoice_file = None
                award_comment = None
                lost_comment = None
                if new_status == LeadStatus.AWARDED:
                    st.markdown("**Award details**")
                    award_comment = st.text_area("Award comment", key=f"award_comment_{lead.id}")
                    awarded_invoice_file = st.file_uploader("Upload Invoice File (optional) — only for Awarded", type=["pdf","jpg","jpeg","png","xlsx","csv"], key=f"award_inv_{lead.id}")
                elif new_status == LeadStatus.LOST:
                    st.markdown("**Lost details**")
                    lost_comment = st.text_area("Lost comment", key=f"lost_comment_{lead.id}")

                if st.form_submit_button("💾 Update Lead"):
                    try:
                        db_s = get_session()
                        db_lead = db_s.query(Lead).filter(Lead.id == lead.id).first()
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
                            if new_status == LeadStatus.AWARDED:
                                db_lead.awarded_date = datetime.utcnow()
                                db_lead.awarded_comment = award_comment
                                if awarded_invoice_file is not None:
                                    path = save_uploaded_file(awarded_invoice_file, prefix=f"lead_{db_lead.id}_inv")
                                    db_lead.awarded_invoice = path
                            if new_status == LeadStatus.LOST:
                                db_lead.lost_date = datetime.utcnow()
                                db_lead.lost_comment = lost_comment
                            db_s.add(db_lead); db_s.commit(); db_s.close()
                            st.success(f"Lead #{db_lead.id} updated.")
                        else:
                            st.error("Lead not found.")
                    except Exception as e:
                        st.error(f"Failed to update lead: {e}")
                        st.write(traceback.format_exc())

    s.close()

# ---------------------------
# Page: Analytics & SLA
# ---------------------------
elif page == "Analytics & SLA":
    st.header("📈 Analytics — SLA & Trends")
    st.markdown("<em>Date-filtered SLA overdue trends and pipeline donut.</em>", unsafe_allow_html=True)

    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads to analyze. Add some leads first.")
    else:
        min_date = df["created_at"].min().date()
        max_date = df["created_at"].max().date()
        col_start, col_end = st.columns(2)
        start_date = col_start.date_input("Start date", min_value=min_date, value=min_date)
        end_date = col_end.date_input("End date", min_value=start_date, value=max_date)
        df_range = leads_df(s, start_date=start_date, end_date=end_date)

        # pipeline donut
        st.markdown("#### Pipeline Stages (donut)")
        stage_counts = df_range["status"].value_counts().reindex(LeadStatus.ALL, fill_value=0)
        pie_df2 = pd.DataFrame({"status": stage_counts.index, "count": stage_counts.values})
        if pie_df2["count"].sum() == 0:
            st.info("No leads in selected range.")
        else:
            if px:
                fig = px.pie(pie_df2, names="status", values="count", hole=0.45, color="status", color_discrete_map=stage_colors)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.table(pie_df2)

        # SLA overdue time series (last 30 days)
        st.markdown("---")
        st.subheader("SLA / Overdue Leads")
        st.markdown("<em>Trend of SLA overdue counts (last 30 days) and current overdue leads table.</em>", unsafe_allow_html=True)

        today = datetime.utcnow().date()
        days_back = 30
        ts_rows = []
        for d in range(days_back, -1, -1):
            day = today - pd.Timedelta(days=d)
            day_start = datetime.combine(day, datetime.min.time())
            day_end = datetime.combine(day, datetime.max.time())
            overdue_count = 0
            for _, row in df_range.iterrows():
                sla_entered = row.get("sla_entered_at") or row.get("created_at")
                try:
                    if isinstance(sla_entered, str):
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
                fig = px.line(ts_df, x="date", y="overdue_count", markers=True)
                fig.update_layout(margin=dict(t=6,b=6))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(ts_df)

        # Overdue table (current)
        overdue_rows = []
        for _, row in df_range.iterrows():
            sla_entered = row.get("sla_entered_at") or row.get("created_at")
            try:
                if isinstance(sla_entered, str):
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
    s.close()

# ---------------------------
# Page: CPA & ROI
# ---------------------------
elif page == "CPA & ROI":
    st.header("💰 CPA & ROI Dashboard")
    st.markdown("<em>Track Total Marketing Spend, Conversions, CPA and ROI. Numbers shown in colored Comfortaa font.</em>", unsafe_allow_html=True)

    s = get_session()
    df = leads_df(s)

    # quick date selector
    col1, col2 = st.columns(2)
    today = datetime.utcnow().date()
    start_date = col1.date_input("Start date", value=today)
    end_date = col2.date_input("End date", value=today)

    df_view = df if df.empty else leads_df(s, start_date=start_date, end_date=end_date)

    total_spend = float(df_view["cost_to_acquire"].fillna(0).sum()) if not df_view.empty else 0.0
    conversions = int(df_view[df_view["status"] == LeadStatus.AWARDED].shape[0]) if not df_view.empty else 0
    cpa = (total_spend / conversions) if conversions else 0.0
    revenue = float(df_view[df_view["status"] == LeadStatus.AWARDED]["estimated_value"].fillna(0).sum()) if not df_view.empty else 0.0
    roi = revenue - total_spend
    roi_pct = (roi / total_spend * 100) if total_spend else 0.0

    # display numbers with colored fonts only (no badges)
    st.markdown(f"<div style='font-family:Comfortaa; font-size:18px;'>💰 Total Marketing Spend: <span style='color:#ef4444; font-weight:800; font-size:28px;'>${total_spend:,.2f}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:Comfortaa; font-size:18px;'>✅ Conversions (Won): <span style='color:#2563eb; font-weight:800; font-size:28px;'>{conversions}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:Comfortaa; font-size:18px;'>🎯 CPA: <span style='color:#f97316; font-weight:800; font-size:28px;'>${cpa:,.2f}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:Comfortaa; font-size:18px;'>📈 ROI: <span style='color:#22c55e; font-weight:800; font-size:28px;'>${roi:,.2f} ({roi_pct:.1f}%)</span></div>", unsafe_allow_html=True)

    # Chart: Total Marketing Spend vs Conversions
    if not df_view.empty:
        chart_df = pd.DataFrame({
            "date": df_view["created_at"].dt.date,
            "spend": df_view["cost_to_acquire"],
            "won": df_view["status"].apply(lambda s: 1 if s == LeadStatus.AWARDED else 0)
        })
        agg = chart_df.groupby("date").agg({"spend": "sum", "won": "sum"}).reset_index()
        if px:
            fig = px.line(agg, x="date", y=["spend", "won"], markers=True)
            fig.update_layout(yaxis_title="Value / Conversions")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(agg)

    s.close()

# ---------------------------
# Page: ML (internal info only)
# ---------------------------
elif page == "ML (Internal)":
    st.header("🧠 ML — Internal (no user tuning)")
    st.markdown("<em>Model runs internally. Auto-trains when there are enough labeled leads. Predictions are stored on leads.</em>", unsafe_allow_html=True)
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn not installed; ML unavailable.")
    else:
        s = get_session()
        df = leads_df(s)
        labeled = df[df["status"].isin([LeadStatus.AWARDED, LeadStatus.LOST])].shape[0]
        st.write(f"Labeled leads (awarded/lost): {labeled}")
        if labeled < ML_MIN_LABELS:
            st.info(f"Need at least {ML_MIN_LABELS} labeled leads to auto-train.")
        else:
            if st.button("Run internal training now"):
                model, msg = internal_ml_autorun := internal_ml_autorun if False else internal_ml_autorun  # placeholder to avoid linter
                # call the function properly
                model_result = internal_ml_autorun()
                if model_result is None:
                    st.error("Training failed or unavailable.")
                else:
                    st.success("Training completed (internal)")
        s.close()

# ---------------------------
# Page: Exports
# ---------------------------
elif page == "Exports":
    st.header("📤 Export data")
    s = get_session()
    df_leads = leads_df(s)
    if df_leads.empty:
        st.info("No leads yet to export.")
    else:
        csv = df_leads.to_csv(index=False).encode("utf-8")
        st.download_button("Download leads.csv", csv, file_name="leads.csv", mime="text/csv")
    s.close()

# ---------------------------
# Background: SLA alerts (simple print) and periodic ML autorun
# ---------------------------

def alert_worker(interval=60):
    while True:
        s = get_session()
        df = leads_df(s)
        for _, row in df.iterrows():
            sla_secs, overdue = calculate_remaining_sla(row.get("sla_entered_at"), row.get("sla_hours"))
            if overdue and row.get("status") not in (LeadStatus.AWARDED, LeadStatus.LOST):
                # In production you'd send email/Slack; here we print to logs
                print(f"ALERT: Lead #{row.get('id')} is overdue (status={row.get('status')})")
        s.close()
        # attempt ML autorun periodically
        try:
            internal_ml_autorun()
        except Exception:
            pass
        time.sleep(interval)

# start background thread daemon
bg_thread = threading.Thread(target=alert_worker, args=(120,), daemon=True)
bg_thread.start()

# ---------------------------
# End of file
# ---------------------------
