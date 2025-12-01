# titan.py
"""
TITAN Backend - Single-file fixed app.
Fixes: DataFrame 'df' not defined, fragile .lower() calls, undefined filters.
Includes pages: ML Model, Evaluation Dashboard, AI Recommendations, Pipeline Report, Exports.
Use: streamlit run titan.py
"""

import os
import io
import base64
import traceback
from datetime import datetime, timedelta, date

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, inspect
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import OperationalError

# -------------------------
# CONFIG
# -------------------------
APP_TITLE = "TITAN — Backend Admin (Fixed)"
DB_FILE = "titan_backend.db"
MODEL_FILE = "titan_model.joblib"
PIPELINE_STAGES = [
    "New", "Contacted", "Inspection Scheduled", "Inspection Completed",
    "Estimate Sent", "Qualified", "Won", "Lost"
]
DEFAULT_SLA_HOURS = 72

# -------------------------
# DB setup (SQLite)
# -------------------------
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
    role = Column(String, default="Viewer")
    created_at = Column(DateTime, default=datetime.utcnow)

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String, default="Other")
    source_details = Column(String, nullable=True)
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
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    estimate_submitted = Column(Boolean, default=False)
    estimate_submitted_at = Column(DateTime, nullable=True)
    awarded_date = Column(DateTime, nullable=True)
    lost_date = Column(DateTime, nullable=True)
    qualified = Column(Boolean, default=False)
    ad_cost = Column(Float, default=0.0)
    converted = Column(Boolean, default=False)
    score = Column(Float, nullable=True)
    win_prob = Column(Float, nullable=True)  # optional ML field

class LeadHistory(Base):
    __tablename__ = "lead_history"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, nullable=False)
    changed_by = Column(String, nullable=True)
    field = Column(String, nullable=True)
    old_value = Column(String, nullable=True)
    new_value = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

# create tables if missing
Base.metadata.create_all(bind=engine)

# -------------------------
# DB helpers
# -------------------------
def get_session():
    return SessionLocal()

def leads_to_df(start_date=None, end_date=None):
    s = get_session()
    try:
        rows = s.query(Lead).order_by(Lead.created_at.desc()).all()
        data = []
        for r in rows:
            data.append({
                "id": r.id,
                "lead_id": r.lead_id,
                "created_at": r.created_at,
                "source": r.source or "Other",
                "source_details": getattr(r, "source_details", None),
                "contact_name": getattr(r, "contact_name", None),
                "contact_phone": getattr(r, "contact_phone", None),
                "contact_email": getattr(r, "contact_email", None),
                "property_address": getattr(r, "property_address", None),
                "damage_type": getattr(r, "damage_type", None),
                "assigned_to": getattr(r, "assigned_to", None),
                "notes": r.notes,
                "estimated_value": float(r.estimated_value or 0.0),
                "stage": r.stage or "New",
                "sla_hours": int(r.sla_hours or DEFAULT_SLA_HOURS),
                "sla_entered_at": r.sla_entered_at or r.created_at,
                "contacted": bool(r.contacted),
                "inspection_scheduled": bool(r.inspection_scheduled),
                "inspection_scheduled_at": r.inspection_scheduled_at,
                "inspection_completed": bool(r.inspection_completed),
                "estimate_submitted": bool(r.estimate_submitted),
                "awarded_date": r.awarded_date,
                "lost_date": r.lost_date,
                "qualified": bool(r.qualified),
                "ad_cost": float(r.ad_cost or 0.0),
                "converted": bool(r.converted),
                "score": float(r.score) if r.score is not None else None,
                "win_prob": float(r.win_prob) if getattr(r, "win_prob", None) is not None else None
            })
        df = pd.DataFrame(data)
        if df.empty:
            cols = ["id","lead_id","created_at","source","source_details","contact_name","contact_phone","contact_email",
                    "property_address","damage_type","assigned_to","notes","estimated_value","stage","sla_hours","sla_entered_at",
                    "contacted","inspection_scheduled","inspection_scheduled_at","inspection_completed","estimate_submitted",
                    "awarded_date","lost_date","qualified","ad_cost","converted","score","win_prob"]
            return pd.DataFrame(columns=cols)
        if start_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            df = df[df["created_at"] >= start_dt]
        if end_date:
            end_dt = datetime.combine(end_date, datetime.max.time())
            df = df[df["created_at"] <= end_dt]
        return df.reset_index(drop=True)
    finally:
        s.close()

def upsert_lead_record(payload: dict, actor="admin"):
    s = get_session()
    try:
        lead = s.query(Lead).filter(Lead.lead_id == payload.get("lead_id")).first()
        if lead is None:
            lead = Lead(
                lead_id=payload.get("lead_id"),
                created_at=payload.get("created_at", datetime.utcnow()),
                source=payload.get("source"),
                source_details=payload.get("source_details"),
                contact_name=payload.get("contact_name"),
                contact_phone=payload.get("contact_phone"),
                contact_email=payload.get("contact_email"),
                property_address=payload.get("property_address"),
                damage_type=payload.get("damage_type"),
                assigned_to=payload.get("assigned_to"),
                notes=payload.get("notes"),
                estimated_value=float(payload.get("estimated_value") or 0.0),
                stage=payload.get("stage") or "New",
                sla_hours=int(payload.get("sla_hours") or DEFAULT_SLA_HOURS),
                sla_entered_at=payload.get("sla_entered_at") or datetime.utcnow(),
                ad_cost=float(payload.get("ad_cost") or 0.0),
                converted=bool(payload.get("converted") or False),
                score=payload.get("score"),
                win_prob=payload.get("win_prob")
            )
            s.add(lead)
            s.commit()
            s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor, field="create", old_value=None, new_value=str(lead.stage)))
            s.commit()
            return lead.lead_id
        else:
            changed = []
            for key in ["source","source_details","contact_name","contact_phone","contact_email","property_address",
                        "damage_type","assigned_to","notes","estimated_value","stage","sla_hours","sla_entered_at","ad_cost","converted","score","win_prob"]:
                if key in payload:
                    new = payload.get(key)
                    old = getattr(lead, key)
                    if key in ("estimated_value","ad_cost"):
                        try:
                            new_val = float(new or 0.0)
                        except Exception:
                            new_val = old
                    elif key in ("sla_hours",):
                        try:
                            new_val = int(new or old)
                        except Exception:
                            new_val = old
                    elif key in ("converted",):
                        new_val = bool(new)
                    else:
                        new_val = new
                    if new_val is not None and old != new_val:
                        changed.append((key, old, new_val))
                        setattr(lead, key, new_val)
            s.add(lead)
            for (f, old, new) in changed:
                s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor, field=f, old_value=str(old), new_value=str(new)))
            s.commit()
            return lead.lead_id
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def add_user(username: str, full_name: str = "", role: str = "Viewer"):
    s = get_session()
    try:
        existing = s.query(User).filter(User.username == username).first()
        if existing:
            existing.full_name = full_name
            existing.role = role
            s.add(existing); s.commit()
            return existing.username
        u = User(username=username, full_name=full_name, role=role)
        s.add(u); s.commit()
        return u.username
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

# -------------------------
# ML helpers (simple)
# -------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_internal_model():
    df = leads_to_df()
    if df.empty or df["converted"].nunique() < 2:
        return None, "Not enough data"
    df2 = df.copy()
    df2["age_days"] = (datetime.utcnow() - df2["created_at"]).dt.days
    X = pd.get_dummies(df2[["source","stage"]].astype(str), drop_first=False)
    X["ad_cost"] = df2["ad_cost"]
    X["estimated_value"] = df2["estimated_value"]
    X["age_days"] = df2["age_days"]
    y = df2["converted"].astype(int)
    X = X.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_FILE)
    return acc, "trained"

def load_internal_model():
    if not os.path.exists(MODEL_FILE):
        return None, None
    try:
        obj = joblib.load(MODEL_FILE)
        return obj.get("model"), obj.get("columns")
    except Exception:
        return None, None

def score_dataframe(df, model, cols):
    if model is None or df.empty:
        df["score"] = np.nan
        return df
    df2 = df.copy()
    df2["age_days"] = (datetime.utcnow() - df2["created_at"]).dt.days
    X = pd.get_dummies(df2[["source","stage"]].astype(str), drop_first=False)
    X["ad_cost"] = df2["ad_cost"]
    X["estimated_value"] = df2["estimated_value"]
    X["age_days"] = df2["age_days"]
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    X = X[cols].fillna(0)
    try:
        df["score"] = model.predict_proba(X)[:,1]
    except Exception:
        df["score"] = model.predict(X)
    return df

# -------------------------
# Priority and SLA utilities
# -------------------------
def calculate_remaining_sla(sla_entered_at, sla_hours):
    try:
        if sla_entered_at is None:
            sla_entered_at = datetime.utcnow()
        if isinstance(sla_entered_at, str):
            sla_entered_at = datetime.fromisoformat(sla_entered_at)
        deadline = sla_entered_at + timedelta(hours=int(sla_hours or DEFAULT_SLA_HOURS))
        remain = deadline - datetime.utcnow()
        return max(remain.total_seconds(), 0.0), (remain.total_seconds() <= 0)
    except Exception:
        return float("inf"), False

def compute_priority_for_lead_row(row, weights=None, ml_prob=None):
    # simple composition: use model prob, normalized value, SLA urgency
    if weights is None:
        weights = {"score_w":0.6, "value_w":0.3, "sla_w":0.1, "value_baseline":5000.0}
    try:
        s = float(ml_prob or row.get("score") or 0.0)
    except Exception:
        s = 0.0
    try:
        val = float(row.get("estimated_value") or 0.0)
        vnorm = min(1.0, val / max(1.0, weights["value_baseline"]))
    except Exception:
        vnorm = 0.0
    sla_score = 0.0
    try:
        sla_entered = row.get("sla_entered_at") or row.get("created_at")
        if sla_entered is not None:
            if isinstance(sla_entered, str):
                sla_entered = datetime.fromisoformat(sla_entered)
            time_left_h = max((sla_entered + timedelta(hours=row.get("sla_hours") or DEFAULT_SLA_HOURS) - datetime.utcnow()).total_seconds()/3600.0, 0.0)
            sla_score = max(0.0, (72.0 - min(time_left_h,72.0)) / 72.0)
    except Exception:
        sla_score = 0.0
    total = s*weights["score_w"] + vnorm*weights["value_w"] + sla_score*weights["sla_w"]
    return max(0.0, min(1.0, total))

# -------------------------
# UI & Streamlit setup
# -------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown("""
<style>
body, .stApp { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial; }
.kpi-row-gap { height:16px; }
.priority-card { background:#000; color:#fff; border-radius:12px; padding:12px; margin-bottom:12px; }
.small-muted { color:#94a3b8; }
</style>
""", unsafe_allow_html=True)

# Sidebar / navigation
with st.sidebar:
    st.title("TITAN Admin")
    page = st.radio("Navigate", ["Pipeline Report","ML Model","Evaluation Dashboard","AI Recommendations","Exports","Settings"], index=0)
    st.markdown("---")
    st.markdown("Date range")
    quick = st.selectbox("Quick", ["Today","Last 7 days","Last 30 days","All","Custom"], index=4)
    if quick == "Today":
        st.session_state.start_date = date.today()
        st.session_state.end_date = date.today()
    elif quick == "Last 7 days":
        st.session_state.start_date = date.today() - timedelta(days=6)
        st.session_state.end_date = date.today()
    elif quick == "Last 30 days":
        st.session_state.start_date = date.today() - timedelta(days=29)
        st.session_state.end_date = date.today()
    elif quick == "All":
        st.session_state.start_date = None
        st.session_state.end_date = None
    else:
        sd, ed = st.date_input("Start / End", [date.today() - timedelta(days=29), date.today()])
        st.session_state.start_date = sd
        st.session_state.end_date = ed

    if st.button("Refresh"):
        st.experimental_rerun()

# get date filters
start_dt = st.session_state.get("start_date", None)
end_dt = st.session_state.get("end_date", None)

# load leads safely
try:
    leads_df = leads_to_df(start_dt, end_dt)
except OperationalError as e:
    st.error("Database error: " + str(e))
    st.stop()

# load model if exists and score
model, model_cols = load_internal_model()
if model is not None and not leads_df.empty:
    try:
        leads_df = score_dataframe(leads_df.copy(), model, model_cols)
    except Exception:
        pass

# ensure df variable used in older code is always present
df = leads_df.copy() if not leads_df.empty else pd.DataFrame()

# ---------- Topbar helper ----------
def render_topbar():
    # calculate overdue count
    overdue_count = 0
    for _, r in df.iterrows():
        _, overdue = calculate_remaining_sla(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
        if overdue and r.get("stage") not in ("Won","Lost"):
            overdue_count += 1
    left, right = st.columns([8,2])
    with left:
        st.markdown("")  # spacer
    with right:
        if start_dt and end_dt:
            label = f"{start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}"
        else:
            label = datetime.utcnow().strftime("%Y-%m-%d")
        st.markdown(f"<div style='text-align:right;'><small class='small-muted'>{label}</small><br>🔔 <span style='background:#ef4444;color:white;border-radius:12px;padding:4px 8px;font-weight:700;'>{overdue_count}</span></div>", unsafe_allow_html=True)

# -------------------------
# PAGE: Pipeline Report
# -------------------------
def page_pipeline_report():
    render_topbar()
    st.header("Pipeline Report")
    st.markdown("High-level KPIs and Top-5 Priority Leads (design applied).")

    # KPIs
    df_local = df.copy()
    if df_local.empty:
        st.info("No leads yet. Use Exports > Import or Lead Capture to add leads.")
        return

    total_leads = len(df_local)
    won = df_local[df_local["stage"] == "Won"].shape[0] if "stage" in df_local else 0
    lost = df_local[df_local["stage"] == "Lost"].shape[0] if "stage" in df_local else 0
    closed = won + lost
    conversion_rate = (won / closed * 100) if closed else 0.0
    pipeline_value = float(df_local["estimated_value"].sum()) if "estimated_value" in df_local else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Leads", total_leads)
    c2.metric("Won", won)
    c3.metric("Lost", lost)
    c4.metric("Conversion Rate", f"{conversion_rate:.1f}%")

    st.markdown("<div class='kpi-row-gap'></div>", unsafe_allow_html=True)

    # Top 5 priority leads — robust implementation using the card design provided
    st.markdown("### TOP 5 PRIORITY LEADS")
    st.markdown("<em>Highest urgency leads by priority score (0–1). Address these first.</em>", unsafe_allow_html=True)

    priority_list = []
    weights = st.session_state.get("weights", {"score_w":0.6,"value_w":0.3,"sla_w":0.1,"value_baseline":5000.0})
    for _, row in df_local.iterrows():
        try:
            ml_prob = float(row.get("win_prob")) if row.get("win_prob") is not None else None
        except Exception:
            ml_prob = None
        try:
            score = compute_priority_for_lead_row(row, weights, ml_prob=ml_prob)
        except Exception:
            score = 0.0
        sla_sec, overdue = calculate_remaining_sla(row.get("sla_entered_at") or row.get("created_at"), row.get("sla_hours"))
        time_left_h = sla_sec/3600.0 if sla_sec not in (None, float("inf")) else 9999.0
        priority_list.append({
            "id": int(row.get("id", 0)),
            "contact_name": row.get("contact_name") or "No name",
            "estimated_value": float(row.get("estimated_value") or 0.0),
            "time_left_hours": time_left_h,
            "priority_score": score,
            "status": row.get("stage") or row.get("status") or "New",
            "sla_overdue": overdue,
            "conversion_prob": ml_prob,
            "damage_type": row.get("damage_type", "Unknown")
        })

    pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

    if pr_df.empty:
        st.info("No priority leads to display.")
    else:
        # display top 5 with card design
        stage_colors = {
            "New": "#0ea5a4", "Contacted": "#2563eb", "Inspection Scheduled": "#f97316",
            "Inspection Completed": "#a855f7", "Estimate Sent": "#6d28d9", "Qualified": "#22c55e",
            "Won": "#15803d", "Lost": "#6b7280"
        }
        for _, r in pr_df.head(5).iterrows():
            score = r["priority_score"]
            status = r.get("status")
            status_color = stage_colors.get(status, "#666666")
            if score >= 0.7:
                priority_color = "#ef4444"; priority_label = "🔴 CRITICAL"
            elif score >= 0.45:
                priority_color = "#f97316"; priority_label = "🟠 HIGH"
            else:
                priority_color = "#22c55e"; priority_label = "🟢 NORMAL"
            if r["sla_overdue"]:
                sla_html = "<span style='color:#ef4444;font-weight:700;'>❗ OVERDUE</span>"
            else:
                hours_left = int(r["time_left_hours"])
                mins_left = int((r["time_left_hours"]*60) % 60)
                sla_html = f"<span style='color:#ef4444;font-weight:700;'>⏳ {hours_left}h {mins_left}m left</span>"
            conv_html = ""
            if r["conversion_prob"] is not None:
                conv_pct = r["conversion_prob"] * 100
                conv_color = "#22c55e" if conv_pct > 70 else ("#f97316" if conv_pct > 40 else "#ef4444")
                conv_html = f"<span style='color:{conv_color};font-weight:600;margin-left:12px;'>📊 {conv_pct:.0f}% Win Prob</span>"

            st.markdown(f"""
<div style="background:#000;padding:12px;border-radius:12px;margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div style="flex:1;">
      <div style="margin-bottom:6px;">
        <span style="color:{priority_color};font-weight:800;">{priority_label}</span>
        <span style="display:inline-block;padding:6px 12px;border-radius:18px;font-size:12px;font-weight:600;margin-left:8px;background:{status_color}22;color:{status_color};">{status}</span>
      </div>
      <div style="font-size:20px;font-weight:900;color:#FFFFFF;">
        #{int(r['id'])} — {r['contact_name']}
      </div>
      <div style="font-size:13px;color:#cbd5e1;margin-top:6px;">
        {r['damage_type'].title()} | Est: <span style="font-weight:800;">${r['estimated_value']:,.0f}</span>
      </div>
      <div style="font-size:13px;margin-top:8px;color:#94a3b8;">
        {sla_html} {conv_html}
      </div>
    </div>
    <div style="text-align:right;padding-left:18px;">
      <div style="font-size:26px;font-weight:900;color:{priority_color};">{r['priority_score']:.2f}</div>
      <div style="font-size:11px;text-transform:uppercase;color:#94a3b8;">Priority</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")

    # show a small table of pipeline stages
    st.subheader("Pipeline Stages Distribution")
    stage_counts = df_local["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0)
    stage_df = pd.DataFrame({"stage": stage_counts.index, "count": stage_counts.values})
    fig = px.pie(stage_df, names="stage", values="count", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# PAGE: ML Model
# -------------------------
def page_ml_model():
    render_topbar()
    st.header("ML Model — Training & Scoring")
    st.markdown("Train internal model and score leads. (Internal use only)")

    if st.button("Train model"):
        with st.spinner("Training..."):
            try:
                acc, msg = train_internal_model()
                if acc is None:
                    st.error("Training aborted: " + str(msg))
                else:
                    st.success(f"Model trained (approx accuracy: {acc:.3f})")
            except Exception as e:
                st.error("Training failed: " + str(e))
                st.write(traceback.format_exc())

    model, cols = load_internal_model()
    if model:
        st.success("Model available")
        if st.button("Score leads and persist"):
            df_local = leads_to_df()
            scored = score_dataframe(df_local.copy(), model, cols)
            s = get_session()
            try:
                for _, r in scored.iterrows():
                    lead = s.query(Lead).filter(Lead.lead_id == r["lead_id"]).first()
                    if lead:
                        lead.score = float(r["score"])
                        s.add(lead)
                s.commit()
                st.success("Scores saved to DB")
            except Exception as e:
                s.rollback()
                st.error("Failed to persist: " + str(e))
            finally:
                s.close()

# -------------------------
# PAGE: Evaluation Dashboard
# -------------------------
def page_evaluation_dashboard():
    render_topbar()
    st.header("Evaluation Dashboard")
    st.markdown("High level model + pipeline evaluation metrics")

    df_local = df.copy()
    if df_local.empty:
        st.info("No leads to evaluate.")
        return

    # example: distribution of scores
    if "score" in df_local:
        st.subheader("Score Distribution")
        sd = df_local["score"].dropna()
        if sd.empty:
            st.info("No scores yet.")
        else:
            fig = px.histogram(sd, nbins=20, title="Lead Score Distribution")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top predicted conversions (by score)")
    if "score" in df_local:
        top = df_local.sort_values("score", ascending=False).head(10)
        st.dataframe(top[["lead_id","contact_name","source","stage","estimated_value","score"]])

# -------------------------
# PAGE: AI Recommendations
# -------------------------
def page_ai_recommendations():
    render_topbar()
    st.header("AI Recommendations")
    st.markdown("Simple action recommendations based on priority and SLA")

    df_local = df.copy()
    if df_local.empty:
        st.info("No leads.")
        return

    # naive rules
    recs = []
    for _, r in df_local.iterrows():
        _, overdue = calculate_remaining_sla(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
        score = float(r.get("score") or 0.0)
        if overdue and r.get("stage") not in ("Won","Lost"):
            recs.append((r["lead_id"], "Contact now (SLA overdue)"))
        elif score > 0.7 and r.get("stage") not in ("Won","Lost"):
            recs.append((r["lead_id"], "High conversion probability — follow up and assign estimator"))
        elif score > 0.4:
            recs.append((r["lead_id"], "Nurture lead (email/sequence)"))
    if recs:
        rec_df = pd.DataFrame(recs, columns=["lead_id","recommendation"])
        st.dataframe(rec_df)
    else:
        st.info("No recommendations at this time.")

# -------------------------
# PAGE: Exports & Imports
# -------------------------
def page_exports():
    render_topbar()
    st.header("Exports & Imports")
    st.markdown("Export current leads or import via CSV/XLSX (upsert by lead_id).")

    df_local = leads_to_df()
    if not df_local.empty:
        towrite = io.BytesIO()
        df_local.to_excel(towrite, index=False, engine="openpyxl")
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"
        st.markdown(f'<a href="{href}" download="leads_export.xlsx">Download leads_export.xlsx</a>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload leads CSV/XLSX", type=["csv","xlsx"])
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_in = pd.read_csv(uploaded)
            else:
                df_in = pd.read_excel(uploaded)
            if "lead_id" not in df_in.columns:
                st.error("File must include 'lead_id' column")
            else:
                count = 0
                for _, r in df_in.iterrows():
                    try:
                        upsert_lead_record({
                            "lead_id": str(r["lead_id"]),
                            "created_at": pd.to_datetime(r.get("created_at")) if r.get("created_at") is not None else datetime.utcnow(),
                            "source": r.get("source"),
                            "contact_name": r.get("contact_name"),
                            "contact_phone": r.get("contact_phone"),
                            "contact_email": r.get("contact_email"),
                            "property_address": r.get("property_address"),
                            "damage_type": r.get("damage_type"),
                            "assigned_to": r.get("assigned_to"),
                            "notes": r.get("notes"),
                            "estimated_value": float(r.get("estimated_value") or 0.0),
                            "ad_cost": float(r.get("ad_cost") or 0.0),
                            "stage": r.get("stage") or "New",
                            "converted": bool(r.get("converted") or False)
                        }, actor="import")
                        count += 1
                    except Exception:
                        continue
                st.success(f"Imported/Upserted {count} rows.")
        except Exception as e:
            st.error("Import failed: " + str(e))

# -------------------------
# PAGE: Settings (basic)
# -------------------------
def page_settings():
    render_topbar()
    st.header("Settings & Users")
    st.subheader("Add / Update user")
    with st.form("user_form"):
        username = st.text_input("Username")
        fullname = st.text_input("Full name")
        role = st.selectbox("Role", ["Admin","Estimator","Adjuster","Tech","Viewer"])
        if st.form_submit_button("Save user"):
            if not username:
                st.error("Username required")
            else:
                add_user(username.strip(), fullname.strip(), role)
                st.success("User saved")

# -------------------------
# Router mapping and execution
# -------------------------
PAGE_MAP = {
    "Pipeline Report": page_pipeline_report,
    "ML Model": page_ml_model,
    "Evaluation Dashboard": page_evaluation_dashboard,
    "AI Recommendations": page_ai_recommendations,
    "Exports": page_exports,
    "Settings": page_settings
}

selected_page = PAGE_MAP.get(page)
if selected_page:
    try:
        selected_page()
    except Exception as e:
        st.error("Page crashed: " + str(e))
        st.code(traceback.format_exc())
else:
    st.info("Page not implemented yet.")

# Footer
st.markdown("---")
st.markdown("<small class='small-muted'>TITAN Backend — fixed runtime issues; keep original logic. Contact me if you want selective diffs instead of full replacement.</small>", unsafe_allow_html=True)
