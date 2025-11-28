import os
import threading
from datetime import datetime, timedelta, date

import streamlit as st
import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Boolean, DateTime, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import inspect, text

# -----------------
# CONFIG
# -----------------
DB_FILE = "project_x_pipeline.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"
MODEL_FILE = "lead_win_model.pkl"

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


# -----------------
# MODELS
# -----------------
class LeadStatus:
    NEW = "New"
    CONTACTED = "Contacted"
    INSPECTION_SCHEDULED = "Inspection Scheduled"
    INSPECTION_COMPLETED = "Inspection Completed"
    ESTIMATE_SUBMITTED = "Estimate Submitted"
    AWARDED = "Awarded"
    LOST = "Lost"
    ALL = [
        NEW, CONTACTED,
        INSPECTION_SCHEDULED, INSPECTION_COMPLETED,
        ESTIMATE_SUBMITTED, AWARDED, LOST
    ]


class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True)
    source = Column(String, default="Not set")
    source_details = Column(String, nullable=True)

    name = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)

    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)
    assigned_to = Column(String, nullable=True)
    notes = Column(Text, nullable=True)

    estimated_value = Column(Float, default=0.0)

    status = Column(String, default=LeadStatus.NEW)
    created_at = Column(DateTime, default=datetime.utcnow)

    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime, nullable=True)

    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_completed = Column(Boolean, default=False)
    estimate_submitted = Column(Boolean, default=False)

    awarded_date = Column(DateTime, nullable=True)
    lost_date = Column(DateTime, nullable=True)

    cost_to_acquire = Column(Float, default=0.0)
    qualified = Column(Boolean, default=False)
    predicted_prob = Column(Float, nullable=True)


# -----------------
# DB Init + Migration
# -----------------
def init_db():
    Base.metadata.create_all(bind=engine)
    insp = inspect(engine)
    cols = [c['name'] for c in insp.get_columns('leads')]
    with engine.connect() as conn:
        if "cost_to_acquire" not in cols:
            try: conn.execute(text("ALTER TABLE leads ADD COLUMN cost_to_acquire FLOAT")); 
            except: pass
        if "qualified" not in cols:
            try: conn.execute(text("ALTER TABLE leads ADD COLUMN qualified BOOL")); 
            except: pass
        if "predicted_prob" not in cols:
            try: conn.execute(text("ALTER TABLE leads ADD COLUMN predicted_prob FLOAT")); 
            except: pass
        if "sla_entered_at" not in cols:
            try: conn.execute(text("ALTER TABLE leads ADD COLUMN sla_entered_at DATETIME")); 
            except: pass


init_db()


def get_session():
    return SessionLocal()


# ---------------
# Lead Capture
# ---------------
def add_lead(session, **kwargs):
    cost = float(kwargs.get("cost_to_acquire") or 0.0)
    val = float(kwargs.get("estimated_value") or 0.0)
    lead = Lead(
        source=kwargs.get("source") or "Not set",
        source_details=kwargs.get("source_details"),
        name=kwargs.get("name"),
        phone=kwargs.get("phone"),
        email=kwargs.get("email"),

        property_address=kwargs.get("property_address"),
        damage_type=kwargs.get("damage_type"),
        assigned_to=kwargs.get("assigned_to"),
        notes=kwargs.get("notes"),
        estimated_value=val,

        sla_hours=int(kwargs.get("sla_hours") or 24),
        sla_entered_at=datetime.utcnow(),
        cost_to_acquire=cost,
        qualified=True if kwargs.get("qualified") else False
    )
    session.add(lead)
    session.commit()
    session.refresh(lead)
    return lead


# ---------------
# Read leads into DataFrame with date filter
# ---------------
def leads_df(session, start: date=None, end: date=None):
    rows = session.query(Lead).order_by(Lead.created_at.desc()).all()
    data = []
    for r in rows:
        data.append({
            "id": r.id,
            "source": r.source,
            "source_details": r.source_details,
            "name": r.name,
            "phone": r.phone,
            "email": r.email,
            "property_address": r.property_address,
            "damage_type": r.damage_type,
            "assigned_to": r.assigned_to,
            "notes": r.notes,
            "estimated_value": float(r.estimated_value or 0.0),
            "status": r.status,
            "created_at": r.created_at,
            "sla_hours": r.sla_hours,
            "sla_entered_at": r.sla_entered_at,
            "awarded_date": r.awarded_date,
            "lost_date": r.lost_date,
            "qualified": bool(r.qualified),
            "cost_to_acquire": float(r.cost_to_acquire or 0.0),
            "predicted_prob": r.predicted_prob,
            "contacted": bool(r.contacted),
            "inspection_scheduled": bool(r.inspection_scheduled),
            "inspection_completed": bool(r.inspection_completed),
            "estimate_submitted": bool(r.estimate_submitted)
        })
    df = pd.DataFrame(data)
    if start and end and not df.empty:
        sdt = datetime.combine(start, datetime.min.time())
        edt = datetime.combine(end, datetime.max.time())
        df = df[(df['created_at'] >= sdt) & (df['created_at'] <= edt)]
    return df


# ---------------
# SLA compute
# ---------------
def calculate_remaining_sla(sla_entered_at, sla_hours):
    if isinstance(sla_entered_at, str):
        try: sla_entered_at = datetime.fromisoformat(sla_entered_at)
        except: sla_entered_at = datetime.utcnow()
    entered = sla_entered_at or datetime.utcnow()
    deadline = entered + timedelta(hours=int(sla_hours or 24))
    remain = deadline - datetime.utcnow()
    return max(remain.total_seconds(), 0), remain.total_seconds() <= 0


###############################################################################
# ML (INTERNAL ONLY — silent autorun)
###############################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def auto_train_model(min_labeled_required=12):
    s = get_session()
    try:
        df = leads_df(s)
        labeled = df[df['status'].isin([LeadStatus.AWARDED, LeadStatus.LOST])]
        if labeled.empty or len(labeled) < min_labeled_required:
            return None

        numeric_cols = ["estimated_value", "sla_hours", "cost_to_acquire"]
        cat_cols = ["damage_type", "source", "assigned_to"]

        X = labeled[numeric_cols + cat_cols].fillna({
            "damage_type": "other", "source":"Not set", "assigned_to":"unassigned"
        })
        X[numeric_cols] = X[numeric_cols].fillna(0)
        y = (labeled['status'] == LeadStatus.AWARDED).astype(int)

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])

        pipe = Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=140, max_depth=6, random_state=42))
        ])

        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(Xa, ya)

        # Save model silently
        joblib.dump(pipe, MODEL_FILE)

        # Store probabilities
        probs = pipe.predict_proba(X)[:,1]
        for lid, p in zip(labeled['id'], probs):
            ld = s.query(Lead).filter(Lead.id==int(lid)).first()
            ld.predicted_prob = float(p)
            s.add(ld)
        s.commit()

        return pipe
    except Exception:
        return None
    finally:
        s.close()

# Trigger once internally at load time
ML_ENABLED = True
lead_model = auto_train_model()


###############################################################################
# NOTIFICATION HOOKS (UI level)
###############################################################################
def sla_notification_monitor():
    s = get_session()
    try:
        df = leads_df(s)
        overdue = 0
        for _,row in df.iterrows():
            _,ov = calculate_remaining_sla(row.get("sla_entered_at"), row.get("sla_hours"))
            if ov and row['status'] not in (LeadStatus.AWARDED, LeadStatus.LOST):
                overdue +=1
        return overdue
    except:
        return 0
    finally:
        s.close()


###############################################################################
# LEAD SCORING (backend)
###############################################################################
def compute_priority_for_lead_row(row):
    # Simple internal scoring 0–1
    try:
        val = float(row.get("estimated_value") or 0.0)
        baseline = 5000.0
        value_component = min(1.0, val / baseline)
    except:
        value_component = 0.0

    try:
        sla_entered = row.get("sla_entered_at") or row.get("created_at") or datetime.utcnow()
        if isinstance(sla_entered,str):
            sla_entered = datetime.fromisoformat(sla_entered)
        deadline = sla_entered + timedelta(hours=int(row.get("sla_hours") or 24))
        hours_left = max((deadline - datetime.utcnow()).total_seconds() / 3600.0, 0.0)
        sla_component = max(0.0, (72 - min(hours_left,72))/72)
    except:
        sla_component = 0.2

    flags = 0
    flags += (0 if row.get("contacted") else 1)
    flags += (0 if row.get("inspection_scheduled") else 1)
    flags += (0 if row.get("estimate_submitted") else 1)
    urgency_component = flags/3

    score = 0.5*value_component + 0.35*sla_component + 0.15*urgency_component
    return max(0.0, min(score,1.0))


###############################################################################
# FRONTEND APP
###############################################################################
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@400;600;700&display=swap');
body, .stApp { font-family:'Comfortaa', cursive; background:#0b0f18; color:#fff; }
.header { font-size:22px; font-weight:700; color:#fff; padding:10px 0; }
.metric-card { background:#000; border-radius:10px; padding:18px; margin:8px; color:#fff; min-width:200px; }
.progress-bar{ width:100%; height:8px; background:#222; border-radius:6px; overflow:hidden; }
.progress-fill{ height:100%; border-radius:6px; transition: width .4s ease; }
"""

st.set_page_config(page_title="Restoration Pipeline", layout="wide")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
st.markdown("<div class='header'>TOTAL LEAD PIPELINE KEY PERFORMANCE INDICATOR</div>", unsafe_allow_html=True)
st.write("*Lead flow, compliance, conversions, spend, ROI, and ML predictions — optimized for restoration teams.*")

# Top right date filter like Ads Dashboard
today = date.today()
with st.container():
    cA, cB = st.columns([6,2])
    with cB:
        st.markdown("**Data view range**")
        sel_start = st.date_input("Start", value=today, key="sel_start")
        sel_end = st.date_input("End", value=today, key="sel_end")

# KPI
s = get_session()
df = leads_df(s, sel_start, sel_end)
s.close()

if df.empty:
    st.info("No saved lead data for selected date range.")
else:
    total = len(df)
    qual = int(df[df['qualified']==True].shape[0])
    awarded = int(df[df['status']==LeadStatus.AWARDED].shape[0])
    est_sent = int(df[df['estimate_submitted']==True].shape[0])

    insp_booked = int(df[df['inspection_scheduled']==True].shape[0])
    insp_success_pct = insp_booked/qual*100 if qual else 0

    # SLA success pct = contacted within SLA
    sla_checks = df.apply(lambda r: calculate_remaining_sla(r.get('sla_entered_at'), r.get('sla_hours')), axis=1)
    # For simplicity, if contacted mark = SLA success
    sla_success = df[df['contacted']==True].shape[0]
    sla_success_pct = sla_success/total*100 if total else 0

    # qualification rate
    qual_rate = qual/total*100 if total else 0
    # conversion rate: AWARDED + approved estimates
    approved_est = df[(df['status']==LeadStatus.AWARDED)]
    conversion = len(approved_est)/total*100 if total else 0

    # marketing spend & ROI
    spend = df['cost_to_acquire'].sum()
    revenue = df['estimated_value'].sum()
    roi = revenue/spend*100 if spend else 0

    # 7 cards in 2 rows
    KPI_DATA = [
        ("#22c55e","Active Leads", f"{total-awarded}", sla_success_pct, "Currently open"),
        ("#2563eb","SLA Success", f"{sla_success_pct:.1f}%", sla_success_pct, "Lead touched in SLA"),
        ("#f97316","Qualification Rate", f"{qual_rate:.1f}%", qual_rate, "Qualified leads pct"),
        ("#0ea5e9","Conversion Rate", f"{conversion:.1f}%", conversion, "Won + est approved"),
        ("#ef4444","Inspections Booked", f"{insp_success_pct:.1f}%", insp_success_pct, "Qualified→Inspection"),
        ("#a855f7","Estimates Sent", f"{est_sent}", est_sent, "Estimator submitted"),
        ("#eab308","Pipeline Job Values", f"${revenue:,.0f}", roi, "Total pipeline value")
    ]

    # render grid
    st.markdown("<div style='display:flex; flex-wrap:wrap; gap:6px;'>",unsafe_allow_html=True)
    for color,title,value,pct,note in KPI_DATA:
        st.markdown(f"""
          <div class='metric-card'>
            <div style='font-size:14px;font-weight:700;color:#fff;'>{title}</div>
            <div style='font-size:26px;font-weight:800;color:{color};'>{value}</div>
            <div class='progress-bar'><div class='progress-fill' style='width:{pct}%; background:{color};'></div></div>
            <div style='font-size:11px;color:#bbb;margin-top:6px;'>{note}</div>
          </div>
        """,unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

    # Pipeline donut pie
    st.markdown("---")
    st.markdown("## Lead Pipeline Stages")
    st.write("*Lead distribution across statuses — reveals bottlenecks & drop-offs.*")
    stage_colors = {
        LeadStatus.NEW:'#2563eb',LeadStatus.CONTACTED:'#eab308',
        LeadStatus.INSPECTION_SCHEDULED:'#f97316',LeadStatus.INSPECTION_COMPLETED:'#14b8a6',
        LeadStatus.ESTIMATE_SUBMITTED:'#a855f7',LeadStatus.AWARDED:'#22c55e',LeadStatus.LOST:'#ef4444'
    }
    stage_counts = df['status'].value_counts().reindex(LeadStatus.ALL,fill_value=0)

    pie_df = pd.DataFrame({"status":stage_counts.index,"count":stage_counts.values})
    if px:
        fig = px.pie(pie_df, names="status", values="count", hole=0.45, color='status', color_discrete_map=stage_colors)
        fig.update_layout(margin=dict(t=6,b=6))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.table(pie_df)

    # SLA overdue chart + table
    st.markdown("---")
    st.markdown("## SLA / Overdue Leads")
    st.write("*SLA breach trend & open overdue leads.*")

    over_count = sla_notification_monitor()
    st.write(f"🔔 Overdue SLA leads: **{over_count}**")

    ts_rows=[]
    s2=get_session()
    df2=leads_df(s2, sel_start, sel_end)
    for i in range(30,-1,-1):
        d = today - pd.Timedelta(days=i)
        ds = datetime.combine(d, datetime.min.time()); de=datetime.combine(d,datetime.max.time())
        cnt=0
        for _,row in df2.iterrows():
            sla=row.get('sla_entered_at') or row.get('created_at')
            if isinstance(sla,str): sla=datetime.fromisoformat(sla)
            dl=sla + timedelta(hours=int(row.get('sla_hours') or 24))
            if dl < de and row['status'] not in (LeadStatus.AWARDED,LeadStatus.LOST): cnt+=1
        ts_rows.append({"date":d,"overdue":cnt})
    s2.close()
    ts_df=pd.DataFrame(ts_rows)
    if px:
        fig=px.line(ts_df,x='date',y='overdue', labels={'overdue':'Overdue'}, color_discrete_sequence=None)
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.table(ts_df)

    overdue=df2.copy()
    overdue['deadline']=overdue.apply(lambda r: (r.get('sla_entered_at') or r.get('created_at')) + timedelta(hours=int(r.get('sla_hours') or 24)),axis=1)
    overdue=overdue[overdue['deadline'] < datetime.utcnow()][~overdue['status'].isin([LeadStatus.AWARDED,LeadStatus.LOST])]
    if not overdue.empty:
        st.dataframe(overdue[['id','name','status','deadline']])
    else:
        st.success("No open overdue leads 🎉")

    # Lead Management
    st.markdown("---")
    st.markdown("## TOP 5 PRIORITY LEADS")
    st.write("*AI+SLA urgency ranked leads that need action first.*")

    s3=get_session(); df3 = leads_df(s3, sel_start, sel_end); s3.close()
    df3['priority']=df3.apply(lambda r: compute_priority_for_lead_row(r),axis=1)
    df3=df3.sort_values("priority",ascending=False)
    for _,r in df3.head(5).iterrows():
        tag="🔴 CRITICAL" if r['priority']>=0.7 else "🟠 HIGH" if r['priority']>=0.45 else "🟢 NORMAL"
        st.markdown(f"<div class='stage-card'><b>{tag}</b> #{r['id']} — {r.get('name') or 'No name'} (${r['estimated_value']:,.0f})</div>",unsafe_allow_html=True)

    # Lead search + edit
    st.markdown("---")
    st.markdown("## All Leads")
    st.write("*Search, update, and change pipeline statuses. All records are stored and retrievable by date filter above.*")

    q = st.text_input("Search leads…", "")
    s4=get_session()
    df4=leads_df(s4, sel_start, sel_end)
    if q:
        q2=q.lower()
        df4=df4[
            df4['name'].str.lower().str.contains(q2)|
            df4['phone'].str.lower().str.contains(q2)|
            df4['email'].str.lower().str.contains(q2)|
            df4['property_address'].str.lower().str.contains(q2)
        ]
    if not df4.empty:
        st.dataframe(df4)
    else:
        st.warning("No leads found.")

    # Optional ML predictions viewer (internal trained silently)
    if ML_ENABLED:
        st.markdown("---")
        st.markdown("## ML — Internal Win Predictions")
        dfp=df4.copy()
        dfp['win_probability']=dfp['predicted_prob'].fillna(0)*100
        dfp=dfp.sort_values("win_probability",ascending=False)
        st.dataframe(dfp[['id','name','status','win_probability']])

    s4.close()
