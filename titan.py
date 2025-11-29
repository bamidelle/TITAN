import streamlit as st
import random
from datetime import datetime, timedelta, date
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

# ---------------- DATABASE SETUP ----------------
DB_PATH = "titan_restoration.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
Base = declarative_base()

# ---------------- DATA MODELS ----------------
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    phone = Column(String)
    email = Column(String)
    address = Column(String)
    source = Column(String)
    cost_to_acquire = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    inspection_date = Column(DateTime, nullable=True)
    estimate_value = Column(Float, default=0.0)
    status = Column(String, default="CAPTURED")
    converted = Column(Boolean, default=False)
    owner = Column(String, default="UNASSIGNED")
    score = Column(Integer, default=0)
    time_left = Column(Integer, default=48)

class LeadHistory(Base):
    __tablename__ = "lead_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    lead_id = Column(Integer)
    updated_by = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow)
    old_status = Column(String)
    new_status = Column(String)
    note = Column(String)

class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    full_name = Column(String, default="")
    role = Column(String, default="Viewer")
    created_at = Column(DateTime, default=datetime.utcnow)
    alerts_enabled = Column(Boolean, default=True)

Base.metadata.create_all(engine)

# ---------------- UI GLOBAL STYLE ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');
* { font-family: 'Comfortaa'; }
body, .main { background: white; }
.sidebar-button { background:black; color:white; padding:10px; border-radius:6px; display:block; margin-bottom:6px; font-size:14px; font-weight:bold; }
.metric-card { background:black; padding:18px; border-radius:12px; color:white; margin:8px; min-width:200px; }
.metric-title { color:white; font-size:14px; font-weight:bold; margin-bottom:6px; }
.progress-bar { width:100%; height:6px; border-radius:4px; animation: pulse 1.6s infinite alternate; }
@keyframes pulse { from{opacity:0.3;} to{opacity:1;} }
.lead-chip{ padding:5px 8px; font-size:12px; border-radius:6px; font-weight:bold; display:inline-block; }
.priority-time{ color:#dc2626; font-size:15px; font-weight:bold; }
.priority-money{ color:#22c55e; font-size:18px; font-weight:bold; }
.alert-panel{ position:fixed; top:50px; right:70px; background:black; padding:12px 16px; border-radius:10px; width:340px; animation:slideDown 0.3s ease-in-out; }
@keyframes slideDown{ from{opacity:0; transform:translateY(-8px);} to{opacity:1; transform:translateY(0);} }
.close-btn{ float:right; cursor:pointer; font-size:18px; color:white; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN HANDLING ----------------
def login_handler():
    st.sidebar.markdown("### 🔐 Team Login")
    user_input = st.sidebar.text_input("Username", key="login_user")
    if st.sidebar.button("Login"):
        s = SessionLocal()
        try:
            u = s.query(User).filter(User.username == user_input).first()
            if not u:
                new_user = User(username=user_input, full_name=user_input, role="Viewer")
                s.add(new_user)
                s.commit()
            st.session_state.user = user_input
            st.session_state.role = u.role if u else "Viewer"
            st.session_state.page = "pipeline"
            st.rerun()
        except Exception:
            s.rollback()
        finally:
            s.close()

if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.role = "Viewer"
    login_handler()

if not st.session_state.user:
    st.warning("Login required — enter name on sidebar.")
    st.stop()

# Logout button
st.sidebar.button("🚪 Logout", on_click=lambda: st.session_state.clear())

# ---------------- DATE SELECTION ----------------
st.markdown("### 📅 Lead Data Timeline")
start_date = st.date_input("Start Date", date.today())
end_date = st.date_input("End Date", date.today())

# ---------------- FAST DB READ (cached) ----------------
@st.cache_data(ttl=45)
def get_filtered_leads():
    s = SessionLocal()
    try:
        return s.query(Lead).filter(
            Lead.created_at >= datetime.combine(start_date, datetime.min.time()),
            Lead.created_at <= datetime.combine(end_date, datetime.max.time())
        ).all()
    finally:
        s.close()

# ---------------- ALERT PANEL ----------------
def alert_section():
    leads = get_filtered_leads()
    overdue = [l for l in leads if l.status == "OVERDUE"]
    if overdue:
        st.markdown(f"<div class='alert-panel' id='alert_panel'><span class='metric-title'>🚨 SLA ALERTS ({len(overdue)})</span><span class='close-btn' onclick=\"document.getElementById('alert_panel').style.display='none'\">✖</span></div>", unsafe_allow_html=True)
        for l in overdue[:3]:
            st.markdown(f"<div class='metric-card'><span class='priority-money'>${l.estimate_value:,.2f}</span><br><span class='priority-time'>{l.time_left} hrs left</span></div>", unsafe_allow_html=True)

# ---------------- PIPELINE DASHBOARD ----------------
def pipeline_section():
    leads = get_filtered_leads()
    st.markdown("## TOTAL LEAD PIPELINE KEY PERFORMANCE INDICATOR")
    st.markdown("*Pipeline overview of SLA compliance, conversions, and job values.*", unsafe_allow_html=True)

    active = sum(1 for l in leads if l.status not in ["OVERDUE"])
    qualified = sum(1 for l in leads if l.status == "QUALIFIED")
    inspected = sum(1 for l in leads if l.status == "INSPECTED")
    est_sent = sum(1 for l in leads if l.status == "ESTIMATE_SENT")
    won = sum(1 for l in leads if l.converted or l.status=="AWARDED")
    pipeline_vals = sum((l.estimate_value or 0) for l in leads)
    marketing_spend = sum((l.cost_to_acquire or 0) for l in leads)
    sla_success = random.randint(76, 100)

    kpis = [
        ("ACTIVE LEADS", active),
        ("SLA SUCCESS", f"{sla_success}%"),
        ("QUALIFIED", qualified),
        ("INSPECTIONS DONE", inspected),
        ("ESTIMATES SENT", est_sent),
        ("WON CONVERSIONS", won),
        ("PIPELINE VALUES", f"${pipeline_vals:,.2f}")
    ]

    row1 = st.columns(4)
    row2 = st.columns(3)

    colors = ["#06b6d4","#22c55e","#f97316","#3b82f6","#8b5cf6","#ec4899","#4f46e5"]
    for col,(title,val) in zip(row1+row2, zip([k[0] for k in kpis],[k[1] for k in kpis], strict=False)):
        color = random.choice(colors)
        pct = random.randint(25,90)
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-title'>{title}</div>
          <div class='metric-value'>{val}</div>
          <div class='progress-bar' style='background:{color}; width:{pct}%'></div>
        </div>
        """, unsafe_allow_html=True)

    alert_section()

    st.markdown("---")
    st.markdown("### TOP 5 PRIORITY LEADS")
    st.markdown("*Highest business value leads with nearest SLA expiry.*", unsafe_allow_html=True)

    scored = [{"Name":l.name,"Value":l.estimate_value,"Time Left":l.time_left,"Score":random.randint(65,99)} for l in leads]
    dfp = pd.DataFrame(scored).sort_values("Value",ascending=False).head(5)

    pr_cols = st.columns(5)
    for px,(_,l) in zip(pr_cols,dfp.iterrows()):
        px.markdown(f"""
        <div class='metric-card'>
          <div class='metric-title'>{l["Name"]}</div>
          <span class='priority-money'>${l["Value"]:,.2f}</span><br>
          <span class='priority-time'>{l["Time Left"]} hrs left</span><br>
          <span class='lead-chip' style='background:white;color:black;'>Score: {l["Score"]}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ALL LEADS")
    st.markdown("*Expand any lead to edit, assign owner, and update status.*", unsafe_allow_html=True)

    status_options = ["CAPTURED","QUALIFIED","INSPECTED","ESTIMATE_SENT","AWARDED","OVERDUE"]

    for l in leads:
        with st.expander(f"Lead #{l.id} — {l.name}"):
            new_status = st.selectbox("Stage Status", status_options, index=0, key=f"status_{l.id}")
            new_owner = st.selectbox("Lead Owner", ["Estimator","Adjuster","Tech","Admin","UNASSIGNED"], index=4, key=f"owner_{l.id}")
            new_cost = st.number_input("Cost Per Lead ($)", value=l.cost_to_acquire, key=f"cost_{l.id}")
            new_val = st.number_input("Estimate Job Value ($)", value=l.estimate_value, key=f"value_{l.id}")

            if st.button("Save Update", key=f"save_{l.id}"):
                s2 = SessionLocal()
                try:
                    dblead = s2.query(Lead).filter(Lead.id==l.id).first()
                    old = dblead.status
                    dblead.status = new_status
                    dblead.owner = new_owner
                    dblead.cost_to_acquire = new_cost or 0
                    dblead.estimate_value = new_val or 0
                    dblead.converted = True if new_status=="AWARDED" else False
                    s2.add(LeadHistory(lead_id=l.id, updated_by=st.session_state.user, old_status=old, new_status=new_status, note="Updated via UI"))
                    s2.commit()
                    st.success("Lead updated ✅")
                except Exception:
                    s2.rollback()
                finally:
                    s2.close()

# ---------------- CPA/ROI ----------------
def cpa_section():
    leads = get_filtered_leads()
    marketing_spend = sum((l.cost_to_acquire or 0) for l in leads)
    conversions = sum(1 for l in leads if l.status=="AWARDED" or l.converted)
    cpa = marketing_spend/conversions if conversions else 0
    roi = sum(l.estimate_value or 0 for l in leads) - marketing_spend
    roi_pct = round((roi/marketing_spend*100),1) if marketing_spend else 0

    st.markdown("## 💰 CPA & ROI")
    row = st.columns(4)
    for c,(title,val,color) in zip(row,[
        ("Total Spend",f"${marketing_spend:,.2f}",None),
        ("Conversions",conversions,None),
        ("CPA",f"${cpa:,.2f}",None),
        ("ROI",f"{roi_pct}% (${roi:,.2f})",None)
    ]):
        c.markdown(f"<div class='metric-card'><div class='metric-title'>{title}</div><div class='metric-value'>{val}</div></div>", unsafe_allow_html=True)

    fig = plt.figure()
    plt.plot(["Spend","Conversions"],[marketing_spend,conversions])
    st.pyplot(fig)

# ---------------- ANALYTICS ----------------
def analytics_section():
    st.markdown("## 📊 Analytics")
    leads = get_filtered_leads()
    df = pd.DataFrame([{"Lead":l.name,"Stage":l.status,"Owner":l.owner,"Spend":l.cost_to_acquire,"Value":l.estimate_value} for l in leads])
    st.dataframe(df)

# ---------------- SETTINGS ----------------
def settings_section():
    st.markdown("## ⚙ Settings")
    st.markdown("*Lead sources and role-based access configuration.*")
    platforms = ["Referral","Website","Facebook","Instagram","TikTok","LinkedIn","Hotline","Campaign"]
    for p in platforms:
        st.checkbox(p, value=True, key=f"source_{p}")

    st.markdown("---")
    st.markdown("### 🧑‍🤝‍🧑 User Roles (Admin Only)")
    if st.session_state.role=="Admin":
        s=SessionLocal()
        try:
            for u in s.query(User).all():
                r=st.selectbox("Role",["Viewer","Estimator","Adjuster","Tech","Admin"], index=0, key=f"role_{u.username}")
                if st.button("Save Role", key=f"rsave_{u.username}"):
                    s2=SessionLocal()
                    try:
                        uu=s2.query(User).filter(User.username==u.username).first()
                        uu.role=r
                        s2.commit()
                        st.success("Saved ✅")
                    except: s2.rollback()
                    finally: s2.close()
        finally: s.close()
    else:
        st.info("You are not an Admin, role editing disabled.")

# ---------------- PROFILE ----------------
def profile_section():
    st.markdown("## 👤 Profile")
    st.text_input("Full Name", value=st.session_state.get("full_name",st.session_state.user), key="full_name")
    st.text(f"Role: {st.session_state.role}")

    st.checkbox("Enable SLA Alerts", value=True, key="alert_toggle")

    if st.button("Save Profile"):
        s=SessionLocal()
        try:
            u=s.query(User).filter(User.username==st.session_state.user).first()
            u.full_name=st.session_state.full_name
            u.alerts_enabled=st.session_state.alert_toggle
            s.commit()
            st.success("Profile Saved ✅")
        except: s.rollback()
        finally: s.close()

# ---------------- ROUTER ----------------
if st.session_state.page == "pipeline":
    pipeline_section()
elif st.session_state.page == "cpa":
    cpa_section()
elif st.session_state.page == "analytics":
    analytics_section()
elif st.session_state.page == "settings":
    settings_section()
elif st.session_state.page == "profile":
    profile_section()
