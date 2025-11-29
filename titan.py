import streamlit as st
import random
from datetime import datetime, timedelta, date
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

# ---------- DATABASE SETUP ----------
DB_PATH = "titan_restoration.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
Base = declarative_base()

# ---------- TABLE MODELS ----------
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

class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    full_name = Column(String)
    role = Column(String)

Base.metadata.create_all(engine)

# ---------- UI STYLING ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');
* {font-family:'Comfortaa';}
body, .main {background:#ffffff;}
.sidebar .sidebar-button {
  background:black;color:white;padding:10px;border-radius:6px;text-align:center;font-size:14px;font-weight:bold;
}
.metric-card {
  background:#000;padding:16px;border-radius:12px;margin:6px;color:#fff;text-align:left;
}
.metric-title {
  font-size:14px;color:white;margin-bottom:5px;font-weight:bold;
}
.metric-value {
  font-size:26px;font-weight:bold;
}
.progress-bar {
  height:6px;border-radius:4px;width:100%;animation:pulse 1.8s infinite alternate;
}
@keyframes pulse {
  0%{opacity:0.35;}100%{opacity:1;}
}
.sla-badge {
  background:#dc2626;color:white;padding:6px 12px;border-radius:6px;font-size:14px;font-weight:bold;cursor:pointer;float:right;
}
.priority-time {color:#dc2626;font-size:14px;font-weight:bold;}
.priority-money {color:#22c55e;font-size:18px;font-weight:bold;}
.close-btn{float:right;cursor:pointer;font-size:16px;color:white;}
</style>
""", unsafe_allow_html=True)

# ---------- LOGIN ----------
def login_page():
    st.sidebar.title("Shake5 Login")
    user_input = st.sidebar.text_input("Enter Username", key="login_user")
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
            st.session_state.page = "pipeline"  # force UI after login
            st.sidebar.success(f"Logged in as {user_input}")
            st.rerun()
        except Exception:
            s.rollback()
        finally:
            s.close()

if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.role = "Viewer"
    login_page()

if not st.session_state.user:
    st.warning("Login required")
    st.stop()

if "page" not in st.session_state:
    st.session_state.page = "pipeline"

st.sidebar.button("Logout", on_click=lambda: st.session_state.clear())

# ---------- NAVIGATION ----------
if st.sidebar.button("📊 Pipeline"):
    st.session_state.page = "pipeline"
if st.sidebar.button("💰 Cost Per Acquisition"):
    st.session_state.page = "cpa"
if st.sidebar.button("📈 Analytics"):
    st.session_state.page = "analytics"
if st.sidebar.button("⚙ Settings"):
    st.session_state.page = "settings"
if st.sidebar.button("👤 User Profile"):
    st.session_state.page = "profile"

start_date = st.date_input("Start Date", date.today())
end_date = st.date_input("End Date", date.today())

# ---------- DATA CACHING ----------
@st.cache_data(ttl=40)
def get_filtered_leads():
    s = SessionLocal()
    try:
        return s.query(Lead).filter(
            Lead.created_at >= datetime.combine(start_date, datetime.min.time()),
            Lead.created_at <= datetime.combine(end_date, datetime.max.time())
        ).all()
    finally:
        s.close()

# ---------- ALERT BELL PANEL ----------
def overdue_alert_panel(leads):
    overdue = [l for l in leads if l.status == "OVERDUE"]
    if not overdue:
        return

    if "show_bell" not in st.session_state:
        st.session_state.show_bell = False

    if st.markdown(f"<div class='sla-badge' onclick='bell_toggle()'>🔔 {len(overdue)}</div>", unsafe_allow_html=True):
        st.session_state.show_bell = True

    if overdue and st.session_state.show_bell:
        for l in overdue[:5]:
            countdown = random.randint(-3, 72)
            st.markdown(f"""
            <div class='metric-card'>
                <span class='priority-money'>${l.estimate_value:,.2f}</span><br>
                <span class='priority-time'>{countdown} hrs left</span>
                <span class='close-btn' onclick='close_alert()'>✖</span>
            </div>
            """, unsafe_allow_html=True)

# ---------- PIPELINE DASHBOARD ----------
def pipeline_dashboard():
    leads = get_filtered_leads()
    if not leads:
        st.info("No leads found in selected date range")
        return

    active = len(leads)
    qualified = len([l for l in leads if l.status == "QUALIFIED"])
    inspected = len([l for l in leads if l.status == "INSPECTED"])
    estimate_sent = len([l for l in leads if l.status == "ESTIMATE_SENT"])
    awarded = len([l for l in leads if l.status == "AWARDED"])
    sla_success = random.randint(76, 100)
    qual_rate = round(qualified/active*100,1) if active else 0
    conv_rate = round(awarded/qualified*100,1) if qualified else 0
    pipeline_val = sum(l.estimate_value or 0 for l in leads)
    inspected_val = awarded + inspected  # inspection booked interpreted as inspected stages

    # 2 rows professional cards
    row1 = st.columns(4)
    row2 = st.columns(3)

    metrics = [
        ("ACTIVE LEADS", active,  "#06b6d4"),
        ("SLA SUCCESS",  f"{sla_success}%",  "#22c55e"),
        ("QUALIFICATION RATE",  f"{qual_rate}%",  "#f97316"),
        ("CONVERSION RATE", f"{conv_rate}%",  "#3b82f6"),
        ("INSPECTION BOOKED", inspected_val, "#8b5cf6"),
        ("ESTIMATE SENT", estimate_sent, "#ec4899"),
        ("PIPELINE JOB VALUES", f"${pipeline_val:,.2f}", "#22c55e")
    ]

    # apply
    for col,(title,val,color) in zip(row1+row2,metrics):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>{title}</div>
            <div class='metric-value' style='color:white'>{val}</div>
            <div class='progress-bar' style='background:{color};width:100%'></div>
        </div>
        """, unsafe_allow_html=True)

    overdue_alert_panel(leads)

    st.markdown("---")
    st.markdown("### TOP 5 PRIORITY LEADS")
    st.markdown("*Urgent high value leads close to SLA expiry*")

    scored = [{"Name":l.name,"Value":l.estimate_value,"Score":random.randint(65,99)} for l in leads]
    dfp = pd.DataFrame(scored).sort_values("Value",ascending=False).head(5)
    top_cols = st.columns(5)
    for c,(_,l) in zip(top_cols, dfp.iterrows()):
        hrs = random.randint(-5,48)
        c.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>{l["Name"]}</div>
            <span class='priority-money'>${l["Value"]:,.2f}</span><br>
            <span class='priority-time'>{hrs} hrs left</span><br>
            <span class='lead-chip' style='background:{color};'>Score: {l["Score"]}</span>
            <span class='close-btn' onclick='close_alert()'>✖</span>
        </div>
        """, unsafe_allow_html=True)

    # Editable leads section
    st.markdown("---")
    st.markdown("### ALL LEADS")
    st.markdown("*Expand to edit, assign owner, update status*")
    for l in leads:
        with st.expander(f"Lead #{l.id} — {l.name}"):
            new_owner = st.selectbox("Assign Owner",["UNASSIGNED","Estimator","Adjuster","Tech","Admin"], key=f"own_{l.id}")
            new_status = st.selectbox("Status",["CAPTURED","QUALIFIED","INSPECTED","ESTIMATE_SENT","AWARDED","OVERDUE"], key=f"stat_{l.id}")
            cost = st.number_input("Cost to Acquire Lead ($)", value=l.cost_to_acquire or 0.0, key=f"cost_{l.id}")
            if cost == 0: cost = 0.0

            if st.button("Save Lead Update", key=f"save_{l.id}"):
                s2 = SessionLocal()
                try:
                    dblead = s2.query(Lead).filter(Lead.id==l.id).first()
                    old = dblead.status
                    dblead.owner = new_owner
                    dblead.status = new_status
                    dblead.cost_to_acquire = cost
                    dblead.converted = True if new_status=="AWARDED" else False
                    dblead.score = random.randint(60,100)
                    s2.commit()
                    st.success("Updated ✅")
                except Exception:
                    s2.rollback()
                finally:
                    s2.close()

# ---------- CPA / ROI ----------
def cpa_dashboard():
    leads = get_filtered_leads()
    total_spend = sum(l.cost_to_acquire or 0 for l in leads)
    won = sum(1 for l in leads if l.status=="AWARDED" or l.converted)
    total_value = sum(l.estimate_value or 0 for l in leads)

    cpa = total_spend/won if won else 0
    roi = total_value - total_spend if won else 0
    roi_pct = (roi/total_spend*100) if total_spend else 0

    st.markdown("## CPA & ROI Dashboard")

    st.markdown(f"""
    <div class='metric-card'><div class='metric-title'>💰 Total Marketing Spend</div><div class='metric-value'>${total_spend:,.2f}</div></div>
    <div class='metric-card'><div class='metric-title'>✅ Conversions (Won)</div><div class='metric-value'>{won}</div></div>
    <div class='metric-card'><div class='metric-title'>🎯 CPA</div><div class='metric-value'>${round(cpa,2)}</div></div>
    <div class='metric-card'><div class='metric-title'>📈 ROI</div><div class='metric-value'>${roi:,.2f} ({round(roi_pct,1)}%)</div></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Marketing Spend vs Conversions (Won)")
    fig = plt.figure()
    plt.plot([total_spend, won])
    st.pyplot(fig)

# ---------- ANALYTICS ----------
def analytics_dashboard():
    leads = get_filtered_leads()
    df = pd.DataFrame([{
        "Stage":l.status,
        "Created":l.created_at,
        "Cost":l.cost_to_acquire,
        "Value":l.estimate_value
    } for l in leads])

    st.markdown("## Analytics Dashboard")

    st.markdown("### Workflow Activity Log")
    dh = df.groupby("Stage").size()
    fig = plt.figure()
    plt.plot(dh)
    st.pyplot(fig)

    st.dataframe(df)

# ---------- SETTINGS ----------
def settings_dashboard():
    st.markdown("## Settings")

    st.markdown("### Lead Source Platforms")
    platforms = ["Referral","Walk-In","Website","Facebook","Instagram","TikTok","Google Ads","Hotline"]
    for p in platforms:
        st.checkbox(p, value=True, key=f"plat_{p}")

    # CPA chart in settings
    st.markdown("---")
    st.markdown("### CPA Overview")
    leads = get_filtered_leads()
    total_spend = sum(l.cost_to_acquire or 0 for l in leads)
    won = sum(1 for l in leads if l.status=="AWARDED" or l.converted)

    fig = plt.figure()
    plt.plot([total_spend, won])
    st.pyplot(fig)

# ---------- PROFILE ----------
def profile_dashboard():
    st.markdown("## User Profile")
    st.text(f"Logged in as: {st.session_state.user} ({st.session_state.role})")
    st.button("Refresh View", on_click=lambda: st.rerun())

# ---------- ROUTER ----------
if st.session_state.page == "pipeline":
    pipeline_dashboard()
elif st.session_state.page == "cpa":
    cpa_dashboard()
elif st.session_state.page == "analytics":
    analytics_dashboard()
elif st.session_state.page == "settings":
    settings_dashboard()
elif st.session_state.page == "profile":
    profile_dashboard()
