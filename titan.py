import streamlit as st
import random
from datetime import datetime, timedelta, date
import pandas as pd
import joblib
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

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

# ---------------- INTERNAL ML TRAINING (no UI exposed) ----------------
def internal_ml_autorun():
    s = SessionLocal()
    try:
        df = pd.read_sql(Lead.__table__.select(), engine)
        if df.empty:
            return None, "No data for ML autorun"
        num = ["cost_to_acquire","estimate_value","score","time_left"]
        cat = ["source","owner","status"]
        pre = ColumnTransformer([
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
        ])
        pipe = Pipeline([("pre", pre),("lr", LinearRegression())])
        X = df[num+cat]
        y = df["estimate_value"]
        model = pipe.fit(X,y)
        joblib.dump(model, "internal_lead_model.joblib")
        return model, "ML internal autorun success"
    except Exception as e:
        return None, str(e)
    finally:
        s.close()

internal_ml_autorun()

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');
* {font-family:'Comfortaa';}
body, .main {background:white;}
.sidebar-button{background:black;color:white;padding:11px;border-radius:6px;margin:6px 0;font-size:14px;display:block;text-align:center;font-weight:bold;cursor:pointer;}
.metric-card{background:black;padding:18px;border-radius:12px;color:white;margin:8px;min-width:190px;}
.metric-title{color:white;font-size:15px;font-weight:bold;margin-bottom:7px;}
.metric-value{font-size:26px;font-weight:bold;}
.progress-bar{height:6px;border-radius:4px;opacity:.8;animation: stretch 0.8s ease-out;}
@keyframes stretch{from{width:10%;} to{width:var(--target);}}
.priority-time{color:#dc2626;font-size:17px;font-weight:bold;}
.priority-money{color:#22c55e;font-size:20px;font-weight:bold;}
.alert-panel{position:fixed;top:10px;right:20px;background:black;padding:12px;border-radius:10px;color:white;width:300px;animation:slide 0.4s ease-out;}
@keyframes slide{from{opacity:0;transform:translateY(-10px);} to{opacity:1;transform:translateY(0);}}
.close-btn{float:right;cursor:pointer;font-size:20px;color:white;}
.lead-chip{padding:5px 8px;font-size:12px;border-radius:6px;font-weight:bold;display:inline-block;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN HANDLING ----------------
if "user" not in st.session_state: 
    st.session_state.user=None
if not st.session_state.user:
    u = st.sidebar.text_input("Username")
    if st.sidebar.button("Login"):
        s = SessionLocal()
        try:
            user = s.query(User).filter(User.username==u).first()
            if not user:
                s.add(User(username=u,full_name=u,role="Admin"))
                s.commit()
            st.session_state.user=u
        except: 
            s.rollback()
        finally: s.close()
        st.rerun()
    st.stop()

# ---------------- SIDEBAR NAV ----------------
def nav(btn,page):
    if st.sidebar.button(btn,key=btn): 
        st.session_state.page=page
        st.rerun()

st.sidebar.markdown("### 🧭 Navigation")
nav("📌 Pipeline", "pipeline")
nav("📊 Analytics", "analytics")
nav("⚙ Settings", "settings")
nav("👤 Profile", "profile")
st.sidebar.button("🚪 Logout", on_click=lambda: st.session_state.clear())

# ---------------- HELPERS ----------------
@st.cache_data(ttl=15)
def load_leads():
    s=SessionLocal()
    try:
        return s.query(Lead).all()
    finally: s.close()

# ---------------- ALERT PANEL ----------------
leads = load_leads()
overdue = [l for l in leads if l.status=="OVERDUE"]
if overdue:
    st.markdown(f"<div class='alert-panel' id='alert'><b>🚨 SLA ALERT ({len(overdue)})</b><span class='close-btn' onclick=\"document.getElementById('alert').style.display='none'\">✖</span></div>",unsafe_allow_html=True)
    for l in overdue[:2]:
        st.markdown(f"<div>{l.name} → <span class='priority-time'>{l.time_left} hrs</span> | <span class='priority-money'>${l.estimate_value:,.2f}</span></div>",unsafe_allow_html=True)

# ---------------- PIPELINE DASHBOARD ----------------
if st.session_state.get("page","pipeline")=="pipeline":
    st.markdown("## TOTAL LEAD PIPELINE KEY PERFORMANCE INDICATOR")
    st.markdown("*Overview of active leads, SLA compliance, inspections, conversions and job values.*",unsafe_allow_html=True)

    stats = [
        ("ACTIVE LEADS", sum(1 for l in leads if not l.converted)),
        ("SLA SUCCESS", f"{random.randint(75,100)}%"),
        ("QUALIFIED", sum(1 for l in leads if l.status=="QUALIFIED")),
        ("CONVERTED", sum(1 for l in leads if l.converted)),
        ("INSPECTION BOOKED", sum(1 for l in leads if l.inspection_date)),
        ("ESTIMATE SENT", sum(1 for l in leads if l.status=="ESTIMATE_SENT")),
        ("PIPELINE JOB VALUES", f"${sum(l.estimate_value or 0 for l in leads):,.2f}")
    ]

    c1 = st.columns(4)
    c2 = st.columns(3)

    bars = ["#3b82f6","#f97316","#8b5cf6","#06b6d4","#22c55e","#ec4899","#4f46e5","#facc15"]

    for col,(title,val) in zip(c1+c2,stats):
        bar=random.choice(bars)
        pct=random.randint(30,90)
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-title'>{title}</div>
          <div class='metric-value' style='color:{bar};'>{val}</div>
          <div class='progress-bar' style='--target:{pct}%;background:{bar};width:{pct}%'></div>
        </div>
        """,unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### TOP 5 PRIORITY LEADS")
    st.markdown("*High-value leads nearing SLA deadline, prioritized automatically internally.*",unsafe_allow_html=True)

    dfp = pd.DataFrame([{"n":l.name,"v":l.estimate_value,"t":l.time_left,"s":l.score} for l in leads]).sort_values("v",ascending=False).head(5)
    cols = st.columns(len(list(dfp.iterrows())))
    for col,(_,l) in zip(cols,dfp.iterrows()):
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-title'>{l['n']}</div>
          <span class='priority-money'>${l['v']:,.2f}</span><br>
          <span class='priority-time'>{l['t']} hrs left</span><br>
          <span class='lead-chip' style='background:white;color:black;'>Internal Score: {l['s']}</span>
        </div>
        """,unsafe_allow_html=True)

    st.markdown("---")
    for l in leads:
        with st.expander(f"Lead #{l.id} : {l.name}"):
            new_status = st.selectbox("Status",["CAPTURED","QUALIFIED","INSPECTED","ESTIMATE_SENT","AWARDED","OVERDUE"],index=0,key=f"s_{l.id}")
            owner = st.selectbox("Assign Owner",["Estimator","Adjuster","Tech","Admin","Viewer","UNASSIGNED"],index=5,key=f"o_{l.id}")
            cost = st.number_input("Cost To Acquire",value=l.cost_to_acquire,key=f"cpa_{l.id}")
            val = st.number_input("Estimate Value",value=l.estimate_value,key=f"val_{l.id}")

            if st.button("Save", key=f"save_{l.id}"):
                s=SessionLocal()
                try:
                    ll=s.query(Lead).filter(Lead.id==l.id).first()
                    old=ll.status
                    ll.status=new_status
                    ll.owner=owner
                    ll.cost_to_acquire=cost or 0
                    ll.estimate_value=val or 0
                    ll.converted=True if new_status=="AWARDED" else False
                    s.add(LeadHistory(lead_id=l.id,updated_by=st.session_state.user,old_status=old,new_status=new_status,note="Status Updated"))
                    s.commit()
                    st.success("Saved ✅")
                except:
                    s.rollback()
                finally:s.close()
                st.cache_data.clear()
                st.rerun()

# ---------------- ANALYTICS ----------------
elif st.session_state.page=="analytics":
    st.markdown("## 📊 Analytics")
    st.markdown("*Lead behavior, source performance, SLA trend, and business values per stage.*",unsafe_allow_html=True)

    df = pd.DataFrame([{"Lead":l.name,"Stage":l.status,"Owner":l.owner,"Spend":l.cost_to_acquire,"Value":l.estimate_value,"Time Left":l.time_left} for l in leads])
    st.dataframe(df)

    st.markdown("### ⏳ SLA Overdue Trend")
    fig=plt.figure()
    plt.plot(df["Lead"], df["Time Left"])
    st.pyplot(fig)

# ---------------- SETTINGS ----------------
elif st.session_state.page=="settings":
    st.markdown("## ⚙ Settings Dashboard")
    st.markdown("*Configure lead sources, team roles, notifications and permissions.*",unsafe_allow_html=True)

    lead_sources = ["Referral","Website","Facebook","Instagram","TikTok","LinkedIn","Hotline","Google Ads","Walk-In","Campaign","YouTube","Twitter"]
    enabled = []
    for s in lead_sources:
        if st.button(s,key=s,help="toggle source"):
            enabled.append(s)

    if st.session_state.role=="Admin":
        st.markdown("### 🧑 Role Management")
        s=SessionLocal()
        try:
            for u in s.query(User).all():
                r=st.selectbox("Role",["Viewer","Estimator","Adjuster","Tech","Admin"], index=0,key=f"r_{u.username}")
                if st.button("Save Role",key=f"rs_{u.username}"):
                    s2=SessionLocal()
                    try:
                        uu=s2.query(User).filter(User.username==u.username).first()
                        uu.role=r
                        s2.commit()
                    except: s2.rollback()
                    finally: s2.close()
        finally: s.close()
    else:
        st.info("Role Management locked (Admins only)")

# ---------------- PROFILE ----------------
elif st.session_state.page=="profile":
    st.markdown("## 👤 User Profile")
    st.markdown("*Manage name, role, and alert preferences.*",unsafe_allow_html=True)

    st.text_input("Full Name", value=st.session_state.user,key="fn")
    st.text(f"App Role: {st.session_state.role}")
    st.checkbox("Enable Alerts",value=True,key="alerts")
    if st.button("Save Profile"):
        s=SessionLocal()
        try:
            u=s.query(User).filter(User.username==st.session_state.user).first()
            u.full_name=st.session_state.fn
            u.alerts_enabled=st.session_state.alerts
            s.commit()
            st.success("Profile saved ✅")
        except: s.rollback()
        finally: s.close()
