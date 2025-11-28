"""
TITAN - Single-file Streamlit application
Combined features and updates (dashboard, lead pipeline, CPA, imports, simple ML placeholder)

How to run:
1. Install dependencies: pip install streamlit pandas numpy scikit-learn plotly openpyxl
2. Run: streamlit run titan_full_app.py

Notes:
- This is an integrated, runnable single-file prototype. Replace mock data and placeholders with real DB/services.
- WordPress integration: use REST API from settings -> you'll need site URL and credentials; placeholder included.

Author: Generated for user
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

# ----------------------------
# Styling
# ----------------------------
st.set_page_config(page_title="TITAN - Lead Pipeline & Restoration Suite", layout="wide")

APP_CSS = """
<style>
body {background-color: #0f172a; color: #e6eef8}
.header {display:flex; align-items:center; gap:12px}
.metric-card {background:linear-gradient(135deg,#0ea5a0,#06b6d4); padding:18px; border-radius:12px; color:white}
.small {font-size:12px; opacity:0.9}
.kpi {font-size:28px; font-weight:700}
.card-row {display:flex; gap:12px; flex-wrap:wrap}
.table-container {background:#0b1220; padding:12px; border-radius:8px}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)

# ----------------------------
# Utility helpers
# ----------------------------

def generate_mock_leads(n=200):
    rng = np.random.default_rng(42)
    created = [datetime.now() - timedelta(days=int(x)) for x in rng.integers(0,90,size=n)]
    sources = rng.choice(['Google Ads','Organic','Referral','Facebook Ads','Direct','Partner'], size=n)
    stages = rng.choice(['New','Contacted','Qualified','Estimate Sent','Won','Lost'], size=n, p=[0.2,0.25,0.2,0.15,0.1,0.1])
    est_value = np.round(rng.normal(2000,1500,size=n).clip(200,20000),2)
    cost = np.round(rng.normal(45,30,size=n).clip(0,500),2)
    lead_id = [f"L{100000+i}" for i in range(n)]
    df = pd.DataFrame({
        'lead_id': lead_id,
        'created_at': created,
        'source': sources,
        'stage': stages,
        'estimated_value': est_value,
        'ad_cost': cost,
        'converted': np.where(stages=='Won',1,0)
    })
    return df

@st.cache_data
def load_data():
    return generate_mock_leads(500)


def download_link(df: pd.DataFrame, filename: str):
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"
    return f'<a href="{href}" download="{filename}">Download {filename}</a>'

# ----------------------------
# Lead pipeline KPI calculations
# ----------------------------

def compute_kpis(df: pd.DataFrame):
    total_leads = len(df)
    new_leads = (df['stage']=='New').sum()
    contacted = (df['stage']=='Contacted').sum()
    conversion_rate = df['converted'].mean() if total_leads>0 else 0
    avg_value = df.loc[df['converted']==1,'estimated_value'].mean() if df['converted'].sum()>0 else df['estimated_value'].mean()
    total_revenue = df.loc[df['converted']==1,'estimated_value'].sum()
    avg_cpa = df['ad_cost'].sum() / df['converted'].sum() if df['converted'].sum()>0 else df['ad_cost'].mean()
    return {
        'total_leads': total_leads,
        'new_leads': new_leads,
        'contacted': contacted,
        'conversion_rate': conversion_rate,
        'avg_value': avg_value,
        'total_revenue': total_revenue,
        'avg_cpa': avg_cpa
    }

# ----------------------------
# Simple Lead Scoring model placeholder
# ----------------------------

def train_lead_scoring(df: pd.DataFrame):
    # create features from source/stage/ad_cost/estimated_value
    df2 = df.copy()
    df2['age_days'] = (datetime.now() - pd.to_datetime(df2['created_at'])).dt.days
    X = pd.get_dummies(df2[['source','stage']].astype(str))
    X['ad_cost'] = df2['ad_cost']
    X['estimated_value'] = df2['estimated_value']
    X['age_days'] = df2['age_days']
    y = df2['converted']
    if len(y.unique())==1:
        # cannot train - return None
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc

# ----------------------------
# Main UI
# ----------------------------

df = load_data()

st.sidebar.title('TITAN Control Panel')
page = st.sidebar.selectbox('Select page', ['Dashboard','Leads','Cost per Acquisition','ML Lead Scoring','Imports & Exports','Settings','Reports'])

# Settings cache
if 'app_settings' not in st.session_state:
    st.session_state['app_settings'] = {'wordpress_url':'', 'wordpress_user':'', 'wordpress_app_pass':''}

# ----------------------------
# Dashboard
# ----------------------------
if page == 'Dashboard':
    st.markdown("<div class='header'><h1>📊 TOTAL LEAD PIPELINE KEY PERFORMANCE INDICATOR</h1></div>", unsafe_allow_html=True)
    st.markdown("*\_A high-level snapshot of leads, conversion, and acquisition efficiency across channels_*")
    KPIs = compute_kpis(df)

    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        st.markdown(f"<div class='metric-card'><div class='kpi'> {KPIs['total_leads']} </div><div class='small'>Total leads</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div class='kpi'> {KPIs['new_leads']} </div><div class='small'>New leads (current period)</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='kpi'> {KPIs['contacted']} </div><div class='small'>Contacted</div></div>", unsafe_allow_html=True)
    with col4:
        conv = f"{KPIs['conversion_rate']*100:.1f}%"
        st.markdown(f"<div class='metric-card'><div class='kpi'> {conv} </div><div class='small'>Conversion Rate</div></div>", unsafe_allow_html=True)

    # trend chart by source
    st.subheader('Leads by Source (last 90 days)')
    df_time = df.copy()
    df_time['day'] = pd.to_datetime(df_time['created_at']).dt.date
    grouped = df_time.groupby(['day','source']).size().reset_index(name='count')
    fig = px.area(grouped, x='day', y='count', color='source', title='Leads by Source over time')
    st.plotly_chart(fig, use_container_width=True)

    # pipeline funnel
    st.subheader('Pipeline stages')
    stage_counts = df['stage'].value_counts().reset_index()
    stage_counts.columns = ['stage','count']
    fig2 = px.bar(stage_counts, x='stage', y='count', title='Count per pipeline stage')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader('Recent Leads')
    st.dataframe(df.sort_values('created_at', ascending=False).head(15))

# ----------------------------
# Leads page
# ----------------------------
elif page == 'Leads':
    st.header('Leads Management')
    left, right = st.columns([2,1])
    with left:
        st.subheader('All leads')
        edited = st.data_editor(df, num_rows='dynamic')
        st.markdown(download_link(edited, 'leads_export.xlsx'), unsafe_allow_html=True)
    with right:
        st.subheader('Filters')
        src = st.multiselect('Source', options=df['source'].unique(), default=df['source'].unique())
        stg = st.multiselect('Stage', options=df['stage'].unique(), default=df['stage'].unique())
        min_date = st.date_input('From date', value=(datetime.now()-timedelta(days=90)).date())
        filtered = df[(df['source'].isin(src)) & (df['stage'].isin(stg)) & (pd.to_datetime(df['created_at']).dt.date >= min_date)]
        st.markdown(f"**Filtered leads: {len(filtered)}**")
        if st.button('Export filtered to Excel'):
            st.markdown(download_link(filtered, 'filtered_leads.xlsx'), unsafe_allow_html=True)

# ----------------------------
# Cost per Acquisition
# ----------------------------
elif page == 'Cost per Acquisition':
    st.header('Cost per Acquisition (CPA)')
    st.markdown('Analyze how much each conversion costs grouped by channel and campaign-level placeholder')
    by_source = df.groupby('source').agg({'ad_cost':'sum','converted':'sum'})
    by_source['cpa'] = by_source['ad_cost'] / by_source['converted'].replace(0,np.nan)
    by_source = by_source.reset_index()
    st.dataframe(by_source.sort_values('cpa'))
    fig = px.bar(by_source, x='source', y='cpa', title='CPA by Source')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('CPA Time Series')
    df_daily = df.copy()
    df_daily['day'] = pd.to_datetime(df_daily['created_at']).dt.date
    daily = df_daily.groupby('day').agg({'ad_cost':'sum','converted':'sum'})
    daily['cpa'] = daily['ad_cost'] / daily['converted'].replace(0,np.nan)
    daily = daily.reset_index()
    st.line_chart(daily.set_index('day')['cpa'])

# ----------------------------
# ML Lead Scoring
# ----------------------------
elif page == 'ML Lead Scoring':
    st.header('Lead Scoring (ML Placeholder)')
    st.markdown('Train a simple model to predict which leads convert (this is a demo; replace with production data and validation)')
    model, acc = train_lead_scoring(df)
    if model is None:
        st.warning('Not enough target variability to train a model.')
    else:
        st.success(f'Model trained. Test accuracy: {acc:.2f}')
        # score current leads
        df2 = df.copy()
        df2['age_days'] = (datetime.now() - pd.to_datetime(df2['created_at'])).dt.days
        X_full = pd.get_dummies(df2[['source','stage']].astype(str))
        # ensure columns match training columns
        # hack: train again to get feature set
        all_columns = pd.get_dummies(df[['source','stage']].astype(str)).columns
        for c in all_columns:
            if c not in X_full.columns:
                X_full[c] = 0
        X_full['ad_cost'] = df2['ad_cost']
        X_full['estimated_value'] = df2['estimated_value']
        X_full['age_days'] = df2['age_days']
        X_full = X_full.reindex(sorted(X_full.columns), axis=1)
        try:
            scores = model.predict_proba(X_full)[:,1]
            df2['score'] = scores
            st.subheader('Top 10 leads by score')
            st.dataframe(df2.sort_values('score', ascending=False).head(10))
        except Exception as e:
            st.error(f'Error scoring leads: {e}')

# ----------------------------
# Imports & Exports
# ----------------------------
elif page == 'Imports & Exports':
    st.header('Import leads / Export reports')
    uploaded = st.file_uploader('Upload leads CSV or Excel (expects lead_id,created_at,source,stage,estimated_value,ad_cost,converted)', type=['csv','xlsx'])
    if uploaded is not None:
        try:
            if uploaded.type == 'text/csv':
                new = pd.read_csv(uploaded)
            else:
                new = pd.read_excel(uploaded)
            st.success(f'Loaded {len(new)} rows')
            st.dataframe(new.head())
            if st.button('Append to dataset'):
                df = pd.concat([df,new], ignore_index=True)
                st.success('Appended to session dataset. Note: persist to DB or file for long-term storage.')
        except Exception as e:
            st.error(f'Error reading file: {e}')
    st.markdown(download_link(df, 'titan_all_leads.xlsx'), unsafe_allow_html=True)

# ----------------------------
# Settings
# ----------------------------
elif page == 'Settings':
    st.header('Application Settings')
    st.subheader('WordPress homepage integration (placeholder)')
    wp_url = st.text_input('WordPress Site URL', value=st.session_state['app_settings']['wordpress_url'])
    wp_user = st.text_input('WordPress username', value=st.session_state['app_settings']['wordpress_user'])
    wp_app_pass = st.text_input('WordPress Application Password (leave blank to disable)', value=st.session_state['app_settings']['wordpress_app_pass'])
    if st.button('Save settings'):
        st.session_state['app_settings']['wordpress_url'] = wp_url
        st.session_state['app_settings']['wordpress_user'] = wp_user
        st.session_state['app_settings']['wordpress_app_pass'] = wp_app_pass
        st.success('Settings saved to session. To actually push data to WordPress use the REST API (see docs).')

    st.subheader('Email / Notification Settings')
    smtp_host = st.text_input('SMTP Host (placeholder)')
    smtp_port = st.number_input('SMTP Port', value=587)
    smtp_user = st.text_input('SMTP Username')
    smtp_pass = st.text_input('SMTP Password', type='password')
    if st.button('Test email'):
        st.info('Email sending not implemented in this demo. Hook your SMTP settings and send via smtplib or an email service API.')

# ----------------------------
# Reports
# ----------------------------
elif page == 'Reports':
    st.header('Reports & Exports')
    st.subheader('Monthly performance')
    df['month'] = pd.to_datetime(df['created_at']).dt.to_period('M')
    monthly = df.groupby('month').agg(total_leads=('lead_id','count'), conversions=('converted','sum'), revenue=('estimated_value',lambda x: x[df['converted']==1].sum()))
    st.dataframe(monthly)
    st.markdown('You can export reports to Excel using the Exports tab or schedule them via external schedulers.')

# ----------------------------
# Footer
# ----------------------------
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('TITAN - prototype single-file app • Replace mock data with your production DB and secure integrations before public use.')
