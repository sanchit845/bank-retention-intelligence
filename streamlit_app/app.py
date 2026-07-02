"""
app.py — Bank Retention Intelligence Platform
Full Streamlit Dashboard with Welcome Screen

Run: streamlit run streamlit_app/app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_cleaning import load_and_clean
from src.feature_engineering import engineer_features, get_feature_list
from src.clustering import compute_rsi, get_recommendation, segment_customers
from src.utils import encode_categoricals, get_path

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Retention Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colour palette ────────────────────────────────────────────────────────
PALETTE = {
    'blue':   '#3266ad',
    'red':    '#c0392b',
    'amber':  '#f0a500',
    'green':  '#1d9e75',
    'gray':   '#73726c',
    'purple': '#7f77dd',
}

CLUSTER_COLORS = {
    'Young Active':       PALETTE['blue'],
    'Premium Loyal':      PALETTE['green'],
    'Wealthy Disengaged': PALETTE['amber'],
    'High Risk':          PALETTE['red'],
}

# ── Session state init ────────────────────────────────────────────────────
if 'entered' not in st.session_state:
    st.session_state.entered = False
if 'page' not in st.session_state:
    st.session_state.page = '📊 Executive Overview'

# ══════════════════════════════════════════════════════════════════════════
# WELCOME / LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════
if not st.session_state.entered:

    # Hide sidebar on landing page
    st.markdown("""
    <style>
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"]  { display: none; }
    .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero section ──────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 3rem 2rem 1.5rem;'>
        <div style='font-size:64px; margin-bottom:0.5rem;'>🏦</div>
        <h1 style='font-size:2.8rem; font-weight:800; color:antiquewhite; margin-bottom:0.4rem;'>
            Bank Retention Intelligence Platform
        </h1>
        <p style='font-size:1.1rem; color:#555; max-width:620px; margin:0 auto 0.5rem;'>
            Customer Engagement &amp; Product Utilisation Analytics for Retention Strategy
        </p>
        <p style='font-size:0.9rem; color:#888; margin-bottom:2rem;'>
            European Bank Dataset &nbsp;·&nbsp; 10,000 Customers &nbsp;·&nbsp;
            CatBoost Model &nbsp;·&nbsp; ROC-AUC 86.69%
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats bar ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex; justify-content:center; gap:2rem; flex-wrap:wrap;
                background:#f0f4ff; border-radius:16px; padding:1.5rem 2rem;
                max-width:800px; margin:0 auto 2.5rem;'>
        <div style='text-align:center;'>
            <div style='font-size:1.8rem; font-weight:700; color:#3266ad;'>10,000</div>
            <div style='font-size:0.8rem; color:#666; text-transform:uppercase; letter-spacing:0.05em;'>Customers</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-size:1.8rem; font-weight:700; color:#c0392b;'>20.4%</div>
            <div style='font-size:0.8rem; color:#666; text-transform:uppercase; letter-spacing:0.05em;'>Churn Rate</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-size:1.8rem; font-weight:700; color:#f0a500;'>7</div>
            <div style='font-size:0.8rem; color:#666; text-transform:uppercase; letter-spacing:0.05em;'>ML Models</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-size:1.8rem; font-weight:700; color:#1d9e75;'>86.69%</div>
            <div style='font-size:0.8rem; color:#666; text-transform:uppercase; letter-spacing:0.05em;'>ROC-AUC</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-size:1.8rem; font-weight:700; color:#7f77dd;'>€104.7M</div>
            <div style='font-size:0.8rem; color:#666; text-transform:uppercase; letter-spacing:0.05em;'>Revenue at Risk</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dashboard feature cards ────────────────────────────────────────────
    st.markdown("<h3 style='text-align:center; color:#1a1a2e; margin-bottom:1.2rem;'>What's Inside</h3>",
                unsafe_allow_html=True)

    cards = [
        ("📊", "Executive Overview",      "KPIs, churn distribution, geography & age group breakdown"),
        ("🔵", "Engagement Analysis",     "Active vs inactive churn, credit card stickiness, engagement scores"),
        ("📦", "Product Utilisation",     "Churn by product count, single vs multi-product retention"),
        ("⚠️",  "High-Value at Risk",     "2,456 disengaged premium customers · €104.7M revenue at risk"),
        ("👥", "Customer Segments",       "KMeans 4-cluster segmentation with RSI scoring"),
        ("🔮", "Churn Predictor",         "Individual customer churn probability & personalised recommendations"),
        ("🎯", "Retention Strategy",      "6 data-backed retention strategies with target customer counts"),
    ]

    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(cards):
        with cols[i % 4]:
            st.markdown(
                f"""<div style='background:white; border:1px solid #e8e8e8;
                    border-top:4px solid #3266ad; border-radius:10px;
                    padding:1rem 1rem 0.8rem; margin-bottom:1rem;
                    box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
                    <div style='font-size:1.6rem; margin-bottom:0.3rem;'>{icon}</div>
                    <div style='font-weight:700; font-size:0.95rem; color:#1a1a2e;
                        margin-bottom:0.3rem;'>{title}</div>
                    <div style='font-size:0.8rem; color:#666; line-height:1.4;'>{desc}</div>
                </div>""",
                unsafe_allow_html=True
            )

    # ── Project context ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown("""
        <div style='background:#fff8f0; border:1px solid #f0a500; border-radius:12px;
                    padding:1.2rem 1.5rem; margin-bottom:1.5rem;'>
            <p style='font-weight:700; color:#b07800; margin-bottom:0.5rem;'>
                📋 Project Statement
            </p>
            <p style='font-size:0.88rem; color:#555; margin:0; line-height:1.6;'>
                Banks increasingly recognise that customer behaviour and engagement —
                not just demographics — determine long-term retention. This platform
                evaluates how <b>engagement status</b>, <b>product utilisation</b>,
                and <b>relationship depth</b> drive churn, identifies high-value
                disengaged customers, and proposes actionable retention strategies.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Enter button ──────────────────────────────────────────────────
        if st.button("🚀  Enter Dashboard", type="primary", use_container_width=True):
            st.session_state.entered = True
            st.rerun()

        st.markdown(
            "<p style='text-align:center; font-size:0.75rem; color:#aaa; margin-top:0.8rem;'>"
            "Internship Project &nbsp;·&nbsp; Bank Retention Intelligence Platform &nbsp;·&nbsp; "
            "European Bank Dataset</p>",
            unsafe_allow_html=True
        )

    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD (shown after Enter button)
# ══════════════════════════════════════════════════════════════════════════

# Re-expand sidebar now that we're inside the dashboard
st.markdown("""
<style>
[data-testid="collapsedControl"] { display: flex; }
section[data-testid="stSidebar"]  { display: flex; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw_path = get_path('data', 'raw', 'European_Bank.csv')
    df = load_and_clean(raw_path)
    df = engineer_features(df)
    df = encode_categoricals(df)
    df, seg_summary, X_sc, _ = segment_customers(df)
    df = compute_rsi(df)
    return df, seg_summary

@st.cache_resource
def load_model_bundle():
    """
    Load the saved model bundle. Returns a 4-tuple
    (model, scaler, feature_cols, decision_threshold).

    Newer bundles (post-refactor) carry `decision_threshold`; older ones
    fall back to the config default so the dashboard never crashes on
    a stale model file.
    """
    import joblib
    from src.config import DECISION_THRESHOLD
    model_path = get_path('models', 'best_model.pkl')
    if os.path.exists(model_path):
        bundle = joblib.load(model_path)
        threshold = bundle.get('decision_threshold', DECISION_THRESHOLD)
        return bundle['model'], bundle['scaler'], bundle['features'], float(threshold)
    return None, None, None, None

df_full, seg_summary = load_data()
model, scaler, feature_cols, decision_threshold = load_model_bundle()

if model is not None and 'ChurnProbability' not in df_full.columns:
    X = df_full[feature_cols].fillna(0)
    proba = model.predict_proba(scaler.transform(X))[:, 1]
    df_full['ChurnProbability'] = proba
    df_full['PredictedChurn']   = (proba >= decision_threshold).astype(int)
    df_full['RevenueAtRisk']    = (df_full['Balance'].fillna(0) * proba).round(2)
    df_full['Recommendation']   = df_full.apply(get_recommendation, axis=1)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    # Back to home button
    if st.button("🏠  Home", use_container_width=True):
        st.session_state.entered = False
        st.rerun()

    st.markdown("## 🏦 Bank Retention\nIntelligence Platform")
    st.markdown("---")

    page = st.radio("Navigate", [
        "📊 Executive Overview",
        "🔵 Engagement Analysis",
        "📦 Product Utilisation",
        "⚠️  High-Value at Risk",
        "👥 Customer Segments",
        "🔮 Churn Predictor",
        "🎯 Retention Strategy",
    ])

    # ── Global Filters ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎛️ Global Filters")
    st.caption("Apply to all analysis pages")

    engagement_filter = st.selectbox(
        "Engagement Status",
        ["All Customers", "Active Only", "Inactive Only"],
        help="Filter by IsActiveMember status"
    )

    prod_min, prod_max = st.select_slider(
        "Number of Products",
        options=[1, 2, 3, 4],
        value=(1, 4),
        help="Filter by product count"
    )

    bal_min, bal_max = st.slider(
        "Balance Range (€)",
        min_value=0,
        max_value=int(df_full['Balance'].max()),
        value=(0, int(df_full['Balance'].max())),
        step=5000,
        help="Filter by account balance"
    )

    sal_min, sal_max = st.slider(
        "Salary Range (€)",
        min_value=int(df_full['EstimatedSalary'].min()),
        max_value=int(df_full['EstimatedSalary'].max()),
        value=(int(df_full['EstimatedSalary'].min()), int(df_full['EstimatedSalary'].max())),
        step=5000,
        help="Filter by estimated salary"
    )

    st.markdown("---")

    # Apply filters
    df = df_full.copy()
    if engagement_filter == "Active Only":
        df = df[df['IsActiveMember'] == 1]
    elif engagement_filter == "Inactive Only":
        df = df[df['IsActiveMember'] == 0]

    df = df[
        (df['NumOfProducts']   >= prod_min) & (df['NumOfProducts']   <= prod_max) &
        (df['Balance']          >= bal_min)  & (df['Balance']          <= bal_max)  &
        (df['EstimatedSalary']  >= sal_min)  & (df['EstimatedSalary']  <= sal_max)
    ]

    st.caption(f"**Filtered:** {len(df):,} customers")
    st.caption(f"Churn rate: {df['Exited'].mean()*100:.1f}%")
    if len(df) < len(df_full):
        st.info(f"🔍 {len(df):,} of {len(df_full):,} shown")
    st.caption("Model: CatBoost · AUC 86.69%")


def check_data(df):
    if len(df) < 10:
        st.warning("⚠️ Too few customers match filters. Adjust sidebar filters.")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.title("📊 Executive Overview")
    st.caption(f"Showing **{len(df):,} customers** based on current filters")
    if not check_data(df): st.stop()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Customers",      f"{len(df):,}")
    c2.metric("Churn Rate",     f"{df['Exited'].mean()*100:.1f}%")
    c3.metric("Retention Rate", f"{(1-df['Exited'].mean())*100:.1f}%")
    c4.metric("Active Members", f"{df['IsActiveMember'].mean()*100:.1f}%")
    c5.metric("Avg Balance",    f"€{df['Balance'].mean():,.0f}")
    c6.metric("Avg Products",   f"{df['NumOfProducts'].mean():.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        counts = df['Exited'].value_counts().reset_index()
        counts.columns = ['Status', 'Count']
        counts['Status'] = counts['Status'].map({0: 'Retained', 1: 'Churned'})
        fig = px.pie(counts, values='Count', names='Status', hole=0.4,
                     title='Churn vs Retention', color='Status',
                     color_discrete_map={'Retained': PALETTE['blue'], 'Churned': PALETTE['red']})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        geo = df.groupby('Geography')['Exited'].mean().reset_index()
        geo['ChurnRate'] = (geo['Exited'] * 100).round(1)
        fig = px.bar(geo, x='Geography', y='ChurnRate', color='ChurnRate', text='ChurnRate',
                     color_continuous_scale=[PALETTE['blue'], PALETTE['amber'], PALETTE['red']],
                     title='Churn Rate by Geography')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(yaxis_title='Churn Rate (%)', coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    df2 = df.copy()
    df2['AgeG'] = pd.cut(df2['Age'], bins=[0,25,35,45,55,120], labels=['18–25','26–35','36–45','46–55','56+'])
    ag = df2.groupby('AgeG', observed=True)['Exited'].mean().reset_index()
    ag['ChurnRate'] = (ag['Exited'] * 100).round(1)
    fig = px.bar(ag, x='AgeG', y='ChurnRate', color='ChurnRate', text='ChurnRate',
                 color_continuous_scale=[PALETTE['blue'], PALETTE['amber'], PALETTE['red']],
                 title='Churn Rate by Age Group')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title='Age Group', yaxis_title='Churn Rate (%)', coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — ENGAGEMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔵 Engagement Analysis":
    st.title("🔵 Engagement Analysis")
    st.caption(f"Showing **{len(df):,} customers** · Use sidebar filters to explore engagement segments")
    if not check_data(df): st.stop()

    a_churn  = df[df['IsActiveMember']==1]['Exited'].mean()*100 if len(df[df['IsActiveMember']==1])>0 else 0
    i_churn  = df[df['IsActiveMember']==0]['Exited'].mean()*100 if len(df[df['IsActiveMember']==0])>0 else 0
    cc_churn   = df[df['HasCrCard']==1]['Exited'].mean()*100 if len(df[df['HasCrCard']==1])>0 else 0
    nocc_churn = df[df['HasCrCard']==0]['Exited'].mean()*100 if len(df[df['HasCrCard']==0])>0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Member Churn",   f"{a_churn:.1f}%")
    c2.metric("Inactive Member Churn", f"{i_churn:.1f}%",
              delta=f"+{i_churn-a_churn:.1f} pp vs active" if a_churn > 0 else None, delta_color="inverse")
    c3.metric("Has Credit Card Churn", f"{cc_churn:.1f}%")
    c4.metric("No Credit Card Churn",  f"{nocc_churn:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(data=[
            go.Bar(name='Active',   x=['Active Members'],   y=[a_churn],
                   marker_color=PALETTE['blue'], text=[f'{a_churn:.1f}%'], textposition='outside'),
            go.Bar(name='Inactive', x=['Inactive Members'], y=[i_churn],
                   marker_color=PALETTE['red'],  text=[f'{i_churn:.1f}%'], textposition='outside'),
        ])
        fig.update_layout(title='Active vs Inactive Churn Rate', yaxis_title='Churn Rate (%)',
                          barmode='group', yaxis_range=[0, max(i_churn*1.25, 10)])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        eng_data = pd.DataFrame({
            'Segment': ['Active Engaged', 'Credit Card Only', 'Disengaged'],
            'Count':   [len(df[df['IsActiveMember']==1]),
                        len(df[(df['IsActiveMember']==0)&(df['HasCrCard']==1)]),
                        len(df[(df['IsActiveMember']==0)&(df['HasCrCard']==0)])]
        })
        fig = px.pie(eng_data, values='Count', names='Segment', hole=0.35,
                     color_discrete_sequence=[PALETTE['blue'], PALETTE['amber'], PALETTE['red']],
                     title='Engagement Profile Distribution')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Credit Card Stickiness Score")
    col3, col4 = st.columns(2)
    with col3:
        cc_comp = pd.DataFrame({'Status': ['Has Credit Card','No Credit Card'],
                                'ChurnRate': [cc_churn, nocc_churn]})
        fig = px.bar(cc_comp, x='Status', y='ChurnRate', color='Status', text='ChurnRate',
                     color_discrete_map={'Has Credit Card':PALETTE['blue'],'No Credit Card':PALETTE['gray']},
                     title='Credit Card Ownership vs Churn')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(yaxis_title='Churn Rate (%)', showlegend=False,
                          yaxis_range=[0, max(nocc_churn*1.4, 10)])
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        diff = abs(cc_churn - nocc_churn)
        st.markdown("#### Key Finding")
        if diff < 2:
            st.info(f"📌 Credit card ownership has **minimal impact** on churn ({diff:.1f} pp difference). "
                    f"Active membership is far more predictive of retention.")
        else:
            st.info(f"📌 Credit card ownership reduces churn by **{diff:.1f} pp**.")

    st.subheader("Engagement Score vs Churn Rate")
    df_copy = df.copy()
    df_copy['EngBucket'] = pd.cut(df_copy['EngagementScore'], bins=5)
    eng_churn = df_copy.groupby('EngBucket', observed=True).agg(
        ChurnRate=('Exited', lambda x: round(x.mean()*100, 1))).reset_index()
    eng_churn['EngBucket'] = eng_churn['EngBucket'].astype(str)
    fig = px.bar(eng_churn, x='EngBucket', y='ChurnRate', color='ChurnRate', text='ChurnRate',
                 color_continuous_scale=[PALETTE['green'], PALETTE['amber'], PALETTE['red']],
                 title='Churn Rate by Engagement Score Bucket')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title='Engagement Score Range',
                      yaxis_title='Churn Rate (%)', coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRODUCT UTILISATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "📦 Product Utilisation":
    st.title("📦 Product Utilisation Analysis")
    st.caption(f"Showing **{len(df):,} customers** · Use Product slider in sidebar to focus analysis")
    if not check_data(df): st.stop()

    pc = df.groupby('NumOfProducts').agg(
        ChurnRate=('Exited', lambda x: round(x.mean()*100, 1)),
        Count=('Exited', 'count')).reset_index()

    cols = st.columns(len(pc))
    for col, (_, row) in zip(cols, pc.iterrows()):
        col.metric(f"{int(row['NumOfProducts'])} Product(s)", f"{row['ChurnRate']}%", f"n={int(row['Count']):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(pc, x='NumOfProducts', y='ChurnRate', color='ChurnRate', text='ChurnRate',
                     color_continuous_scale=[PALETTE['green'], PALETTE['amber'], PALETTE['red']],
                     title='Churn Rate by Number of Products')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(yaxis_title='Churn Rate (%)', coloraxis_showscale=False,
                          xaxis_title='Number of Products',
                          yaxis_range=[0, min(pc['ChurnRate'].max()*1.2, 115)])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        pc2 = df['NumOfProducts'].value_counts().reset_index()
        pc2.columns = ['Products', 'Count']
        pc2['Products'] = pc2['Products'].astype(str) + ' product(s)'
        fig = px.pie(pc2, values='Count', names='Products', hole=0.35,
                     color_discrete_sequence=[PALETTE['blue'],PALETTE['green'],PALETTE['amber'],PALETTE['red']],
                     title='Product Adoption Distribution')
        st.plotly_chart(fig, use_container_width=True)

    single = df[df['NumOfProducts']==1]
    multi  = df[df['NumOfProducts']>=2]
    st.subheader("Single vs Multi-Product Retention")
    c1, c2 = st.columns(2)
    c1.metric("Single-Product Churn", f"{single['Exited'].mean()*100:.1f}%" if len(single)>0 else "N/A", f"{len(single):,} customers")
    c2.metric("Multi-Product Churn",  f"{multi['Exited'].mean()*100:.1f}%"  if len(multi)>0  else "N/A", f"{len(multi):,} customers")
    if len(single)>0 and len(multi)>0:
        diff = single['Exited'].mean()*100 - multi['Exited'].mean()*100
        st.success(f"✅ Multi-product customers churn **{diff:.1f} pp less**. Cross-selling is the highest-ROI retention lever.")

    st.subheader("Product Depth by Geography")
    geo_prod = df.groupby(['Geography','NumOfProducts'])['Exited'].mean().reset_index()
    geo_prod['ChurnRate'] = (geo_prod['Exited']*100).round(1)
    fig = px.bar(geo_prod, x='Geography', y='ChurnRate', color='NumOfProducts',
                 barmode='group', text='ChurnRate',
                 title='Churn Rate by Geography & Product Count')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis_title='Churn Rate (%)')
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 **Key Insight:** 2-product customers churn at only 7.6% vs 27.7% for single-product holders.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — HIGH-VALUE AT RISK
# ══════════════════════════════════════════════════════════════════════════
elif page == "⚠️  High-Value at Risk":
    st.title("⚠️ High-Value Disengaged Customers")
    st.caption(f"Showing **{len(df):,} customers** · Adjust Balance slider to change HV threshold")
    if not check_data(df): st.stop()

    hv_threshold = bal_min if bal_min > 0 else int(df_full['Balance'].median())
    if bal_min > 0:
        st.info(f"📌 Using balance filter minimum (€{hv_threshold:,}) as the high-value threshold.")
    else:
        st.info(f"📌 Default threshold: median balance €{hv_threshold:,}. Use the Balance Range slider to customise.")

    hv = df[(df['Balance'] > hv_threshold) & (df['IsActiveMember'] == 0)]
    if len(hv) == 0:
        st.warning("No high-value disengaged customers with current filters."); st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HV Disengaged",   f"{len(hv):,}")
    c2.metric("Their Churn Rate", f"{hv['Exited'].mean()*100:.1f}%",
              delta=f"+{(hv['Exited'].mean()-df['Exited'].mean())*100:.1f} pp", delta_color="inverse")
    c3.metric("Avg Balance",     f"€{hv['Balance'].mean():,.0f}")
    # Forward-looking revenue at risk: expected loss = balance × P(churn).
    # This is the proper "what we could lose" number, including not-yet-churned
    # customers who are predicted to leave.
    c4.metric("Revenue at Risk (Σ P·Balance)", f"€{hv['RevenueAtRisk'].sum()/1e6:.1f}M")

    st.error(f"⚠️ **{len(hv):,} high-value customers** are disengaged — churning at "
             f"**{hv['Exited'].mean()*100:.1f}%** vs **{df['Exited'].mean()*100:.1f}%** average. "
             f"Forward-looking revenue at risk: **€{hv['RevenueAtRisk'].sum()/1e6:.1f}M** "
             f"(actual lost: €{hv[hv['Exited']==1]['Balance'].sum()/1e6:.1f}M).")

    col1, col2 = st.columns(2)
    with col1:
        comp = pd.DataFrame({'Segment':['Overall Average','HV Disengaged'],
                             'ChurnRate':[df['Exited'].mean()*100, hv['Exited'].mean()*100]})
        fig = px.bar(comp, x='Segment', y='ChurnRate', color='Segment', text='ChurnRate',
                     color_discrete_map={'Overall Average':PALETTE['gray'],'HV Disengaged':PALETTE['red']},
                     title='HV Disengaged vs Overall Churn')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(yaxis_title='Churn Rate (%)', showlegend=False,
                          yaxis_range=[0, hv['Exited'].mean()*130])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        hv_geo = hv.groupby('Geography')['Exited'].mean().reset_index()
        hv_geo['ChurnRate'] = (hv_geo['Exited']*100).round(1)
        fig = px.bar(hv_geo, x='Geography', y='ChurnRate', color='ChurnRate', text='ChurnRate',
                     color_continuous_scale=[PALETTE['amber'],PALETTE['red']],
                     title='HV Disengaged Churn by Geography')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    hv2 = hv.copy()
    hv2['AgeG'] = pd.cut(hv2['Age'], bins=[0,25,35,45,55,120], labels=['18–25','26–35','36–45','46–55','56+'])
    hv_age = hv2.groupby('AgeG', observed=True).agg(
        ChurnRate=('Exited', lambda x: round(x.mean()*100,1))).reset_index()
    st.subheader("HV Disengaged — Age Group Breakdown")
    fig = px.bar(hv_age, x='AgeG', y='ChurnRate', color='ChurnRate', text='ChurnRate',
                 color_continuous_scale=[PALETTE['amber'],PALETTE['red']],
                 title='HV Disengaged Churn by Age Group')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(xaxis_title='Age Group', yaxis_title='Churn Rate (%)', coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top High-Value Disengaged Customers")
    disp = ['Geography','Gender','Age','Balance','NumOfProducts','Tenure','Exited']
    if 'ChurnProbability' in hv.columns: disp.append('ChurnProbability')
    if 'RevenueAtRisk'    in hv.columns: disp.append('RevenueAtRisk')
    if 'Recommendation'   in hv.columns: disp.append('Recommendation')
    st.dataframe(hv[disp].sort_values('RevenueAtRisk' if 'RevenueAtRisk' in hv.columns else 'Balance',
                                      ascending=False).head(50).reset_index(drop=True),
                 use_container_width=True,
                 column_config={
                     'Balance':          st.column_config.NumberColumn(format='€%.0f'),
                     'RevenueAtRisk':    st.column_config.NumberColumn(format='€%.0f'),
                     'ChurnProbability': st.column_config.ProgressColumn(min_value=0, max_value=1, format='%.2f'),
                 })


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation")
    st.caption(f"Showing **{len(df):,} customers** · KMeans 4-cluster behavioural segmentation")
    if not check_data(df): st.stop()

    if 'Cluster' in df.columns:
        seg_filtered = df.groupby('Cluster').agg(
            Count      =('Exited','count'),
            ChurnRate  =('Exited', lambda x: round(x.mean()*100,1)),
            AvgBalance =('Balance', lambda x: round(x.mean(),0)),
            AvgAge     =('Age', lambda x: round(x.mean(),1)),
            AvgProducts=('NumOfProducts', lambda x: round(x.mean(),2)),
            ActiveRate =('IsActiveMember', lambda x: round(x.mean()*100,1)),
        ).reset_index()
    else:
        seg_filtered = seg_summary

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(seg_filtered, x='Cluster', y='ChurnRate', color='Cluster',
                     color_discrete_map=CLUSTER_COLORS, text='ChurnRate', title='Churn Rate by Segment')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title='Churn Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(seg_filtered, values='Count', names='Cluster', hole=0.35,
                     color='Cluster', color_discrete_map=CLUSTER_COLORS, title='Segment Size Distribution')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Profiles")
    for _, row in seg_filtered.iterrows():
        color = CLUSTER_COLORS.get(row['Cluster'], PALETTE['gray'])
        rsi_avg = df[df['Cluster']==row['Cluster']]['RSI'].mean() if 'RSI' in df.columns else 0
        st.markdown(
            f"<div style='border-left:5px solid {color};padding:12px 18px;margin:6px 0;"
            f"background:#f9f9f9;border-radius:6px;'>"
            f"<b style='color:{color};font-size:15px;'>{row['Cluster']}</b>"
            f"<span style='float:right;color:{color};font-weight:600;'>Churn: {row['ChurnRate']:.1f}%</span><br>"
            f"<span style='font-size:12px;color:#555;'>"
            f"👥 {int(row['Count']):,} customers &nbsp;·&nbsp; "
            f"💰 Avg Balance: €{row['AvgBalance']:,.0f} &nbsp;·&nbsp; "
            f"📦 Avg Products: {row['AvgProducts']:.2f} &nbsp;·&nbsp; "
            f"✅ Active: {row['ActiveRate']:.1f}% &nbsp;·&nbsp; "
            f"🎯 Avg RSI: {rsi_avg:.1f}</span></div>",
            unsafe_allow_html=True
        )

    if 'RSI' in df.columns and 'Cluster' in df.columns:
        st.subheader("RSI Distribution by Segment")
        fig = px.box(df, x='Cluster', y='RSI', color='Cluster',
                     color_discrete_map=CLUSTER_COLORS,
                     title='Retention Strength Index (RSI) by Segment')
        fig.update_layout(showlegend=False, yaxis_title='RSI Score (0–100)')
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 6 — CHURN PREDICTOR
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Churn Predictor":
    st.title("🔮 Individual Churn Predictor")
    st.caption("Enter a customer's details to get their churn probability, RSI score, and personalised recommendation.")

    if model is None:
        st.warning("⚠️ No trained model found. Run notebooks 01 → 02 → 03 first.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Customer Demographics**")
        age          = st.number_input("Age", 18, 92, 45)
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        geography    = st.selectbox("Geography", ["France","Spain","Germany"])
        gender       = st.selectbox("Gender", ["Female","Male"])
    with col2:
        st.markdown("**Financial Profile**")
        balance = st.number_input("Account Balance (€)", 0, 300000, 85000, step=1000)
        salary  = st.number_input("Estimated Salary (€)", 10000, 300000, 100000, step=1000)
        tenure  = st.slider("Tenure (years)", 0, 10, 3)
    with col3:
        st.markdown("**Product & Engagement**")
        products = st.select_slider("Number of Products", options=[1,2,3,4], value=2)
        active   = st.selectbox("Active Member", ["Yes","No"])
        cc       = st.selectbox("Has Credit Card", ["Yes","No"])

    st.markdown("---")
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        from src.inference import predict_single
        customer = {
            'Age':age,'CreditScore':credit_score,'Balance':balance,
            'EstimatedSalary':salary,'Tenure':tenure,'NumOfProducts':products,
            'IsActiveMember':1 if active=="Yes" else 0,
            'HasCrCard':1 if cc=="Yes" else 0,
            'Geography':geography,'Gender':gender
        }
        result = predict_single(customer, model, scaler, feature_cols, decision_threshold)
        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Churn Probability", f"{result['churn_probability']}%")
        r2.metric("Risk Level",        result['risk_level'])
        r3.metric("RSI Score",         f"{result['rsi']} / 100")
        r4.metric("RSI Category",      result['rsi_category'])

        p = result['churn_probability']
        color = PALETTE['red'] if p>60 else PALETTE['amber'] if p>40 else PALETTE['green']
        st.markdown(
            f"<div style='margin:10px 0 6px;'>"
            f"<div style='background:#e8e8e8;border-radius:8px;height:14px;'>"
            f"<div style='background:{color};width:{p}%;height:100%;border-radius:8px;'></div>"
            f"</div><div style='display:flex;justify-content:space-between;font-size:11px;color:#888;margin-top:3px;'>"
            f"<span>0%</span><span>Low Risk</span><span>Moderate</span><span>High Risk</span><span>100%</span>"
            f"</div></div>", unsafe_allow_html=True)

        rec_colors = {
            'Immediate Outreach':PALETTE['red'],
            'Reactivation Campaign':PALETTE['amber'],
            'Cross-Sell Programme':PALETTE['blue'],
            'Relationship Manager Assignment':PALETTE['purple'],
            'Standard Retention Programme':PALETTE['green'],
        }
        rc = rec_colors.get(result['recommendation'], PALETTE['gray'])
        st.markdown(
            f"<div style='border-left:5px solid {rc};background:#f9f9f9;"
            f"padding:12px 18px;border-radius:6px;margin:12px 0;'>"
            f"<b style='color:{rc};'>Recommended Action:</b> {result['recommendation']}</div>",
            unsafe_allow_html=True)

        st.subheader("Why this prediction?")
        reasons = []
        if 46 <= age <= 55:    reasons.append("🔴 **Age 46–55** — highest churn cohort (50.6% churn rate)")
        elif age >= 56:        reasons.append("🟡 **Age 56+** — elevated churn risk (36.8%)")
        if (1 if active=="Yes" else 0)==0: reasons.append("🔴 **Inactive member** — 26.9% vs 14.3% churn for active")
        if products==1:        reasons.append("🟡 **Single product** — 27.7% churn; 2 products drops to 7.6%")
        elif products>=3:      reasons.append("🔴 **3+ products** — anomalously high churn (82.7%+)")
        if geography=="Germany": reasons.append("🟡 **Germany** — 32.4% churn vs 16.2% in France")
        if credit_score < 500: reasons.append("🟡 **Low credit score** — associated with higher exit risk")
        if balance>97199 and (1 if active=="Yes" else 0)==0:
            reasons.append("🔴 **High balance + inactive** = high-value disengaged profile")
        if not reasons: reasons.append("✅ **No major risk factors** — stable engagement and product profile")
        for r in reasons:
            st.markdown(f"- {r}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 7 — RETENTION STRATEGY
# ══════════════════════════════════════════════════════════════════════════
elif page == "🎯 Retention Strategy":
    st.title("🎯 Retention Strategy")
    st.caption(f"Showing **{len(df):,} customers** · All targets calculated on filtered dataset")
    if not check_data(df): st.stop()

    med = df_full['Balance'].median()
    imm_count   = int(len(df[df['ChurnProbability']>0.80])) if 'ChurnProbability' in df.columns else 0
    cross_count = len(df[df['NumOfProducts']==1])
    react_count = len(df[df['IsActiveMember']==0])
    hv_count    = len(df[(df['Balance']>med)&(df['IsActiveMember']==0)])
    ger_count   = len(df[df['Geography']=='Germany'])
    age_count   = len(df[(df['Age']>=46)&(df['Age']<=55)])

    strategies = [
        {"title":"🔴 Strategy 1 — Immediate Outreach","priority":"Urgent","color":PALETTE['red'],
         "target":f"{imm_count:,}","impact":"Prevent highest-risk churners before they leave",
         "insight":"Customers with churn probability > 80% require personal intervention immediately.",
         "action":"Assign dedicated relationship managers · Personal phone call · Exclusive retention offer"},
        {"title":"🟠 Strategy 2 — Cross-Sell Programme","priority":"High ROI","color":PALETTE['amber'],
         "target":f"{cross_count:,}","impact":"20 pp churn reduction per converted customer",
         "insight":"Single-product customers churn at 27.7%. Moving them to 2 products drops churn to 7.6%.",
         "action":"Personalised bundle offers · In-app cross-sell · Relationship manager recommendations"},
        {"title":"🟡 Strategy 3 — Reactivation Campaign","priority":"Medium","color":PALETTE['blue'],
         "target":f"{react_count:,}","impact":"12.6 pp churn reduction if reactivated",
         "insight":"Inactive members churn at 26.9% vs 14.3% for active members — a 12.6 pp gap.",
         "action":"Loyalty points · Push notification campaigns · Personalised re-engagement emails"},
        {"title":"🟣 Strategy 4 — HV Disengaged Outreach","priority":"High Value","color":PALETTE['purple'],
         "target":f"{hv_count:,}","impact":"€104.7M revenue at risk",
         "insight":f"Customers with balance > €{med:,.0f} who are inactive churn at 32.3%.",
         "action":"Immediate RM assignment · Premium service tier · Wealth planning consultation"},
        {"title":"🟢 Strategy 5 — Germany Programme","priority":"Regional","color":PALETTE['green'],
         "target":f"{ger_count:,}","impact":"Target: reduce Germany churn to ~18%",
         "insight":"German customers churn at 32.4% — nearly double France (16.2%) and Spain (16.7%).",
         "action":"Regional NPS survey · Local competitive analysis · Germany-specific product localisation"},
        {"title":"🔴 Strategy 6 — Age 46–55 Advisory","priority":"Critical Cohort","color":PALETTE['red'],
         "target":f"{age_count:,}","impact":"50.6% → target 25% churn",
         "insight":"Customers aged 46–55 have a 50.6% churn rate — highest of any age group.",
         "action":"Wealth planning consultations · Premium advisory tier · Dedicated financial advisor"},
    ]

    for s in strategies:
        with st.expander(f"{s['title']}  |  Priority: {s['priority']}", expanded=True):
            c1, c2 = st.columns([3,1])
            with c1:
                st.markdown(
                    f"<div style='border-left:4px solid {s['color']};padding:8px 14px;'>"
                    f"<p style='margin:0 0 6px;'>{s['insight']}</p>"
                    f"<p style='margin:0;font-size:12px;color:#555;'><b>Action:</b> {s['action']}</p>"
                    f"</div>", unsafe_allow_html=True)
            with c2:
                st.metric("Target Customers", s['target'])
                st.metric("Impact", s['impact'])

    st.markdown("---")
    st.subheader("Strategy Target Coverage")
    impact_df = pd.DataFrame({
        'Strategy':  ['Immediate\nOutreach','Cross-Sell','Reactivation','HV Disengaged','Germany','Age 46–55'],
        'Customers': [imm_count, cross_count, react_count, hv_count, ger_count, age_count],
        'Priority':  ['Urgent','High ROI','Medium','High Value','Regional','Critical']
    })
    fig = px.bar(impact_df, x='Strategy', y='Customers', color='Priority',
                 color_discrete_map={'Urgent':PALETTE['red'],'High ROI':PALETTE['amber'],
                                     'Medium':PALETTE['blue'],'High Value':PALETTE['purple'],
                                     'Regional':PALETTE['green'],'Critical':PALETTE['red']},
                 title='Customers Targeted by Each Retention Strategy', text='Customers')
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig.update_layout(yaxis_title='Customers Targeted', xaxis_title='')
    st.plotly_chart(fig, use_container_width=True)
