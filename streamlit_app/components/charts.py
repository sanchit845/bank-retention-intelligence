"""
charts.py
Reusable Plotly chart components for the Streamlit dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PALETTE = {
    'blue':   '#3266ad',
    'red':    '#c0392b',
    'amber':  '#f0a500',
    'green':  '#1d9e75',
    'gray':   '#73726c',
    'purple': '#7f77dd'
}


def churn_donut(df: pd.DataFrame, title: str = "Churn Distribution"):
    counts = df['Exited'].value_counts().reset_index()
    counts.columns = ['Status', 'Count']
    counts['Status'] = counts['Status'].map({0: 'Retained', 1: 'Churned'})
    fig = px.pie(
        counts, values='Count', names='Status', title=title, hole=0.4,
        color='Status',
        color_discrete_map={'Retained': PALETTE['blue'], 'Churned': PALETTE['red']}
    )
    fig.update_layout(legend_title_text='Status')
    return fig


def churn_by_category(df: pd.DataFrame, col: str, title: str = None):
    """Bar chart of churn rate by a categorical column."""
    grp = df.groupby(col)['Exited'].mean().reset_index()
    grp['ChurnRate'] = (grp['Exited'] * 100).round(1)
    grp = grp.sort_values('ChurnRate', ascending=False)
    fig = px.bar(
        grp, x=col, y='ChurnRate',
        color='ChurnRate',
        color_continuous_scale=[PALETTE['blue'], PALETTE['amber'], PALETTE['red']],
        title=title or f'Churn Rate by {col}',
        text='ChurnRate'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        yaxis_title='Churn Rate (%)',
        coloraxis_showscale=False
    )
    return fig


def engagement_comparison(active_churn: float, inactive_churn: float):
    fig = go.Figure(data=[
        go.Bar(name='Active', x=['Active Members'],
               y=[active_churn], marker_color=PALETTE['blue'],
               text=[f'{active_churn:.1f}%'], textposition='outside'),
        go.Bar(name='Inactive', x=['Inactive Members'],
               y=[inactive_churn], marker_color=PALETTE['red'],
               text=[f'{inactive_churn:.1f}%'], textposition='outside')
    ])
    fig.update_layout(
        title='Active vs Inactive Churn Rate',
        yaxis_title='Churn Rate (%)',
        barmode='group',
        yaxis_range=[0, 35]
    )
    return fig


def product_churn_bar(df: pd.DataFrame):
    prod = df.groupby('NumOfProducts')['Exited'].mean().reset_index()
    prod['ChurnRate'] = (prod['Exited'] * 100).round(1)
    colors = [PALETTE['amber'], PALETTE['blue'], PALETTE['red'], '#8B0000']
    fig = go.Figure()
    for i, row in prod.iterrows():
        fig.add_trace(go.Bar(
            x=[f"{int(row['NumOfProducts'])} Product(s)"],
            y=[row['ChurnRate']],
            marker_color=colors[i % len(colors)],
            text=[f"{row['ChurnRate']}%"],
            textposition='outside',
            showlegend=False
        ))
    fig.update_layout(
        title='Churn Rate by Number of Products',
        yaxis_title='Churn Rate (%)',
        yaxis_range=[0, 115]
    )
    return fig


def roc_curve_plot(fpr, tpr, auc_score: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f'CatBoost (AUC = {auc_score:.4f})',
        line=dict(color=PALETTE['blue'], width=2),
        fill='tozeroy', fillcolor='rgba(50, 102, 173, 0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Random baseline',
        line=dict(color=PALETTE['gray'], width=1, dash='dash')
    ))
    fig.update_layout(
        title='ROC Curve — CatBoost',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.6, y=0.1)
    )
    return fig
