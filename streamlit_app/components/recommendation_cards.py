"""
recommendation_cards.py
Reusable recommendation card components for the Streamlit dashboard.
"""

import streamlit as st


RECOMMENDATION_CONFIG = {
    'Immediate Outreach': {
        'color': '#c0392b',
        'bg': '#fff0f0',
        'icon': '🚨',
        'description': 'This customer is at critical churn risk (>80%). Assign a relationship manager immediately for personal outreach.'
    },
    'Reactivation Campaign': {
        'color': '#f0a500',
        'bg': '#fffbf0',
        'icon': '🔄',
        'description': 'This customer is inactive. Target with a reactivation campaign — loyalty points, mobile app engagement, or a personalised offer.'
    },
    'Cross-Sell Programme': {
        'color': '#3266ad',
        'bg': '#f0f4ff',
        'icon': '📦',
        'description': 'Single-product customer. Introducing a second product drops churn risk from 27.7% to 7.6%. Recommend bundles.'
    },
    'Relationship Manager Assignment': {
        'color': '#7f77dd',
        'bg': '#f4f3ff',
        'icon': '🤝',
        'description': 'High-value customer. Assign a dedicated relationship manager to deepen engagement and prevent silent churn.'
    },
    'Standard Retention Programme': {
        'color': '#1d9e75',
        'bg': '#f0faf6',
        'icon': '✅',
        'description': 'Low risk customer. Include in standard loyalty and engagement programmes.'
    }
}


def recommendation_card(recommendation: str, churn_prob: float = None, rsi: float = None):
    """
    Render a styled recommendation card.
    """
    config = RECOMMENDATION_CONFIG.get(recommendation, {
        'color': '#73726c', 'bg': '#f8f8f8',
        'icon': 'ℹ️', 'description': recommendation
    })

    prob_html = ""
    if churn_prob is not None:
        prob_html = f"<div style='font-size:12px;color:#666;margin-top:6px;'>Churn probability: <b>{churn_prob:.1f}%</b></div>"

    rsi_html = ""
    if rsi is not None:
        rsi_html = f"<div style='font-size:12px;color:#666;'>RSI: <b>{rsi:.1f} / 100</b></div>"

    st.markdown(
        f"""<div style='background:{config["bg"]};border-left:5px solid {config["color"]};
            padding:14px 18px;border-radius:8px;margin:8px 0;'>
            <div style='font-size:18px;margin-bottom:6px;'>{config["icon"]}
                <span style='font-size:15px;font-weight:700;color:{config["color"]};
                    margin-left:8px;'>{recommendation}</span>
            </div>
            <div style='font-size:13px;color:#555;line-height:1.5;'>
                {config["description"]}
            </div>
            {prob_html}
            {rsi_html}
        </div>""",
        unsafe_allow_html=True
    )


def strategy_summary_card(title: str, priority: str, customers: int,
                           impact: str, action: str, color: str = '#3266ad'):
    """
    Render a strategy summary card for the retention strategy page.
    """
    st.markdown(
        f"""<div style='border:1px solid #e0e0e0;border-radius:10px;
            padding:16px 20px;margin:8px 0;
            border-top:4px solid {color};'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                <div>
                    <div style='font-size:11px;color:#888;font-weight:600;
                        text-transform:uppercase;margin-bottom:4px;'>{priority}</div>
                    <div style='font-size:16px;font-weight:700;margin-bottom:8px;'>{title}</div>
                    <div style='font-size:12px;color:#555;'><b>Action:</b> {action}</div>
                </div>
                <div style='text-align:right;min-width:120px;'>
                    <div style='font-size:20px;font-weight:700;color:{color};'>{customers:,}</div>
                    <div style='font-size:11px;color:#888;'>customers</div>
                    <div style='font-size:12px;font-weight:600;margin-top:6px;
                        color:{color};'>{impact}</div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True
    )
