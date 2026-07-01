"""
metric_cards.py
Reusable metric card components for the Streamlit dashboard.
"""

import streamlit as st


def kpi_card(label: str, value: str, delta: str = None,
             delta_color: str = "normal", help: str = None):
    """Render a single KPI metric card."""
    st.metric(label=label, value=value, delta=delta,
              delta_color=delta_color, help=help)


def kpi_row(metrics: list):
    """
    Render a row of KPI cards.

    Args:
        metrics: list of dicts with keys: label, value, delta (opt), delta_color (opt)
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            kpi_card(
                label=m.get('label', ''),
                value=m.get('value', ''),
                delta=m.get('delta'),
                delta_color=m.get('delta_color', 'normal'),
                help=m.get('help')
            )


def colored_metric_card(label: str, value: str, color: str = "#3266ad",
                         subtitle: str = None):
    """
    Render a styled colored metric card using HTML.
    """
    subtitle_html = f"<div style='font-size:11px;color:#888;margin-top:2px'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""<div style='background:{color}11;border-left:4px solid {color};
            padding:12px 16px;border-radius:8px;margin:4px 0;'>
            <div style='font-size:11px;color:#666;text-transform:uppercase;
                letter-spacing:0.04em;margin-bottom:2px;'>{label}</div>
            <div style='font-size:24px;font-weight:700;color:{color};'>{value}</div>
            {subtitle_html}
        </div>""",
        unsafe_allow_html=True
    )


def rsi_indicator(rsi: float):
    """
    Render a visual RSI bar with category label.
    """
    if rsi <= 30:
        color, cat = '#c0392b', 'High Risk'
    elif rsi <= 60:
        color, cat = '#f0a500', 'Moderate Risk'
    elif rsi <= 80:
        color, cat = '#3266ad', 'Stable'
    else:
        color, cat = '#1d9e75', 'Loyal'

    st.markdown(
        f"""<div style='margin:8px 0;'>
            <div style='display:flex;justify-content:space-between;
                font-size:12px;margin-bottom:4px;'>
                <span>RSI: <b>{rsi:.1f}</b> / 100</span>
                <span style='color:{color};font-weight:600;'>{cat}</span>
            </div>
            <div style='background:#eee;border-radius:6px;height:10px;'>
                <div style='background:{color};width:{rsi}%;
                    height:100%;border-radius:6px;'></div>
            </div>
        </div>""",
        unsafe_allow_html=True
    )
