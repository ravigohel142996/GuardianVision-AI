"""
app.py (Dashboard)
===================
Streamlit dashboard for GuardianVision AI — the operator-facing UI
showing live camera status, worker compliance, and incident history.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import time

import streamlit as st

from app.utils.config_loader import get_config
from app.utils.database import get_incident_stats, get_recent_incidents, init_db
from app.utils.logger import get_logger
from app.utils.visualization import (
    build_camera_comparison_chart,
    build_missing_ppe_breakdown_chart,
    build_violations_by_risk_chart,
    build_violations_timeline_chart,
    incidents_to_dataframe,
)
from dashboard.components import (
    render_alert_feed,
    render_camera_grid,
    render_incident_table,
    render_summary_cards,
    render_sidebar,
)

logger = get_logger(__name__)


def configure_page() -> None:
    """Set Streamlit page config and inject the dark theme CSS."""
    st.set_page_config(
        page_title="GuardianVision AI — Safety Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main { background-color: #0e1117; }
        .stMetric { background-color: #1a1d24; border-radius: 8px; padding: 12px; }
        .block-container { padding-top: 1.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Main Streamlit entrypoint."""
    configure_page()
    config = get_config()
    init_db(config.reports.database_path)

    st.title("🛡️ GuardianVision AI")
    st.caption("Real-Time PPE Compliance & Industrial Safety Monitoring")

    selected_camera = render_sidebar(config)

    stats = get_incident_stats()
    render_summary_cards(stats, config)

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📹 Live Camera Feeds")
        render_camera_grid(config, selected_camera)

    with col_right:
        st.subheader("🚨 Live Alert Feed")
        incidents = get_recent_incidents(limit=10, camera_id=selected_camera)
        render_alert_feed(incidents)

    st.divider()

    st.subheader("📊 Analytics")
    all_incidents = get_recent_incidents(limit=1000)
    df = incidents_to_dataframe(all_incidents)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(build_violations_by_risk_chart(df), use_container_width=True)
        st.plotly_chart(build_camera_comparison_chart(df), use_container_width=True)
    with chart_col2:
        st.plotly_chart(build_violations_timeline_chart(df), use_container_width=True)
        st.plotly_chart(build_missing_ppe_breakdown_chart(df), use_container_width=True)

    st.divider()

    st.subheader("📋 Incident History")
    render_incident_table(all_incidents)

    if config.dashboard.refresh_interval_seconds > 0:
        time.sleep(config.dashboard.refresh_interval_seconds)
        st.rerun()


if __name__ == "__main__":
    main()
