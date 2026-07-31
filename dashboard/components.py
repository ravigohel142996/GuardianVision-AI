"""
components.py (Dashboard)
===========================
Reusable Streamlit UI components for the GuardianVision AI dashboard.
Separated from `app.py` to keep the main page flow readable and to
allow individual widgets to be reused (e.g. in a future admin page).
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import streamlit as st

from app.utils.config_loader import AppConfig


def render_sidebar(config: AppConfig) -> Optional[str]:
    """
    Render the sidebar navigation and camera filter.

    Returns
    -------
    Optional[str]
        The selected camera_id, or None for "All Cameras".
    """
    with st.sidebar:
        st.header("⚙️ Controls")

        camera_options = ["All Cameras"] + [cam.id for cam in config.cameras]
        camera_labels = {"All Cameras": "All Cameras"}
        camera_labels.update({cam.id: cam.name for cam in config.cameras})

        selected = st.selectbox(
            "Filter by Camera",
            options=camera_options,
            format_func=lambda cid: camera_labels.get(cid, cid),
        )

        st.divider()
        st.subheader("System Info")
        st.text(f"Version: {config.version}")
        st.text(f"Environment: {config.environment}")
        st.text(f"Cameras configured: {len(config.cameras)}")
        st.text(f"Zones configured: {len(config.zones)}")

        st.divider()
        st.caption("GuardianVision AI © 2026")

        return None if selected == "All Cameras" else selected


def render_summary_cards(stats: dict, config: AppConfig) -> None:
    """Render the top-row KPI metric cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Incidents", stats.get("total_incidents", 0))
    with col2:
        st.metric("High-Risk Incidents", stats.get("high_risk_incidents", 0))
    with col3:
        st.metric("Active Cameras", len(config.enabled_cameras()))
    with col4:
        st.metric("Monitored Zones", len(config.zones))


def render_camera_grid(config: AppConfig, selected_camera: Optional[str]) -> None:
    """
    Render camera feed placeholders in a responsive grid.

    Note: Actual live frame rendering is wired up by the pipeline runner
    pushing frames into `st.session_state`; this renders the layout and
    a static placeholder when no live frame is yet available, so the UI
    is inspectable even before a camera stream is attached.
    """
    cameras = config.cameras if not selected_camera else [
        c for c in config.cameras if c.id == selected_camera
    ]

    if not cameras:
        st.info("No cameras match the current filter.")
        return

    cols = st.columns(2)
    for idx, camera in enumerate(cameras):
        with cols[idx % 2]:
            status_icon = "🟢" if camera.enabled else "🔴"
            st.markdown(f"**{status_icon} {camera.name}** (`{camera.id}`)")

            frame_key = f"latest_frame_{camera.id}"
            if frame_key in st.session_state:
                st.image(st.session_state[frame_key], channels="BGR", use_container_width=True)
            else:
                st.info("Waiting for stream connection...")


def render_alert_feed(incidents: List[dict]) -> None:
    """Render a scrollable feed of the most recent violation alerts."""
    if not incidents:
        st.success("No active violations. All monitored workers compliant. ✅")
        return

    risk_colors = {"high": "🔴", "medium": "🟠", "low": "🟡"}

    for incident in incidents:
        icon = risk_colors.get(incident["risk_level"], "⚪")
        missing = ", ".join(incident["missing_ppe"])
        with st.container(border=True):
            st.markdown(
                f"{icon} **Worker #{incident['worker_id']}** — {incident['camera_id']}"
            )
            st.caption(f"Missing: {missing} · {incident['timestamp']}")


def render_incident_table(incidents: List[dict]) -> None:
    """Render the full incident history as a sortable, filterable table."""
    if not incidents:
        st.info("No incidents logged yet.")
        return

    df = pd.DataFrame(incidents)
    df["missing_ppe"] = df["missing_ppe"].apply(
        lambda items: ", ".join(items) if isinstance(items, list) else items
    )
    df = df[
        ["id", "timestamp", "worker_id", "camera_id", "zone_id",
         "missing_ppe", "risk_level", "confidence"]
    ]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_level": st.column_config.TextColumn("Risk"),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=1, format="%.2f"
            ),
        },
    )
