import streamlit as st
import json
import os
import time
from PIL import Image

# FusionX UI Configuration
st.set_page_config(page_title="FusionX Command Center", layout="wide", initial_sidebar_state="expanded")

# Custom Styling for "WOW" factor
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stAlert { border-radius: 10px; }
    .narrator-box { 
        background: linear-gradient(90deg, #1e3a8a, #1e40af); 
        padding: 20px; border-radius: 15px; border-left: 5px solid #3b82f6;
        margin-bottom: 20px;
    }
    .evidence-card {
        border: 1px solid #374151; padding: 10px; border-radius: 10px; background: #1f2937;
    }
    </style>
""", unsafe_allow_html=True)

def load_intelligence():
    log_file = "FusionX_Intelligence.json"
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                data = json.load(f)
                return data
        except: return []
    return []

def main():
    st.title("🛡️ FusionX AI: Situational Command Center")
    st.write("Real-time Multi-Agent Traffic Perception & Reasoning Engine")

    # Load live data
    intel_logs = load_intelligence()
    
    if not intel_logs:
        st.warning("Waiting for Perception Engine to start... (Run run_official_reltr.py first)")
        if st.button("Refresh Dashboard"): st.rerun()
        return

    latest = intel_logs[-1]

    # --- TOP ROW: THE NARRATOR ---
    st.markdown(f"""
    <div class="narrator-box">
        <h3 style='margin:0; color:#93c5fd;'>🎙️ LIVE AI NARRATION</h3>
        <p style='font-size:1.2rem; margin-top:10px;'>"{latest.get('last_narration', 'Observing traffic pattern...')}"</p>
    </div>
    """, unsafe_allow_html=True)

    # --- MAIN CONTENT: LEFT (Consensus) | RIGHT (Evidence) ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🤖 Swarm Intelligence")
        verdict = latest.get('swarm_verdict', 'Stable')
        if "ACCIDENT" in verdict.upper() or "CRASH" in verdict.upper():
            st.error(f"### CRITICAL ALERT: {verdict}")
        else:
            st.success(f"### Status: {verdict}")
        
        st.write("---")
        st.write("**Detected Objects:**")
        st.write(", ".join(latest.get('objs', [])))
        
        st.write("**Active Hazards:**")
        if latest.get('hazards'):
            for h in latest['hazards']: st.warning(h)
        else:
            st.write("No physical hazards detected.")

    with col2:
        st.subheader("📸 Latest Evidence Capture")
        # Find the most recent accident image
        evidence_frames = [log for log in intel_logs if log.get('is_accident') and log.get('evidence_img')]
        if evidence_frames:
            last_evidence = evidence_frames[-1]
            img_path = last_evidence['evidence_img']
            if os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, caption=f"IMPACT EVIDENCE: {last_evidence['timestamp']}", use_column_width=True)
                st.markdown(f"<div class='evidence-card'><b>Verdict:</b> {last_evidence['swarm_verdict']}</div>", unsafe_allow_html=True)
            else:
                st.info("Evidence file found in logs but missing on disk.")
        else:
            st.info("No accident evidence captured yet. Monitoring flow...")

    # --- BOTTOM ROW: LOG HISTORY ---
    st.write("---")
    st.subheader("📜 Session Global History")
    history_df = []
    for log in reversed(intel_logs):
        history_df.append({
            "Time": log['timestamp'],
            "Agent Consensus": log['swarm_verdict'],
            "Objects": ", ".join(log['objs']),
            "Alert": "⚠️" if log['is_accident'] else "✅"
        })
    st.table(history_df[:10])

    # Auto-refresh every 3 seconds
    time.sleep(3)
    st.rerun()

if __name__ == "__main__":
    main()
