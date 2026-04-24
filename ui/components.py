import streamlit as st
import pandas as pd
from datetime import datetime
from config import DOMAINS

def render_match_dashboard(domain_state):
    """Render the domain dashboard with current statistics"""
    
    # Get domain config
    selected_domain = st.session_state.get('selected_domain', 'Cricket')
    domain_cfg = DOMAINS[selected_domain]
    
    # Metrics
    metrics = domain_cfg.get('metrics', [])
    cols = st.columns(len(metrics))
    
    for i, metric in enumerate(metrics):
        with cols[i]:
            val = domain_state.get(metric['key'], 0)
            # Special handling for overs_display in Cricket
            if metric['key'] == 'overs_display' and selected_domain == 'Cricket':
                val = f"{domain_state.get('overs', 0)}.{domain_state.get('balls', 0)}"
            elif metric['key'] == 'run_rate' and selected_domain == 'Cricket':
                overs = domain_state.get('overs', 0)
                balls = domain_state.get('balls', 0)
                runs = domain_state.get('runs', 0)
                val = f"{(runs / max(overs + balls/6, 1)):.2f}"
            
            st.metric(
                metric['label'],
                f"{val}{metric['suffix']}",
                delta=None
            )
    
    st.divider()
    
    # Domain information
    col1, col2 = st.columns(2)
    
    if selected_domain == "Cricket":
        with col1:
            st.markdown("**🏏 On Strike**")
            st.markdown(f"### {domain_state.get('striker', 'N/A')}")
            st.caption(f"Facing {domain_state.get('bowler', 'N/A')}")
        
        with col2:
            st.markdown("**🎯 Bowling**")
            st.markdown(f"### {domain_state.get('bowler', 'N/A')}")
            st.caption(f"Phase: {domain_state.get('phase', 'N/A')}")
    elif selected_domain == "Security":
        with col1:
            st.markdown("**📍 Location**")
            st.markdown(f"### {domain_state.get('location', 'N/A')}")
        with col2:
            st.markdown("**🛡️ Status**")
            st.markdown(f"### {domain_state.get('security_level', 'Normal')}")
    elif selected_domain == "Traffic":
        with col1:
            st.markdown("**🚦 Intersection**")
            st.markdown(f"### {domain_state.get('intersection', 'N/A')}")
        with col2:
            st.markdown("**📊 Phase**")
            st.markdown(f"### {domain_state.get('phase', 'N/A')}")

def render_video_player(video_path):
    """Render video player interface"""
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video file")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        st.write(f"**Video Info:** {total_frames} frames @ {fps:.1f} FPS")
        
        cap.release()
    
    except Exception as e:
        st.error(f"Error loading video: {e}")

def render_event_stream(events_log, match_state):
    """Render the event detection stream"""
    
    if not events_log:
        st.info("No events detected yet. Events will appear here as they're detected by the DSG engine.")
        return
    
    # Display recent events
    st.write(f"**Total Events Detected:** {len(events_log)}")
    
    # Create event dataframe
    event_data = []
    for idx, event in enumerate(reversed(events_log[-10:])):  # Show last 10 events
        event_data.append({
            'Timestamp': event.get('timestamp', 'N/A'),
            'Event Type': event.get('event', 'Unknown'),
            'Primary Actor': event.get('primary_actor', {}).get('type', 'N/A'),
            'Secondary Actor': event.get('secondary_actor', {}).get('type', 'N/A'),
            'Status': '✅ Triggered'
        })
    
    if event_data:
        df = pd.DataFrame(event_data)
        st.dataframe(df, use_container_width=True)
    
    st.divider()
    
    # Event statistics
    col1, col2, col3 = st.columns(3)
    
    event_types = {}
    for event in events_log:
        event_type = event.get('event', 'Unknown')
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    with col1:
        st.metric("Unique Event Types", len(event_types))
    
    with col2:
        most_common = max(event_types.items(), key=lambda x: x[1])[0] if event_types else "N/A"
        st.metric("Most Common", most_common)
    
    with col3:
        st.metric("Total Events", len(events_log))
    
    # Event type breakdown
    st.write("**Event Breakdown**")
    if event_types:
        chart_data = pd.DataFrame(list(event_types.items()), columns=['Event Type', 'Count'])
        st.bar_chart(chart_data.set_index('Event Type'))

def render_llm_commentary(commentary_history):
    """Render live commentary from LLM agents"""
    
    if not commentary_history:
        st.info("💭 Waiting for commentary generation...")
        return
    
    # Display commentary in reverse chronological order (most recent first)
    for comment in reversed(commentary_history[-5:]):  # Show last 5 comments
        timestamp = comment.get('timestamp', '')
        if timestamp:
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%H:%M:%S")
        else:
            time_str = "Now"
        
        st.markdown(f"""
        <div class="commentary-box">
            <small>⏰ {time_str}</small><br>
            <strong>Event:</strong> {comment.get('event', 'N/A')}<br>
            <strong>Commentary:</strong> <em>{comment.get('commentary', 'N/A')}</em>
        </div>
        """, unsafe_allow_html=True)

def render_agent_outputs(active_agents, match_state):
    """Render outputs from different LLM agents"""
    
    if not active_agents:
        st.warning("No agents selected. Configure agents in the sidebar.")
        return
    
    st.write(f"**Active Agents:** {', '.join(active_agents)}")
    st.divider()
    
    # Create tabs for each agent
    tabs = st.tabs(active_agents)
    
    for i, agent in enumerate(active_agents):
        with tabs[i]:
            # Placeholder for agent output
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"**{agent}**")
                st.markdown(f"""
                <div class="agent-output">
                    <strong>Status:</strong> Ready<br>
                    <strong>Last Updated:</strong> Just now<br>
                    <strong>Context Windows:</strong> {agent}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Performance Metrics**")
                st.metric("Response Time", "~1.2s", delta=None)
                st.metric("Context Used", "45%", delta=None)
            
            # Sample output
            st.markdown("**Latest Output**")
            if agent == "GPT-4o-mini":
                sample_output = "The bowler is running up to deliver from over the wicket. The batsman has taken guard outside off stump and is preparing for a quick delivery..."
            elif agent == "Gemini 1.5 Flash":
                sample_output = "Exceptional pace being bowled here! Ball is set to pitch on good length with slight away movement..."
            else:
                sample_output = "The fielding setup suggests a defensive approach. All-rounders are positioned for quick turnaround..."
            
            st.info(sample_output)

def render_match_controls():
    """Render match control buttons"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ Play", use_container_width=True):
            st.session_state.playing = True
            st.rerun()
    
    with col2:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.playing = False
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.current_frame = 0
            st.rerun()
    
    with col4:
        if st.button("📊 Analyze", use_container_width=True):
            st.info("Analysis feature coming soon!")

def render_event_triggers(config):
    """Render configured event triggers from DSG engine"""
    
    if not config:
        st.warning("No configuration loaded")
        return
    
    st.write("**Configured Event Triggers**")
    
    triggers = config.get('triggers_to_watch', [])
    
    for trigger in triggers:
        with st.expander(f"🎯 {trigger.get('trigger_name')}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Node A:** {trigger.get('node_A_type')}")
            
            with col2:
                st.markdown(f"**Condition:** {trigger.get('condition')}")
            
            with col3:
                st.markdown(f"**Node B:** {trigger.get('node_B_type')}")

def render_frame_analysis(frame_data):
    """Render analysis of current frame"""
    
    if not frame_data:
        st.info("No frame data available")
        return
    
    selected_domain = st.session_state.get('selected_domain', 'Cricket')
    cols = st.columns(3)
    
    # Generic display of detected objects
    for i, (obj_name, obj_info) in enumerate(list(frame_data.items())[:3]):
        with cols[i]:
            icon = "🔹"
            if obj_name.lower() in ['ball', 'vehicle', 'intruder']:
                icon = "🔴"
            elif obj_name.lower() in ['bat', 'person', 'pedestrian']:
                icon = "🟢"
            
            st.markdown(f"**{icon} {obj_name}**")
            for key, val in obj_info.items():
                if key != 'box':
                    st.markdown(f"- {key.replace('_', ' ').title()}: {val}")
                else:
                    st.markdown(f"- Position: {val}")
