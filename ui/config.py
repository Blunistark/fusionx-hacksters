"""
Configuration file for FusionX Streamlit Frontend
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
UI_ROOT = Path(__file__).parent
ASSETS_PATH = PROJECT_ROOT / "assets"
ENGINE_PATH = PROJECT_ROOT / "engine"

# Create directories if they don't exist
ASSETS_PATH.mkdir(exist_ok=True)
ENGINE_PATH.mkdir(exist_ok=True)

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "FusionX: Real-Time Event Engine",
    "page_icon": "🚀",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Domain Configuration
DOMAINS = {
    "Cricket": {
        "icon": "🏏",
        "description": "Real-time cricket match analysis and commentary",
        "event_types": [
            "SHOT_PLAYED", "BALL_PITCHED", "CATCH", "RUN_SCORED", 
            "WICKET_FALLEN", "WIDE", "NO_BALL", "BOUNDARY"
        ],
        "metrics": [
            {"label": "🏃 Runs", "key": "runs", "suffix": ""},
            {"label": "😢 Wickets", "key": "wickets", "suffix": "/10"},
            {"label": "📊 Overs", "key": "overs_display", "suffix": ""},
            {"label": "⚡ Run Rate", "key": "run_rate", "suffix": ""}
        ],
        "actors": ["Striker", "Bowler", "Fielder"],
        "phases": ["Powerplay", "Middle Overs", "Death Overs"],
        "default_state": {
            "runs": 0, "wickets": 0, "overs": 0, "balls": 0,
            "striker": "Player 1", "bowler": "Bowler 1", "phase": "Powerplay"
        }
    },
    "Security": {
        "icon": "🛡️",
        "description": "Real-time security monitoring and threat detection",
        "event_types": [
            "INTRUSION_DETECTED", "UNAUTHORIZED_ACCESS", "LOITERING", 
            "SUSPICIOUS_OBJECT", "GEOFENCE_BREACH", "CROWD_GATHERING"
        ],
        "metrics": [
            {"label": "👤 Persons", "key": "person_count", "suffix": ""},
            {"label": "⚠️ Alerts", "key": "alert_count", "suffix": ""},
            {"label": "🔒 Sec. Level", "key": "security_level", "suffix": ""},
            {"label": "⏱️ Uptime", "key": "uptime", "suffix": "h"}
        ],
        "actors": ["Security Guard", "Authorized Personnel", "Intruder"],
        "phases": ["Normal", "Heightened", "Emergency"],
        "default_state": {
            "person_count": 0, "alert_count": 0, "security_level": "Normal",
            "uptime": 24, "location": "Main Entrance", "phase": "Normal"
        }
    },
    "Traffic": {
        "icon": "🚦",
        "description": "Real-time traffic flow analysis and incident detection",
        "event_types": [
            "TRAFFIC_CONGESTION", "ACCIDENT_DETECTED", "SPEEDING", 
            "ILLEGAL_PARKING", "PEDESTRIAN_CROSSING", "SIGNAL_VIOLATION"
        ],
        "metrics": [
            {"label": "🚗 Vehicles", "key": "vehicle_count", "suffix": ""},
            {"label": "📉 Flow Rate", "key": "flow_rate", "suffix": " v/m"},
            {"label": "🛑 Incidents", "key": "incident_count", "suffix": ""},
            {"label": "🛣️ Avg. Speed", "key": "avg_speed", "suffix": " km/h"}
        ],
        "actors": ["Vehicle", "Pedestrian", "Emergency Vehicle"],
        "phases": ["Morning Peak", "Off-Peak", "Evening Peak"],
        "default_state": {
            "vehicle_count": 0, "flow_rate": 0, "incident_count": 0,
            "avg_speed": 40, "intersection": "Crossroad A", "phase": "Off-Peak"
        }
    }
}

# Video configuration
VIDEO_CONFIG = {
    "supported_formats": {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"},
    "max_file_size_mb": 500,
    "default_fps": 30
}

# DSG Engine configuration
DSG_CONFIG = {
    "config_file": ENGINE_PATH / "config.json",
    "detection_threshold": 0.5,
    "delta_filter_enabled": True
}

# LLM Agent configuration
LLM_CONFIG = {
    "default_agents": ["GPT-4o-mini"],
    "supported_agents": ["GPT-4o-mini", "Gemini 1.5 Flash", "Claude 3.5"],
    "timeout_seconds": 30,
    "max_retries": 3,
    "retry_delay_seconds": 2,
    "temperature": 0.7,
    "max_tokens": 150
}

# Event configuration
EVENT_CONFIG = {
    "max_log_size": 1000,
    "display_recent": 10
}

# UI customization
UI_CONFIG = {
    "primary_color": "#667eea",
    "secondary_color": "#764ba2",
    "success_color": "#4caf50",
    "warning_color": "#ff9800",
    "error_color": "#f44336"
}

# API endpoints (if using external services)
API_CONFIG = {
    "event_endpoint": os.getenv("EVENT_API_ENDPOINT", "http://localhost:8000/events"),
    "commentary_endpoint": os.getenv("COMMENTARY_API_ENDPOINT", "http://localhost:8001/commentary"),
    "timeout": 10
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": UI_ROOT / "logs" / "app.log"
}

# Create logs directory
LOGGING_CONFIG["log_file"].parent.mkdir(exist_ok=True)

# Cache configuration
CACHE_CONFIG = {
    "video_cache_dir": UI_ROOT / ".cache" / "videos",
    "cache_ttl_hours": 24,
    "max_cache_size_mb": 1000
}

# Create cache directory
CACHE_CONFIG["video_cache_dir"].mkdir(parents=True, exist_ok=True)

# Feature flags
FEATURES = {
    "enable_real_time_events": True,
    "enable_agent_fallback": True,
    "enable_event_replay": True,
    "enable_match_analytics": True,
    "enable_performance_metrics": True,
    "debug_mode": os.getenv("DEBUG_MODE", "false").lower() == "true"
}

