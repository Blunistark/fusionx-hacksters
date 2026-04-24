# FusionX Streamlit Frontend

This is the interactive Streamlit frontend for the **FusionX Real-Time Cricket Commentary System**. It provides a comprehensive dashboard for analyzing cricket videos with live commentary powered by LLM agents.

## Features

✨ **Key Features:**
- 📺 **Video Player** - Play cricket videos from the assets folder with frame-by-frame analysis
- 🎙️ **Live Commentary** - Real-time commentary generation from multiple LLM agents (GPT-4o-mini, Gemini 1.5 Flash, Claude 3.5)
- 📊 **Event Stream** - Real-time event detection using the Dynamic Scene Graph (DSG) engine
- 🤖 **Multi-Agent Support** - Switch between different LLM providers for diverse commentary styles
- 📈 **Match Dashboard** - Track match statistics (runs, wickets, overs, run rate)
- 🔍 **Frame Analysis** - Detailed analysis of individual frames with detected objects
- 💾 **Event Logging** - Complete history of detected events and triggered actions

## Architecture

The Streamlit frontend integrates with the FusionX platform:

```
Video Input (assets/)
    ↓
Computer Vision (YOLOv8, MediaPipe)
    ↓
Dynamic Scene Graph Engine (dsg_core.py)
    ↓
Event Triggers (Ball-Bat Contact, etc.)
    ↓
LLM Agent Pool (Commentary Generation)
    ↓
Streamlit UI (Live Display & Interaction)
```

## Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Setup Steps

1. **Navigate to the UI folder:**
```bash
cd ui
```

2. **Create a virtual environment (recommended):**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up LLM API keys:**
Create a `.env` file in the `ui` folder with your API keys:

```env
# OpenAI API Key
OPENAI_API_KEY=sk-xxx...

# Google Gemini API Key
GOOGLE_API_KEY=xxx...

# Anthropic Claude API Key
ANTHROPIC_API_KEY=xxx...

# Optional Debug Mode
DEBUG_MODE=false
```

5. **Add test videos:**
Place cricket videos in the `assets/` folder at the project root.

## Running the Application

### Start the Streamlit App

```bash
streamlit run app.py
```

This will:
- Launch the app in your default web browser at `http://localhost:8501`
- Display the interactive dashboard
- Be ready to process videos and generate commentary

### Advanced Options

```bash
# Run with specific host/port
streamlit run app.py --server.port 8888 --server.address 0.0.0.0

# Run in developer mode
streamlit run app.py --logger.level=debug

# Run headless (no browser)
streamlit run app.py --server.headless true
```

## Usage Guide

### 1. **Selecting a Video**
- Use the sidebar to select a video from the assets folder
- Videos are automatically discovered and listed
- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.webm`

### 2. **Configuring Match Setup**
In the sidebar, fill in:
- **Striker**: Name of the batsman
- **Bowler**: Name of the bowler
- **Match Phase**: Powerplay / Middle Overs / Death Overs

### 3. **Selecting LLM Agents**
- Choose one or more agents from the "LLM Agents" section
- Available agents:
  - **GPT-4o-mini** (OpenAI) - Fast, accurate
  - **Gemini 1.5 Flash** (Google) - Multimodal capable
  - **Claude 3.5** (Anthropic) - High quality reasoning

### 4. **Analyzing Videos**

**Video & Commentary Tab:**
- Use the frame slider to navigate through the video
- Play, pause, or reset the video using the control buttons
- View live commentary on the right side
- Each frame is analyzed for events

**Event Stream Tab:**
- See all detected events in real-time
- View event statistics and breakdowns
- Track the most common event types

**Agent Outputs Tab:**
- See outputs from each active agent
- Compare different agent perspectives
- View performance metrics

**Match Dashboard Tab:**
- Real-time match statistics
- Current run rate and phase
- Player information (striker/bowler)
- Wicket count and overs

### 5. **Generate Live Commentary**
- In the "Generate Live Commentary" section, describe what's happening in the frame
- Click "Generate Commentary" to get live commentary from the agents
- Commentary appears below and is logged in the sidebar

## Project Structure

```
ui/
├── app.py                 # Main Streamlit application
├── components.py          # UI component functions
├── config.py              # Configuration and constants
├── agent_integration.py    # LLM agent integration
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .env                   # Environment variables (create this)
```

## Configuration

### config.py
- Adjust video formats, paths, and LLM settings
- Configure event types and UI colors
- Set cache and logging parameters

### agent_integration.py
- Add new LLM providers
- Customize commentary prompts
- Modify agent fallback behavior

## API Integration

The frontend can integrate with external APIs:

### Event Stream API
If running a separate event detection service:
```python
# Modify EVENT_API_ENDPOINT in config.py
EVENT_API_ENDPOINT = "http://your-server:8000/events"
```

### Commentary Generation API
For external LLM services:
```python
COMMENTARY_API_ENDPOINT = "http://your-server:8001/commentary"
```

## Advanced Features

### Real-time Event Processing
The frontend automatically processes DSG events:
- Detects frame-to-frame changes
- Triggers LLM commentary generation
- Maintains match context window

### Context Window Management
Agents maintain conversation history for coherent commentary:
- Previous 5 events are considered
- Match state is continuously updated
- Fallback to template-based commentary if LLM unavailable

### Event Replay
- Select any frame to analyze
- Re-generate commentary for specific moments
- Compare different agent outputs

### Performance Metrics
- Monitor LLM response times
- Track context window usage
- View API call statistics

## Troubleshooting

### "No videos found" error
- Ensure videos are in the `assets/` folder
- Check file extensions are supported
- Verify folder exists

### LLM Commentary not generating
- Check `.env` file for API keys
- Verify internet connection
- Check API quota/rate limits
- See sidebar for agent status

### Video playback issues
- Install ffmpeg: `pip install ffmpeg-python`
- Ensure video codec is supported
- Check file integrity

### Performance issues
- Reduce video resolution
- Enable debug mode to identify bottlenecks
- Check system memory and CPU

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black app.py components.py config.py agent_integration.py
```

### Type Checking
```bash
mypy app.py --ignore-missing-imports
```

## Extending the Frontend

### Adding Custom Event Types
Edit `EVENT_CONFIG` in `config.py`:
```python
EVENT_CONFIG = {
    "event_types": [
        "SHOT_PLAYED",
        "YOUR_CUSTOM_EVENT",
        ...
    ]
}
```

### Adding New LLM Providers
In `agent_integration.py`, add to `_call_agent()` method:
```python
elif provider == "your_provider":
    return self._call_your_provider(...)
```

### Custom Commentary Prompts
Modify `_build_system_prompt()` in `agent_integration.py` to customize the commentary style.

## Performance Tips

1. **Use GPT-4o-mini** for fastest responses
2. **Enable caching** in Streamlit with `@st.cache_data`
3. **Process videos offline** with pre-saved detections
4. **Use event replay** instead of re-analyzing videos

## Security Considerations

1. **API Keys**: Never commit `.env` to version control
2. **Rate Limiting**: Implement request throttling for production
3. **Input Validation**: Sanitize video file paths
4. **Error Handling**: Don't expose sensitive info in error messages

## License

This project is part of the FusionX framework. See LICENSE in the project root.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the PROJECT_ARCHITECTURE.md in the root
3. Check DEBUG_MODE logs in `ui/logs/app.log`

---

**Version**: 1.0.0  
**Last Updated**: April 2026  
**Maintainer**: FusionX Team
