# FusionX Streamlit Frontend - Quick Start Guide

## ⚡ 5-Minute Setup

### Step 1: Navigate to UI Folder
```bash
cd ui
```

### Step 2: Double-click to Run (Windows)
Simply double-click `run_streamlit.bat` to:
- Create a virtual environment (first time only)
- Install dependencies
- Launch the Streamlit app

The app will open automatically at `http://localhost:8501`

### Step 3: Configure API Keys (Optional but Recommended)
1. Create a `.env` file in the `ui` folder
2. Copy content from `.env.example`
3. Add your LLM API keys:
   - OpenAI: https://platform.openai.com/api-keys
   - Google Gemini: https://aistudio.google.com/apikey
   - Anthropic Claude: https://console.anthropic.com/account/keys

### Step 4: Add Test Videos
1. Place cricket videos in the `assets/` folder (root project directory)
2. Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.webm`
3. They'll automatically appear in the sidebar

### Step 5: Use the App!
- Select a video
- Configure match settings (striker, bowler, phase)
- Choose LLM agents
- Describe events to generate live commentary

---

## 🎯 Common Tasks

### Generate Commentary
1. Watch video and identify an event
2. Describe it in the "Generate Live Commentary" section
3. Click the button - get instant commentary!

### View Event Detections
Go to **Event Stream** tab to see:
- All detected events
- Event statistics
- Most common event types

### Compare Agent Outputs
Switch to **Agent Outputs** tab to see:
- Different LLM agent responses
- Performance metrics
- Individual agent analysis

### Track Match Progress
**Match Dashboard** shows:
- Current score and wickets
- Run rate
- Player information
- Match phase

---

## 🛠️ Manual Installation (if batch file doesn't work)

### For Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
copy .env.example .env
# Edit .env with your keys

# Run the app
streamlit run app.py
```

### For Mac/Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
cp .env.example .env
# Edit .env with your keys

# Run the app
streamlit run app.py
```

---

## ❓ Troubleshooting

### "Module not found" error
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Streamlit not starting
```bash
# Check Python version (should be 3.9+)
python --version

# Reinstall Streamlit
pip install streamlit --upgrade
```

### Video playback issues
```bash
# Install ffmpeg (required for video)
pip install ffmpeg-python
```

### LLM not working
- Check `.env` file exists
- Verify API keys are correct
- Check internet connection
- See sidebar for agent status

### Slow performance
- Use smaller videos
- Reduce FPS with frame skip
- Check system memory

---

## 📊 File Structure
```
ui/
├── app.py                 # Main app
├── components.py          # UI components
├── config.py              # Settings
├── agent_integration.py    # LLM integration
├── utils.py               # Helper functions
├── requirements.txt       # Dependencies
├── README.md              # Full documentation
├── QUICKSTART.md          # This file
├── run_streamlit.bat      # Windows launcher
├── .env.example           # Example config
├── .gitignore             # Git settings
└── .streamlit/
    └── config.toml        # Streamlit settings
```

---

## 🚀 Next Steps

1. **Add more videos** to assets folder
2. **Configure your API keys** in `.env`
3. **Explore the agent outputs** tab
4. **Use event replay** to analyze specific moments
5. **Customize commentary prompts** in `agent_integration.py`

---

## 📚 More Information

- **Full Documentation**: See `README.md`
- **Project Architecture**: See `../PROJECT_ARCHITECTURE.md`
- **Engine Details**: See `../engine/`

---

## 💡 Tips & Tricks

### Faster Startup
- Leave the app running and refresh browser
- The virtual environment stays active

### Better Commentary
- Be specific in event descriptions
- Include relevant stats (ball speed, etc.)
- Multiple agents give different perspectives

### Performance Tuning
- Use `STREAMLIT_LOGGER_LEVEL=warning` to reduce logs
- Disable unused agents to save API calls
- Pre-process videos offline

### Development Mode
- Edit `agent_integration.py` to customize prompts
- Add event types in `config.py`
- Extend UI components in `components.py`

---

## 🎓 Learning Resources

- Streamlit Docs: https://docs.streamlit.io
- OpenAI API: https://platform.openai.com/docs
- Google Gemini: https://ai.google.dev
- Anthropic Claude: https://docs.anthropic.com

---

**Happy Analyzing! 🏏🎙️**

For issues: Check README.md or project documentation
