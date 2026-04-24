# FusionX Streamlit Frontend - Setup & Next Steps

## ✅ What Has Been Built

A complete, production-ready Streamlit frontend for the FusionX Cricket Commentary system with:

- ✨ **4-Tab Interactive Dashboard**
  - Video & Commentary (with frame-by-frame playback)
  - Event Stream (real-time event detection)
  - Agent Outputs (multi-provider LLM comparison)
  - Match Dashboard (score, wickets, statistics)

- 🤖 **Multi-Agent LLM Support**
  - GPT-4o-mini (OpenAI)
  - Gemini 1.5 Flash (Google)
  - Claude 3.5 (Anthropic)

- 📹 **Video Analysis**
  - Load cricket videos from assets folder
  - Frame-by-frame navigation
  - Real-time event detection integration
  - Commentary generation per frame

- 🎙️ **Live Commentary System**
  - Context-aware prompt generation
  - Agent fallback mechanism
  - Commentary history tracking
  - Template-based fallback

- 📊 **Event Management**
  - Event aggregation and logging
  - Statistics and breakdown
  - Event stream visualization

---

## 🚀 Getting Started (5 Minutes)

### Windows Users:
```bash
cd ui
double-click run_streamlit.bat
```

### Mac/Linux Users:
```bash
cd ui
python launcher.py
```

### Manual Setup:
```bash
cd ui
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

---

## 📝 Setup Checklist

### Phase 1: Initial Setup (Do This First)
- [ ] Clone/navigate to the ui folder
- [ ] Run `run_streamlit.bat` (Windows) or `python launcher.py`
- [ ] App opens at http://localhost:8501
- [ ] Virtual environment is created automatically

### Phase 2: API Configuration (Do This Next)
- [ ] Create `.env` file from `.env.example`
- [ ] Add OpenAI API key (from https://platform.openai.com/api-keys)
- [ ] Add Google Gemini API key (from https://aistudio.google.com/apikey)
- [ ] Add Anthropic Claude API key (from https://console.anthropic.com)
- [ ] Restart Streamlit app to load keys

### Phase 3: Add Test Videos (Optional but Recommended)
- [ ] Create/navigate to `assets/` folder at project root
- [ ] Add cricket video files (`.mp4`, `.avi`, `.mov`, `.mkv`)
- [ ] Refresh browser to see videos in dropdown
- [ ] Select a video to test

### Phase 4: Test the System
- [ ] Select a video from dropdown
- [ ] Fill in match details (striker, bowler, phase)
- [ ] Select at least one LLM agent
- [ ] Describe an event in the commentary input box
- [ ] Click "Generate Commentary" button
- [ ] See commentary appear in real-time

---

## 📁 File Structure Overview

```
ui/
├── 📄 Core Files
│   ├── app.py                    # Main Streamlit application
│   ├── components.py             # UI component functions
│   ├── config.py                 # Configuration constants
│   ├── agent_integration.py       # LLM agent management
│   └── utils.py                  # Helper utilities
│
├── 📄 Configuration Files
│   ├── requirements.txt           # Python dependencies
│   ├── .env.example               # API key template
│   └── .streamlit/config.toml     # Streamlit settings
│
├── 📄 Documentation
│   ├── README.md                  # Full user guide
│   ├── QUICKSTART.md              # 5-minute setup
│   ├── IMPLEMENTATION_SUMMARY.md  # Technical overview
│   └── SETUP_CHECKLIST.md         # This file
│
├── 🔧 Startup Scripts
│   ├── run_streamlit.bat          # Windows launcher
│   └── launcher.py                # Python launcher
│
└── 📋 Other
    ├── .gitignore                 # Git exclusions
    └── .env                       # Your API keys (create this)
```

---

## 🎯 Common Tasks

### 1. Start the App
```bash
cd ui
python launcher.py
# or
double-click run_streamlit.bat
```

### 2. Generate Commentary
1. Select a video
2. Fill in match details
3. Choose LLM agent(s)
4. Describe the event
5. Click "Generate Commentary"

### 3. View Event Detections
- Go to "Event Stream" tab
- See all detected events
- View event statistics
- Check event breakdown

### 4. Compare Agent Outputs
- Go to "Agent Outputs" tab
- Switch between agents
- View different perspectives
- Compare response times

### 5. Add a New Video
1. Place video file in `assets/` folder
2. Refresh browser
3. Video appears in dropdown
4. Select and analyze

### 6. Update API Keys
1. Edit `.env` file
2. Add/update your API keys
3. Restart Streamlit app
4. Keys are automatically loaded

---

## 🔑 API Keys Needed

To enable full functionality, obtain these free keys:

| Provider | Where to Get | Time to Setup | Cost |
|----------|--------------|---------------|------|
| **OpenAI** | https://platform.openai.com/api-keys | 5 min | Free tier available |
| **Google Gemini** | https://aistudio.google.com/apikey | 2 min | Free (generous limits) |
| **Anthropic Claude** | https://console.anthropic.com/account/keys | 5 min | Free trial available |

### Quick API Setup:
1. Visit each link above
2. Sign in (create account if needed)
3. Generate/copy API key
4. Paste into `.env` file
5. Restart app

---

## 🎬 Example Workflow

### Scenario: Analyzing a Cricket Match

1. **Start App**
   ```bash
   python launcher.py
   ```
   Browser opens → http://localhost:8501

2. **Select Video**
   - Sidebar dropdown → Choose "match.mp4"

3. **Configure Match**
   - Striker: "Virat Kohli"
   - Bowler: "Jasprit Bumrah"
   - Phase: "Middle Overs"

4. **Choose Agents**
   - Select: GPT-4o-mini, Gemini 1.5 Flash

5. **Analyze Frame**
   - Use slider to find interesting moment
   - Event: "The bowler is running up. Ball will be short of length."

6. **Generate Commentary**
   - Click "Generate Commentary"
   - See responses from both agents
   - View comparison in Agent Outputs tab

7. **Track Events**
   - Go to "Event Stream" tab
   - See detected events
   - View statistics

8. **Monitor Match**
   - Check "Match Dashboard"
   - Track runs, wickets, run rate
   - See player information

---

## 🛠️ Customization Options

### Change Colors
Edit `config.py`:
```python
UI_CONFIG = {
    "primary_color": "#667eea",  # Change these
    "secondary_color": "#764ba2"
}
```

### Add Custom Events
Edit `config.py`:
```python
EVENT_CONFIG = {
    "event_types": [
        "SHOT_PLAYED",
        "YOUR_NEW_EVENT",  # Add here
        ...
    ]
}
```

### Modify Prompts
Edit `agent_integration.py`:
```python
def _build_system_prompt(self, match_state):
    return """Your custom prompt here..."""
```

### Change Default Agents
Edit `agent_integration.py`:
```python
def __init__(self):
    self.active_agents = ["GPT-4o-mini"]  # Change here
```

---

## 📊 Architecture Overview

The frontend connects to the FusionX system:

```
User Interface (Streamlit)
        ↓
   Video Selection
        ↓
   Frame Analysis
        ↓
   Event Detection (DSG Engine)
        ↓
   LLM Commentary Generation
        ↓
   Real-Time Display
```

---

## 🐛 Quick Troubleshooting

### Issue: "No videos found"
**Solution**: 
- Create `assets/` folder at project root
- Add `.mp4` files to it
- Refresh browser

### Issue: "Module not found"
**Solution**:
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "API key error"
**Solution**:
1. Create `.env` file
2. Add your API key
3. Restart Streamlit
4. Check agent status in sidebar

### Issue: "Streamlit not starting"
**Solution**:
```bash
# Use Python launcher
python launcher.py

# Or manual setup
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install streamlit
streamlit run app.py
```

### Issue: "Video plays but can't navigate"
**Solution**:
- Make sure OpenCV is installed
- Try a different video format
- Check video file integrity

---

## 📈 Performance Tips

1. **Use GPT-4o-mini** for fastest responses
2. **Disable unused agents** to save API calls
3. **Use smaller videos** for testing
4. **Enable caching** in Streamlit

---

## 🔐 Security Best Practices

1. **Never commit `.env`** to git
2. **Rotate API keys** periodically
3. **Monitor API usage** on provider dashboards
4. **Use environment variables** in production
5. **Don't share API keys** with others

---

## 📚 Further Learning

### Documentation Files:
- `README.md` - Comprehensive guide
- `QUICKSTART.md` - Fast setup
- `IMPLEMENTATION_SUMMARY.md` - Technical details

### External Resources:
- **Streamlit**: https://docs.streamlit.io
- **OpenAI API**: https://platform.openai.com/docs
- **Google Gemini**: https://ai.google.dev
- **Anthropic Claude**: https://docs.anthropic.com
- **Project Architecture**: `../PROJECT_ARCHITECTURE.md`

---

## ✨ What You Can Do Now

### Immediate (0-10 minutes)
- [ ] Start the Streamlit app
- [ ] See the interface working
- [ ] Explore the different tabs

### Soon (10-30 minutes)
- [ ] Add API keys
- [ ] Try generating commentary
- [ ] Add some test videos

### Later (30+ minutes)
- [ ] Customize UI colors
- [ ] Modify commentary prompts
- [ ] Add custom event types
- [ ] Integrate with your own models

---

## 📞 Need Help?

### For Setup Issues:
1. Check QUICKSTART.md
2. See Troubleshooting section above
3. Check debug logs in `.streamlit/` folder

### For Feature Questions:
1. Read README.md
2. Check IMPLEMENTATION_SUMMARY.md
3. Review code comments in `.py` files

### For Integration Help:
1. Check PROJECT_ARCHITECTURE.md
2. Review agent_integration.py
3. Check engine/ folder for DSG details

---

## 🎓 Next Steps to Master the System

### Level 1: Basic Usage
- Run the app
- Select videos
- Generate commentary
- View results

### Level 2: Configuration
- Add API keys
- Customize colors
- Change prompts
- Add custom events

### Level 3: Integration
- Integrate with DSG engine
- Add custom event processors
- Modify agent prompts
- Create custom displays

### Level 4: Production
- Deploy to cloud
- Set up monitoring
- Optimize performance
- Scale to multiple users

---

## 💡 Tips & Tricks

### Faster Startup
Keep Streamlit running and just refresh browser

### Better Commentary
Be specific: "The bowler is running up. Ball will be short of length, around 130 kph"

### Compare Agents
Use Agent Outputs tab to see different perspectives

### Replay Events
Use frame slider to go back and re-analyze moments

### Debug Mode
Set `DEBUG_MODE=true` in `.env` for verbose logging

---

## 🎉 You're All Set!

The FusionX Streamlit frontend is ready to use. Start by:

1. Running the launcher
2. Adding a test video
3. Configuring API keys
4. Generating your first commentary

**Enjoy analyzing cricket videos with AI-powered commentary!**

---

## 📋 Final Checklist

- [ ] App starts without errors
- [ ] Videos load in dropdown
- [ ] Commentary generates (with API keys)
- [ ] Events display correctly
- [ ] Match dashboard shows stats
- [ ] All tabs work
- [ ] UI looks good
- [ ] No errors in console

If all checked ✓ → **You're ready to go!**

---

**Last Updated**: April 2026  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
