#!/usr/bin/env python3
"""
FusionX Streamlit Frontend Launcher
Handles environment setup and application startup with proper error handling
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import json


class StreamlitLauncher:
    """Launcher for FusionX Streamlit application"""
    
    def __init__(self):
        self.ui_dir = Path(__file__).parent
        self.project_root = self.ui_dir.parent
        self.venv_dir = self.ui_dir / "venv"
        self.python_exe = sys.executable
        self.os_type = platform.system()
    
    def print_header(self):
        """Print application header"""
        print("\n" + "="*50)
        print("  FusionX Cricket Commentary System")
        print("  Streamlit Frontend Launcher")
        print("="*50 + "\n")
    
    def check_python(self) -> bool:
        """Check if Python version is compatible"""
        if sys.version_info < (3, 9):
            print(f"❌ Python 3.9+ required, but you have {sys.version_info.major}.{sys.version_info.minor}")
            return False
        
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        return True
    
    def setup_venv(self) -> bool:
        """Create and activate virtual environment"""
        if self.venv_dir.exists():
            print("✓ Virtual environment already exists")
            return True
        
        print("Creating virtual environment...")
        try:
            subprocess.run([self.python_exe, "-m", "venv", str(self.venv_dir)], check=True)
            print("✓ Virtual environment created")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def get_pip_exe(self) -> Path:
        """Get the pip executable path"""
        if self.os_type == "Windows":
            pip_exe = self.venv_dir / "Scripts" / "pip.exe"
        else:
            pip_exe = self.venv_dir / "bin" / "pip"
        return pip_exe
    
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        pip_exe = self.get_pip_exe()
        requirements_file = self.ui_dir / "requirements.txt"
        
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        print("Checking dependencies...")
        try:
            # Check if streamlit is already installed
            result = subprocess.run(
                [str(pip_exe), "show", "streamlit"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✓ Dependencies already installed")
                return True
            
            print("Installing dependencies (this may take a minute)...")
            subprocess.run(
                [str(pip_exe), "install", "-r", str(requirements_file)],
                check=True
            )
            print("✓ Dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def check_env_file(self):
        """Check for .env file and provide setup guidance"""
        env_file = self.ui_dir / ".env"
        env_example = self.ui_dir / ".env.example"
        
        if not env_file.exists():
            print("\n⚠️  WARNING: .env file not found")
            print("   API keys are not configured")
            print("   Commentary generation from LLM agents may not work\n")
            
            if env_example.exists():
                print("   To configure API keys:")
                print("   1. Copy .env.example to .env")
                print("   2. Edit .env and add your API keys:")
                print("      - OpenAI: https://platform.openai.com/api-keys")
                print("      - Gemini: https://aistudio.google.com/apikey")
                print("      - Claude: https://console.anthropic.com/account/keys\n")
        else:
            print("✓ .env file found")
    
    def check_assets(self):
        """Check for videos in assets folder"""
        assets_dir = self.project_root / "assets"
        
        if not assets_dir.exists():
            print("ℹ️  No assets folder found - it will be created on first run")
            return
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}
        videos = [f for f in assets_dir.iterdir() 
                 if f.suffix.lower() in video_extensions]
        
        if videos:
            print(f"✓ Found {len(videos)} video(s) in assets folder")
        else:
            print("ℹ️  No videos in assets folder - add some to get started")
    
    def get_streamlit_exe(self) -> Path:
        """Get the streamlit executable path"""
        if self.os_type == "Windows":
            streamlit_exe = self.venv_dir / "Scripts" / "streamlit.exe"
        else:
            streamlit_exe = self.venv_dir / "bin" / "streamlit"
        return streamlit_exe
    
    def launch_app(self) -> bool:
        """Launch the Streamlit application"""
        streamlit_exe = self.get_streamlit_exe()
        app_file = self.ui_dir / "app.py"
        
        if not app_file.exists():
            print(f"❌ app.py not found at {app_file}")
            return False
        
        if not streamlit_exe.exists():
            print(f"❌ Streamlit not found at {streamlit_exe}")
            return False
        
        print("\n" + "="*50)
        print("Launching Streamlit application...")
        print("The app will open at: http://localhost:8501")
        print("Press Ctrl+C to stop the server")
        print("="*50 + "\n")
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            # Launch streamlit
            subprocess.run(
                [str(streamlit_exe), "run", str(app_file)],
                cwd=str(self.ui_dir),
                env=env
            )
            return True
        except KeyboardInterrupt:
            print("\n\nStreamlit server stopped")
            return True
        except Exception as e:
            print(f"❌ Failed to launch application: {e}")
            return False
    
    def run(self) -> bool:
        """Run the complete setup and launch process"""
        self.print_header()
        
        # Check Python version
        if not self.check_python():
            print("Please install Python 3.9 or higher")
            return False
        
        # Setup virtual environment
        if not self.setup_venv():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Check configuration
        self.check_env_file()
        self.check_assets()
        
        print()
        
        # Launch application
        return self.launch_app()


def main():
    """Main entry point"""
    launcher = StreamlitLauncher()
    
    try:
        success = launcher.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nLauncher interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
