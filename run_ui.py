#!/usr/bin/env python3
"""
Launch script for the modern LOB Trading System UI.
Run this to start the Streamlit web interface.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app."""
    print("🚀 Starting LOB Trading System UI...")
    print("📱 Inspired by Nothing Phone's minimalist design")
    print("🌐 Opening web interface...")
    
    # Change to the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run Streamlit
    port = os.getenv('PORT', '8501')
    address = os.getenv('RENDER') == 'true' and '0.0.0.0' or 'localhost'
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "modern_ui.py",
            "--server.port", port,
            "--server.address", address,
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down UI...")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        print("💡 Make sure you have installed all requirements: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
