#!/usr/bin/env python3
"""
Runner script for the Advanced Data Science & ML Pipeline Streamlit app
"""
import subprocess
import sys
import os

def run_streamlit_app():
    """Run the Streamlit application"""
    try:
        print("🚀 Starting Advanced Data Science & ML Pipeline...")
        print("📊 Loading modules and dependencies...")
        
        # Change to app directory
        os.chdir('/app')
        
        # Run streamlit app
        cmd = [sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.port', '8501', '--server.address', '0.0.0.0']
        
        print("🌐 Starting Streamlit server...")
        print("   📍 URL: http://localhost:8501")
        print("   🔧 Use Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\
🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit_app()
