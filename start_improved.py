#!/usr/bin/env python3
"""
Railway DSS - Improved Simulation
Uses real GeoJSON data with enhanced features
"""

import sys
import os
import subprocess
import webbrowser
import time

def check_files():
    """Check if all required files exist"""
    required_files = [
        'bangalore_mysore_stations.geojson',
        'bangalore_mysore_tracks.geojson',
        'data/sbc_mys_schedules.csv',
        'backend/improved_app.py',
        'frontend/improved.html'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("✗ Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print("✓ All required files present")
        return True

def check_basic_dependencies():
    """Check basic dependencies"""
    try:
        import flask
        print("✓ Flask available")
        return True
    except ImportError:
        print("✗ Flask not available")
        print("Install with: pip install flask")
        return False

def start_improved_backend():
    """Start the improved backend"""
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    app_file = os.path.join(backend_dir, 'improved_app.py')
    
    print("Starting improved backend with real GeoJSON data...")
    try:
        process = subprocess.Popen([
            sys.executable, app_file
        ], cwd=backend_dir)
        
        time.sleep(3)  # Give more time for GeoJSON loading
        
        if process.poll() is None:
            print("✓ Backend started successfully!")
            return process
        else:
            print("✗ Backend failed to start!")
            return None
            
    except Exception as e:
        print(f"✗ Error starting backend: {e}")
        return None

def main():
    print("=" * 50)
    print("🚂 Railway DSS - Simulation")
    print("=" * 50)
    
    print("\n1. Checking files...")
    if not check_files():
        print("\nMake sure you have the GeoJSON files in the project root:")
        print("- bangalore_mysore_stations.geojson")
        print("- bangalore_mysore_tracks.geojson")
        return 1
    
    print("\n2. Checking dependencies...")
    if not check_basic_dependencies():
        return 1
    
    print("\n3. Starting improved backend...")
    backend_process = start_improved_backend()
    if not backend_process:
        return 1
    
    print("\n4. Opening improved frontend...")
    frontend_file = os.path.join(os.path.dirname(__file__), 'frontend', 'improved.html')
    webbrowser.open(f'file://{os.path.abspath(frontend_file)}')
    
    print("\n" + "=" * 50)
    print("🎉 Railway DSS Started!")
    print("=" * 50)
    print("\n✨ FEATURES:")
    print("✓ Real GeoJSON track data")
    print("✓ Station locations (SBC, Mandya, Mysore)")
    print("✓ Variable simulation speed")
    print("✓ Train movement visualization")
    print("✓ Delay injection & special trains")
    print("\nBackend: http://localhost:5000")
    print("Frontend: Opened in browser")
    print("\nPress Ctrl+C to stop")
    
    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n\nStopping improved system...")
        backend_process.terminate()
        print("✓ Stopped!")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
