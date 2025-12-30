# standalone_launcher.py
"""
Standalone desktop application launcher using PyWebView.
This creates a native desktop window for the Dash application.
"""
import webview
import sys
import os
import logging
from threading import Thread, Event
import time
import socket


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def setup_paths():
    """Setup paths for database and logs when running as standalone"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        app_data_dir = os.path.join(os.getenv('APPDATA'), 'InteractiveTorqueApp')
        os.makedirs(app_data_dir, exist_ok=True)
        
        # Set environment variables for the app
        os.environ['DATABASE_URL'] = f"sqlite:///{os.path.join(app_data_dir, 'torque_data.db')}"
        os.environ['ALLOWED_DIRECTORY'] = os.path.join(app_data_dir, 'data')
        os.makedirs(os.environ['ALLOWED_DIRECTORY'], exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(app_data_dir, 'app.log')
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s:%(message)s",
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        print(f"Running in standalone mode. Data directory: {app_data_dir}")
        print(f"Log file: {log_file}")
        logging.info(f"Running in standalone mode. Data directory: {app_data_dir}")
    else:
        # Running in development mode
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'app.log')
        logging.basicConfig(
            level=logging.DEBUG, 
            format="%(asctime)s %(levelname)s:%(message)s",
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        print("Running in development mode")
        logging.info("Running in development mode")


# Global variable to track server startup
server_ready = Event()
server_error = None


def check_server_ready(host='127.0.0.1', port=8050, timeout=30):
    """Check if server is ready by attempting to connect"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return True
        except:
            pass
        time.sleep(0.5)
    return False


def start_dash_server():
    """Start the Dash server in a background thread"""
    global server_error
    try:
        print("Importing app module...")
        from app import create_app
        print("Creating Dash app...")
        app = create_app()
        print("Dash app created successfully!")
        print("Starting Dash server on http://127.0.0.1:8050")
        # Set ready AFTER app is created, BEFORE server starts
        server_ready.set()
        # Run server without debug mode and without auto-reloading
        app.run(debug=False, port=8050, host='127.0.0.1', use_reloader=False)
    except Exception as e:
        server_error = e
        print(f"ERROR starting Dash server: {e}")
        logging.error(f"Error starting Dash server: {e}", exc_info=True)
        server_ready.set()  # Set anyway to unblock main thread
        raise


def main():
    """Main entry point for standalone desktop application"""
    global server_error
    
    print("="*60)
    print("Interactive Torque Analysis - Standalone Application")
    print("="*60)
    
    setup_paths()
    
    # Start Dash server in background thread
    print("Starting Dash server thread...")
    server_thread = Thread(target=start_dash_server, daemon=True)
    server_thread.start()
    
    # Wait for server to be ready (with timeout)
    print("Waiting for server to start...")
    if not server_ready.wait(timeout=30):  # Increased from 10 to 30 seconds
        print("ERROR: Server failed to start within 30 seconds")
        logging.error("Server failed to start within timeout")
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Startup Error",
            "The Dash server failed to start within 10 seconds.\n\nPlease check the log file for details."
        )
        sys.exit(1)
    
    # Check if there was an error during startup
    if server_error:
        print(f"ERROR: Server failed with error: {server_error}")
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Startup Error",
            f"Failed to start the Dash server:\n{str(server_error)}\n\nPlease check the log file for details."
        )
        sys.exit(1)
    
    # Additional check: verify server is actually responding
    print("Verifying server is responding...")
    if not check_server_ready():
        print("ERROR: Server is not responding on port 8050")
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Startup Error",
            "The Dash server started but is not responding.\n\nPlease check the log file for details."
        )
        sys.exit(1)
    
    print("Server is ready! Opening desktop window...")
    
    # Create PyWebView window
    try:
        window = webview.create_window(
            'Interactive Torque Analysis',
            'http://127.0.0.1:8050',
            width=1400,
            height=900,
            resizable=True,
            fullscreen=False,
            min_size=(1000, 700)
        )
        print("Starting PyWebView...")
        webview.start()
    except Exception as e:
        print(f"ERROR creating desktop window: {e}")
        logging.error(f"Error creating desktop window: {e}", exc_info=True)
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Startup Error",
            f"Failed to create the desktop window:\n{str(e)}\n\nPlease check the log file for details."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
