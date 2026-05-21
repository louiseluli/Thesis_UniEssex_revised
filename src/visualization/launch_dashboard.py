#!/usr/bin/env python3
"""
launch_dashboard.py
===================

Launch the interactive dashboard in a web browser for presentation.
"""

import webbrowser
import http.server
import socketserver
import threading
from pathlib import Path

def serve_dashboard(port=8000):
    """Serve the dashboard using Python's built-in HTTP server."""
    
    # Path to outputs directory
    output_dir = Path(__file__).resolve().parents[2] / "outputs"
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(output_dir), **kwargs)
    
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Dashboard server running at http://localhost:{port}/")
        print(f"Open: http://localhost:{port}/interactive/21_interactive_dashboard.html")
        print("Press Ctrl+C to stop the server")
        httpd.serve_forever()

if __name__ == "__main__":
    # Start server in background thread
    port = 8000
    thread = threading.Thread(target=serve_dashboard, args=(port,))
    thread.daemon = True
    thread.start()
    
    # Open browser
    webbrowser.open(f"http://localhost:{port}/interactive/21_interactive_dashboard.html")
    
    # Keep main thread alive
    try:
        thread.join()
    except KeyboardInterrupt:
        print("\nShutting down server...")