# app/main.py
from . import create_app

app = create_app()
server = app.server

def get_app():
    """Get the Dash app instance (useful for standalone launcher)"""
    return app

if __name__ == "__main__":
    # Development mode: run with debug enabled and auto-reload
    app.run(debug=True, port=8050, host="0.0.0.0")
