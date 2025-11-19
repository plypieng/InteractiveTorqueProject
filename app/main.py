# app/main.py
from . import create_app

app = create_app()
server = app.server

if __name__ == "__main__":
    # Bind to all interfaces and run Dash server with default settings
    app.run(debug=True, port=8050, host="0.0.0.0")
