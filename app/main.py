# app/main.py
from . import create_app

app = create_app()
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8050)
