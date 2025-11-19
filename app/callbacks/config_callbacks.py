from dash import Input, Output, State, no_update, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import os
import base64
from dotenv import set_key
from ..config import Config
import run_migrations
from ..database import session as db_session
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

def register_config_callbacks(app):
    @app.callback(
        [Output("settings-feedback", "children"), Output("confirm-create-db", "displayed")],
        [Input("save-settings-btn", "n_clicks"), Input("confirm-create-db", "submit_n_clicks")],
        [State("allowed-directory-input", "value"),
         State("database-path-input", "value"),
         State("developer-password-input", "value"),
         State("developer-password-confirm-input", "value")],
        prevent_initial_call=True
    )
    def manage_settings(save_clicks, confirm_clicks, allowed_dir, db_path, new_pw, new_pw_conf):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger = ctx.triggered[0]["prop_id"].split('.')[0]
        dotenv_path = os.path.join(os.getcwd(), ".env")
        # Handle save button
        if trigger == "save-settings-btn":
            if new_pw and new_pw != new_pw_conf:
                return dbc.Alert("パスワードが一致しません", color="danger"), False
            if allowed_dir:
                Config.ALLOWED_DIRECTORY = allowed_dir
                set_key(dotenv_path, "ALLOWED_DIRECTORY", allowed_dir)
            if db_path:
                if not os.path.isfile(db_path):
                    return no_update, True
                Config.DATABASE_URL = f"sqlite:///{db_path}"
                set_key(dotenv_path, "DATABASE_URL", Config.DATABASE_URL)
                db_session.engine.dispose()
                db_session.engine = create_engine(Config.DATABASE_URL, echo=False)
                db_session.SessionLocal = sessionmaker(bind=db_session.engine)
                run_migrations.run_migrations()
                insp = inspect(db_session.engine)
                if insp.has_table("measurements"):
                    cols = [c["name"] for c in insp.get_columns("measurements")]
                    if "order_id" not in cols:
                        with db_session.engine.connect() as conn:
                            conn.execute(text('ALTER TABLE measurements ADD COLUMN order_id TEXT'))
            if new_pw:
                Config.DEVELOPER_PASSWORD = new_pw
                set_key(dotenv_path, "DEVELOPER_PASSWORD", new_pw)
            return dbc.Alert("設定を保存しました", color="success"), False
        # Handle confirm dialog
        if trigger == "confirm-create-db":
            dirn = os.path.dirname(db_path)
            if dirn and not os.path.isdir(dirn):
                os.makedirs(dirn, exist_ok=True)
            Config.DATABASE_URL = f"sqlite:///{db_path}"
            set_key(dotenv_path, "DATABASE_URL", Config.DATABASE_URL)
            db_session.engine.dispose()
            db_session.engine = create_engine(Config.DATABASE_URL, echo=False)
            db_session.SessionLocal = sessionmaker(bind=db_session.engine)
            run_migrations.run_migrations()
            insp = inspect(db_session.engine)
            if insp.has_table("measurements"):
                cols = [c["name"] for c in insp.get_columns("measurements")]
                if "order_id" not in cols:
                    with db_session.engine.connect() as conn:
                        conn.execute(text('ALTER TABLE measurements ADD COLUMN order_id TEXT'))
            return dbc.Alert("新しいデータベースを作成しました", color="success"), False

    @app.callback(
        Output("database-path-input", "value"),
        Input("upload-db-file", "contents"),
        State("upload-db-file", "filename"),
        prevent_initial_call=True
    )
    def handle_db_upload(contents, filename):
        """Save uploaded DB file from client, update path."""
        if contents and filename:
            header, content = contents.split(",", 1)
            decoded = base64.b64decode(content)
            uploads_dir = os.path.join(os.getcwd(), "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            save_path = os.path.join(uploads_dir, filename)
            with open(save_path, "wb") as f:
                f.write(decoded)
            Config.DATABASE_URL = f"sqlite:///{save_path}"
            return save_path
        return no_update
