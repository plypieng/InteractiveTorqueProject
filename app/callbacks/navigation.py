# app/callbacks/navigation.py
from dash import Input, Output, State, callback_context
from ..config import Config

def register_navigation_callbacks(app):
    @app.callback(
        Output("modal", "is_open"),
        [Input("open-modal", "n_clicks"), Input("close-modal", "n_clicks")],
        [State("modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

    @app.callback(
        [Output("user-mode", "data"),
         Output("mode-label", "children"),
         Output("password-error", "children"),
         Output("password-modal", "is_open")],
        [Input("toggle-mode-btn", "n_clicks"),
         Input("password-submit-btn", "n_clicks"),
         Input("password-cancel-btn", "n_clicks")],
        [State("password-input", "value"),
         State("user-mode", "data")],
        prevent_initial_call=True
    )
    def handle_mode(n_toggle, n_submit, n_cancel, password, current_mode):
        ctx = callback_context
        trig = ctx.triggered[0]["prop_id"].split(".")[0]
        # Toggle switch
        if trig == "toggle-mode-btn":
            if current_mode == "operator":
                # prompt for password
                return current_mode, f"{current_mode.capitalize()} Mode", "", True
            else:
                # switch back to operator
                return "operator", "Operator Mode", "", False
        # Cancel password entry
        if trig == "password-cancel-btn":
            return current_mode, f"{current_mode.capitalize()} Mode", "", False
        # Submit password
        if trig == "password-submit-btn":
            if password == Config.DEVELOPER_PASSWORD:
                return "developer", "Developer Mode", "", False
            # incorrect
            return current_mode, f"{current_mode.capitalize()} Mode", "Incorrect password", True
        # default
        return current_mode, f"{current_mode.capitalize()} Mode", "", False

    @app.callback(
        [Output("dev-params-row1", "style"),
         Output("dev-params-row2", "style"),
         Output("features-table-2", "style")],
        [Input("user-mode", "data")]
    )
    def toggle_dev_ui(user_mode):
        # Show developer widgets only in developer mode
        style = {} if user_mode == "developer" else {"display": "none"}
        return style, style, style

    @app.callback(
        Output("tab-6", "disabled"),
        [Input("user-mode", "data")]
    )
    def toggle_settings_tab(user_mode):
        # Enable Settings tab only for developer mode
        return False if user_mode == "developer" else True
