from dash import html
import dash_bootstrap_components as dbc
from ...config import Config

def get_settings_content():
    return dbc.Container([
        html.H3("設定", className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader("一般設定", className="fw-bold"),
            dbc.CardBody([
                html.P(f"現在のモード: {Config.APP_MODE}"),
                html.P(f"監視ディレクトリ: {Config.ALLOWED_DIRECTORY}"),
            ])
        ], className="mb-4"),

        # Add more settings here as needed
    ])
