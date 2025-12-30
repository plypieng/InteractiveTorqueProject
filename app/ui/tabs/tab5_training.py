from dash import html, dcc
import dash_bootstrap_components as dbc
from ...config import Config

def get_tab5_content():
    hidden_mode = {"display": "none"} if Config.APP_MODE == "operator" else {}
    return dbc.Container([
        html.H3("ステップ 5: モデルのトレーニング (DBから)", className="mb-4"),

        dbc.Alert(
            id="db-training-status-alert",
            is_open=False,
            duration=None,
            className="mb-4"
        ),

        dbc.Card(style=hidden_mode, children=[
            dbc.CardHeader([
                html.I(className="fas fa-database me-2"),
                "DBラベルデータで学習"
            ], className="fw-bold"),
            dbc.CardBody([
                html.P("『合格/不合格』とラベル付けされたDBデータを利用してモデルを学習します。"),

                dbc.Row([
                    dbc.Col([
                        html.Label("モデルの名前:", className="fw-bold"),
                        dbc.Input(
                            id="training-model-name",
                            type="text",
                            placeholder="best_model_v1, MyLogistic2025, etc.",
                            value="MyModel",
                            className="mb-2",
                        ),
                        dbc.FormText("拡張子 .pkl は自動で付与されます。"),
                    ], md=8),
                    dbc.Col([
                        dbc.Button(
                            id="start-db-training-btn",
                            children="モデルを再トレーニング",
                            color="primary",
                            className="mt-4 me-2",
                            n_clicks=0
                        ),
                        dbc.Button(
                            "トレーニング中止",
                            id="cancel-training-btn",
                            color="danger",
                            className="mt-4",
                            disabled=True,
                            n_clicks=0
                        ),
                    ], md=4, className="d-flex align-items-end"),
                ]),

                html.Hr(),

                html.Div([
                    html.H5("学習ログ"),
                    html.Pre(
                        id="db-training-log",
                        style={
                            "backgroundColor": "#1e1e1e",
                            "color": "#00ff00",
                            "padding": "10px",
                            "height": "200px",
                            "overflowY": "scroll",
                            "border": "1px solid #444"
                        }
                    )
                ])
            ]),
        ], className="mb-4"),

        # Database Review Section (Editable Table)
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-table me-2"),
                "データベースの確認・編集"
            ], className="fw-bold"),
            dbc.CardBody([
                html.Div(id="db-review-table-container"),
                dbc.Button("DB再読み込み", id="refresh-db-btn", color="secondary", className="mt-2", n_clicks=0)
            ])
        ])
    ])
