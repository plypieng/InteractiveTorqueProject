from dash import html
import dash_bootstrap_components as dbc

def get_tab3_content():
    return dbc.Container([
        html.H3("ステップ 3: データのラベリング", className="mb-4"),
        
        # File Summary Card
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-file-alt me-2"),
                "選択されたファイルの概要"
            ], className="fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.P([
                            html.Strong("ファイル: "),
                            html.Span(id="selected-file-name", style={"whiteSpace": "pre-wrap"})
                        ]),
                        html.P([
                            html.Strong("ボールサイズ: "),
                            html.Span(id="selected-ball-size-name")
                        ]),
                        html.P([
                            html.Strong("オペレーター: "),
                            html.Span(id="selected-operator-name")
                        ]),
                        html.P([
                            html.Strong("オーダーID: "),
                            html.Span(id="selected-order-id")
                        ]),
                    ], md=6),
                    dbc.Col([
                        html.P([
                            html.Strong("スパイク分析結果: "),
                            html.Span(id="analysis-summary", className="fw-bold")
                        ]),
                        html.P([
                            html.Span(id="model-prediction-summary", className="fw-bold")
                        ]),
                    ], md=6),
                ]),
            ]),
        ], className="mb-4"),

        # features Card
        dbc.Card(style={"display": "none"}, children=[
            dbc.CardHeader([
                html.I(className="fas fa-chart-bar me-2"),
                "特徴量"
            ], className="fw-bold"),
            dbc.CardBody([
                html.Div(
                    id="tab-3-features",
                    style={"whiteSpace": "pre-wrap"},
                    className="mb-4"
                ),
            ]),
        ], className="mb-4"),
        
        # Labeling Card
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-tags me-2"),
                "ラベルを割り当てる"
            ], className="fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("ラベルを選択:", className="fw-bold mb-2"),
                        dbc.RadioItems(
                            id="data-label-dropdown",
                            options=[
                                {"label": "合格", "value": "PASS"},
                                {"label": "不合格", "value": "FAIL"},
                                {"label": "検査が必要", "value": "REQUIRES_INSPECTION"},
                            ],
                            inline=True,
                            className="mb-3",
                        ),
                        dbc.FormText([
                            "分析に基づいて適切なラベルを選択してください。",
                            html.Br(),
                            html.Span("ショートカットキー: [P] 合格, [F] 不合格, [Enter] 提出", className="text-info small")
                        ]),
                    ], width=12),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("追加のノート:", className="fw-bold mb-2"),
                        dbc.Textarea(
                            id="label-notes",
                            placeholder="追加の観察やノートを入力してください...",
                            style={"height": "100px"},
                            className="mb-3",
                        ),
                    ], width=12),
                ]),
            ]),
        ], className="mb-4"),
        
        # Alert for feedback
        dbc.Alert(
            id="save-alert-message",
            dismissable=True,
            duration=4000,
            is_open=False,
        ),
        
        # Navigation Buttons
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [
                        html.I(className="fas fa-arrow-left me-2"),
                        "可視化に戻る"
                    ],
                    id="back-to-visualization-btn",
                    color="secondary",
                    className="me-2",
                ),
                dbc.Button(
                    [
                        html.I(className="fas fa-save me-2"),
                        "ラベルを保存して進む"
                    ],
                    id="save-label-btn",
                    color="success",
                    disabled=True,
                ),
            ], className="d-flex justify-content-between"),
        ]),
    ])
