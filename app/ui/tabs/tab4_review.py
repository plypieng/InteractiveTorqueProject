from dash import html
import dash_bootstrap_components as dbc

def get_tab4_content():
    return dbc.Container([
        html.H3("ステップ 4: レビューと提出", className="mb-4"),
        
        # Summary Card
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-clipboard-list me-2"),
                "レビュー概要"
            ], className="fw-bold"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5([
                            html.I(className="fas fa-file me-2"),
                            "選択されたファイル"
                        ], className="mb-3"),
                        html.Div(
                            id="review-selected-files",
                            className="mb-4"
                        ),
                    ], md=6),
                    dbc.Col([
                        html.H5([
                            html.I(className="fas fa-tag me-2"),
                            "割り当てられたラベル"
                        ], className="mb-3"),
                        html.Div(
                            id="review-assigned-labels",
                            className="mb-4"
                        ),
                        html.P([
                            html.Strong("オーダーID: "),
                            html.Span(id="review-order-id", className="ms-2")
                        ]),
                    ], md=6),
                ]),
                # review features
                dbc.Row([
                    html.H5("最終確認用特徴量", className="mb-4"),
                    html.Div(id="features-table-4", className="mb-4"),
                ])
            ]),
        ], className="mb-4"),
        
        # Submit Alert
        dbc.Alert(
            id="submit-alert-message",
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
                        "ラベリングに戻る"
                    ],
                    id="back-to-labeling-btn",
                    color="secondary",
                    className="me-2",
                ),
                dbc.Button(
                    [
                        html.I(className="fas fa-check-circle me-2"),
                        "ラベルを提出"
                    ],
                    id="submit-labels-btn",
                    color="success",
                    disabled=True,
                ),
            ], className="d-flex justify-content-between"),
        ]),
    ])
