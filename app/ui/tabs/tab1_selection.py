from dash import html
import dash_bootstrap_components as dbc
from ...config import Config

def get_tab1_content():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("ステップ 1: 測定結果選択", className="mb-0"),
            ], width=8),
            dbc.Col([
                html.Small("Network Status:", className="text-muted me-2"),
                dbc.Badge(
                    "Checking...",
                    id="directory-status-badge",
                    color="secondary",
                    className="p-2",
                    style={"borderRadius": "20px"}
                ),
            ], width=4, className="text-end d-flex align-items-center justify-content-end"),
        ], className="mb-4 align-items-center"),
        dbc.Row([
            dbc.Col([
                html.Label("オペレーターID:", className="fw-bold"),
                dbc.Input(
                    id="operator-id-input",
                    type="text",
                    placeholder="IDを入力してください",
                    className="mb-2",
                    valid=False,  # Initialize as not valid
                    invalid=False,  # Initialize as not invalid
                ),
                dbc.FormFeedback(
                    "オペレーターIDを入力してください",
                    id="operator-id-feedback",
                ),
                dbc.Tooltip(
                    "一意のオペレーター識別子を入力してください",
                    target="operator-id-input",
                    placement="right",
                    delay={"show": 200, "hide": 0},
                ),
            ], width=12, md=6),
        ], className="mb-4"),
        # New Order ID input row
        dbc.Row([
            dbc.Col([
                html.Label("オーダー番号:", className="fw-bold"),
                dbc.Input(
                    id="order-id-input",
                    type="text",
                    placeholder="オーダー番号を入力してください",
                    className="mb-2",
                    value=None,
                ),
                dbc.FormFeedback(
                    "オーダー番号を入力してください",
                    id="order-id-feedback",
                ),
                dbc.Tooltip(
                    "Order or lot identifier",
                    target="order-id-input",
                    placement="right",
                    delay={"show": 200, "hide": 0},
                ),
            ], width=12, md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                html.Label("ボールサイズ:", className="fw-bold"),
                dbc.Input(
                    id="ball-size-input",
                    type="number",
                    placeholder="ボールサイズを入力してください",
                    className="mb-2",
                    value=None,
                    step=0.001,
                ),
                dbc.FormFeedback(
                    "ボールサイズを入力してください",
                    id="ball-size-feedback",
                ),
                dbc.Tooltip(
                    "トルク測定に適したボールサイズを入力してください",
                    target="ball-size-input",
                    placement="right",
                    delay={"show": 200, "hide": 0},
                ),
            ], width=12, md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                html.Label("測定ファイル 1 選択:", className="fw-bold"),
                dbc.Select(
                    id="file-dropdown-1",
                    options=[],
                    placeholder="CSVファイルを選択してください",
                    className="mb-2",
                    value=None,
                ),
                dbc.FormFeedback(
                    "主要なCSVファイルを選択してください",
                    id="file-1-feedback",
                ),
                dbc.Tooltip(
                    "主要な測定用CSVファイルを選択してください",
                    target="file-dropdown-1",
                    placement="right",
                    delay={"show": 200, "hide": 0},
                ),
            ], width=12, md=6),
        ], className="mb-4"),
        # ===== NEW: Model selection dropdown =====
        dbc.Row([
            dbc.Col([
                html.Label("使用するモデル:", className="fw-bold"),
                dbc.Select(
                    id="model-dropdown",
                    options=[],  # We'll populate via callback
                    placeholder="トレーニング済みモデルを選択してください",
                    className="mb-2",
                    value=None,
                ),
                dbc.FormText("予測に使用する.pklモデルを選択"),
            ], width=12, md=6),
        ], className="mb-4"),
        dbc.Button(
            [
                html.I(className="fas fa-arrow-right me-2"),
                "可視化に進む"
            ],
            id="proceed-visualization-btn",
            n_clicks=0,
            color="primary",
            className="mt-3",
            disabled=True,
            size="lg",
        ),
    ])
