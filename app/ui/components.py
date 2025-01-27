# app/ui/components.py
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from ..config import Config


def get_loading_overlay():
    return dbc.Spinner(
        html.Div(id="loading-overlay"),
        fullscreen=True,
        color="primary",
        type="grow",
    )

def get_confirmation_modal():
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="fas fa-check-circle me-2"),
            "提出の確認"
        ])),
        dbc.ModalBody([
            html.P("以下のデータを提出してもよろしいですか？"),
            html.Div(id="confirmation-content", className="mt-3"),
        ]),
        dbc.ModalFooter([
            dbc.Button(
                "キャンセル",
                id="cancel-submission-btn",
                color="secondary",
                className="me-2"
            ),
            dbc.Button(
                "提出を確認",
                id="confirm-submission-btn",
                color="success"
            ),
        ]),
    ], id="confirmation-modal", is_open=False)

def get_progress_indicator():
    return html.Div([
        dbc.Progress(
            value=25,
            id="progress-bar",
            striped=True,
            animated=True,
            className="mb-3",
            style={"height": "8px"}
        ),
        html.Div(
            id="progress-label",
            className="text-center text-muted small mb-4",
        ),
    ])

def get_help_modal():
    help_text = """
    ### このアプリケーションの使い方
    - **測定ファイルの選択:** 比較するために、ドロップダウンから最大2つのCSVファイルを選択します。
    - **ボールサイズの選択:** トルク測定に適したボールサイズを選択します。
    - **パラメータの調整:**
      - **ハイパスフィルターカットオフ周波数（Hz）:** Hz単位で値を入力します。
      - **RMSウィンドウサイズ:** 移動RMSを計算するためのウィンドウサイズを入力します。
      - **HPF_RMSしきい値:** HPF_RMSのしきい値を設定します。
      - **スパイクしきい値:** RMSデータのスパイク検出のためのしきい値を設定します。
      - **初期トルク入力（Nm）:** 初期トルク測定値を入力して、ボールサイズの推奨を受けます。
      - **Y軸スケール:** スライダーを使用して範囲を設定します。
    - **グラフの表示:** 選択に基づいてプロットが更新されます。
    - **分析結果の表示:** 分析に基づいてPASSまたはFAILEDの結果が表示されます。
    - **データのダウンロード:** リンクを使用してCSVまたはPDFファイルをダウンロードします。
    - **データのラベル付け:** モデルのトレーニングのために各データセットにラベルを割り当てます。
    """
    
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="fas fa-book me-2"),
            "ユーザーガイド"
        ])),
        dbc.ModalBody(dcc.Markdown(help_text)),
        dbc.ModalFooter(
            dbc.Button("閉じる", id="close-modal", className="ms-auto")
        ),
    ], id="modal", is_open=False, size="lg")

def get_tab1_content():
    return dbc.Container([
        html.H3("ステップ 1: 測定結果選択", className="mb-4"),
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

        dbc.Row([
            dbc.Col([
                html.Label("ボールサイズ:", className="fw-bold"),
                dbc.Select(
                    id="ball-size-dropdown",
                    options=[],
                    placeholder="ボールサイズを選択してください",
                    className="mb-2",
                    value=None,
                ),
                dbc.FormFeedback(
                    "ボールサイズを選択してください",
                    id="ball-size-feedback",
                ),
                dbc.Tooltip(
                    "トルク測定に適したボールサイズを選択してください",
                    target="ball-size-dropdown",
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

        dbc.Row([
            dbc.Col([
                html.Label("測定ファイル 2 選択（WIP）:", className="fw-bold"),
                dbc.Select(
                    id="file-dropdown-2",
                    options=[],
                    placeholder="CSVファイルを選択してください（任意）",
                    className="mb-2",
                    value=None,
                ),
                dbc.Tooltip(
                    "比較用に追加の測定用CSVファイルを選択してください（任意）",
                    target="file-dropdown-2",
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

def get_tab2_content():
    hidden_mode = {"display": "none"} if Config.APP_MODE == "operator" else {}
    
    return dbc.Container([
        html.H3("ステップ 2: データの可視化", className="mb-4"),
        
        # Model Prediction Alert (optional, can remove if not needed)
        dbc.Alert(
            id="model-prediction",
            className="mb-4",
            dismissable=True,
            duration=4000,
            style={"fontSize": "1.1em"}
        ),
        
        # Anomaly Alert with icon
        dbc.Alert(
            [
                html.I(className="fas fa-exclamation-triangle me-2"),
                html.Span(id="anomaly-alert-text")
            ],
            id="anomaly-alert",
            is_open=False,
            duration=4000,
            className="mb-4"
        ),
        
        # ====== Card 1: Graphs Section ======
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-sliders-h me-2"),
                "グラフ表示"
            ], className="fw-bold"),
            
            dbc.CardBody([
                # Parameter inputs row is optional—moved if you like
                dbc.Row(style=hidden_mode, children=[
                    dbc.Col([
                        html.Label("ハイパスフィルターカットオフ周波数（Hz）:", className="fw-bold"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="cutoff-input",
                                type="number",
                                min=0.01,
                                step=0.01,
                                value=10,
                                className="mb-2",
                            ),
                            dbc.InputGroupText("Hz"),
                        ]),
                        dbc.FormText("推奨範囲: 0.01 - 50 Hz"),
                    ], md=4),
                    dbc.Col([
                        html.Label("RMSウィンドウサイズ:", className="fw-bold"),
                        dbc.Input(
                            id="rms-window-size",
                            type="number",
                            min=1,
                            step=1,
                            value=300,
                            className="mb-2",
                        ),
                        dbc.FormText("推奨範囲: 100 - 500"),
                    ], md=4),
                    dbc.Col([
                        html.Label("HPF_RMSしきい値:", className="fw-bold"),
                        dbc.Input(
                            id="hpf-rms-threshold",
                            type="number",
                            min=0.005,
                            step=0.005,
                            value=0.05,
                            className="mb-2",
                        ),
                        dbc.FormText("推奨範囲: 0.005 - 0.1"),
                    ], md=4),
                ], className="mb-3"),
                
                dbc.Row(style=hidden_mode, children=[
                    dbc.Col([
                        html.Label("スパイクしきい値:", className="fw-bold"),
                        dbc.Input(
                            id="spike-threshold",
                            type="number",
                            min=0.0,
                            step=0.01,
                            value=0.1,
                            className="mb-2",
                        ),
                        dbc.FormText("推奨範囲: 0.05 - 0.2"),
                    ], md=4),
                    dbc.Col([
                        html.Label("初期トルク（Nm）:", className="fw-bold"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="initial-torque-input",
                                type="number",
                                min=0.0,
                                step=0.01,
                                value=0.0,
                                className="mb-2",
                            ),
                            dbc.InputGroupText("Nm"),
                        ]),
                    ], md=4),
                    dbc.Col([
                        html.Label("Y軸スケール:", className="fw-bold"),
                        dcc.RangeSlider(
                            id="y-axis-slider",
                            min=0,
                            max=20,
                            step=0.5,
                            value=[0, 10],
                            marks={i: str(i) for i in range(0, 21, 5)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mb-2",
                        ),
                    ], md=4),
                ]),

                
                
                # Graphs
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.I(className="fas fa-chart-line me-2"),
                                "生データ"
                            ], className="fw-bold"),
                            dbc.CardBody(
                                dcc.Loading(
                                    id="loading-normal-graph",
                                    type="circle",
                                    children=dcc.Graph(id="normal-graph"),
                                )
                            ),
                        ]),
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.I(className="fas fa-filter me-2"),
                                "フィルタ済みデータ"
                            ], className="fw-bold"),
                            dbc.CardBody(
                                dcc.Loading(
                                    id="loading-filtered-graph",
                                    type="circle",
                                    children=dcc.Graph(id="filtered-graph"),
                                )
                            ),
                        ]),
                    ], md=6),
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.I(className="fas fa-wave-square me-2"),
                                "FFT分析"
                            ], className="fw-bold"),
                            dbc.CardBody(
                                dcc.Loading(
                                    id="loading-fft-graph",
                                    type="circle",
                                    children=dcc.Graph(id="fft-graph"),
                                )
                            ),
                        ]),
                    ], md=6),
                ]),
            ]),
        ], className="mb-4"),

        # ====== Card 2: Analysis Summary & Feature Table ======
        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-info-circle me-2"),
                "解析情報"
            ], className="fw-bold"),
            dbc.CardBody([
                # Row with basic info
                dbc.Row([
                    dbc.Col([
                        html.P([
                            html.Strong("ファイル名: "),
                            html.Span(id="analysis-file-name", className="ms-2")
                        ]),
                        html.P([
                            html.Strong("オペレーター: "),
                            html.Span(id="analysis-operator-id", className="ms-2")
                        ]),
                        html.P([
                            html.Strong("ボールサイズ: "),
                            html.Span(id="analysis-ball-size", className="ms-2")
                        ]),
                        html.P([
                            html.Strong("Measurement Date: "),
                            html.Span(id="analysis-measurement-date", className="ms-2")
                        ]),
                        html.P([
                            html.Strong("Analysis Date: "),
                            html.Span(id="analysis-analysis-date", className="ms-2")
                        ]),
                    ], md=6),

                    dbc.Col([
                        html.P([
                            html.Strong("Size: "),
                            html.Span(id="analysis-size", className="ms-2")
                        ]),
                        html.P([
                            html.Strong("Number: "),
                            html.Span(id="analysis-number", className="ms-2")
                        ]),
                        html.P([
                            html.Strong("RPM: "),
                            html.Span(id="analysis-rpm", className="ms-2")
                        ]),

                        # Big PASS / FAIL
                        html.Div(
                            id="analysis-result-big",
                            style={"fontSize": "1.8em", "fontWeight": "bold"},
                            className="mt-2"
                        ),
                        # Confidence
                        html.Div(
                            id="analysis-confidence",
                            style={"fontSize": "1.2em"},
                            className="text-muted",                         
                        ),
                        html.Div(
                            "※信頼スコア: 0.0はNG、1.0はOK",
                            style={"fontSize": "0.8em"},
                            className="mt-2"
                        ),
                        html.Div(
                            id="analysis-result",
                            style={"fontSize": "0.6em"},
                            className="mt-2"
                        ),
                    ], md=6),
                ]),
                
                html.Hr(),

                # The new feature table
                html.H5("抽出された特徴量", className="mt-3", style=hidden_mode),
                
                html.Div(
                    id="features-table-2", 
                    className="mb-4",
                    style=hidden_mode
                ),
            ]),
        ], className="mb-4"),
        
        # Navigation Buttons
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [
                        html.I(className="fas fa-arrow-left me-2"),
                        "ファイル選択に戻る"
                    ],
                    id="back-to-file-selection-btn",
                    color="secondary",
                    className="me-2",
                ),
                dbc.Button(
                    [
                        html.I(className="fas fa-file-csv me-2"),
                        "CSVをダウンロード",
                        dcc.Download(id="download-csv"),
                    ],
                    id="download-csv-btn",
                    color="primary",
                ),
                dbc.Button(
                    [
                        html.I(className="fas fa-file-pdf ms-2"),
                        "PDFをダウンロード",
                        dcc.Download(id="download-pdf"),
                    ],
                    id="download-pdf-btn",
                    color="primary",
                ),
                dbc.Button(
                    [
                        html.I(className="fas fa-arrow-right me-2"),
                        "ラベリングに進む"
                    ],
                    id="proceed-labeling-btn",
                    color="success",
                    disabled=True,
                ),
            ], className="d-flex justify-content-between"),
        ]),
    ])

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
                        dbc.FormText("分析に基づいて適切なラベルを選択してください"),
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
                            "モデルを再トレーニング",
                            id="start-db-training-btn",
                            color="primary",
                            className="mt-4",
                            n_clicks=0
                        )
                    ], md=4, className="d-flex align-items-end"),
                ]),

                html.Hr(),

                html.Div(
                    id="db-training-log",
                    style={
                        "whiteSpace": "pre-wrap",
                        "fontSize": "0.9em",
                        "backgroundColor": "#222",
                        "color": "white",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "minHeight": "200px"
                    }
                ),
            ]),
        ], className="mb-4"),
        
        html.Hr(),
        html.H4("DBに保存されているMeasurementsを参照・編集", className="mt-4"),

        dbc.Button("テーブルをリフレッシュ", id="refresh-db-table-btn", color="info", className="mb-2"),
        dash_table.DataTable(
            id="db-review-table",
            columns=[],
            data=[],
            editable=True,  # let user edit cells
            row_selectable="multi",
            selected_rows=[],
            style_table={"overflowX": "auto", 
                         "backgroundColor": "#222"},
            style_header={"fontWeight": "bold", 
                          "backgroundColor": "#333", 
                          "color": "white"},
            style_cell={"textAlign": "center",
                        "backgroundColor": "#444",
                        "color": "white",
                        "minWidth": "120px", 
                        "width": "120px", 
                        "maxWidth": "250px"},
        ),
        dbc.Button(
            "選択行を削除",
            id="delete-selected-rows-btn",
            color="danger",
            className="mt-2"
        ),
        html.Div(id="db-review-output", className="mt-3 text-info"),
    ])