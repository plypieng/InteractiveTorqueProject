from dash import html, dcc
import dash_bootstrap_components as dbc

def get_loading_overlay():
    return html.Div([
        dbc.Spinner(
            color="primary",
            type="grow",
            size="lg",
        ),
        html.H5("データを処理中...", className="mt-3 text-light", style={"fontFamily": "Orbitron"})
    ], id="loading-overlay", className="loading-overlay-container")

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
