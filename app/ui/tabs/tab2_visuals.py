from dash import html, dcc
import dash_bootstrap_components as dbc
from ...config import Config

def get_tab2_content():
    hidden_mode = {"display": "none"} if Config.APP_MODE == "operator" else {}
    
    return dbc.Container([
        html.H3("ステップ 2: データの可視化", className="mb-4"),
        
        # --- Parameter Sidebar (Offcanvas) ---
        html.Div([
            dbc.Button(
                [html.I(className="fas fa-sliders-h me-2"), "パラメータ調整"],
                id="open-params-canvas",
                n_clicks=0,
                color="secondary",
                className="mb-3"
            ),
        ], style=hidden_mode),

        dbc.Offcanvas(
            html.Div([
                html.H5("信号処理パラメータ", className="border-bottom pb-2 mb-3"),
                
                html.Label("HPFカットオフ (Hz):", className="fw-bold mt-2"),
                dbc.Input(id="cutoff-input", type="number", value=10, min=0.01, step=0.01),
                dbc.FormText("推奨: 0.01 - 50 Hz"),
                
                html.Label("RMSウィンドウサイズ:", className="fw-bold mt-3"),
                dbc.Input(id="rms-window-size", type="number", value=300, min=1, step=1),
                
                html.Label("HPF_RMSしきい値:", className="fw-bold mt-3"),
                dbc.Input(id="hpf-rms-threshold", type="number", value=0.05, min=0.001, step=0.001),
                
                html.Label("スパイクしきい値:", className="fw-bold mt-3"),
                dbc.Input(id="spike-threshold", type="number", value=0.1, min=0.01, step=0.01),
                
                html.Label("初期トルク (Nm):", className="fw-bold mt-3"),
                dbc.Input(id="initial-torque-input", type="number", value=0.0, step=0.01),

                html.Label("Y軸スケール:", className="fw-bold mt-3"),
                dcc.RangeSlider(
                    id="y-axis-slider",
                    min=0, max=20, step=0.5,
                    value=[0, 10],
                    marks={i: str(i) for i in range(0, 21, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ]),
            id="params-offcanvas",
            title="調整パネル",
            is_open=False,
            placement="end",
            style={"background": "rgba(22, 27, 34, 0.95)", "color": "#c9d1d9"}
        ),

        # --- Alerts ---
        dbc.Alert(id="model-prediction", color="primary", dismissable=True, is_open=False, className="mb-2"),
        dbc.Alert([html.I(className="fas fa-exclamation-triangle me-2"), html.Span(id="anomaly-alert-text")], 
                  id="anomaly-alert", is_open=False, color="danger", className="mb-4"),

        # --- Grid Layout for Graphs (2x2) ---
        dbc.Row([
            # Col 1: Raw Data
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.I(className="fas fa-chart-line me-2"), "生データ"], className="py-1"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id="normal-graph", style={"height": "300px"}), type="circle"), className="p-0")
            ], className="h-100"), md=6, className="mb-3"),

            # Col 2: Filtered Data
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.I(className="fas fa-filter me-2"), "フィルタ済み"], className="py-1"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id="filtered-graph", style={"height": "300px"}), type="circle"), className="p-0")
            ], className="h-100"), md=6, className="mb-3"),
        ]),

        dbc.Row([
            # Col 3: FFT
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.I(className="fas fa-wave-square me-2"), "FFT分析"], className="py-1"),
                dbc.CardBody(dcc.Loading(dcc.Graph(id="fft-graph", style={"height": "300px"}), type="circle"), className="p-0")
            ], className="h-100"), md=6, className="mb-3"),

            # Col 4: Analysis & Features
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.I(className="fas fa-info-circle me-2"), "解析結果"], className="py-1"),
                dbc.CardBody([
                   html.Div(id="analysis-result-big", className="text-center my-2", style={"fontSize": "1.5rem"}),
                   dbc.Progress(id="analysis-confidence-bar", value=0, striped=True, animated=True, className="mb-2", style={"height": "10px"}),
                   html.Div(id="analysis-confidence-text", className="text-end small text-muted mb-3"),
                   html.Hr(),
                   html.Div(id="features-table-2", style=hidden_mode) # Feature Table
                ], className="p-3 overflow-auto", style={"height": "300px"})
            ], className="h-100"), md=6, className="mb-3"),
        ]),

        # --- Hidden Data Stores (for callbacks compatibility if needed) ---
        # Note: Analysis info spans (rpm, size, etc) were removed from main view for density.
        # We can add them to a popover or tooltip if needed, or put them back in "Analysis Result" card.

        # --- Navigation ---
        dbc.Row([
            dbc.Col([
                dbc.Button([html.I(className="fas fa-arrow-left me-2"), "戻る"], id="back-to-file-selection-btn", color="secondary"),
                html.Div([
                    dbc.Button([html.I(className="fas fa-file-csv me-2"), "CSV"], id="download-csv-btn", color="primary", size="sm", className="me-1"),
                    dcc.Download(id="download-csv"),
                    dbc.Button([html.I(className="fas fa-file-pdf me-2"), "PDF"], id="download-pdf-btn", color="primary", size="sm", className="me-2"),
                    dcc.Download(id="download-pdf"),
                    dbc.Button([html.I(className="fas fa-arrow-right me-2"), "ラベリング"], id="proceed-labeling-btn", color="success", disabled=True),
                ], className="d-inline-block float-end")
            ])
        ], className="mt-2 mb-5")
    ], fluid=True)
