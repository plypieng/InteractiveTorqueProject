# app/ui/layout.py
import dash_bootstrap_components as dbc
from dash import html, dcc
from ..config import Config
from .tabs.common import (
    get_confirmation_modal,
    get_help_modal,
    get_progress_indicator,
    get_loading_overlay
)
from .tabs.tab1_selection import get_tab1_content
from .tabs.tab2_visuals import get_tab2_content
from .tabs.tab3_labeling import get_tab3_content
from .tabs.tab4_review import get_tab4_content
from .tabs.tab5_training import get_tab5_content
from .tabs.settings import get_settings_content

def create_layout():
    return dbc.Container([
        # オーバーレイとモーダル
        get_confirmation_modal(),
        get_help_modal(),
        
        # タイトルとヘルプボタンを含むヘッダー
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className="fas fa-microchip me-3"),
                    "AIトルクデータ解析 V3.0"
                ], className="text-center mb-0", style={"fontFamily": "Orbitron", "fontWeight": "700", "color": "#58a6ff", "textShadow": "0 0 10px rgba(88, 166, 255, 0.3)"}),
                html.P("Interactive AI Torque-analysis system for precision engineering", className="text-center text-muted mb-4", style={"letterSpacing": "1px", "fontSize": "0.9rem"}),
            ], width=12),
            dbc.Col([
                dbc.Button([
                    html.I(className="fas fa-question-circle me-2"),
                    "ヘルプガイド"
                ], id="open-modal", color="info", size="sm", className="me-2 outline-info"),
                # Mode toggle button
                dbc.Button([
                    html.I(className="fas fa-user-cog me-2"),
                    html.Span(f"{Config.APP_MODE.capitalize()} Mode", id="mode-label")
                ], id="toggle-mode-btn", n_clicks=0, color="secondary", size="sm", className="outline-secondary"),
            ], width=12, className="text-center"),
        ], className="py-4 mb-4", style={"background": "linear-gradient(180deg, rgba(33,38,45,0.8) 0%, rgba(13,17,23,0) 100%)", "borderBottom": "1px solid rgba(48,54,61,0.5)"}),
        
        # プログレスインジケーター
        get_progress_indicator(),
        
        # 定期的な更新用のインターバルコンポーネント
        dcc.Interval(
            id="interval-component",
            interval=10*1000,  # 10秒
            n_intervals=0,
        ),
        
        # 状態管理用の隠しストア
        dcc.Store(id="selected-file-1"),
        dcc.Store(id="selected-file-2"),
        dcc.Store(id="selected-ball-size"),
        dcc.Store(id="selected-model"),
        dcc.Store(id="labels-data", data={}),
        dcc.Store(id="prev-file-list", data=[]),
        dcc.Store(id="features-data", data={}),
        dcc.Store(id="analysis-model-info", data=""),
        dcc.Store(id="user-mode", data=Config.APP_MODE),
        # Store for keyboard events
        dcc.Store(id="keyboard-event"),

        # Password modal for developer mode
        dbc.Modal([
            dbc.ModalHeader("開発者モード認証"),
            dbc.ModalBody([
                dbc.Label("開発者パスワードを入力:", html_for="password-input"),
                dbc.Input(id="password-input", type="password"),
                html.Div(id="password-error", style={"color":"red","marginTop":"10px"}),
            ]),
            dbc.ModalFooter([
                dbc.Button("キャンセル", id="password-cancel-btn", n_clicks=0, color="secondary", className="me-2"),
                dbc.Button("送信", id="password-submit-btn", n_clicks=0, color="primary"),
            ]),
        ], id="password-modal", is_open=False, centered=True),
        
        # メインタブ
        dbc.Tabs([
            dbc.Tab(label="ファイル選択", tab_id="tab-1", id="tab-1", children=[
                get_tab1_content()
            ]),
            dbc.Tab(label="データの可視化", tab_id="tab-2", id="tab-2", children=[
                get_tab2_content()
            ], disabled=False),
            dbc.Tab(label="データのラベリング", tab_id="tab-3", id="tab-3", children=[
                get_tab3_content()
            ], disabled=True),
            dbc.Tab(label="レビューと提出", tab_id="tab-4", id="tab-4", children=[
                get_tab4_content()
            ], disabled=True),
            dbc.Tab(label="DBからモデル学習", tab_id="tab-5", id="tab-5", children=[
                get_tab5_content()], disabled=False),
            dbc.Tab(label="設定", tab_id="tab-6", id="tab-6", children=[
                get_settings_content()
            ], disabled=False),
        ], id="tabs", active_tab="tab-1"),

        # Full Page Loading Overlay
        get_loading_overlay(),

        html.Footer(
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.P([
                            html.I(className="fas fa-database me-2", style={"color": "#238636"}),
                            "DB: Connected",
                            html.Span(className="mx-3", style={"borderLeft": "1px solid #30363d"}),
                            html.I(className="fas fa-folder-open me-2", style={"color": "#e3b341"}),
                            "Dir: Validating...",
                            html.Span(id="footer-dir-status", className="ms-1"),
                        ], className="text-muted small mb-0"),
                    ], md=6, className="d-flex align-items-center"),
                    dbc.Col([
                        html.P("© 2025 黒田精工株式会社 | Version 3.0.4", className="text-end text-muted small mb-0"),
                    ], md=6),
                ], className="py-2")
            ], fluid=True),
            className="fixed-bottom",
            style={
                "backgroundColor": "rgba(22, 27, 34, 0.95)",
                "borderTop": "1px solid #30363d",
                "zIndex": "1001",
            },
        ),
    ], fluid=True, style={"paddingBottom": "10rem"})