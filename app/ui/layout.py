# app/ui/layout.py
import dash_bootstrap_components as dbc
from dash import html, dcc
from ..config import Config
from .components import (
    get_confirmation_modal,
    get_help_modal,
    get_progress_indicator,
    get_tab1_content,
    get_tab2_content,
    get_tab3_content,
    get_tab4_content,
    get_tab5_content,
    get_settings_content
)

def create_layout():
    return dbc.Container([
        # オーバーレイとモーダル
        get_confirmation_modal(),
        get_help_modal(),
        
        # タイトルとヘルプボタンを含むヘッダー
        dbc.Row([
            dbc.Col([
                html.H1("AIトルクデータ解析 V3.0", className="text-center text-primary mb-2"),
                html.P("Interactive AI Torque-analysis system", className="text-center text-muted mb-4"),
            ], width=12),
            dbc.Col([
                dbc.Button([
                    html.I(className="fas fa-question-circle me-2"),
                    "ヘルプガイド"
                ], id="open-modal", color="info", size="sm", className="me-2"),
                # Mode toggle button
                dbc.Button([
                    html.I(className="fas fa-user-cog me-2"),
                    html.Span(f"{Config.APP_MODE.capitalize()} Mode", id="mode-label")
                ], id="toggle-mode-btn", n_clicks=0, color="secondary", size="sm"),
            ], width=12, className="text-center"),
        ]),
        
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
            ], disabled=True),
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
        html.Footer(
            html.P("© 黒田精工株式会社", className="text-center text-muted mb-3"),
            className="bg-light",
            style={
                "position": "fixed",
                "bottom": "0",
                "width": "100%",
                "backgroundColor": "#f8f9fa",
                "padding": "0.5rem 0",
                "zIndex": "1000",
            },
        ),
    ], fluid=True, style={"paddingBottom": "4rem"})