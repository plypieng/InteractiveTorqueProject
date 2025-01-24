# app/ui/layout.py
import dash_bootstrap_components as dbc
from dash import html, dcc
from .components import (
    get_loading_overlay,
    get_confirmation_modal,
    get_progress_indicator,
    get_help_modal,
    get_tab1_content,
    get_tab2_content,
    get_tab3_content,
    get_tab4_content,
    get_tab5_content
)

def create_layout():
    return dbc.Container([
        # オーバーレイとモーダル
        get_loading_overlay(),
        get_confirmation_modal(),
        get_help_modal(),
        
        # タイトルとヘルプボタンを含むヘッダー
        dbc.Row([
            dbc.Col([
                html.H1("トルクデータ解析 V2.0", className="text-center text-primary mb-2"),
                html.P("Torque Data Analysis System", className="text-center text-muted mb-4"),
            ], width=12),
            dbc.Col([
                dbc.Button([
                    html.I(className="fas fa-question-circle me-2"),
                    "ヘルプガイド"
                ], id="open-modal", color="info", size="sm", className="mb-3"),
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
                get_tab5_content()],  disabled=False),
        ], id="tabs", active_tab="tab-1"),
    ], fluid=True)