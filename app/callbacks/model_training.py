# app/callbacks/model_training.py

import os
import logging
import pandas as pd
import numpy as np

from dash import Input, Output, State, callback_context, html, dcc
from dash.exceptions import PreventUpdate

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek
import joblib

from ..database.session import SessionLocal
from ..database.session import SessionLocal
from ..database.models import Measurement, AuditLog, ModelRegistry
import json
import uuid


def register_db_review_callbacks(app):
    """
    Shows a table with all measurements. Allows basic CRUD: e.g. update label, delete row.
    """

    @app.callback(
        [Output("db-review-table", "data"), Output("db-review-table", "columns"), Output("db-review-table", "tooltip_data")],
        Input("tabs", "active_tab"),
        Input("refresh-db-table-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def refresh_db_table(active_tab, n_clicks):
        ctx = callback_context
        triggered = ctx.triggered[0]["prop_id"].split(".")[0]
        # Only refresh on Tab5 activation or manual click
        if triggered == "tabs":
            if active_tab != "tab-5":
                raise PreventUpdate
        elif triggered == "refresh-db-table-btn":
            if not n_clicks:
                raise PreventUpdate
        else:
            raise PreventUpdate
        with SessionLocal() as session:
            measurements = session.query(Measurement).all()
            rows = []
            tooltip_data = []
            for m in measurements:
                row = {
                    "id": m.id,
                    "submitted_timestamp": m.submitted_timestamp,
                    "file_path": m.file_path,
                    "order_id": m.order_id,
                    "operator_id": m.operator_id,
                    "label": m.label,
                    "predicted_label": m.predicted_label,
                    "confidence": m.prediction_confidence,
                    "model_version": m.model_version,
                    "notes": m.notes,
                }
                rows.append(row)
                tooltip_data.append({
                    "submitted_timestamp": m.submitted_timestamp,
                    "file_path": m.file_path,
                    "notes": m.notes, 
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return [], [{"name": "No data", "id": "no_data"}], []

        # Define columns with custom names and IDs
        columns = [
            {"name": "ID", "id": "id", "editable": False},
            {"name": "ファイルパス", "id": "file_path", "editable": False},
            {"name": "オーダー番号", "id": "order_id", "editable": False},
            {"name": "オペレーターID", "id": "operator_id", "editable": False},
            {"name": "ラベル", "id": "label", "editable": False},
            {"name": "予測ラベル", "id": "predicted_label", "editable": False},
            {"name": "予測信頼度", "id": "confidence", "editable": False},
            {"name": "モデルバージョン", "id": "model_version", "editable": False},
            {"name": "提出日時", "id": "submitted_timestamp", "editable": False},
            {"name": "備考", "id": "notes", "editable": True, "presentation": "input"},
        ]
        return df.to_dict("records"), columns, tooltip_data

    @app.callback(
        Output("db-review-output", "children"),
        Input("db-review-table", "data_timestamp"),
        State("db-review-table", "data"),
        prevent_initial_call=True
    )
    def update_database_on_edit(timestamp, table_data):
        """
        This callback is triggered when the user edits a cell in the DataTable.
        We'll update the DB with the new label/notes/predicted_label, etc.
        """
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        changed_rows = table_data  # entire table's data
        messages = []
        with SessionLocal() as session:
            for row in changed_rows:
                # row["id"] is the primary key
                mid = row["id"]
                measurement = session.query(Measurement).filter_by(id=mid).first()
                if measurement:
                    # Detect changes
                    changes = []
                    if measurement.label != row["label"]:
                        changes.append(f"Label: {measurement.label} -> {row['label']}")
                        measurement.label = row["label"]
                    
                    if measurement.notes != row["notes"]:
                        measurement.notes = row["notes"]
                        # We might not log simple note changes, or maybe we do. Let's log major ones.
                    
                    if changes:
                        audit = AuditLog(
                            measurement_id=measurement.id,
                            changed_by="Admin/Reviewer", # We don't have operator ID here easily unless we store it in session
                            previous_status=measurement.status,
                            new_status=measurement.status,
                            previous_label=None, # Already captured in reason
                            new_label=None,
                            change_reason=f"DB Review Table Edit: {'; '.join(changes)}"
                        )
                        session.add(audit)
                        
                    measurement.file_path = row["file_path"]
                    measurement.predicted_label = row["predicted_label"]
                    measurement.prediction_confidence = float(row["confidence"]) if row["confidence"] else None
            try:
                session.commit()
                messages.append("Database updated successfully.")
            except Exception as e:
                logging.error(f"Error updating DB: {e}", exc_info=True)
                session.rollback()
                messages.append("Error updating the database.")
        return html.Div(messages)

    @app.callback(
        Output("db-review-output", "children", allow_duplicate=True),
        Input("delete-selected-rows-btn", "n_clicks"),
        State("db-review-table", "selected_rows"),
        State("db-review-table", "data"),
        prevent_initial_call=True
    )
    def delete_selected_rows(n, selected_rows, table_data):
        if not n or not selected_rows:
            raise PreventUpdate

        ids_to_delete = [table_data[i]["id"] for i in selected_rows]
        with SessionLocal() as session:
            for mid in ids_to_delete:
                session.query(Measurement).filter_by(id=mid).delete()
            try:
                session.commit()
                msg = f"Deleted {len(ids_to_delete)} rows from DB."
            except Exception as e:
                logging.error(f"Error deleting rows: {e}", exc_info=True)
                session.rollback()
                msg = "Error deleting rows."
        return msg

    @app.callback(
        Output("download-db-csv", "data"),
        Input("download-db-csv-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def download_db_csv(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        with SessionLocal() as session:
            measurements = session.query(Measurement).all()
            # Extract all columns (including features) dynamically
            rows = []
            for m in measurements:
                row = {col.name: getattr(m, col.name) for col in Measurement.__table__.columns}
                rows.append(row)
        df = pd.DataFrame(rows)
        return dcc.send_data_frame(df.to_csv, "measurements.csv", index=False)

def register_model_training_callbacks(app):

    @app.callback(
        output=[
            Output("db-training-log", "children"),
            Output("db-training-status-alert", "is_open"),
            Output("db-training-status-alert", "children"),
            Output("db-training-status-alert", "color"),
        ],
        inputs=[Input("start-db-training-btn", "n_clicks")],
        state=[
            State("db-training-log", "children"),
            State("training-model-name", "value"),
        ],
        running=[
            (Output("start-db-training-btn", "disabled"), True, False),
            (Output("cancel-training-btn", "disabled"), False, True),
            (Output("db-training-log", "style"), {"opacity": "0.5"}, {"opacity": "1.0"}),
        ],
        cancel=[Input("cancel-training-btn", "n_clicks")],
        progress=[Output("db-training-log", "children")],
        prevent_initial_call=True,
        background=True
    )
    def train_model_from_db_wide(set_progress, n_clicks, current_log, model_name):
        """
        Trains a new model from DB using wide columns on Measurement.
        Saves to {model_name}.pkl if model_name is given, else "best_model.pkl".
        """         
        if not n_clicks:
            raise PreventUpdate

        new_log = current_log or ""
        update_log = lambda msg: f"{new_log}\n{msg}" if new_log else msg
        
        new_log += "\n[INFO] Training started..."
        set_progress(new_log)

        if not model_name:
            model_name = "best_model"
        # sanitize model_name if necessary to avoid illegal file chars
        # or you can do more checks e.g. length, special characters, etc.
        model_name = model_name.strip().replace(" ", "_")
        if not model_name.endswith(".pkl"):
            model_file = f"{model_name}.pkl"
        else:
            model_file = model_name

        new_log += f"\n[INFO] Model Name: {model_file}"
        
        # ========================
        # PROJECT ROOT SETUP
        # ========================
        # We'll go 2 directories up from: InteractiveTorqueProjectv2/app/callbacks/
        # So we land at: InteractiveTorqueProjectv2/
        PROJECT_ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        models_dir = os.path.join(PROJECT_ROOT, "trained_models")
        os.makedirs(models_dir, exist_ok=True)  # ensure folder exists

        # 1) Query DB for pass/fail
        session = SessionLocal()
        try:
            measurements = session.query(Measurement).filter(
                Measurement.label.in_(["PASS", "FAIL"])
            ).all()
        except Exception as e:
            session.rollback()
            new_log += f"\n[ERROR] DB query failed: {e}"
            return (new_log, True, "DBエラーが発生しました", "danger")
        finally:
            session.close()

        if not measurements:
            new_log += "\n[WARN] No PASS/FAIL-labeled data found."
            return (new_log, True, "ラベル付きデータがありません。", "warning")

        new_log += f"\n[INFO] Found {len(measurements)} labeled measurements."
        set_progress(new_log)

        # 2) Convert to DataFrame from wide columns
        feature_cols = [
            "mean","median","mad","standard_deviation","rms","shape_factor","crest_factor","entropy",
            "skewness","kurtosis","gradient_mean","gradient_std_dev","rolling_median_mean","rolling_mad_mean",
            "spectral_centroid","spectral_entropy","peak_frequency","spectral_flatness","spectral_spread",
            "spectral_roll_off","low_band_energy","mid_band_energy","high_band_energy","spectral_crest",
            "spectral_flux","spectral_kurtosis","spectral_skewness","spectral_slope"
        ]
        rows = []
        for m in measurements:
            label_val = 1 if m.label == "PASS" else 0
            row_dict = {}
            for col in feature_cols:
                row_dict[col] = getattr(m, col, None)
            row_dict["Label"] = label_val
            rows.append(row_dict)

        df = pd.DataFrame(rows)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        set_progress(new_log + "\n[INFO] Data cleaning complete. Starting Grid Search...")

        if df.empty:
            new_log += "\n[WARN] All data is NaN or inf after cleaning."
            return (new_log, True, "特徴量が欠損しており学習できません。", "warning")

        new_log += f"\n[INFO] Final shape after dropna: {df.shape}"

        X = df.drop(columns=["Label"])
        y = df["Label"]

        # 3) Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        # 4) SMOTETomek
        sm = SMOTETomek(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)

        # 5) Pipeline & params
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("clf", LogisticRegression(random_state=42))
        ])
        param_dist = [
            {
                "pca__n_components": [5, 10, None],
                "clf": [LogisticRegression(max_iter=1000, random_state=42)],
                "clf__C": [0.01, 0.1, 1, 10],
            },
            {
                "pca__n_components": [5, 10, None],
                "clf": [RandomForestClassifier(random_state=42)],
                "clf__n_estimators": [50, 100],
                "clf__max_depth": [5, 10, None],
            }
        ]

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        rand_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=5,
            cv=cv,
            scoring="accuracy",
            refit=True,
            random_state=42,
            n_jobs=-1
        )
        rand_search.fit(X_res, y_res)

        best_model = rand_search.best_estimator_
        new_log += f"\n[INFO] Best Params: {rand_search.best_params_}"
        new_log += f"\n[INFO] Best CV Score: {rand_search.best_score_:.3f}"

        test_acc = best_model.score(X_test, y_test)
        new_log += f"\n[INFO] Test Accuracy: {test_acc:.3f}"
        set_progress(new_log + "\n[INFO] Saving model...")

        # 6) Save pipeline to the project-root/trained_models folder
        model_path = os.path.join(models_dir, model_file)
        joblib.dump(best_model, model_path)
        new_log += f"\n[SAVED] {model_file} -> {model_path}\n"

        # 7) Register in DB
        try:
            # Extract algorithm name and params
            final_step = best_model.named_steps["clf"]
            algo_type = final_step.__class__.__name__
            best_params_json = json.dumps(rand_search.best_params_, default=str)
            
            # Generate a version string or use the filename if unique
            version_id = model_name if model_name != "best_model" else f"best_model_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

            with SessionLocal() as session:
                new_model_record = ModelRegistry(
                    version=version_id,
                    algorithm_type=algo_type,
                    hyperparameters=best_params_json,
                    test_accuracy=float(test_acc),
                    file_path=model_file,
                    created_by="System_Training", # or pass user ID if available
                    is_active=1
                )
                session.add(new_model_record)
                session.commit()
                new_log += f"\n[REGISTRY] Model registered in DB (ID: {new_model_record.id})"
        except Exception as e:
            new_log += f"\n[WARN] Failed to register model in DB: {e}"

        return (new_log, True, "モデル学習が成功しました！", "success")
