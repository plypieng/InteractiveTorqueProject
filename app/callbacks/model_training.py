# app/callbacks/model_training.py

import os
import logging
import pandas as pd
import numpy as np

from dash import Input, Output, State, callback_context, html
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
from ..database.models import Measurement


def register_db_review_callbacks(app):
    """
    Shows a table with all measurements. Allows basic CRUD: e.g. update label, delete row.
    """

    @app.callback(
        Output("db-review-table", "data"),
        Output("db-review-table", "columns"),
        Output("db-review-table", "tooltip_data"),
        Input("refresh-db-table-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def refresh_db_table(n):
        if not n:
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
            {"name": "ID", "id": "id"},
            {"name": "File Path", "id": "file_path"},
            {"name": "Label", "id": "label"},
            {"name": "Predicted", "id": "predicted_label"},
            {"name": "Confidence", "id": "confidence"},
            {"name": "Model Version", "id": "model_version"},
            {"name": "Submitted-Timestamp", "id": "submitted_timestamp"},
            {"name": "Notes", "id": "notes"},
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
                    measurement.file_path = row["file_path"]
                    measurement.label = row["label"]
                    measurement.predicted_label = row["predicted_label"]
                    measurement.prediction_confidence = float(row["confidence"]) if row["confidence"] else None
                    measurement.notes = row["notes"]
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

def register_model_training_callbacks(app):

    @app.callback(
        [
            Output("db-training-log", "children"),
            Output("db-training-status-alert", "is_open"),
            Output("db-training-status-alert", "children"),
            Output("db-training-status-alert", "color"),
        ],
        [Input("start-db-training-btn", "n_clicks")],
        [
            State("db-training-log", "children"),
            State("training-model-name", "value"),
        ],
        prevent_initial_call=True
    )
    def train_model_from_db_wide(n_clicks, current_log, model_name):
        """
        Trains a new model from DB using wide columns on Measurement.
        Saves to {model_name}.pkl if model_name is given, else "best_model.pkl".
        """
        if not n_clicks:
            raise PreventUpdate

        new_log = current_log or ""
        new_log += "\n[INFO] Training started..."

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

        # 6) Save pipeline to the project-root/trained_models folder
        model_path = os.path.join(models_dir, model_file)
        joblib.dump(best_model, model_path)
        new_log += f"\n[SAVED] {model_file} -> {model_path}\n"

        return (new_log, True, "モデル学習が成功しました！", "success")
