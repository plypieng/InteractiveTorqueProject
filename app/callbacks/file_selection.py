# app/callbacks/file_selection.py
from dash import Input, Output, State, no_update, html
import dash_bootstrap_components as dbc
from ..config import Config
from ..database.session import SessionLocal
from ..database.models import BallSize
import os
import logging

def register_file_selection_callbacks(app):
    @app.callback(
        [
            Output("file-dropdown-1", "options"),
            Output("file-dropdown-1", "value"),
            Output("file-dropdown-2", "options"),
            Output("file-dropdown-2", "value"),
            Output("ball-size-dropdown", "options"),
            Output("prev-file-list", "data"),
        ],
        [   
            Input("interval-component", "n_intervals")
        ],
        [
            State("file-dropdown-1", "value"), 
            State("file-dropdown-2", "value"), 
            State("prev-file-list", "data")
        ],
        prevent_initial_call=True
    )
    def update_file_and_ball_size_options(
        n_intervals, 
        selected_file1, 
        selected_file2, 
        prev_file_list
    ):
        """_summary_
            Periodically checks for new CSVs in ALLOWED_DIRECTORY and updates the file dropdowns.
        Also pulls ball sizes from DB. Only changes 'value' if the previously selected file is gone.
        """
        
        directory = Config.ALLOWED_DIRECTORY
        try:
            if not os.path.exists(directory):
                logging.warning(f"Allowed directory does not exist: {directory}")
                current_files = []
            else:
                current_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
            files_sorted = sorted(
                current_files,
                key=lambda x: os.path.getmtime(os.path.join(directory, x)),
                reverse=True
            )
            options_sorted = [{"label": f, "value": os.path.join(directory, f)} for f in files_sorted]
            
            # Pull ball sizes from DB
            session = SessionLocal()
            try:
                ball_sizes = session.query(BallSize).all()
                ball_size_options = [{"label": bs.size, "value": bs.id} for bs in ball_sizes]
            except Exception as e:
                logging.error(f"Error fetching ball sizes: {e}", exc_info=True)
                ball_size_options = []
            finally:
                session.close()
            
            # Compare with previous file list
            if sorted(current_files) != sorted(prev_file_list):
                valid_values = [option["value"] for option in options_sorted]
                value1 = selected_file1 if selected_file1 in valid_values else None
                value2 = selected_file2 if selected_file2 in valid_values else None
                return (options_sorted, value1, options_sorted, value2, ball_size_options, current_files)
            else:
                return [no_update]*6
        except Exception as e:
            logging.error(f"Error updating file and ball size options: {e}", exc_info=True)
            return [no_update]*6
        
        
    @app.callback(
        Output("model-dropdown", "options"),       
        Input("interval-component", "n_intervals"),
        prevent_initial_call=True
    )
    def refresh_model_dropdown_list(n):
        """
        Periodically scans the 'trained_models/' folder for .pkl files and updates ONLY the .options
        of model-dropdown. We do NOT set .value here, so the user's chosen model remains stable
        unless it no longer exists in the folder.
        """
        PROJECT_ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        models_dir = os.path.join(PROJECT_ROOT, "trained_models")
        
        if not os.path.exists(models_dir):
            logging.warning(f"Models directory not found: {models_dir}")
            return []
        
        pkl_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
        pkl_files_sorted = sorted(pkl_files)  # optional sorting
        options = [{"label": f, "value": f} for f in pkl_files_sorted]
        return options
        

    @app.callback(
        [
            Output("proceed-visualization-btn", "disabled"),
            Output("proceed-visualization-btn", "children"),
            Output("selected-file-1", "data"),
            Output("selected-file-2", "data"),
            Output("selected-ball-size", "data"),
            Output("selected-model", "data"),
        ],
        [
            Input("file-dropdown-1", "value"),
            Input("ball-size-dropdown", "value"),
            Input("operator-id-input", "value"),
            Input("model-dropdown", "value"),
            
        ],
        [State("file-dropdown-2", "value")],
        prevent_initial_call=True,
    )
    def update_proceed_button(
        file1, 
        ball_size_id, 
        operator_id, 
        model_name, 
        file2
    ):
        """
        Enables the 'Proceed' button only if file1, ball_size_id, operator_id, and model_name are all selected.
        Also sets the hidden stores: selected-file-1, selected-file-2, selected-ball-size, selected-model.
        """
        if all([file1, ball_size_id, operator_id, model_name]):
            return (
                False,
                [html.I(className="fas fa-arrow-right me-2"), "Proceed to Visualization"],
                file1,
                file2,
                ball_size_id,
                model_name
            )
        return (
            True,
            [html.I(className="fas fa-arrow-right me-2"), "Complete All Required Fields"],
            no_update,
            no_update,
            no_update,
            no_update
        )
    
    @app.callback(
        [
            Output("operator-id-input", "valid"),
            Output("operator-id-feedback", "children"),
            Output("ball-size-dropdown", "valid"),
            Output("ball-size-feedback", "children"),
            Output("file-dropdown-1", "valid"),
            Output("file-1-feedback", "children"),
        ],
        [
            Input("operator-id-input", "value"),
            Input("ball-size-dropdown", "value"),
            Input("file-dropdown-1", "value"),
        ],
    )
    def update_form_validation(operator_id, ball_size, file1):
        # Operator ID validation
        if not operator_id:
            operator_valid = False
            operator_message = "Please enter your operator ID"
        else:
            operator_valid = True
            operator_message = "Operator ID valid"
        
        # Ball size validation
        if not ball_size:
            ball_size_valid = False
            ball_size_message = "Please select a ball size"
        else:
            ball_size_valid = True
            ball_size_message = "Ball size selected"
        
        # File validation
        if not file1:
            file_valid = False
            file_message = "Please select a primary CSV file"
        else:
            file_valid = True
            file_message = "File selected"
        
        return [
            operator_valid,
            operator_message,
            ball_size_valid,
            ball_size_message,
            file_valid,
            file_message,
        ]
    
    @app.callback(
        [Output("tabs", "active_tab", allow_duplicate=True),
         Output("tab-2", "disabled")],
        [Input("proceed-visualization-btn", "n_clicks")],
        [State("tabs", "active_tab")],
        prevent_initial_call=True
    )
    def switch_to_visualization_tab(n_clicks, current_tab):
        if n_clicks:
            return "tab-2", False
        return no_update, no_update
    
    
    
