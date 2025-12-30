# app/callbacks/file_selection.py
from dash import Input, Output, State, no_update, html
from ..config import Config
from ..database.session import SessionLocal
# from ..database.models import BallSize  <-- REMOVED
import os
import logging
import re
from ..services.file_watcher import FileWatcherService

def register_file_selection_callbacks(app):
    @app.callback(
        [
            Output("file-dropdown-1", "options"),
            Output("file-dropdown-1", "value"),
            Output("prev-file-list", "data"),
            Output("directory-status-badge", "children"),
            Output("directory-status-badge", "color"),
            Output("footer-dir-status", "children"),
        ],
        [   
            Input("interval-component", "n_intervals")
        ],
        [
            State("file-dropdown-1", "value"), 
            State("prev-file-list", "data")
        ],
        prevent_initial_call=True
    )
    def update_file_options(
        n_intervals, 
        selected_file1, 
        prev_file_list
    ):
        """_summary_
            Periodically checks for new CSVs in ALLOWED_DIRECTORY and updates the file dropdowns.
        Also pulls ball sizes from DB. Only changes 'value' if the previously selected file is gone.
        """
        
        directory = Config.ALLOWED_DIRECTORY
        try:
            # New Event-Driven Logic
            # Read from memory cache (O(1)) instead of disk (O(N))
            watcher = FileWatcherService()
            files_unsorted = watcher.get_files(cache_key="csv_data")
            
            # If watcher is empty or not started, we might want to fallback or just trust it.
            # But the watcher is robust.
            
            # We still need to sort them by date for better UX.
            # Watchdog cache is just filenames. We need metadata for sorting.
            # BUT: Touching 1000 files to get st_mtime is still slow.
            # OPTIMIZATION: If list hasn't changed, don't re-sort.
            # Dash callback fires on "interval", so we check if set(files) == set(current_files)
            
            # Simplified approach for v1: Just sort the cached list.
            # We can construct full paths
            full_paths_with_filenames = [(f, os.path.join(directory, f)) for f in files_unsorted]
            
            # Sort by modification time (newest first)
            # This part still touches disk, but only for files in the list.
            # Ideally FileWatcher would cache mtime too. 
            # For now, this is still better than os.listdir() overhead on some systems, 
            # but mainly it centralizes the file list.
            
            # Filter out files that might have been deleted between cache update and mtime check
            # and handle potential errors during os.path.getmtime
            valid_full_paths = []
            for filename, full_path in full_paths_with_filenames:
                if os.path.exists(full_path):
                    valid_full_paths.append((filename, full_path))
                else:
                    logging.warning(f"File {full_path} found in cache but not on disk. Skipping.")

            valid_full_paths.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
            
            current_files = [x[0] for x in valid_full_paths] # List of filenames
            options_sorted = [{"label": f, "value": os.path.join(directory, f)} for f in current_files]
            
            # Ball size population REMOVED
            
            # Compare with previous file list
            if sorted(current_files) != sorted(prev_file_list):
                valid_values = [option["value"] for option in options_sorted]
                value1 = selected_file1 if selected_file1 in valid_values else None
                status_text = "Connected" if os.path.exists(directory) else "Disconnected"
                status_color = "success" if os.path.exists(directory) else "danger"
                footer_text = f"({directory})" if os.path.exists(directory) else "(Unreachable)"
                return (options_sorted, value1, current_files, status_text, status_color, footer_text)
            else:
                status_text = "Connected" if os.path.exists(directory) else "Disconnected"
                status_color = "success" if os.path.exists(directory) else "danger"
                footer_text = f"({directory})" if os.path.exists(directory) else "(Unreachable)"
                return [no_update]*3 + [status_text, status_color, footer_text]
        except Exception as e:
            logging.error(f"Error updating file options: {e}", exc_info=True)
            return [no_update]*3 + ["Error", "warning", "(Error)"]
        
        
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
            Output("proceed-visualization-btn", "disabled", allow_duplicate=True),
            Output("proceed-visualization-btn", "children", allow_duplicate=True),
            Output("selected-file-1", "data"),
            Output("selected-ball-size", "data"),
            Output("selected-model", "data"),
        ],
        [
            Input("file-dropdown-1", "value"),
            Input("ball-size-input", "value"),
            Input("operator-id-input", "value"),
            Input("model-dropdown", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_proceed_button(
        file1, 
        ball_size_val, 
        operator_id, 
        model_name
    ):
        """
        Enables the 'Proceed' button only if file1, ball_size_val, operator_id, and model_name are all selected.
        Also sets the hidden stores: selected-file-1, selected-ball-size, selected-model.
        """
        if all([file1, ball_size_val, operator_id, model_name]):
            return (
                False,
                [html.I(className="fas fa-arrow-right me-2"), "可視化に進む"],
                file1,
                ball_size_val,
                model_name
            )
        return (
            True,
            [html.I(className="fas fa-arrow-right me-2"), "Complete All Required Fields"],
            no_update,
            no_update,
            no_update
        )
    
    @app.callback(
        [
            Output("operator-id-input", "valid"),
            Output("operator-id-input", "invalid"),
            Output("operator-id-feedback", "children"),
            Output("ball-size-input", "valid"),
            Output("ball-size-input", "invalid"),
            Output("ball-size-feedback", "children"),
            Output("file-dropdown-1", "valid"),
            Output("file-1-feedback", "children"),
        ],
        [
            Input("operator-id-input", "value"),
            Input("ball-size-input", "value"),
            Input("file-dropdown-1", "value"),
        ],
    )
    def update_form_validation(operator_id, ball_size, file1):
        # Operator ID validation: Alphanumeric, 3-20 chars
        operator_id = (operator_id or "").strip()
        if not operator_id:
            op_valid, op_invalid = False, False
            op_msg = "Please enter your operator ID"
        elif not re.match(r"^[a-zA-Z0-9_\-]+$", operator_id):
            op_valid, op_invalid = False, True
            op_msg = "ID must be alphanumeric (dashes/underscores allowed)"
        elif len(operator_id) < 3:
            op_valid, op_invalid = False, True
            op_msg = "ID too short (min 3 chars)"
        else:
            op_valid, op_invalid = True, False
            op_msg = "Operator ID confirmed"
        
        # Ball size validation: Numeric range 0.1 to 50.0
        if ball_size is None:
            bs_valid, bs_invalid = False, False
            bs_msg = "Please enter a ball size"
        elif not (0.1 <= float(ball_size) <= 50.0):
            bs_valid, bs_invalid = False, True
            bs_msg = "Invalid range (Expected 0.1 - 50.0 mm)"
        else:
            bs_valid, bs_invalid = True, False
            bs_msg = "Ball size within spec"
        
        # File validation
        f_valid = bool(file1)
        f_msg = "File selected" if f_valid else "Select a CSV to proceed"
        
        return [
            op_valid, op_invalid, op_msg,
            bs_valid, bs_invalid, bs_msg,
            f_valid, f_msg,
        ]
    
    @app.callback(
        [
            Output("tabs", "active_tab", allow_duplicate=True),
            Output("tab-2", "disabled"),
            Output("proceed-visualization-btn", "children", allow_duplicate=True),
            Output("proceed-visualization-btn", "disabled", allow_duplicate=True),
            Output("loading-overlay", "style", allow_duplicate=True),
        ],
        [Input("proceed-visualization-btn", "n_clicks")],
        [State("tabs", "active_tab")],
        prevent_initial_call=True
    )
    def switch_to_visualization_tab(n_clicks, current_tab):
        if n_clicks:
            # disable button and indicate processing
            # disable button and indicate processing, show overlay
            return (
                "tab-2",
                False,
                [html.I(className="fas fa-arrow-right me-2"), "Proceeding..."],
                True,
                {"display": "flex"}
            )
        return no_update, no_update, no_update, no_update, {"display": "none"}
