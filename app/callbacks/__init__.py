# app/callbacks/__init__.py
def register_callbacks(app):
    from . import navigation
    from . import file_selection
    from . import data_visualization
    from . import data_labeling
    from . import review_submission
    
    # Register callbacks
    navigation.register_navigation_callbacks(app)
    file_selection.register_file_selection_callbacks(app)
    data_visualization.register_data_visualization_callbacks(app)
    data_labeling.register_data_labeling_callbacks(app)
    review_submission.register_review_submission_callbacks(app)
    # Add other callback registrations as needed
