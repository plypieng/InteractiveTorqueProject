import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from dash import no_update

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.callbacks.data_visualization import register_data_visualization_callbacks
from app.callbacks.data_labeling import register_data_labeling_callbacks
from app.callbacks.file_selection import register_file_selection_callbacks

# Import the actual callback functions (we need to access them from the module or register them to a dummy app)
# Since the callbacks are inner functions in register_*, we might need to refactor or use a trick.
# However, looking at the code, the functions are defined inside register_*. 
# A better approach for testing is to extract the logic or use a mock app to capture the callbacks.

# Let's try to import the modules and see if we can access the functions if they were top-level.
# Wait, they are inner functions. This makes unit testing hard without refactoring.
# BUT, I can use a trick: I can pass a mock app to register_* and capture the decorated functions.

class TestButtonCallbacks(unittest.TestCase):
    def setUp(self):
        self.callbacks = {}
        self.mock_app = MagicMock()
        
        # Capture callbacks
        def mock_callback(*args, **kwargs):
            def decorator(func):
                # Store function by name or some identifier. 
                # Since we don't have easy IDs here, we'll use function name.
                self.callbacks[func.__name__] = func
                return func
            return decorator
        
        self.mock_app.callback = mock_callback
        self.mock_app.clientside_callback = MagicMock()
        
        # Register callbacks to capture them
        register_data_visualization_callbacks(self.mock_app)
        register_data_labeling_callbacks(self.mock_app)
        register_file_selection_callbacks(self.mock_app)

    def test_switch_to_file_selection_tab(self):
        """Test 'Back to File Selection' button"""
        func = self.callbacks.get('switch_to_file_selection_tab')
        self.assertIsNotNone(func)
        
        # Test click
        result = func(1)
        self.assertEqual(result, "tab-1")
        
        # Test no click
        result = func(None)
        self.assertEqual(result, no_update)

    def test_switch_to_visualization_tab(self):
        """Test 'Proceed to Visualization' button"""
        func = self.callbacks.get('switch_to_visualization_tab')
        self.assertIsNotNone(func)
        
        # Test click
        result = func(1, "tab-1")
        self.assertEqual(result[0], "tab-2")
        self.assertEqual(result[1], False) # disabled=False
        self.assertTrue("処理中" in result[2][1]) # loading text
        self.assertEqual(result[3], True) # button disabled=True

    def test_switch_to_labeling_tab(self):
        """Test 'Proceed to Labeling' button"""
        func = self.callbacks.get('switch_to_labeling_tab')
        self.assertIsNotNone(func)
        
        # Test click with valid data
        result = func(1, "Op1", 10.5, "path/to/file.csv")
        self.assertEqual(result[0], "tab-3")
        self.assertEqual(result[1], False) # disabled=False
        self.assertEqual(result[2], "file.csv")
        self.assertEqual(result[3], "10.5")
        self.assertEqual(result[4], "Op1")

    @patch('app.callbacks.data_visualization.generate_pdf')
    @patch('app.callbacks.data_visualization.shutil')
    @patch('app.callbacks.data_visualization.os.path.expanduser')
    def test_download_pdf(self, mock_expanduser, mock_shutil, mock_generate_pdf):
        """Test 'Download PDF' button"""
        func = self.callbacks.get('download_pdf')
        self.assertIsNotNone(func)
        
        mock_generate_pdf.return_value = "generated.pdf"
        mock_expanduser.return_value = "C:/Users/Test"
        
        # Test click
        result = func(1, "path/to/file.csv")
        
        # Verify generate_pdf called
        mock_generate_pdf.assert_called_with("path/to/file.csv")
        
        # Verify copy
        mock_shutil.copy2.assert_called()
        
        # Verify alert return
        self.assertEqual(result[1], True) # is_open
        self.assertIn("saved to Downloads", result[2])
        self.assertEqual(result[3], "success")

    @patch('app.callbacks.data_visualization.shutil')
    @patch('app.callbacks.data_visualization.os.path.expanduser')
    def test_download_csv(self, mock_expanduser, mock_shutil):
        """Test 'Download CSV' button"""
        func = self.callbacks.get('download_csv')
        self.assertIsNotNone(func)
        
        mock_expanduser.return_value = "C:/Users/Test"
        
        # Test click
        result = func(1, "path/to/file.csv")
        
        # Verify copy
        mock_shutil.copy2.assert_called()
        
        # Verify alert return
        self.assertEqual(result[1], True)
        self.assertIn("saved to Downloads", result[2])
        self.assertEqual(result[3], "success")

    @patch('app.callbacks.data_labeling.SessionLocal')
    def test_save_label_button(self, mock_session_cls):
        """Test 'Save Label' button"""
        func = self.callbacks.get('handle_label_submit_button_and_back_button')
        self.assertIsNotNone(func)
        
        # Mock DB session
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.query.return_value.filter_by.return_value.first.return_value = MagicMock() # Existing record
        
        # Mock context
        with patch('app.callbacks.data_labeling.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': 'save-label-btn.n_clicks'}]
            
            # Call function with all 13 args
            # save_n, back_n, label, notes, file, labels_data, active_tab, review_files, review_labels, sel_ball, ball_val, op_id, model
            result = func(
                1, 0, "PASS", "Note", "file.csv", 
                {}, "tab-3", [], [], {}, 10.0, "Op1", "ModelX"
            )
            
            # Verify DB commit
            mock_session.commit.assert_called()
            
            # Verify return
            self.assertEqual(result[3], "success") # alert color
            self.assertEqual(result[4], "tab-4") # active tab

    def test_back_to_visualization_from_labeling(self):
        """Test 'Back to Visualization' button on labeling page"""
        func = self.callbacks.get('handle_label_submit_button_and_back_button')
        
        with patch('app.callbacks.data_labeling.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': 'back-to-visualization-btn.n_clicks'}]
            
            result = func(
                0, 1, "PASS", "Note", "file.csv", 
                {}, "tab-3", [], [], {}, 10.0, "Op1", "ModelX"
            )
            
            # Should return tab-2
            self.assertEqual(result[4], "tab-2")

if __name__ == '__main__':
    unittest.main()
