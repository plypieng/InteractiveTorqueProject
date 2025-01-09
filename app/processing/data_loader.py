# app/processing/data_loader.py
import pandas as pd
from ..utils.file_security import is_safe_path
from ..config import Config
import logging

def load_data(file_path):
    try:
        if not is_safe_path(Config.ALLOWED_DIRECTORY, file_path):
            raise ValueError("Attempted to access a file outside the allowed directory.")
        
        data = pd.read_csv(file_path, encoding='shift-jis')
        data.columns = [col.strip() for col in data.columns]
        return data
    except UnicodeDecodeError:
        # Attempt with 'utf-8' encoding
        try:
            data = pd.read_csv(file_path, encoding='utf-8')
            data.columns = [col.strip() for col in data.columns]
            return data
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {e}")
            raise e
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise e
