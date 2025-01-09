# app/utils/helpers.py
import logging

def log_error(e, message="An error occurred"):
    logging.error(f"{message}: {e}", exc_info=True)
