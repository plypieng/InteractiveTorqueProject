# app/utils/file_security.py
import os

def is_safe_path(basedir, path):
    basedir = os.path.abspath(basedir)
    path = os.path.abspath(path)
    return os.path.commonpath([basedir, path]) == basedir
