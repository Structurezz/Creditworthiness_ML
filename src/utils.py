import os

def ensure_directory(path: str):
    """Ensure that a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
