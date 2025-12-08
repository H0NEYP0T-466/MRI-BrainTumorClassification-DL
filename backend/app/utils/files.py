"""File utilities for path management and file handling."""
import os
from pathlib import Path
from typing import List

def ensure_dir(directory: Path) -> None:
    """Create directory if it doesn't exist."""
    directory.mkdir(parents=True, exist_ok=True)

def get_image_files(directory: Path, extensions: List[str] = None) -> List[Path]:
    """Get all image files from a directory."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    if not directory.exists():
        return []
    
    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f'*{ext}'))
        image_files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(image_files)

def safe_filename(filename: str) -> str:
    """Make a filename safe for filesystem."""
    # Remove or replace unsafe characters
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in '._-':
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    return ''.join(safe_chars)
