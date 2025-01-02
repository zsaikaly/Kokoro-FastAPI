import os
from typing import List, Optional, Tuple
import datetime
from .config import INPUTS_DIR, OUTPUTS_DIR, AUDIO_FORMATS

def list_input_files() -> List[str]:
    """List all input text files."""
    return [f for f in os.listdir(INPUTS_DIR) if f.endswith('.txt')]

def list_output_files() -> List[str]:
    """List all output audio files."""
    return [os.path.join(OUTPUTS_DIR, f) 
            for f in os.listdir(OUTPUTS_DIR) 
            if any(f.endswith(ext) for ext in AUDIO_FORMATS)]

def read_text_file(filename: str) -> str:
    """Read content of a text file."""
    if not filename:
        return ""
    try:
        file_path = os.path.join(INPUTS_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""

def save_text(text: str, filename: Optional[str] = None) -> Optional[str]:
    """Save text to a file. Returns the filename if successful."""
    if not text.strip():
        return None
        
    if filename is None:
        # Use input_1.txt, input_2.txt, etc.
        base = "input"
        counter = 1
        while True:
            filename = f"{base}_{counter}.txt"
            if not os.path.exists(os.path.join(INPUTS_DIR, filename)):
                break
            counter += 1
    else:
        # Handle duplicate filenames by adding _1, _2, etc.
        base = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1] or '.txt'
        counter = 1
        while os.path.exists(os.path.join(INPUTS_DIR, filename)):
            filename = f"{base}_{counter}{ext}"
            counter += 1
        
    filepath = os.path.join(INPUTS_DIR, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        return filename
    except Exception as e:
        print(f"Error saving file: {e}")
        return None

def process_uploaded_file(file_path: str) -> bool:
    """Save uploaded file to inputs directory. Returns True if successful."""
    if not file_path:
        return False
        
    try:
        filename = os.path.basename(file_path)
        if not filename.endswith('.txt'):
            return False
            
        # Create target path in inputs directory
        target_path = os.path.join(INPUTS_DIR, filename)
        
        # If file exists, add number suffix
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(target_path):
            new_name = f"{base}_{counter}{ext}"
            target_path = os.path.join(INPUTS_DIR, new_name)
            counter += 1
            
        # Copy file to inputs directory
        import shutil
        shutil.copy2(file_path, target_path)
        return True
        
    except Exception as e:
        print(f"Error saving uploaded file: {e}")
        return False
