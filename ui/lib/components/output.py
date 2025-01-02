import gradio as gr
from typing import Tuple
from .. import files

def create_output_column() -> Tuple[gr.Column, dict]:
    """Create the output column with audio player and file list."""
    with gr.Column(scale=1) as col:
        gr.Markdown("### Latest Output")
        audio_output = gr.Audio(
            label="Generated Speech",
            type="filepath"
        )
        
        gr.Markdown("### Generated Files")
        output_files = gr.Dropdown(
            label="Previous Outputs",
            choices=files.list_output_files(),
            value=None,
            allow_custom_value=False
        )
        
        play_btn = gr.Button("▶️ Play Selected", size="sm")
        
        selected_audio = gr.Audio(
            label="Selected Output",
            type="filepath",
            visible=False
        )
    
    components = {
        "audio_output": audio_output,
        "output_files": output_files,
        "play_btn": play_btn,
        "selected_audio": selected_audio
    }
    
    return col, components
