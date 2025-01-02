import gradio as gr
from typing import Tuple, Optional
from .. import api, config

def create_model_column(voice_ids: Optional[list] = None) -> Tuple[gr.Column, dict]:
    """Create the model settings column."""
    if voice_ids is None:
        voice_ids = []
        
    with gr.Column(scale=1) as col:
        gr.Markdown("### Model Settings")
        
        # Status button with embedded status
        is_available, _ = api.check_api_status()
        status_btn = gr.Button(
            f"Checking TTS Service: {'Available' if is_available else 'Not Yet Available'}",
            variant="secondary"
        )
        
        voice_input = gr.Dropdown(
            choices=voice_ids,
            label="Voice",
            value=voice_ids[0] if voice_ids else None,
            interactive=True
        )
        format_input = gr.Dropdown(
            choices=config.AUDIO_FORMATS,
            label="Audio Format",
            value="mp3"
        )
        speed_input = gr.Slider(
            minimum=0.5,
            maximum=2.0,
            value=1.0,
            step=0.1,
            label="Speed"
        )
        
        submit_btn = gr.Button(
            "Generate Speech",
            variant="primary",
            size="lg"
        )
    
    components = {
        "status_btn": status_btn,
        "voice": voice_input,
        "format": format_input,
        "speed": speed_input,
        "submit": submit_btn
    }
    
    return col, components
