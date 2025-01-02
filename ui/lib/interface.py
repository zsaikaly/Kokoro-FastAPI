import gradio as gr
from . import api
from .components import create_input_column, create_model_column, create_output_column
from .handlers import setup_event_handlers

def create_interface():
    """Create the main Gradio interface."""
    # Initial status check
    is_available, available_voices = api.check_api_status()

    with gr.Blocks(
    title="Kokoro TTS Demo",
    theme=gr.themes.Monochrome()
) as demo:
        gr.HTML(value='<div style="display: flex; gap: 0;">'
                '<a href="https://huggingface.co/hexgrad/Kokoro-82M" target="_blank" style="color: #2196F3; text-decoration: none; margin: 2px; border: 1px solid #2196F3; padding: 4px 8px; height: 24px; box-sizing: border-box; display: inline-flex; align-items: center;">Kokoro-82M HF Repo</a>'
                '<a href="https://github.com/remsky/Kokoro-FastAPI" target="_blank" style="color: #2196F3; text-decoration: none; margin: 2px; border: 1px solid #2196F3; padding: 4px 8px; height: 24px; box-sizing: border-box; display: inline-flex; align-items: center;">Kokoro-FastAPI Repo</a>'
                '</div>', show_label=False)
    
        # Main interface
        with gr.Row():
            # Create columns
            input_col, input_components = create_input_column()
            model_col, model_components = create_model_column(available_voices)  # Pass initial voices
            output_col, output_components = create_output_column()
            
            # Collect all components
            components = {
                "input": input_components,
                "model": model_components,
                "output": output_components
            }
            
            # Set up event handlers
            setup_event_handlers(components)
            
        # Add periodic status check with Timer
        def update_status():
            is_available, voices = api.check_api_status()
            status = "Available" if is_available else "Unavailable"
            return {
                components["model"]["status_btn"]: gr.update(value=f"🔄 TTS Service: {status}"),
                components["model"]["voice"]: gr.update(choices=voices, value=voices[0] if voices else None)
            }
            
        timer = gr.Timer(10, active=True)  # Check every 10 seconds
        timer.tick(
            fn=update_status,
            outputs=[
                components["model"]["status_btn"],
                components["model"]["voice"]
            ]
        )

    return demo
