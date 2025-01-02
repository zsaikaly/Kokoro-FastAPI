import gradio as gr
from typing import Tuple
from .. import files

def create_input_column() -> Tuple[gr.Column, dict]:
    """Create the input column with text input and file handling."""
    with gr.Column(scale=1) as col:
        with gr.Tabs() as tabs:
            # Set first tab as selected by default
            tabs.selected = 0
            # Direct Input Tab
            with gr.TabItem("Direct Input"):
                text_input = gr.Textbox(
                    label="Text to speak",
                    placeholder="Enter text here...",
                    lines=4
                )
                text_submit = gr.Button(
                    "Generate Speech",
                    variant="primary",
                    size="lg"
                )
            
            # File Input Tab
            with gr.TabItem("From File"):
                # Existing files dropdown
                input_files_list = gr.Dropdown(
                    label="Select Existing File",
                    choices=files.list_input_files(),
                    value=None
                )
                
                # Simple file upload
                file_upload = gr.File(
                    label="Upload Text File (.txt)",
                    file_types=[".txt"]
                )
                
                file_preview = gr.Textbox(
                    label="File Content Preview",
                    interactive=False,
                    lines=4
                )
                
                with gr.Row():
                    file_submit = gr.Button(
                        "Generate Speech",
                        variant="primary",
                        size="lg"
                    )
                    clear_files = gr.Button(
                        "Clear Files",
                        variant="secondary",
                        size="lg"
                    )
    
    components = {
        "tabs": tabs,
        "text_input": text_input,
        "file_select": input_files_list,
        "file_upload": file_upload,
        "file_preview": file_preview,
        "text_submit": text_submit,
        "file_submit": file_submit,
        "clear_files": clear_files
    }
    
    return col, components
