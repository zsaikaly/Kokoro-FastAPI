import gradio as gr
import os
import shutil
from . import api, files

def setup_event_handlers(components: dict):
    """Set up all event handlers for the UI components."""
    
    def refresh_status():
        is_available, voices = api.check_api_status()
        status = "Available" if is_available else "Unavailable"
        btn_text = f"🔄 TTS Service: {status}"
        
        if is_available and voices:
            return {
                components["model"]["status_btn"]: gr.update(value=btn_text),
                components["model"]["voice"]: gr.update(choices=voices, value=voices[0] if voices else None)
            }
        return {
            components["model"]["status_btn"]: gr.update(value=btn_text),
            components["model"]["voice"]: gr.update(choices=[], value=None)
        }
    
    def handle_file_select(filename):
        if filename:
            try:
                text = files.read_text_file(filename)
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    return gr.update(value=preview)
            except Exception as e:
                print(f"Error reading file: {e}")
        return gr.update(value="")
    
    def handle_file_upload(file):
        if file is None:
            return gr.update(choices=files.list_input_files())
            
        try:
            # Copy file to inputs directory
            filename = os.path.basename(file.name)
            target_path = os.path.join(files.INPUTS_DIR, filename)
            
            # Handle duplicate filenames
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(target_path):
                new_name = f"{base}_{counter}{ext}"
                target_path = os.path.join(files.INPUTS_DIR, new_name)
                counter += 1
                
            shutil.copy2(file.name, target_path)
            
        except Exception as e:
            print(f"Error uploading file: {e}")
            
        return gr.update(choices=files.list_input_files())
    
    def generate_speech(text, selected_file, voice, format, speed):
        is_available, _ = api.check_api_status()
        if not is_available:
            gr.Warning("TTS Service is currently unavailable")
            return {
                components["output"]["audio_output"]: None,
                components["output"]["output_files"]: gr.update(choices=files.list_output_files())
            }
        
        # Use text input if provided, otherwise use file content
        if text and text.strip():
            files.save_text(text)
            final_text = text
        elif selected_file:
            final_text = files.read_text_file(selected_file)
        else:
            gr.Warning("Please enter text or select a file")
            return {
                components["output"]["audio_output"]: None,
                components["output"]["output_files"]: gr.update(choices=files.list_output_files())
            }
        
        result = api.text_to_speech(final_text, voice, format, speed)
        if result is None:
            gr.Warning("Failed to generate speech. Please try again.")
            return {
                components["output"]["audio_output"]: None,
                components["output"]["output_files"]: gr.update(choices=files.list_output_files())
            }
        
        return {
            components["output"]["audio_output"]: result,
            components["output"]["output_files"]: gr.update(choices=files.list_output_files(), value=os.path.basename(result))
        }

    def play_selected(file_path):
        if file_path and os.path.exists(file_path):
            return gr.update(value=file_path, visible=True)
        return gr.update(visible=False)

    # Connect event handlers
    components["model"]["status_btn"].click(
        fn=refresh_status,
        outputs=[
            components["model"]["status_btn"],
            components["model"]["voice"]
        ]
    )
    
    components["input"]["file_select"].change(
        fn=handle_file_select,
        inputs=[components["input"]["file_select"]],
        outputs=[components["input"]["file_preview"]]
    )
    
    components["input"]["file_upload"].upload(
        fn=handle_file_upload,
        inputs=[components["input"]["file_upload"]],
        outputs=[components["input"]["file_select"]]
    )
    
    components["output"]["play_btn"].click(
        fn=play_selected,
        inputs=[components["output"]["output_files"]],
        outputs=[components["output"]["selected_audio"]]
    )
    
    components["model"]["submit"].click(
        fn=generate_speech,
        inputs=[
            components["input"]["text_input"],
            components["input"]["file_select"],
            components["model"]["voice"],
            components["model"]["format"],
            components["model"]["speed"]
        ],
        outputs=[
            components["output"]["audio_output"],
            components["output"]["output_files"]
        ]
    )
