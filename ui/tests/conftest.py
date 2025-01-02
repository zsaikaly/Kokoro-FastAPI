import pytest
import gradio as gr


@pytest.fixture
def mock_gr_context():
    """Provides a context for testing Gradio components"""
    with gr.Blocks():
        yield
