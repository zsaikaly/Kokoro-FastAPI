import gradio as gr
import pytest


@pytest.fixture
def mock_gr_context():
    """Provides a context for testing Gradio components"""
    with gr.Blocks():
        yield
