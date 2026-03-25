import panel as pn
import uuid
import base64
import io
from PIL import Image

class ImageViewer(pn.reactive.ReactiveHTML):
    """
    Enhanced Image Viewer with Zoom, Pan, and Fullscreen support.
    Uses custom JS logic for smooth interaction.
    """
    
    object = pn.param.Parameter(default=None)
    viewer_id = pn.param.String(default=lambda: f"viewer-{uuid.uuid4().hex[:8]}")
    
    _template = """
    <div id="${viewer_id}" class="image-viewer-container">
        <div class="image-viewer-surface">
            <img src="${_image_src}" />
        </div>
        
        <div class="image-viewer-controls">
            <button class="viewer-btn" data-action="zoom-in" title="Zoom In">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
            </button>
            <button class="viewer-btn" data-action="zoom-out" title="Zoom Out">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
            </button>
            <button class="viewer-btn" data-action="reset" title="Reset View">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
            </button>
            <button class="viewer-btn" data-action="fullscreen" title="Toggle Fullscreen">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
            </button>
        </div>
        
        <div class="zoom-indicator">
            <span>100%</span>
        </div>
    </div>
    """

    _scripts = {
        'render': 'if (window.initImageViewer) window.initImageViewer(data.viewer_id);',
        'object': 'if (window.initImageViewer) window.initImageViewer(data.viewer_id);'
    }

    def __init__(self, **params):
        super().__init__(**params)
        # Ensure CSS/JS are loaded
        pn.extension(
            css_files=['static/css/image_viewer.css'],
            js_files={'image_viewer': 'static/js/image_viewer.js'}
        )

    @pn.depends('object')
    def _image_src(self):
        if self.object is None:
            return ""
        if isinstance(self.object, str):
            return self.object
        
        # Handle PIL Image
        if isinstance(self.object, Image.Image):
            buffered = io.BytesIO()
            self.object.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
        
        # Handle raw bytes
        if isinstance(self.object, bytes):
            img_str = base64.b64encode(self.object).decode()
            return f"data:image/jpeg;base64,{img_str}"
        
        return ""
