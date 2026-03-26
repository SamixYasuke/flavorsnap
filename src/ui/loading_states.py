import panel as pn
import uuid

class LoadingUI(pn.reactive.ReactiveHTML):
    """
    Enhanced Loading UI with Progress Bar, Stage Messages, 
    and Time Remaining estimations.
    """
    
    progress = pn.param.Integer(default=0, bounds=(0, 100))
    message = pn.param.String(default="Preparing classification...")
    visible = pn.param.Boolean(default=False)
    estimation = pn.param.String(default="Estimated: 3s")
    
    _template = """
    <div id="loading-overlay" class="processing-overlay ${visible_class}">
        <div class="loader-content">
            <h3 style="margin-bottom: 2rem;">🧠 Analyzing Your Dish</h3>
            
            <div class="progress-container">
                <div id="pbar" class="progress-bar" style="width: ${progress}%"></div>
            </div>
            
            <div id="msg" class="loading-message">
                <div class="bk-root bk-indicator-spinner bk-spinner-border bk-spinner-xs"></div>
                ${message}
            </div>
            
            <div id="est" class="time-estimation">
                ${estimation}
            </div>
            
            <button id="cancel-btn" class="viewer-btn cancel-btn" onclick="${_on_cancel}">
                Cancel Analysis
            </button>
        </div>
    </div>
    """

    @pn.depends('visible')
    def visible_class(self):
        return "visible" if self.visible else ""

    def _on_cancel(self, event):
        self.visible = False
        self.message = "Analysis canceled."

    _scripts = {
        'render': "if (window.initProgressTracker) window.initProgressTracker('loading-overlay', 'pbar', 'msg', 'est');",
        'progress': "if (window.progressTracker) window.progressTracker.update(data.progress, data.message);",
        'message': "if (window.progressTracker) window.progressTracker.update(data.progress, data.message);"
    }

    def __init__(self, **params):
        super().__init__(**params)
        pn.extension(
            css_files=['static/css/loading.css'],
            js_files={'progress_tracker': 'static/js/progress_tracker.js'}
        )

class SkeletonCard(pn.reactive.ReactiveHTML):
    """Skeleton component for placeholder during image loading."""
    _template = '<div class="skeleton-box skeleton-card"></div>'
    
    def __init__(self, **params):
        super().__init__(**params)
        pn.extension(css_files=['static/css/loading.css'])
