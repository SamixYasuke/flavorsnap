import panel as pn
from src.ui.shortcut_help import ShortcutHelpPanel

class MainInterface:
    def __init__(self, classify_fn, save_image_fn):
        self.classify_fn = classify_fn
        self.save_image_fn = save_image_fn
        
        self.image_input = pn.widgets.FileInput(accept='image/*')
        self.output = pn.pane.Markdown("Upload an image of food 🍲")
        self.image_preview = pn.pane.Image(width=300, height=300, visible=False)
        self.spinner = pn.indicators.LoadingSpinner(value=False, width=50)
        
        # Buttons
        self.run_button = pn.widgets.Button(name='Classify', button_type='primary')
        self.run_button.on_click(self.classify_fn)
        
        self.help_button = pn.widgets.Button(name='Help', button_type='light', width=80)
        self.help_button.on_click(lambda event: self.toggle_help())
        
        self.history_button = pn.widgets.Button(name='History', button_type='light', width=80)
        self.history_button.on_click(lambda event: self.toggle_history())
        
        # Panels
        self.history_panel = pn.Column(
            "### 📜 History",
            pn.pane.Markdown("No history yet."),
            visible=False,
            width=300,
            margin=(0, 20)
        )
        self.help_panel = pn.Column(
            ShortcutHelpPanel().get_panel(),
            visible=False,
            width=300,
            margin=(0, 20)
        )
        
        # Main layout
        self.main_content = pn.Column(
            self.image_input,
            pn.Row(self.run_button, self.help_button, self.history_button),
            self.spinner,
            self.image_preview,
            self.output,
            width=500
        )
        
        self.layout = pn.Column(
            "# 🍽️ FlavorSnap Dashboard",
            "Upload an image and click the button or press **Enter** to classify your food!",
            pn.Row(self.main_content, self.history_panel, self.help_panel)
        )
        
    def toggle_history(self):
        self.history_panel.visible = not self.history_panel.visible
        
    def toggle_help(self):
        self.help_panel.visible = not self.help_panel.visible
        
    def clear_image(self):
        # Reset visual components
        self.image_preview.visible = False
        self.output.object = "Upload an image of food 🍲"
        if hasattr(self.image_input, 'value'):
            self.image_input.value = None
            
    def trigger_export(self):
        # We only export if there is an image to export and it has been classified
        if self.image_preview.visible:
            self.save_image_fn()
            
    def get_layout(self):
        return self.layout
