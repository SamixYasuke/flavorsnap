import panel as pn

class ShortcutHelpPanel:
    def __init__(self):
        self.shortcuts = {
            "Ctrl+O": "Open file dialog",
            "Ctrl+V": "Paste from clipboard",
            "Enter": "Trigger classification",
            "Escape": "Clear current image",
            "Ctrl+S": "Export results",
            "Ctrl+H": "Toggle history panel",
            "Ctrl+D": "Toggle dark mode",
            "F11": "Toggle fullscreen"
        }
        
    def get_panel(self):
        markdown_str = "### ⌨️ Keyboard Shortcuts\n\n"
        markdown_str += "| Shortcut | Action |\n|---|---|\n"
        for key, desc in self.shortcuts.items():
            markdown_str += f"| `<kbd>{key}</kbd>` | {desc} |\n"
            
        return pn.pane.Markdown(
            markdown_str, 
            css_classes=['shortcut-panel'],
            margin=(10, 20)
        )
