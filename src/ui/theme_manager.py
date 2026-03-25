import panel as pn
import os

class ThemeManager:
    """
    Manages the application's UI theme mode, bridging the Python backend 
    with the client-side CSS variables.
    """
    def __init__(self, default_theme='light'):
        self.current_theme = default_theme
        self._css_files = [
            'static/css/themes.css',
            'static/css/main.css'
        ]
        self._js_files = {
            'theme_toggle': 'static/js/theme-toggle.js'
        }

    def get_header_toggle_btn(self):
        """Returns a Panel widget representing the theme toggle button with JS logic."""
        theme_toggle = pn.widgets.Button(
            name='🌓 Dark Mode', 
            css_classes=['theme-toggle-btn'],
            width=150,
            button_type='default'
        )
        
        # JS logic for toggling and updating button label
        theme_toggle.js_on_click(
            args={'btn': theme_toggle},
            code="""
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            
            // Update button label
            btn.name = newTheme === 'dark' ? '☀️ Light Mode' : '🌓 Dark Mode';
            
            // Dispatch event for other components
            window.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme: newTheme } }));
            """
        )
        return theme_toggle

    def apply_to_app(self):
        """Configures general application theme settings."""
        pn.extension(
            css_files=['static/css/themes.css', 'static/css/main.css'],
            js_files={'theme_toggle_init': 'static/js/theme-toggle.js'}
        )

theme_manager = ThemeManager()
