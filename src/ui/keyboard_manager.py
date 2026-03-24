import panel as pn

class KeyboardManager:
    def __init__(self, target_callback):
        # We use a hidden TextInput as a bridge for JS events
        self.keyboard_event = pn.widgets.TextInput(
            name='Keyboard Bridge',
            value='',
            css_classes=['keyboard-target'],
            height=0,
            width=0,
            margin=0,
            sizing_mode='fixed'
        )
        self.target_callback = target_callback
        self.keyboard_event.param.watch(self._handle_event, 'value')
        
    def _handle_event(self, event):
        combo = event.new
        if combo:
            self.target_callback(combo)
            self.keyboard_event.value = "" # reset
            
    def get_widget(self):
        return self.keyboard_event
