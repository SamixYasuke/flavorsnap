document.addEventListener('keydown', function(e) {
    // Ignore keydown if user is typing in an input field (other than our hidden one)
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        if (!e.target.classList.contains('bk-input')) {
            // Panel inputs often have bokeh classes, but we can just check if it's our target
            if (!e.target.closest('.keyboard-target')) {
                 return;
            }
        }
    }

    let combo = [];
    if (e.ctrlKey) combo.push('ctrl');
    if (e.shiftKey) combo.push('shift');
    if (e.altKey) combo.push('alt');
    
    let key = e.key.toLowerCase();
    if (key === 'control' || key === 'shift' || key === 'alt') return;
    
    if (key === ' ') {
        combo.push('space');
    } else {
        combo.push(key);
    }
    
    const comboStr = combo.join('+');
    const knownShortcuts = ['ctrl+o', 'enter', 'escape', 'ctrl+s', 'ctrl+h', 'ctrl+d', 'f11'];
    
    if (knownShortcuts.includes(comboStr)) {
        e.preventDefault();
        
        if (comboStr === 'f11') {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                }
            }
            return;
        }
        
        if (comboStr === 'ctrl+o') {
            const fileInputs = document.querySelectorAll('input[type="file"]');
            if (fileInputs.length > 0) {
                fileInputs[0].click();
            }
            return;
        }

        // Send other shortcuts to Python explicitly via our hidden target
        const targetContainers = document.querySelectorAll('.keyboard-target');
        if (targetContainers.length > 0) {
            const input = targetContainers[0].querySelector('input');
            if (input) {
                input.value = comboStr;
                // dispatch a change event so that Panel is notified
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    }
});

// Handle Ctrl+V (Paste) from anywhere
document.addEventListener('paste', function(e) {
    if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length > 0) {
        const fileInputs = document.querySelectorAll('input[type="file"]');
        if (fileInputs.length > 0) {
            fileInputs[0].files = e.clipboardData.files;
            fileInputs[0].dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
});
