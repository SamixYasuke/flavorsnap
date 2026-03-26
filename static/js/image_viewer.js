/**
 * Image Viewer Logic - Interactive Zoom & Pan
 * Optimized for desktop and mobile devices.
 */

class InteractiveViewer {
    constructor(element) {
        this.container = element;
        this.viewSurface = this.container.querySelector('.image-viewer-surface');
        this.img = this.viewSurface.querySelector('img');
        
        // State
        this.scale = 1;
        this.panning = false;
        this.pointX = 0;
        this.pointY = 0;
        this.start = { x: 0, y: 0 };
        this.zoomIndicator = this.container.querySelector('.zoom-indicator span');
        
        // Settings
        this.minScale = 0.5;
        this.maxScale = 8;
        this.step = 0.1;
        
        this.init();
    }

    setTransform() {
        this.viewSurface.style.transform = `translate(${this.pointX}px, ${this.pointY}px) scale(${this.scale})`;
        if (this.zoomIndicator) {
            this.zoomIndicator.textContent = `${Math.round(this.scale * 100)}%`;
        }
    }

    reset() {
        this.scale = 1;
        this.pointX = 0;
        this.pointY = 0;
        this.setTransform();
    }

    zoom(delta, targetX, targetY) {
        // Position relative to viewport
        const rect = this.container.getBoundingClientRect();
        targetX = targetX || rect.width / 2;
        targetY = targetY || rect.height / 2;
        
        // Position relative to scaled surface
        const xs = (targetX - this.pointX) / this.scale;
        const ys = (targetY - this.pointY) / this.scale;
        
        const nextScale = this.scale + delta;
        if (nextScale < this.minScale || nextScale > this.maxScale) return;
        
        this.scale = nextScale;
        this.pointX = targetX - xs * this.scale;
        this.pointY = targetY - ys * this.scale;
        
        this.setTransform();
    }

    init() {
        // Mouse Down / Start Panning
        this.container.onmousedown = (e) => {
            if (e.button !== 0) return; // Only left click
            e.preventDefault();
            this.start = { x: e.clientX - this.pointX, y: e.clientY - this.pointY };
            this.panning = true;
        };

        // Global Mouse Up / Stop Panning
        window.onmouseup = () => {
            this.panning = false;
        };

        // Mouse Move / Update Panning
        this.container.onmousemove = (e) => {
            if (!this.panning) return;
            e.preventDefault();
            this.pointX = e.clientX - this.start.x;
            this.pointY = e.clientY - this.start.y;
            this.setTransform();
        };

        // Scroll Wheel / Zoom
        this.container.onwheel = (e) => {
            e.preventDefault();
            const delta = -e.deltaY / 1000 * this.scale * 2;
            this.zoom(delta, e.offsetX, e.offsetY);
        };

        // Double Click / Quick Zoom
        this.container.ondblclick = (e) => {
            const delta = this.scale === 1 ? 2 : (1 - this.scale);
            this.zoom(delta, e.offsetX, e.offsetY);
        };

        // Control Buttons
        const btns = this.container.querySelectorAll('.viewer-btn');
        btns.forEach(btn => {
            btn.onclick = (e) => {
                const action = btn.dataset.action;
                if (action === 'zoom-in') this.zoom(0.5);
                if (action === 'zoom-out') this.zoom(-0.5);
                if (action === 'reset') this.reset();
                if (action === 'fullscreen') {
                    if (!document.fullscreenElement) {
                        this.container.requestFullscreen();
                    } else {
                        document.exitFullscreen();
                    }
                }
            };
        });

        // Touch Support
        let lastTouchDist = 0;
        this.container.ontouchstart = (e) => {
            if (e.touches.length === 1) {
                const touch = e.touches[0];
                this.start = { x: touch.clientX - this.pointX, y: touch.clientY - this.pointY };
                this.panning = true;
            } else if (e.touches.length === 2) {
                lastTouchDist = Math.hypot(
                    e.touches[0].clientX - e.touches[1].clientX,
                    e.touches[0].clientY - e.touches[1].clientY
                );
            }
        };

        this.container.ontouchmove = (e) => {
            if (e.touches.length === 1 && this.panning) {
                const touch = e.touches[0];
                this.pointX = touch.clientX - this.start.x;
                this.pointY = touch.clientY - this.start.y;
                this.setTransform();
            } else if (e.touches.length === 2) {
                const dist = Math.hypot(
                    e.touches[0].clientX - e.touches[1].clientX,
                    e.touches[0].clientY - e.touches[1].clientY
                );
                const delta = (dist - lastTouchDist) / 100;
                this.zoom(delta);
                lastTouchDist = dist;
            }
        };

        this.container.ontouchend = () => {
            this.panning = false;
        };
    }
}

// Global initialization function for Panel integration
window.initImageViewer = (elementId) => {
    const el = document.getElementById(elementId);
    if (el) new InteractiveViewer(el);
};
