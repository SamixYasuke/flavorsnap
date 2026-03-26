/**
 * Progress Tracking Logic - Estimates and UI Updates
 */

class ProgressTracker {
    constructor(overlayId, progressBarId, messageId, timeId) {
        this.overlay = document.getElementById(overlayId);
        this.progressBar = document.getElementById(progressBarId);
        this.message = document.getElementById(messageId);
        this.timeEstimation = document.getElementById(timeId);
        
        // Progress State
        this.currentProgress = 0;
        this.startTime = 0;
        this.estimatedTime = 3000; // Estimated 3s total for resnet18 inference
        this.interval = null;
    }

    start() {
        this.currentProgress = 0;
        this.startTime = Date.now();
        this.overlay.classList.add('visible');
        this.update(0, "Initiating food analysis...");
        
        // Smooth interpolation for the first 80%
        this.interval = setInterval(() => {
            if (this.currentProgress < 85) {
                const elapsed = Date.now() - this.startTime;
                const target = Math.min(85, Math.floor((elapsed / this.estimatedTime) * 100));
                
                if (target > this.currentProgress) {
                    this.update(target);
                }
            }
        }, 100);
    }

    update(percent, message) {
        if (percent !== undefined) {
          this.currentProgress = percent;
          this.progressBar.style.width = `${percent}%`;
        }
        
        if (message) {
            this.message.textContent = message;
        }

        // Calculate time remaining
        const elapsed = Date.now() - this.startTime;
        if (this.currentProgress > 5 && this.currentProgress < 100) {
            const totalEstimated = (elapsed / this.currentProgress) * 100;
            const remaining = Math.max(0, Math.round((totalEstimated - elapsed) / 1000));
            this.timeEstimation.textContent = `Estimated time remaining: ${remaining}s`;
        } else if (this.currentProgress >= 100) {
            this.timeEstimation.textContent = "Processing complete!";
        }
    }

    finish(finalMessage) {
        clearInterval(this.interval);
        this.update(100, finalMessage || "Analysis complete!");
        
        setTimeout(() => {
            this.overlay.classList.remove('visible');
            // Small delay before actual dismissal
        }, 1000);
    }

    cancel() {
        clearInterval(this.interval);
        this.overlay.classList.remove('visible');
        this.update(0, "Classification canceled.");
    }
}

// Global initialization helper
window.initProgressTracker = (o, p, m, t) => {
    window.progressTracker = new ProgressTracker(o, p, m, t);
};
