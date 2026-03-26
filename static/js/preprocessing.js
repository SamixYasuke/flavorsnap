/**
 * FlavorSnap Real-time Image Preprocessing JavaScript Module
 * 
 * This module handles client-side image preprocessing with real-time preview
 * and communication with the backend for processing.
 */

class ImagePreprocessor {
    constructor() {
        this.originalImage = null;
        this.currentImage = null;
        this.canvas = null;
        this.ctx = null;
        this.isProcessing = false;
        this.debounceTimer = null;
        
        // Processing parameters
        this.params = {
            brightness: 1.0,
            contrast: 1.0,
            rotation: 0,
            cropBox: null,
            aspectRatio: null
        };
        
        this.init();
    }
    
    /**
     * Initialize the preprocessor
     */
    init() {
        this.setupCanvas();
        this.bindEvents();
        this.loadStyles();
    }
    
    /**
     * Setup canvas for image manipulation
     */
    setupCanvas() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.style.display = 'none';
        document.body.appendChild(this.canvas);
    }
    
    /**
     * Load CSS styles
     */
    loadStyles() {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = '/static/css/controls.css';
        document.head.appendChild(link);
    }
    
    /**
     * Bind event listeners
     */
    bindEvents() {
        // Listen for image upload
        const fileInput = document.querySelector('input[type="file"][accept*="image"]');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleImageUpload(e));
        }
        
        // Listen for control changes
        this.bindControlEvents();
        
        // Listen for window resize
        window.addEventListener('resize', () => this.debounceUpdate());
    }
    
    /**
     * Bind control event listeners
     */
    bindControlEvents() {
        // Brightness slider
        const brightnessSlider = document.getElementById('brightness-slider');
        if (brightnessSlider) {
            brightnessSlider.addEventListener('input', (e) => {
                this.params.brightness = parseFloat(e.target.value);
                this.updateSliderValue('brightness', e.target.value);
                this.debounceUpdate();
            });
        }
        
        // Contrast slider
        const contrastSlider = document.getElementById('contrast-slider');
        if (contrastSlider) {
            contrastSlider.addEventListener('input', (e) => {
                this.params.contrast = parseFloat(e.target.value);
                this.updateSliderValue('contrast', e.target.value);
                this.debounceUpdate();
            });
        }
        
        // Rotation slider
        const rotationSlider = document.getElementById('rotation-slider');
        if (rotationSlider) {
            rotationSlider.addEventListener('input', (e) => {
                this.params.rotation = parseFloat(e.target.value);
                this.updateSliderValue('rotation', e.target.value + '°');
                this.debounceUpdate();
            });
        }
        
        // Aspect ratio selector
        const aspectRatioSelect = document.getElementById('aspect-ratio-select');
        if (aspectRatioSelect) {
            aspectRatioSelect.addEventListener('change', (e) => {
                this.params.aspectRatio = e.target.value === 'Original' ? null : e.target.value;
                this.debounceUpdate();
            });
        }
        
        // Crop controls
        this.bindCropEvents();
        
        // Reset button
        const resetButton = document.getElementById('reset-button');
        if (resetButton) {
            resetButton.addEventListener('click', () => this.reset());
        }
        
        // Auto enhance button
        const autoEnhanceButton = document.getElementById('auto-enhance-button');
        if (autoEnhanceButton) {
            autoEnhanceButton.addEventListener('click', () => this.autoEnhance());
        }
    }
    
    /**
     * Bind crop control events
     */
    bindCropEvents() {
        const cropEnabled = document.getElementById('crop-enabled');
        const cropX = document.getElementById('crop-x');
        const cropY = document.getElementById('crop-y');
        const cropWidth = document.getElementById('crop-width');
        const cropHeight = document.getElementById('crop-height');
        
        if (cropEnabled) {
            cropEnabled.addEventListener('change', (e) => {
                if (!e.target.checked) {
                    this.params.cropBox = null;
                    this.debounceUpdate();
                }
            });
        }
        
        [cropX, cropY, cropWidth, cropHeight].forEach(control => {
            if (control) {
                control.addEventListener('input', () => {
                    if (cropEnabled && cropEnabled.checked) {
                        this.updateCropBox();
                        this.debounceUpdate();
                    }
                });
            }
        });
    }
    
    /**
     * Update crop box parameters
     */
    updateCropBox() {
        const cropX = document.getElementById('crop-x');
        const cropY = document.getElementById('crop-y');
        const cropWidth = document.getElementById('crop-width');
        const cropHeight = document.getElementById('crop-height');
        
        if (cropX && cropY && cropWidth && cropHeight) {
            this.params.cropBox = {
                x: parseInt(cropX.value),
                y: parseInt(cropY.value),
                width: parseInt(cropWidth.value),
                height: parseInt(cropHeight.value)
            };
        }
    }
    
    /**
     * Handle image upload
     */
    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file || !file.type.startsWith('image/')) {
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.originalImage = img;
                this.currentImage = img;
                this.reset();
                this.updatePreview();
                this.updateCropControls();
                this.showStatus('Image loaded successfully', 'success');
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    /**
     * Update crop control limits based on image size
     */
    updateCropControls() {
        if (!this.originalImage) return;
        
        const cropX = document.getElementById('crop-x');
        const cropY = document.getElementById('crop-y');
        const cropWidth = document.getElementById('crop-width');
        const cropHeight = document.getElementById('crop-height');
        
        if (cropX) {
            cropX.max = this.originalImage.width - 1;
        }
        if (cropY) {
            cropY.max = this.originalImage.height - 1;
        }
        if (cropWidth) {
            cropWidth.max = this.originalImage.width;
            cropWidth.value = this.originalImage.width;
        }
        if (cropHeight) {
            cropHeight.max = this.originalImage.height;
            cropHeight.value = this.originalImage.height;
        }
    }
    
    /**
     * Debounced update to prevent excessive processing
     */
    debounceUpdate() {
        clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(() => {
            this.updatePreview();
        }, 100);
    }
    
    /**
     * Update image preview with current parameters
     */
    async updatePreview() {
        if (!this.originalImage || this.isProcessing) return;
        
        this.isProcessing = true;
        this.showLoading(true);
        
        try {
            // Apply transformations
            const processedImage = await this.processImage();
            
            // Update preview
            const previewElement = document.getElementById('image-preview');
            if (previewElement) {
                previewElement.src = processedImage;
            }
            
            // Update parameters display
            this.updateParametersDisplay();
            
        } catch (error) {
            console.error('Error processing image:', error);
            this.showStatus('Error processing image', 'error');
        } finally {
            this.isProcessing = false;
            this.showLoading(false);
        }
    }
    
    /**
     * Process image with current parameters
     */
    async processImage() {
        return new Promise((resolve) => {
            // Calculate canvas size based on rotation
            const angle = this.params.rotation * Math.PI / 180;
            const sin = Math.abs(Math.sin(angle));
            const cos = Math.abs(Math.cos(angle));
            
            const width = this.originalImage.width;
            const height = this.originalImage.height;
            
            this.canvas.width = width * cos + height * sin;
            this.canvas.height = width * sin + height * cos;
            
            // Clear canvas
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Save context state
            this.ctx.save();
            
            // Apply transformations
            this.ctx.translate(this.canvas.width / 2, this.canvas.height / 2);
            this.ctx.rotate(angle);
            
            // Apply brightness and contrast
            this.ctx.filter = `brightness(${this.params.brightness}) contrast(${this.params.contrast})`;
            
            // Draw image
            this.ctx.drawImage(
                this.originalImage,
                -width / 2,
                -height / 2,
                width,
                height
            );
            
            // Restore context state
            this.ctx.restore();
            
            // Apply crop if needed
            if (this.params.cropBox) {
                const cropCanvas = document.createElement('canvas');
                const cropCtx = cropCanvas.getContext('2d');
                
                cropCanvas.width = this.params.cropBox.width;
                cropCanvas.height = this.params.cropBox.height;
                
                cropCtx.drawImage(
                    this.canvas,
                    this.params.cropBox.x,
                    this.params.cropBox.y,
                    this.params.cropBox.width,
                    this.params.cropBox.height,
                    0,
                    0,
                    this.params.cropBox.width,
                    this.params.cropBox.height
                );
                
                resolve(cropCanvas.toDataURL());
            } else {
                resolve(this.canvas.toDataURL());
            }
        });
    }
    
    /**
     * Reset all parameters to default
     */
    reset() {
        this.params = {
            brightness: 1.0,
            contrast: 1.0,
            rotation: 0,
            cropBox: null,
            aspectRatio: null
        };
        
        // Reset UI controls
        this.resetControls();
        this.showStatus('Reset to original image', 'success');
    }
    
    /**
     * Reset UI controls to default values
     */
    resetControls() {
        const brightnessSlider = document.getElementById('brightness-slider');
        const contrastSlider = document.getElementById('contrast-slider');
        const rotationSlider = document.getElementById('rotation-slider');
        const aspectRatioSelect = document.getElementById('aspect-ratio-select');
        const cropEnabled = document.getElementById('crop-enabled');
        
        if (brightnessSlider) {
            brightnessSlider.value = 1.0;
            this.updateSliderValue('brightness', '1.0');
        }
        if (contrastSlider) {
            contrastSlider.value = 1.0;
            this.updateSliderValue('contrast', '1.0');
        }
        if (rotationSlider) {
            rotationSlider.value = 0;
            this.updateSliderValue('rotation', '0°');
        }
        if (aspectRatioSelect) {
            aspectRatioSelect.value = 'Original';
        }
        if (cropEnabled) {
            cropEnabled.checked = false;
        }
    }
    
    /**
     * Auto enhance image
     */
    async autoEnhance() {
        if (!this.originalImage) return;
        
        this.showLoading(true);
        this.showStatus('Applying auto enhancement...', 'info');
        
        try {
            // Simulate auto-enhancement with backend call
            const response = await fetch('/api/auto-enhance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_data: this.canvas.toDataURL(),
                    params: this.params
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                
                // Update parameters with auto-enhanced values
                if (result.params) {
                    this.params = { ...this.params, ...result.params };
                    this.updateControlsFromParams();
                }
                
                this.showStatus('Auto enhancement applied', 'success');
            } else {
                throw new Error('Auto enhancement failed');
            }
        } catch (error) {
            console.error('Auto enhance error:', error);
            this.showStatus('Auto enhancement failed', 'error');
        } finally {
            this.showLoading(false);
        }
    }
    
    /**
     * Update controls from current parameters
     */
    updateControlsFromParams() {
        const brightnessSlider = document.getElementById('brightness-slider');
        const contrastSlider = document.getElementById('contrast-slider');
        const rotationSlider = document.getElementById('rotation-slider');
        
        if (brightnessSlider) {
            brightnessSlider.value = this.params.brightness;
            this.updateSliderValue('brightness', this.params.brightness.toFixed(1));
        }
        if (contrastSlider) {
            contrastSlider.value = this.params.contrast;
            this.updateSliderValue('contrast', this.params.contrast.toFixed(1));
        }
        if (rotationSlider) {
            rotationSlider.value = this.params.rotation;
            this.updateSliderValue('rotation', this.params.rotation + '°');
        }
    }
    
    /**
     * Update slider value display
     */
    updateSliderValue(sliderId, value) {
        const valueElement = document.getElementById(`${sliderId}-value`);
        if (valueElement) {
            valueElement.textContent = value;
        }
    }
    
    /**
     * Update parameters display
     */
    updateParametersDisplay() {
        const paramsElement = document.getElementById('parameters-display');
        if (!paramsElement || !this.originalImage) return;
        
        const params = this.getProcessingParams();
        const imageInfo = this.getImageInfo();
        
        paramsElement.innerHTML = `
            <h4>Current Parameters</h4>
            <ul class="parameters-list">
                <li><strong>Brightness:</strong> ${params.brightness.toFixed(1)}</li>
                <li><strong>Contrast:</strong> ${params.contrast.toFixed(1)}</li>
                <li><strong>Rotation:</strong> ${params.rotation}°</li>
                <li><strong>Aspect Ratio:</strong> ${params.aspectRatio || 'Original'}</li>
                <li><strong>Crop:</strong> ${params.cropBox ? `${params.cropBox.width}×${params.cropBox.height}` : 'None'}</li>
            </ul>
            <h4>Image Info</h4>
            <ul class="parameters-list">
                <li><strong>Size:</strong> ${imageInfo.width} × ${imageInfo.height}</li>
                <li><strong>Aspect Ratio:</strong> ${imageInfo.aspectRatio.toFixed(2)}</li>
            </ul>
        `;
    }
    
    /**
     * Get current processing parameters
     */
    getProcessingParams() {
        return { ...this.params };
    }
    
    /**
     * Get image information
     */
    getImageInfo() {
        if (!this.originalImage) {
            return { width: 0, height: 0, aspectRatio: 0 };
        }
        
        return {
            width: this.originalImage.width,
            height: this.originalImage.height,
            aspectRatio: this.originalImage.width / this.originalImage.height
        };
    }
    
    /**
     * Show loading state
     */
    showLoading(show) {
        const loadingElements = document.querySelectorAll('.loading-spinner');
        loadingElements.forEach(element => {
            element.style.display = show ? 'inline-block' : 'none';
        });
    }
    
    /**
     * Show status message
     */
    showStatus(message, type = 'info') {
        const statusElement = document.getElementById('status-message');
        if (statusElement) {
            statusElement.className = `${type}-message fade-in`;
            statusElement.textContent = message;
            
            // Auto-hide after 3 seconds
            setTimeout(() => {
                statusElement.className = '';
            }, 3000);
        }
    }
    
    /**
     * Get processed image data for classification
     */
    async getProcessedImageData() {
        if (!this.originalImage) return null;
        
        return await this.processImage();
    }
}

// Initialize the preprocessor when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.imagePreprocessor = new ImagePreprocessor();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ImagePreprocessor;
}
