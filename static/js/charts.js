/**
 * FlavorSnap Interactive Chart Features
 * 
 * This JavaScript file provides interactive features for the confidence charts,
 * including tooltips, animations, responsive behavior, and accessibility enhancements.
 */

// Global chart management
class FlavorSnapChartManager {
    constructor() {
        this.charts = new Map();
        this.tooltips = new Map();
        this.observers = new Map();
        this.transitionManager = new ChartTransitionManager();
        this.accessibilityManager = new ChartAccessibilityManager();
        this.responsiveManager = new ChartResponsiveManager();
        
        this.init();
    }

    init() {
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupCharts());
        } else {
            this.setupCharts();
        }

        // Setup global event listeners
        this.setupEventListeners();
        
        // Setup responsive behavior
        this.responsiveManager.init();
        
        console.log('FlavorSnap Chart Manager initialized');
    }

    setupCharts() {
        // Find all chart containers
        const chartContainers = document.querySelectorAll('.confidence-chart-container');
        
        chartContainers.forEach((container, index) => {
            const chartId = `chart-${index}`;
            container.setAttribute('data-chart-id', chartId);
            
            // Initialize chart
            const chart = new ConfidenceChart(container, chartId);
            this.charts.set(chartId, chart);
            
            // Setup tooltips
            this.setupTooltips(container, chartId);
            
            // Setup animations
            this.setupAnimations(container, chartId);
            
            // Setup accessibility
            this.accessibilityManager.setup(container, chartId);
            
            // Setup responsive behavior
            this.responsiveManager.setupChart(container, chartId);
        });
    }

    setupTooltips(container, chartId) {
        const tooltip = new ChartTooltip(container);
        this.tooltips.set(chartId, tooltip);
        
        // Add hover effects to chart bars
        const bars = container.querySelectorAll('.chart-bar');
        bars.forEach(bar => {
            bar.addEventListener('mouseenter', (e) => this.handleBarHover(e, chartId));
            bar.addEventListener('mouseleave', (e) => this.handleBarLeave(e, chartId));
            bar.addEventListener('focus', (e) => this.handleBarFocus(e, chartId));
            bar.addEventListener('blur', (e) => this.handleBarBlur(e, chartId));
        });
    }

    setupAnimations(container, chartId) {
        // Setup intersection observer for animations
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.animateChart(entry.target, chartId);
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '50px'
        });

        observer.observe(container);
        this.observers.set(chartId, observer);
    }

    setupEventListeners() {
        // Handle window resize
        window.addEventListener('resize', this.debounce(() => {
            this.handleResize();
        }, 250));

        // Handle theme changes
        if (window.matchMedia) {
            const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
            darkModeQuery.addListener(() => this.handleThemeChange());
        }

        // Handle print events
        window.addEventListener('beforeprint', () => this.handlePrint());
        window.addEventListener('afterprint', () => this.handlePrintEnd());
    }

    handleBarHover(event, chartId) {
        const bar = event.target;
        const className = bar.getAttribute('data-class');
        const confidence = bar.getAttribute('data-confidence');
        
        // Show tooltip
        const tooltip = this.tooltips.get(chartId);
        if (tooltip) {
            tooltip.show(event, className, confidence);
        }

        // Add visual feedback
        bar.style.transform = 'scaleX(1.02)';
        bar.style.filter = 'brightness(1.1)';
        bar.style.cursor = 'pointer';

        // Update ARIA attributes
        bar.setAttribute('aria-expanded', 'true');
    }

    handleBarLeave(event, chartId) {
        const bar = event.target;
        
        // Hide tooltip
        const tooltip = this.tooltips.get(chartId);
        if (tooltip) {
            tooltip.hide();
        }

        // Remove visual feedback
        bar.style.transform = 'scaleX(1)';
        bar.style.filter = 'brightness(1)';

        // Update ARIA attributes
        bar.setAttribute('aria-expanded', 'false');
    }

    handleBarFocus(event, chartId) {
        // Similar to hover but for keyboard navigation
        this.handleBarHover(event, chartId);
    }

    handleBarBlur(event, chartId) {
        // Similar to hover leave but for keyboard navigation
        this.handleBarLeave(event, chartId);
    }

    animateChart(container, chartId) {
        const bars = container.querySelectorAll('.chart-bar');
        const chart = this.charts.get(chartId);
        
        if (!chart) return;

        bars.forEach((bar, index) => {
            const targetWidth = parseFloat(bar.style.width) || 0;
            const delay = index * 100; // Staggered animation
            
            setTimeout(() => {
                this.transitionManager.animateBarWidth(bar, targetWidth, 800);
            }, delay);
        });
    }

    handleResize() {
        // Update all charts for responsive behavior
        this.charts.forEach((chart, chartId) => {
            this.responsiveManager.updateChart(chartId);
        });
    }

    handleThemeChange() {
        // Update chart colors for theme
        this.charts.forEach((chart, chartId) => {
            chart.updateTheme();
        });
    }

    handlePrint() {
        // Optimize charts for printing
        document.querySelectorAll('.confidence-chart-container').forEach(container => {
            container.classList.add('print-mode');
        });
    }

    handlePrintEnd() {
        // Remove print optimizations
        document.querySelectorAll('.confidence-chart-container').forEach(container => {
            container.classList.remove('print-mode');
        });
    }

    // Utility function for debouncing
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Public API
    updateChart(chartId, probabilities, predictedClass) {
        const chart = this.charts.get(chartId);
        if (chart) {
            chart.updateData(probabilities, predictedClass);
        }
    }

    resetChart(chartId) {
        const chart = this.charts.get(chartId);
        if (chart) {
            chart.reset();
        }
    }

    exportChart(chartId, format = 'json') {
        const chart = this.charts.get(chartId);
        if (chart) {
            return chart.exportData(format);
        }
        return null;
    }
}

// Individual Chart Class
class ConfidenceChart {
    constructor(container, chartId) {
        this.container = container;
        this.chartId = chartId;
        this.data = null;
        this.isAnimated = false;
    }

    updateData(probabilities, predictedClass) {
        this.data = {
            probabilities,
            predictedClass
        };

        // Update DOM
        this.updateChartDisplay();
        
        // Trigger animations
        if (!this.isAnimated) {
            this.triggerAnimations();
            this.isAnimated = true;
        }
    }

    updateChartDisplay() {
        if (!this.data) return;

        const { probabilities, predictedClass } = this.data;
        
        // Update chart bars
        const bars = this.container.querySelectorAll('.chart-bar');
        bars.forEach(bar => {
            const className = bar.getAttribute('data-class');
            if (className && probabilities[className] !== undefined) {
                const percentage = probabilities[className] * 100;
                bar.style.width = `${percentage}%`;
                bar.setAttribute('data-confidence', `${percentage.toFixed(2)}%`);
                
                // Update color based on confidence
                const color = this.getConfidenceColor(probabilities[className]);
                bar.style.backgroundColor = color;
            }
        });

        // Update top prediction highlight
        this.updateTopPrediction(predictedClass);
        
        // Update footer
        this.updateFooter(predictedClass, probabilities[predictedClass]);
    }

    updateTopPrediction(predictedClass) {
        // Remove existing highlights
        this.container.querySelectorAll('.chart-bar-container').forEach(container => {
            container.classList.remove('top-prediction');
        });

        // Add highlight to top prediction
        const bars = this.container.querySelectorAll('.chart-bar');
        bars.forEach(bar => {
            if (bar.getAttribute('data-class') === predictedClass) {
                bar.closest('.chart-bar-container').classList.add('top-prediction');
            }
        });
    }

    updateFooter(predictedClass, confidence) {
        const footer = this.container.querySelector('.top-prediction-text');
        if (footer) {
            const percentage = (confidence * 100).toFixed(1);
            footer.innerHTML = `🏆 Top Prediction: <strong>${predictedClass}</strong> (${percentage}% confidence)`;
        }
    }

    getConfidenceColor(confidence) {
        if (confidence > 0.8) {
            return '#22c55e'; // Green
        } else if (confidence >= 0.6) {
            return '#eab308'; // Yellow
        } else {
            return '#ef4444'; // Red
        }
    }

    triggerAnimations() {
        const bars = this.container.querySelectorAll('.chart-bar');
        bars.forEach((bar, index) => {
            setTimeout(() => {
                bar.classList.add('fade-in-up');
            }, index * 100);
        });
    }

    updateTheme() {
        // Update colors based on current theme
        const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        // Theme-specific updates would go here
    }

    reset() {
        this.data = null;
        this.isAnimated = false;
        
        // Reset DOM
        const bars = this.container.querySelectorAll('.chart-bar');
        bars.forEach(bar => {
            bar.style.width = '0%';
            bar.classList.remove('fade-in-up');
        });
        
        // Reset highlights
        this.container.querySelectorAll('.chart-bar-container').forEach(container => {
            container.classList.remove('top-prediction');
        });
    }

    exportData(format = 'json') {
        if (!this.data) return null;

        const { probabilities, predictedClass } = this.data;
        
        if (format === 'json') {
            return JSON.stringify({
                predicted_class: predictedClass,
                probabilities: probabilities,
                timestamp: new Date().toISOString()
            }, null, 2);
        } else if (format === 'csv') {
            let csv = 'Class,Probability\n';
            Object.entries(probabilities).forEach(([className, prob]) => {
                csv += `${className},${prob.toFixed(4)}\n`;
            });
            return csv;
        }
        
        return null;
    }
}

// Tooltip Manager
class ChartTooltip {
    constructor(container) {
        this.container = container;
        this.tooltip = null;
        this.createTooltip();
    }

    createTooltip() {
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'chart-tooltip';
        this.tooltip.setAttribute('role', 'tooltip');
        document.body.appendChild(this.tooltip);
    }

    show(event, className, confidence) {
        if (!this.tooltip) return;

        const confidenceFloat = parseFloat(confidence);
        const level = this.getConfidenceLevel(confidenceFloat);
        
        this.tooltip.innerHTML = `
            <div><strong>${className}</strong></div>
            <div>Confidence: ${confidenceFloat.toFixed(1)}%</div>
            <div>Level: ${level}</div>
        `;

        // Position tooltip
        const rect = event.target.getBoundingClientRect();
        const tooltipRect = this.tooltip.getBoundingClientRect();
        
        let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
        let top = rect.top - tooltipRect.height - 10;
        
        // Adjust if tooltip goes outside viewport
        if (left < 10) left = 10;
        if (left + tooltipRect.width > window.innerWidth - 10) {
            left = window.innerWidth - tooltipRect.width - 10;
        }
        if (top < 10) {
            top = rect.bottom + 10;
        }
        
        this.tooltip.style.left = `${left}px`;
        this.tooltip.style.top = `${top}px`;
        this.tooltip.classList.add('show');
    }

    hide() {
        if (this.tooltip) {
            this.tooltip.classList.remove('show');
        }
    }

    getConfidenceLevel(confidence) {
        if (confidence > 80) return 'High';
        if (confidence >= 60) return 'Medium';
        return 'Low';
    }
}

// Transition Manager
class ChartTransitionManager {
    constructor() {
        this.activeTransitions = new Map();
    }

    animateBarWidth(element, targetWidth, duration = 800) {
        if (this.activeTransitions.has(element)) {
            cancelAnimationFrame(this.activeTransitions.get(element));
        }

        const startWidth = parseFloat(element.style.width) || 0;
        const startTime = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function (ease-in-out)
            const easeProgress = progress < 0.5 
                ? 2 * progress * progress 
                : 1 - Math.pow(-2 * progress + 2, 2) / 2;
            
            const currentWidth = startWidth + (targetWidth - startWidth) * easeProgress;
            element.style.width = `${currentWidth}%`;
            
            if (progress < 1) {
                const animationId = requestAnimationFrame(animate);
                this.activeTransitions.set(element, animationId);
            } else {
                this.activeTransitions.delete(element);
            }
        };

        requestAnimationFrame(animate);
    }

    animateColorChange(element, targetColor, duration = 800) {
        element.style.transition = `background-color ${duration}ms ease-in-out`;
        element.style.backgroundColor = targetColor;
        
        setTimeout(() => {
            element.style.transition = '';
        }, duration);
    }
}

// Accessibility Manager
class ChartAccessibilityManager {
    constructor() {
        this.setupKeyboardNavigation();
    }

    setup(container, chartId) {
        // Add ARIA labels
        this.addAriaLabels(container);
        
        // Setup keyboard navigation
        this.setupKeyboardNavigation(container);
        
        // Setup screen reader announcements
        this.setupScreenReaderAnnouncements(container, chartId);
    }

    addAriaLabels(container) {
        const chartContainer = container.querySelector('.confidence-chart-container');
        if (chartContainer) {
            chartContainer.setAttribute('role', 'application');
            chartContainer.setAttribute('aria-label', 'Confidence scores chart for food classification');
        }

        const bars = container.querySelectorAll('.chart-bar');
        bars.forEach((bar, index) => {
            const className = bar.getAttribute('data-class');
            const confidence = bar.getAttribute('data-confidence');
            bar.setAttribute('role', 'button');
            bar.setAttribute('tabindex', '0');
            bar.setAttribute('aria-label', `${className}: ${confidence} confidence`);
            bar.setAttribute('aria-describedby', `chart-tooltip-${index}`);
        });
    }

    setupKeyboardNavigation(container) {
        const bars = container.querySelectorAll('.chart-bar');
        
        bars.forEach((bar, index) => {
            bar.addEventListener('keydown', (e) => {
                switch (e.key) {
                    case 'ArrowRight':
                    case 'ArrowDown':
                        e.preventDefault();
                        this.focusNextBar(bars, index);
                        break;
                    case 'ArrowLeft':
                    case 'ArrowUp':
                        e.preventDefault();
                        this.focusPreviousBar(bars, index);
                        break;
                    case 'Enter':
                    case ' ':
                        e.preventDefault();
                        this.activateBar(bar);
                        break;
                }
            });
        });
    }

    setupScreenReaderAnnouncements(container, chartId) {
        // Create live region for announcements
        const liveRegion = document.createElement('div');
        liveRegion.setAttribute('aria-live', 'polite');
        liveRegion.setAttribute('aria-atomic', 'true');
        liveRegion.className = 'sr-only';
        liveRegion.id = `chart-live-region-${chartId}`;
        container.appendChild(liveRegion);
    }

    focusNextBar(bars, currentIndex) {
        const nextIndex = (currentIndex + 1) % bars.length;
        bars[nextIndex].focus();
    }

    focusPreviousBar(bars, currentIndex) {
        const prevIndex = (currentIndex - 1 + bars.length) % bars.length;
        bars[prevIndex].focus();
    }

    activateBar(bar) {
        // Trigger the same action as hover/focus
        const event = new MouseEvent('mouseenter', {
            bubbles: true,
            cancelable: true
        });
        bar.dispatchEvent(event);
    }

    announce(message, chartId) {
        const liveRegion = document.getElementById(`chart-live-region-${chartId}`);
        if (liveRegion) {
            liveRegion.textContent = message;
        }
    }
}

// Responsive Manager
class ChartResponsiveManager {
    constructor() {
        this.breakpoints = {
            mobile: 480,
            tablet: 768,
            desktop: 1024
        };
        this.currentBreakpoint = this.getCurrentBreakpoint();
    }

    init() {
        // Setup resize observer for responsive behavior
        this.resizeObserver = new ResizeObserver(entries => {
            entries.forEach(entry => {
                this.handleContainerResize(entry.target);
            });
        });
    }

    setupChart(container, chartId) {
        // Observe container for resize
        this.resizeObserver.observe(container);
        
        // Initial responsive setup
        this.updateChart(chartId);
    }

    getCurrentBreakpoint() {
        const width = window.innerWidth;
        if (width < this.breakpoints.mobile) return 'mobile';
        if (width < this.breakpoints.tablet) return 'tablet';
        if (width < this.breakpoints.desktop) return 'desktop';
        return 'large';
    }

    handleContainerResize(container) {
        const chartId = container.getAttribute('data-chart-id');
        if (chartId) {
            this.updateChart(chartId);
        }
    }

    updateChart(chartId) {
        const container = document.querySelector(`[data-chart-id="${chartId}"]`);
        if (!container) return;

        const width = container.offsetWidth;
        const dimensions = this.getResponsiveDimensions(width);
        
        // Apply responsive styles
        this.applyResponsiveStyles(container, dimensions);
    }

    getResponsiveDimensions(containerWidth) {
        if (containerWidth < this.breakpoints.mobile) {
            return {
                chartHeight: 300,
                barHeight: 25,
                fontSize: 12,
                spacing: 8
            };
        } else if (containerWidth < this.breakpoints.tablet) {
            return {
                chartHeight: 350,
                barHeight: 30,
                fontSize: 13,
                spacing: 10
            };
        } else if (containerWidth < this.breakpoints.desktop) {
            return {
                chartHeight: 400,
                barHeight: 35,
                fontSize: 14,
                spacing: 12
            };
        } else {
            return {
                chartHeight: 450,
                barHeight: 40,
                fontSize: 15,
                spacing: 16
            };
        }
    }

    applyResponsiveStyles(container, dimensions) {
        // Apply styles to chart elements
        const bars = container.querySelectorAll('.chart-bar');
        bars.forEach(bar => {
            bar.style.height = `${dimensions.barHeight}px`;
        });

        const labels = container.querySelectorAll('.chart-label');
        labels.forEach(label => {
            label.style.fontSize = `${dimensions.fontSize}px`;
        });

        // Update container spacing
        const chartBars = container.querySelector('.chart-bars');
        if (chartBars) {
            chartBars.style.gap = `${dimensions.spacing}px`;
        }
    }
}

// Initialize the global chart manager
window.flavorSnapChartManager = new FlavorSnapChartManager();

// Export for external use
window.FlavorSnapChartManager = FlavorSnapChartManager;
window.ConfidenceChart = ConfidenceChart;
window.ChartTooltip = ChartTooltip;

// Utility functions
window.FlavorSnapCharts = {
    // Update chart data
    updateChart: (chartId, probabilities, predictedClass) => {
        return window.flavorSnapChartManager.updateChart(chartId, probabilities, predictedClass);
    },
    
    // Reset chart
    resetChart: (chartId) => {
        return window.flavorSnapChartManager.resetChart(chartId);
    },
    
    // Export chart data
    exportChart: (chartId, format = 'json') => {
        return window.flavorSnapChartManager.exportChart(chartId, format);
    },
    
    // Get chart instance
    getChart: (chartId) => {
        return window.flavorSnapChartManager.charts.get(chartId);
    }
};

// Console info for developers
console.log('FlavorSnap Interactive Charts loaded successfully');
console.log('Available methods: FlavorSnapCharts.updateChart(), FlavorSnapCharts.resetChart(), FlavorSnapCharts.exportChart()');
