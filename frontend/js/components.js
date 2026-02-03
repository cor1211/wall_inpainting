/**
 * UI Components
 * Toast notifications, loading overlay, comparison slider, etc.
 */

const Components = {
    /**
     * Initialize all components
     */
    init() {
        this._setupTabs();
        this._setupComparisonSlider();
    },

    // ========== Toast Notifications ==========

    toastContainer: null,

    /**
     * Show toast notification
     */
    showToast(type, title, message, duration = 5000) {
        if (!this.toastContainer) {
            this.toastContainer = document.getElementById('toast-container');
        }

        const icons = {
            success: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
            error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
            warning: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
        };

        const toast = Utils.createElement(`
            <div class="toast ${type}">
                <span class="toast-icon">${icons[type] || icons.warning}</span>
                <div class="toast-content">
                    <div class="toast-title">${title}</div>
                    ${message ? `<div class="toast-message">${message}</div>` : ''}
                </div>
                <button class="toast-close">
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
        `);

        // Close button handler
        toast.querySelector('.toast-close').addEventListener('click', () => {
            this._removeToast(toast);
        });

        this.toastContainer.appendChild(toast);

        // Auto remove
        if (duration > 0) {
            setTimeout(() => this._removeToast(toast), duration);
        }

        return toast;
    },

    _removeToast(toast) {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    },

    // ========== Loading Overlay ==========

    /**
     * Show loading overlay
     */
    showLoading(title = 'Đang xử lý...', message = '') {
        const overlay = document.getElementById('loading-overlay');
        document.getElementById('loading-title').textContent = title;
        document.getElementById('loading-message').textContent = message;
        overlay.hidden = false;
        AppState.set('isLoading', true);
    },

    /**
     * Hide loading overlay
     */
    hideLoading() {
        document.getElementById('loading-overlay').hidden = true;
        AppState.set('isLoading', false);
    },

    /**
     * Update loading message
     */
    updateLoadingMessage(message) {
        document.getElementById('loading-message').textContent = message;
    },

    // ========== Tabs ==========

    _setupTabs() {
        const tabBtns = document.querySelectorAll('.tab-btn');

        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;

                // Update active state
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                // Show content
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                document.getElementById(`tab-${tab}`).classList.add('active');

                // Update state
                AppState.set('activeTab', tab);
            });
        });
    },

    // ========== Comparison Slider ==========

    _setupComparisonSlider() {
        const container = document.getElementById('comparison-container');
        const slider = document.getElementById('comparison-slider');
        const overlay = document.getElementById('comparison-overlay');

        if (!slider || !overlay) return;

        let isDragging = false;

        const updateSlider = (e) => {
            const rect = container.getBoundingClientRect();
            let x = e.clientX - rect.left;
            x = Utils.clamp(x, 0, rect.width);

            const percent = (x / rect.width) * 100;
            slider.style.left = `${percent}%`;
            overlay.style.clipPath = `inset(0 ${100 - percent}% 0 0)`;
        };

        slider.addEventListener('mousedown', (e) => {
            isDragging = true;
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (isDragging) updateSlider(e);
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
        });

        // Also allow clicking on container
        container.addEventListener('click', updateSlider);
    },

    // ========== Result Panel ==========

    /**
     * Show result panel with images
     */
    showResult(beforeImage, afterImage, processingTime = '') {
        const panel = document.getElementById('result-panel');
        const beforeImg = document.getElementById('result-before');
        const afterImg = document.getElementById('result-after');
        const info = document.getElementById('result-info');
        const overlay = document.getElementById('comparison-overlay');

        // Set images
        if (beforeImage instanceof HTMLImageElement) {
            beforeImg.src = beforeImage.src;
        } else if (beforeImage instanceof Blob) {
            beforeImg.src = URL.createObjectURL(beforeImage);
        }

        if (afterImage instanceof HTMLImageElement) {
            afterImg.src = afterImage.src;
        } else if (afterImage instanceof Blob) {
            afterImg.src = URL.createObjectURL(afterImage);
        }

        // Reset slider position
        document.getElementById('comparison-slider').style.left = '50%';
        overlay.style.clipPath = 'inset(0 50% 0 0)';

        // Show processing time
        if (processingTime) {
            info.textContent = `Thời gian xử lý: ${processingTime}`;
        }

        // Show panel
        panel.hidden = false;
        AppState.set('resultImage', afterImage);
    },

    /**
     * Hide result panel
     */
    hideResult() {
        document.getElementById('result-panel').hidden = true;
    },

    // ========== Color Presets ==========

    /**
     * Setup color presets
     */
    setupColorPresets() {
        const presets = document.querySelectorAll('.color-preset');
        const picker = document.getElementById('color-picker');
        const hexInput = document.getElementById('color-hex');

        presets.forEach(preset => {
            preset.addEventListener('click', () => {
                const color = preset.dataset.color;

                // Update UI
                picker.value = color;
                hexInput.value = color;

                // Update active state
                presets.forEach(p => p.classList.remove('active'));
                preset.classList.add('active');

                // Update state
                AppState.set('selectedColor', color);
            });
        });

        // Sync picker with hex input
        picker.addEventListener('input', () => {
            hexInput.value = picker.value;
            AppState.set('selectedColor', picker.value);
            presets.forEach(p => p.classList.remove('active'));
        });

        hexInput.addEventListener('input', () => {
            let value = hexInput.value;
            if (!value.startsWith('#')) value = '#' + value;
            if (/^#[0-9A-Fa-f]{6}$/.test(value)) {
                picker.value = value;
                AppState.set('selectedColor', value);
            }
            presets.forEach(p => p.classList.remove('active'));
        });
    },

    // ========== Range Inputs ==========

    /**
     * Setup range input with value display
     */
    setupRangeInput(inputId, valueId, formatter = (v) => v) {
        const input = document.getElementById(inputId);
        const display = document.getElementById(valueId);

        if (!input || !display) return;

        input.addEventListener('input', () => {
            display.textContent = formatter(input.value);
        });
    }
};

// Initialize when DOM ready
document.addEventListener('DOMContentLoaded', () => {
    Components.init();
});
