/**
 * Canvas Mask Editor
 * Handles image display and mask painting
 */

const CanvasEditor = {
    // DOM Elements
    container: null,
    mainCanvas: null,
    maskCanvas: null,
    mainCtx: null,
    maskCtx: null,
    brushCursor: null,

    // State
    isInitialized: false,
    isPainting: false,
    lastPoint: null,

    // Image dimensions
    imageWidth: 0,
    imageHeight: 0,
    displayWidth: 0,
    displayHeight: 0,
    offsetX: 0,
    offsetY: 0,
    scale: 1,

    /**
     * Initialize canvas editor
     */
    init() {
        this.container = document.getElementById('canvas-container');
        this.mainCanvas = document.getElementById('main-canvas');
        this.maskCanvas = document.getElementById('mask-canvas');
        this.brushCursor = document.getElementById('brush-cursor');

        if (!this.mainCanvas || !this.maskCanvas) {
            console.error('Canvas elements not found');
            return;
        }

        this.mainCtx = this.mainCanvas.getContext('2d');
        this.maskCtx = this.maskCanvas.getContext('2d');

        this._setupEventListeners();
        this._subscribeToState();

        this.isInitialized = true;
    },

    /**
     * Setup event listeners
     */
    _setupEventListeners() {
        // Mouse events for painting
        this.mainCanvas.addEventListener('mousedown', (e) => this._onMouseDown(e));
        this.mainCanvas.addEventListener('mousemove', (e) => this._onMouseMove(e));
        this.mainCanvas.addEventListener('mouseup', () => this._onMouseUp());
        this.mainCanvas.addEventListener('mouseleave', () => this._onMouseLeave());

        // Prevent context menu on right click
        this.mainCanvas.addEventListener('contextmenu', (e) => e.preventDefault());

        // Track brush cursor
        this.container.addEventListener('mousemove', (e) => this._updateBrushCursor(e));
        this.container.addEventListener('mouseenter', () => this._showBrushCursor());
        this.container.addEventListener('mouseleave', () => this._hideBrushCursor());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this._onKeyDown(e));

        // Window resize
        window.addEventListener('resize', Utils.debounce(() => this._onResize(), 200));
    },

    /**
     * Subscribe to state changes
     */
    _subscribeToState() {
        AppState.subscribe('sourceImage', (img) => {
            if (img) this.loadImage(img);
            else this.clear();
        });

        AppState.subscribe('maskOpacity', (opacity) => {
            this.maskCanvas.style.opacity = opacity;
        });

        AppState.subscribe('brushSize', () => {
            this._updateBrushCursorSize();
        });
    },

    /**
     * Load and display image
     */
    loadImage(image) {
        this.imageWidth = image.naturalWidth || image.width;
        this.imageHeight = image.naturalHeight || image.height;

        this._calculateDisplaySize();
        this._setupCanvases();

        // Draw image
        this.mainCtx.drawImage(image, 0, 0, this.displayWidth, this.displayHeight);

        // Clear mask
        this.maskCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);

        // Show toolbar
        document.getElementById('canvas-toolbar').hidden = false;
        document.getElementById('canvas-empty').hidden = true;
    },

    /**
     * Calculate display size to fit container
     */
    _calculateDisplaySize() {
        const containerRect = this.container.getBoundingClientRect();
        const maxWidth = containerRect.width - 40;
        const maxHeight = containerRect.height - 80;

        const imageAspect = this.imageWidth / this.imageHeight;
        const containerAspect = maxWidth / maxHeight;

        if (imageAspect > containerAspect) {
            this.displayWidth = maxWidth;
            this.displayHeight = maxWidth / imageAspect;
        } else {
            this.displayHeight = maxHeight;
            this.displayWidth = maxHeight * imageAspect;
        }

        this.scale = this.displayWidth / this.imageWidth;
    },

    /**
     * Setup canvas dimensions
     */
    _setupCanvases() {
        // Set canvas sizes
        this.mainCanvas.width = this.displayWidth;
        this.mainCanvas.height = this.displayHeight;
        this.maskCanvas.width = this.displayWidth;
        this.maskCanvas.height = this.displayHeight;

        // Style
        this.mainCanvas.style.width = `${this.displayWidth}px`;
        this.mainCanvas.style.height = `${this.displayHeight}px`;
        this.maskCanvas.style.width = `${this.displayWidth}px`;
        this.maskCanvas.style.height = `${this.displayHeight}px`;
    },

    /**
     * Load mask image
     */
    loadMask(maskImage) {
        // Clear existing
        this.maskCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);

        // Draw mask in red/transparent
        this.maskCtx.save();

        // Draw mask image
        this.maskCtx.drawImage(maskImage, 0, 0, this.displayWidth, this.displayHeight);

        // Convert to colored overlay
        const imageData = this.maskCtx.getImageData(0, 0, this.displayWidth, this.displayHeight);
        const data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
            const brightness = data[i]; // Assuming grayscale mask
            if (brightness > 128) {
                // White = selected -> show as accent color
                data[i] = 99;      // R
                data[i + 1] = 102; // G
                data[i + 2] = 241; // B
                data[i + 3] = 180; // A
            } else {
                // Black = not selected -> transparent
                data[i + 3] = 0;
            }
        }

        this.maskCtx.putImageData(imageData, 0, 0);
        this.maskCtx.restore();

        // Save to state and history
        AppState.set('maskImage', imageData);
        AppState.saveMaskToHistory(imageData);
    },

    /**
     * Get canvas position from mouse event
     */
    _getCanvasPosition(e) {
        const rect = this.mainCanvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    },

    /**
     * Mouse down handler
     */
    _onMouseDown(e) {
        if (!AppState.get('sourceImage')) return;

        this.isPainting = true;
        this.lastPoint = this._getCanvasPosition(e);

        // Determine brush mode based on mouse button
        const mode = e.button === 2 ? 'remove' : AppState.get('brushMode');

        this._paint(this.lastPoint, mode);
    },

    /**
     * Mouse move handler
     */
    _onMouseMove(e) {
        if (!this.isPainting || !this.lastPoint) return;

        const currentPoint = this._getCanvasPosition(e);
        const mode = e.buttons === 2 ? 'remove' : AppState.get('brushMode');

        // Draw line between points for smooth strokes
        this._paintLine(this.lastPoint, currentPoint, mode);

        this.lastPoint = currentPoint;
    },

    /**
     * Mouse up handler
     */
    _onMouseUp() {
        if (this.isPainting) {
            this.isPainting = false;
            this.lastPoint = null;

            // Save to history
            const imageData = this.maskCtx.getImageData(0, 0, this.displayWidth, this.displayHeight);
            AppState.set('maskImage', imageData);
            AppState.saveMaskToHistory(imageData);
        }
    },

    /**
     * Mouse leave handler
     */
    _onMouseLeave() {
        this._onMouseUp();
        this._hideBrushCursor();
    },

    /**
     * Paint at point
     */
    _paint(point, mode) {
        const radius = AppState.get('brushSize') / 2;

        this.maskCtx.beginPath();
        this.maskCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);

        if (mode === 'add') {
            this.maskCtx.fillStyle = 'rgba(99, 102, 241, 0.7)';
            this.maskCtx.fill();
        } else {
            this.maskCtx.save();
            this.maskCtx.globalCompositeOperation = 'destination-out';
            this.maskCtx.fillStyle = 'rgba(0, 0, 0, 1)';
            this.maskCtx.fill();
            this.maskCtx.restore();
        }
    },

    /**
     * Paint line between two points
     */
    _paintLine(from, to, mode) {
        const radius = AppState.get('brushSize') / 2;
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const steps = Math.max(1, Math.floor(distance / (radius / 2)));

        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const point = {
                x: from.x + dx * t,
                y: from.y + dy * t
            };
            this._paint(point, mode);
        }
    },

    /**
     * Update brush cursor position
     */
    _updateBrushCursor(e) {
        if (!AppState.get('sourceImage')) return;

        this.brushCursor.style.left = `${e.clientX}px`;
        this.brushCursor.style.top = `${e.clientY}px`;
    },

    /**
     * Update brush cursor size
     */
    _updateBrushCursorSize() {
        const size = AppState.get('brushSize');
        this.brushCursor.style.width = `${size}px`;
        this.brushCursor.style.height = `${size}px`;
    },

    /**
     * Show brush cursor
     */
    _showBrushCursor() {
        if (AppState.get('sourceImage')) {
            this._updateBrushCursorSize();
            this.brushCursor.style.display = 'block';
        }
    },

    /**
     * Hide brush cursor
     */
    _hideBrushCursor() {
        this.brushCursor.style.display = 'none';
    },

    /**
     * Keyboard shortcuts
     */
    _onKeyDown(e) {
        if (e.ctrlKey || e.metaKey) {
            if (e.key === 'z') {
                e.preventDefault();
                this.undo();
            } else if (e.key === 'y') {
                e.preventDefault();
                this.redo();
            }
        }
    },

    /**
     * Undo
     */
    undo() {
        const maskData = AppState.undo();
        if (maskData) {
            this.maskCtx.putImageData(maskData, 0, 0);
            AppState.set('maskImage', maskData);
        }
    },

    /**
     * Redo
     */
    redo() {
        const maskData = AppState.redo();
        if (maskData) {
            this.maskCtx.putImageData(maskData, 0, 0);
            AppState.set('maskImage', maskData);
        }
    },

    /**
     * Clear mask
     */
    clearMask() {
        this.maskCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);
        const imageData = this.maskCtx.getImageData(0, 0, this.displayWidth, this.displayHeight);
        AppState.set('maskImage', imageData);
        AppState.saveMaskToHistory(imageData);
    },

    /**
     * Clear all
     */
    clear() {
        this.mainCtx.clearRect(0, 0, this.mainCanvas.width, this.mainCanvas.height);
        this.maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);

        document.getElementById('canvas-toolbar').hidden = true;
        document.getElementById('canvas-empty').hidden = false;

        this.imageWidth = 0;
        this.imageHeight = 0;
    },

    /**
     * Export mask as Blob
     */
    async exportMaskAsBlob() {
        // Create temp canvas at original image size
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.imageWidth;
        tempCanvas.height = this.imageHeight;
        const tempCtx = tempCanvas.getContext('2d');

        // Get current mask data
        const maskData = this.maskCtx.getImageData(0, 0, this.displayWidth, this.displayHeight);

        // Create grayscale mask at display size
        const displayCanvas = document.createElement('canvas');
        displayCanvas.width = this.displayWidth;
        displayCanvas.height = this.displayHeight;
        const displayCtx = displayCanvas.getContext('2d');

        // Convert to grayscale mask
        const grayData = new ImageData(this.displayWidth, this.displayHeight);
        for (let i = 0; i < maskData.data.length; i += 4) {
            const alpha = maskData.data[i + 3];
            const value = alpha > 50 ? 255 : 0;
            grayData.data[i] = value;
            grayData.data[i + 1] = value;
            grayData.data[i + 2] = value;
            grayData.data[i + 3] = 255;
        }
        displayCtx.putImageData(grayData, 0, 0);

        // Scale to original size
        tempCtx.drawImage(displayCanvas, 0, 0, this.imageWidth, this.imageHeight);

        return Utils.canvasToBlob(tempCanvas, 'image/png');
    },

    /**
     * Handle resize
     */
    _onResize() {
        const sourceImage = AppState.get('sourceImage');
        if (sourceImage) {
            const maskData = this.maskCtx.getImageData(0, 0, this.displayWidth, this.displayHeight);
            this.loadImage(sourceImage);
            // Restore mask (scaled)
            // For simplicity, you may want to re-segment instead
        }
    },

    /**
     * Set zoom level
     */
    setZoom(level) {
        // TODO: Implement zoom
        AppState.set('zoom', level);
    }
};

// Initialize when DOM ready
document.addEventListener('DOMContentLoaded', () => {
    CanvasEditor.init();
});
