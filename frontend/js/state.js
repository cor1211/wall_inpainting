/**
 * State Management with Observer Pattern
 */

const AppState = {
    // State data
    _state: {
        // Images
        sourceImage: null,      // Original uploaded image (HTMLImageElement)
        sourceFile: null,       // Original file object
        maskImage: null,        // Current mask (ImageData or HTMLImageElement)
        referenceImage: null,   // Reference texture image
        referenceFile: null,    // Reference file object
        resultImage: null,      // Generated result image

        // UI State
        activeTab: 'color',     // 'color' or 'reference'
        selectedColor: '#c8b4a0',
        isLoading: false,
        loadingMessage: '',

        // Canvas state
        zoom: 1,
        panX: 0,
        panY: 0,

        // Brush settings
        brushMode: 'add',       // 'add' or 'remove'
        brushSize: 30,
        maskOpacity: 0.5,

        // History for undo/redo
        maskHistory: [],
        historyIndex: -1,

        // Settings
        includeCeiling: false,
        steps: 30,
        controlnetScale: 0.8,
        ipScale: 0.7,
        seed: null,

        // Connection status
        isConnected: false,

        // Error
        error: null
    },

    // Observers
    _observers: new Map(),

    /**
     * Get state value
     */
    get(key) {
        return this._state[key];
    },

    /**
     * Set state value and notify observers
     */
    set(key, value) {
        const oldValue = this._state[key];
        if (oldValue === value) return;

        this._state[key] = value;
        this._notify(key, value, oldValue);
    },

    /**
     * Set multiple state values
     */
    setMultiple(updates) {
        const changes = [];
        for (const [key, value] of Object.entries(updates)) {
            const oldValue = this._state[key];
            if (oldValue !== value) {
                this._state[key] = value;
                changes.push({ key, value, oldValue });
            }
        }
        changes.forEach(({ key, value, oldValue }) => {
            this._notify(key, value, oldValue);
        });
    },

    /**
     * Subscribe to state changes
     */
    subscribe(key, callback) {
        if (!this._observers.has(key)) {
            this._observers.set(key, new Set());
        }
        this._observers.get(key).add(callback);

        // Return unsubscribe function
        return () => {
            this._observers.get(key).delete(callback);
        };
    },

    /**
     * Subscribe to multiple keys
     */
    subscribeMultiple(keys, callback) {
        const unsubscribes = keys.map(key => this.subscribe(key, callback));
        return () => unsubscribes.forEach(unsub => unsub());
    },

    /**
     * Notify observers
     */
    _notify(key, value, oldValue) {
        if (this._observers.has(key)) {
            this._observers.get(key).forEach(callback => {
                try {
                    callback(value, oldValue, key);
                } catch (e) {
                    console.error('Observer error:', e);
                }
            });
        }
    },

    /**
     * Reset state
     */
    reset() {
        this.setMultiple({
            sourceImage: null,
            sourceFile: null,
            maskImage: null,
            referenceImage: null,
            referenceFile: null,
            resultImage: null,
            maskHistory: [],
            historyIndex: -1,
            zoom: 1,
            panX: 0,
            panY: 0,
            error: null
        });
    },

    /**
     * Check if can generate
     */
    canGenerate() {
        const hasSource = this.get('sourceImage') !== null;
        const hasMask = this.get('maskImage') !== null;
        const hasStyle = this.get('activeTab') === 'color' || this.get('referenceImage') !== null;
        return hasSource && hasMask && hasStyle && !this.get('isLoading');
    },

    // ========== Mask History Methods ==========

    /**
     * Save current mask to history
     */
    saveMaskToHistory(maskImageData) {
        const history = [...this.get('maskHistory')];
        const index = this.get('historyIndex');

        // Remove any redo states
        if (index < history.length - 1) {
            history.splice(index + 1);
        }

        // Add new state (clone the ImageData)
        const clonedData = new ImageData(
            new Uint8ClampedArray(maskImageData.data),
            maskImageData.width,
            maskImageData.height
        );
        history.push(clonedData);

        // Limit history size
        const maxHistory = 30;
        if (history.length > maxHistory) {
            history.shift();
        }

        this.setMultiple({
            maskHistory: history,
            historyIndex: history.length - 1
        });
    },

    /**
     * Undo mask change
     */
    undo() {
        const index = this.get('historyIndex');
        if (index > 0) {
            this.set('historyIndex', index - 1);
            return this.get('maskHistory')[index - 1];
        }
        return null;
    },

    /**
     * Redo mask change
     */
    redo() {
        const history = this.get('maskHistory');
        const index = this.get('historyIndex');
        if (index < history.length - 1) {
            this.set('historyIndex', index + 1);
            return history[index + 1];
        }
        return null;
    },

    /**
     * Check if can undo
     */
    canUndo() {
        return this.get('historyIndex') > 0;
    },

    /**
     * Check if can redo
     */
    canRedo() {
        const history = this.get('maskHistory');
        return this.get('historyIndex') < history.length - 1;
    }
};

// Freeze state manager
Object.freeze(AppState);
