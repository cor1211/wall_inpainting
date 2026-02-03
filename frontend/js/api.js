/**
 * API Service Layer
 * Connects to FastAPI backend
 */

const API = {
    // Base URL - will be updated when connection is established
    baseUrl: 'http://localhost:8000',

    /**
     * Set base URL
     */
    setBaseUrl(url) {
        this.baseUrl = url.replace(/\/$/, ''); // Remove trailing slash
    },

    /**
     * Health check
     */
    async health() {
        const response = await fetch(`${this.baseUrl}/health`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' }
        });

        if (!response.ok) {
            throw new Error('Server không phản hồi');
        }

        return await response.json();
    },

    /**
     * Check connection and return status
     */
    async checkConnection() {
        try {
            const result = await this.health();
            return {
                connected: true,
                cudaAvailable: result.cuda_available,
                device: result.device
            };
        } catch (e) {
            return {
                connected: false,
                error: e.message
            };
        }
    },

    /**
     * Segment wall from image
     * @param {File|Blob} imageFile - Source image file
     * @param {Object} options - Segmentation options
     * @returns {Promise<Blob>} - Mask image as blob
     */
    async segment(imageFile, options = {}) {
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('strategy', options.strategy || 'semantic');
        formData.append('include_ceiling', options.includeCeiling || false);

        const response = await fetch(`${this.baseUrl}/segment`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await this._parseError(response);
            throw new Error(error);
        }

        return await response.blob();
    },

    /**
     * Process image with reference texture
     * @param {File|Blob} sourceFile - Source image
     * @param {File|Blob} referenceFile - Reference texture image
     * @param {File|Blob|null} maskFile - Optional custom mask
     * @param {Object} options - Processing options
     * @returns {Promise<{blob: Blob, processingTime: string}>}
     */
    async process(sourceFile, referenceFile, maskFile = null, options = {}) {
        const formData = new FormData();
        formData.append('source', sourceFile);
        formData.append('reference', referenceFile);

        if (maskFile) {
            formData.append('mask', maskFile);
        }

        formData.append('strategy', options.strategy || 'semantic');
        formData.append('steps', options.steps || 30);
        formData.append('controlnet_scale', options.controlnetScale || 0.8);
        formData.append('ip_scale', options.ipScale || 0.7);

        if (options.seed !== null && options.seed !== undefined) {
            formData.append('seed', options.seed);
        }

        const response = await fetch(`${this.baseUrl}/process`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await this._parseError(response);
            throw new Error(error);
        }

        const blob = await response.blob();
        const processingTime = response.headers.get('X-Processing-Time') || 'N/A';

        return { blob, processingTime };
    },

    /**
     * Process image with solid color
     * @param {File|Blob} sourceFile - Source image
     * @param {string} color - RGB color as "R,G,B"
     * @param {File|Blob|null} maskFile - Optional custom mask
     * @param {Object} options - Processing options
     * @returns {Promise<{blob: Blob, processingTime: string}>}
     */
    async processColor(sourceFile, color, maskFile = null, options = {}) {
        const formData = new FormData();
        formData.append('source', sourceFile);
        formData.append('color', color);

        if (maskFile) {
            formData.append('mask', maskFile);
        }

        formData.append('strategy', options.strategy || 'semantic');
        formData.append('steps', options.steps || 30);
        formData.append('controlnet_scale', options.controlnetScale || 0.8);
        formData.append('ip_scale', options.ipScale || 0.7);

        if (options.seed !== null && options.seed !== undefined) {
            formData.append('seed', options.seed);
        }

        const response = await fetch(`${this.baseUrl}/process-color`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await this._parseError(response);
            throw new Error(error);
        }

        const blob = await response.blob();
        const processingTime = response.headers.get('X-Processing-Time') || 'N/A';

        return { blob, processingTime };
    },

    /**
     * Parse error response
     */
    async _parseError(response) {
        try {
            const data = await response.json();
            return data.detail || data.error || `Lỗi ${response.status}`;
        } catch {
            if (response.status === 422) {
                return 'Không tìm thấy vùng tường trong ảnh';
            }
            if (response.status === 500) {
                return 'Lỗi máy chủ. Có thể do hết VRAM hoặc model chưa được tải.';
            }
            return `Lỗi ${response.status}: ${response.statusText}`;
        }
    },

    /**
     * Convert color object to API format
     */
    colorToString(colorObj) {
        if (typeof colorObj === 'string') {
            // If hex, convert to RGB
            const rgb = Utils.hexToRgb(colorObj);
            if (rgb) {
                return `${rgb.r},${rgb.g},${rgb.b}`;
            }
            return colorObj;
        }
        return `${colorObj.r},${colorObj.g},${colorObj.b}`;
    }
};

// Freeze API object
Object.freeze(API);
