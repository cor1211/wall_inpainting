/**
 * Main Application
 * Wire everything together
 */

const App = {
    /**
     * Initialize application
     */
    async init() {
        console.log('AI Wall Inpainting - Initializing...');

        // Setup UI components
        this._setupUpload();
        this._setupMaskControls();
        this._setupStyleControls();
        this._setupAdvancedSettings();
        this._setupGenerateButton();
        this._setupResultPanel();
        this._setupStateSubscriptions();

        // Check backend connection
        await this._checkConnection();

        console.log('AI Wall Inpainting - Ready!');
    },

    /**
     * Check backend connection
     */
    async _checkConnection() {
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');

        try {
            const result = await API.checkConnection();

            if (result.connected) {
                statusDot.classList.add('connected');
                statusDot.classList.remove('error');
                statusText.textContent = result.cudaAvailable ? 'GPU Ready' : 'CPU Mode';
                AppState.set('isConnected', true);
            } else {
                throw new Error(result.error);
            }
        } catch (e) {
            statusDot.classList.add('error');
            statusDot.classList.remove('connected');
            statusText.textContent = 'Không kết nối';
            AppState.set('isConnected', false);

            Components.showToast(
                'error',
                'Không thể kết nối server',
                'Hãy đảm bảo backend đang chạy tại localhost:8000'
            );
        }
    },

    /**
     * Setup file upload
     */
    _setupUpload() {
        const uploadZone = document.getElementById('upload-zone');
        const sourceInput = document.getElementById('source-input');
        const sourcePreview = document.getElementById('source-preview');
        const sourceThumb = document.getElementById('source-thumb');
        const removeBtn = document.getElementById('btn-remove-source');

        // Click to upload
        uploadZone.addEventListener('click', () => sourceInput.click());

        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');

            const file = e.dataTransfer.files[0];
            if (file) await this._handleSourceUpload(file);
        });

        // File input change
        sourceInput.addEventListener('change', async () => {
            const file = sourceInput.files[0];
            if (file) await this._handleSourceUpload(file);
        });

        // Remove button
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this._clearSource();
        });

        // Reference image upload
        const refZone = document.getElementById('reference-zone');
        const refInput = document.getElementById('reference-input');
        const refPreview = document.getElementById('reference-preview');
        const refThumb = document.getElementById('reference-thumb');
        const removeRefBtn = document.getElementById('btn-remove-reference');

        refZone.addEventListener('click', () => refInput.click());

        refZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            refZone.classList.add('dragover');
        });

        refZone.addEventListener('dragleave', () => {
            refZone.classList.remove('dragover');
        });

        refZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            refZone.classList.remove('dragover');

            const file = e.dataTransfer.files[0];
            if (file) await this._handleReferenceUpload(file);
        });

        refInput.addEventListener('change', async () => {
            const file = refInput.files[0];
            if (file) await this._handleReferenceUpload(file);
        });

        removeRefBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this._clearReference();
        });
    },

    /**
     * Handle source image upload
     */
    async _handleSourceUpload(file) {
        try {
            // Validate file size
            const maxSize = 20 * 1024 * 1024; // 20MB
            if (file.size > maxSize) {
                Components.showToast('error', 'File quá lớn', 'Kích thước tối đa là 20MB');
                return;
            }

            // Load image
            const image = await Utils.loadImageFromFile(file);

            // Update state
            AppState.set('sourceImage', image);
            AppState.set('sourceFile', file);

            // Update UI
            document.getElementById('source-thumb').src = image.src;
            document.getElementById('source-preview').hidden = false;
            document.getElementById('upload-zone').style.display = 'none';

            // Enable auto-segment button
            document.getElementById('btn-auto-segment').disabled = false;

            // Load into canvas
            CanvasEditor.loadImage(image);

            Components.showToast('success', 'Upload thành công', `${file.name}`);

        } catch (e) {
            Components.showToast('error', 'Lỗi upload', e.message);
        }
    },

    /**
     * Clear source image
     */
    _clearSource() {
        AppState.set('sourceImage', null);
        AppState.set('sourceFile', null);
        AppState.set('maskImage', null);

        document.getElementById('source-preview').hidden = true;
        document.getElementById('upload-zone').style.display = '';
        document.getElementById('source-input').value = '';
        document.getElementById('btn-auto-segment').disabled = true;

        CanvasEditor.clear();
        this._updateGenerateButton();
    },

    /**
     * Handle reference image upload
     */
    async _handleReferenceUpload(file) {
        try {
            const image = await Utils.loadImageFromFile(file);

            AppState.set('referenceImage', image);
            AppState.set('referenceFile', file);

            document.getElementById('reference-thumb').src = image.src;
            document.getElementById('reference-preview').hidden = false;
            document.getElementById('reference-zone').style.display = 'none';

            this._updateGenerateButton();

        } catch (e) {
            Components.showToast('error', 'Lỗi upload', e.message);
        }
    },

    /**
     * Clear reference image
     */
    _clearReference() {
        AppState.set('referenceImage', null);
        AppState.set('referenceFile', null);

        document.getElementById('reference-preview').hidden = true;
        document.getElementById('reference-zone').style.display = '';
        document.getElementById('reference-input').value = '';

        this._updateGenerateButton();
    },

    /**
     * Setup mask controls
     */
    _setupMaskControls() {
        // Auto segment button
        document.getElementById('btn-auto-segment').addEventListener('click', () => {
            this._autoSegment();
        });

        // Include ceiling checkbox
        document.getElementById('include-ceiling').addEventListener('change', (e) => {
            AppState.set('includeCeiling', e.target.checked);
        });

        // Brush mode buttons
        document.getElementById('btn-brush-add').addEventListener('click', () => {
            AppState.set('brushMode', 'add');
            document.getElementById('btn-brush-add').classList.add('active');
            document.getElementById('btn-brush-remove').classList.remove('active');
        });

        document.getElementById('btn-brush-remove').addEventListener('click', () => {
            AppState.set('brushMode', 'remove');
            document.getElementById('btn-brush-remove').classList.add('active');
            document.getElementById('btn-brush-add').classList.remove('active');
        });

        // Undo/Redo buttons
        document.getElementById('btn-undo').addEventListener('click', () => {
            CanvasEditor.undo();
        });

        document.getElementById('btn-redo').addEventListener('click', () => {
            CanvasEditor.redo();
        });

        // Clear mask
        document.getElementById('btn-clear-mask').addEventListener('click', () => {
            CanvasEditor.clearMask();
        });

        // Brush size
        const brushSizeInput = document.getElementById('brush-size');
        const brushSizeValue = document.getElementById('brush-size-value');

        brushSizeInput.addEventListener('input', () => {
            const size = parseInt(brushSizeInput.value);
            brushSizeValue.textContent = `${size}px`;
            AppState.set('brushSize', size);
        });

        // Mask opacity
        const maskOpacityInput = document.getElementById('mask-opacity');
        maskOpacityInput.addEventListener('input', () => {
            AppState.set('maskOpacity', maskOpacityInput.value / 100);
        });
    },

    /**
     * Auto segment wall
     */
    async _autoSegment() {
        const sourceFile = AppState.get('sourceFile');
        if (!sourceFile) return;

        Components.showLoading('Đang phân tích ảnh...', 'AI đang tìm vùng tường trong ảnh');

        try {
            const includeCeiling = AppState.get('includeCeiling');

            const maskBlob = await API.segment(sourceFile, {
                strategy: 'semantic',
                includeCeiling: includeCeiling
            });

            // Load mask image
            const maskImage = await Utils.loadImageFromUrl(URL.createObjectURL(maskBlob));

            // Load into canvas editor
            CanvasEditor.loadMask(maskImage);

            Components.hideLoading();
            Components.showToast('success', 'Phân tích thành công', 'Bạn có thể chỉnh sửa mask bằng brush tool');

            this._updateGenerateButton();

        } catch (e) {
            Components.hideLoading();
            Components.showToast('error', 'Lỗi phân tích', e.message);
        }
    },

    /**
     * Setup style controls
     */
    _setupStyleControls() {
        Components.setupColorPresets();
    },

    /**
     * Setup advanced settings
     */
    _setupAdvancedSettings() {
        // Steps
        Components.setupRangeInput('steps', 'steps-value');
        document.getElementById('steps').addEventListener('change', (e) => {
            AppState.set('steps', parseInt(e.target.value));
        });

        // ControlNet scale
        Components.setupRangeInput('controlnet-scale', 'controlnet-value', (v) => (v / 100).toFixed(1));
        document.getElementById('controlnet-scale').addEventListener('change', (e) => {
            AppState.set('controlnetScale', e.target.value / 100);
        });

        // IP-Adapter scale
        Components.setupRangeInput('ip-scale', 'ip-value', (v) => (v / 100).toFixed(1));
        document.getElementById('ip-scale').addEventListener('change', (e) => {
            AppState.set('ipScale', e.target.value / 100);
        });

        // Seed
        document.getElementById('seed').addEventListener('change', (e) => {
            const value = e.target.value ? parseInt(e.target.value) : null;
            AppState.set('seed', value);
        });
    },

    /**
     * Setup generate button
     */
    _setupGenerateButton() {
        document.getElementById('btn-generate').addEventListener('click', () => {
            this._generate();
        });
    },

    /**
     * Update generate button state
     */
    _updateGenerateButton() {
        const btn = document.getElementById('btn-generate');
        btn.disabled = !AppState.canGenerate();
    },

    /**
     * Generate result
     */
    async _generate() {
        const sourceFile = AppState.get('sourceFile');
        const maskImage = AppState.get('maskImage');

        if (!sourceFile || !maskImage) {
            Components.showToast('warning', 'Thiếu thông tin', 'Cần upload ảnh và tạo mask trước');
            return;
        }

        Components.showLoading('Đang tạo ảnh...', 'AI đang thay đổi màu tường, vui lòng đợi (30-60 giây)');

        try {
            // Get mask as blob
            const maskBlob = await CanvasEditor.exportMaskAsBlob();

            // Get options
            const options = {
                steps: AppState.get('steps'),
                controlnetScale: AppState.get('controlnetScale'),
                ipScale: AppState.get('ipScale'),
                seed: AppState.get('seed')
            };

            let result;

            if (AppState.get('activeTab') === 'color') {
                // Process with solid color
                const color = AppState.get('selectedColor');
                const rgb = Utils.hexToRgb(color);
                const colorStr = `${rgb.r},${rgb.g},${rgb.b}`;

                result = await API.processColor(sourceFile, colorStr, maskBlob, options);
            } else {
                // Process with reference image
                const referenceFile = AppState.get('referenceFile');
                if (!referenceFile) {
                    throw new Error('Chưa upload ảnh texture tham chiếu');
                }

                result = await API.process(sourceFile, referenceFile, maskBlob, options);
            }

            Components.hideLoading();

            // Show result
            const sourceImage = AppState.get('sourceImage');
            Components.showResult(sourceImage, result.blob, result.processingTime);

            Components.showToast('success', 'Tạo ảnh thành công!', `Thời gian: ${result.processingTime}`);

        } catch (e) {
            Components.hideLoading();
            Components.showToast('error', 'Lỗi tạo ảnh', e.message);
        }
    },

    /**
     * Setup result panel
     */
    _setupResultPanel() {
        // Close button
        document.getElementById('btn-close-result').addEventListener('click', () => {
            Components.hideResult();
        });

        // Download button
        document.getElementById('btn-download').addEventListener('click', () => {
            this._downloadResult();
        });

        // Regenerate button
        document.getElementById('btn-regenerate').addEventListener('click', () => {
            Components.hideResult();
            this._generate();
        });
    },

    /**
     * Download result
     */
    async _downloadResult() {
        const resultImage = AppState.get('resultImage');
        if (!resultImage) return;

        let blob;
        if (resultImage instanceof Blob) {
            blob = resultImage;
        } else {
            // Convert image to blob
            const canvas = document.createElement('canvas');
            canvas.width = resultImage.naturalWidth || resultImage.width;
            canvas.height = resultImage.naturalHeight || resultImage.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(resultImage, 0, 0);
            blob = await Utils.canvasToBlob(canvas);
        }

        const filename = `wall_inpainting_${Date.now()}.png`;
        Utils.downloadBlob(blob, filename);

        Components.showToast('success', 'Đã tải xuống', filename);
    },

    /**
     * Setup state subscriptions
     */
    _setupStateSubscriptions() {
        // Update undo/redo buttons
        AppState.subscribe('historyIndex', () => {
            document.getElementById('btn-undo').disabled = !AppState.canUndo();
            document.getElementById('btn-redo').disabled = !AppState.canRedo();
        });

        // Update generate button when relevant state changes
        AppState.subscribeMultiple(
            ['sourceImage', 'maskImage', 'referenceImage', 'activeTab', 'isLoading'],
            () => this._updateGenerateButton()
        );

        // Health check button
        document.getElementById('btn-health').addEventListener('click', () => {
            this._checkConnection();
        });
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});
