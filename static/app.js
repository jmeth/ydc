/**
 * YOLO Dataset Creator - Frontend Application
 */

// ============ State ============
const state = {
    currentDataset: null,
    datasets: [],
    images: [],
    classes: {},          // {id: {name, color}}
    currentImage: null,   // {filename, split, path, ...}
    annotations: [],      // [{classId, x, y, width, height}, ...]
    selectedAnnotation: null,
    selectedClass: 0,
    tool: 'select',       // 'select' or 'draw'
    zoom: 1,
    pan: { x: 0, y: 0 },
    isDirty: false,       // unsaved changes
    isDrawing: false,
    drawStart: null,
    imageElement: null,
    // Drag/resize state
    isDragging: false,
    isResizing: false,
    resizeHandle: null,   // 'nw', 'ne', 'sw', 'se', 'n', 's', 'e', 'w'
    dragStart: null,      // {x, y} - mouse position at drag start
    dragOriginal: null,   // original annotation coords at drag start
};

// Default colors for labels
const DEFAULT_COLORS = [
    '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
    '#ff8000', '#8000ff', '#0080ff', '#ff0080', '#80ff00', '#00ff80'
];

// ============ DOM Elements ============
const elements = {
    datasetSelect: document.getElementById('dataset-select'),
    newDatasetBtn: document.getElementById('new-dataset-btn'),
    uploadBtn: document.getElementById('upload-btn'),
    webcamBtn: document.getElementById('webcam-btn'),
    imageList: document.getElementById('image-list'),
    labelList: document.getElementById('label-list'),
    annotationList: document.getElementById('annotation-list'),
    addLabelBtn: document.getElementById('add-label-btn'),
    canvas: document.getElementById('editor-canvas'),
    canvasContainer: document.getElementById('canvas-container'),
    noImageMessage: document.getElementById('no-image-message'),
    toolSelect: document.getElementById('tool-select'),
    toolDraw: document.getElementById('tool-draw'),
    zoomIn: document.getElementById('zoom-in'),
    zoomOut: document.getElementById('zoom-out'),
    zoomFit: document.getElementById('zoom-fit'),
    zoomLevel: document.getElementById('zoom-level'),
    deleteBox: document.getElementById('delete-box'),
    clearAll: document.getElementById('clear-all'),
    deleteImage: document.getElementById('delete-image'),
    splitSelect: document.getElementById('split-select'),
    saveBtn: document.getElementById('save-btn'),
    statusMessage: document.getElementById('status-message'),
    cursorPosition: document.getElementById('cursor-position'),
    fileInput: document.getElementById('file-input'),
    modalOverlay: document.getElementById('modal-overlay'),
    // Modals
    newDatasetModal: document.getElementById('new-dataset-modal'),
    datasetNameInput: document.getElementById('dataset-name'),
    createDatasetBtn: document.getElementById('create-dataset'),
    cancelDatasetBtn: document.getElementById('cancel-dataset'),
    addLabelModal: document.getElementById('add-label-modal'),
    labelNameInput: document.getElementById('label-name'),
    labelColorInput: document.getElementById('label-color'),
    createLabelBtn: document.getElementById('create-label'),
    cancelLabelBtn: document.getElementById('cancel-label'),
    webcamModal: document.getElementById('webcam-modal'),
    webcamVideo: document.getElementById('webcam-video'),
    webcamCanvas: document.getElementById('webcam-canvas'),
    captureWebcamBtn: document.getElementById('capture-webcam'),
    cancelWebcamBtn: document.getElementById('cancel-webcam'),
    // Export
    exportBtn: document.getElementById('export-btn'),
    exportModal: document.getElementById('export-modal'),
    exportInfo: document.getElementById('export-info'),
    exportWarning: document.getElementById('export-warning'),
    confirmExportBtn: document.getElementById('confirm-export'),
    cancelExportBtn: document.getElementById('cancel-export'),
    // Import
    importBtn: document.getElementById('import-btn'),
    importModal: document.getElementById('import-modal'),
    importDropzone: document.getElementById('import-dropzone'),
    importFileInput: document.getElementById('import-file-input'),
    importStatus: document.getElementById('import-status'),
    confirmImportBtn: document.getElementById('confirm-import'),
    cancelImportBtn: document.getElementById('cancel-import'),
};

const ctx = elements.canvas.getContext('2d');

// ============ API Functions ============
const api = {
    async getDatasets() {
        const res = await fetch('/api/datasets');
        return res.json();
    },

    async createDataset(name) {
        const res = await fetch('/api/datasets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        return res.json();
    },

    async getDataset(name) {
        const res = await fetch(`/api/datasets/${name}`);
        return res.json();
    },

    async updateDataset(name, data) {
        const res = await fetch(`/api/datasets/${name}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return res.json();
    },

    async getImages(dataset) {
        const res = await fetch(`/api/datasets/${dataset}/images`);
        return res.json();
    },

    async uploadImage(dataset, file, split) {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('split', split);
        const res = await fetch(`/api/datasets/${dataset}/images`, {
            method: 'POST',
            body: formData
        });
        return res.json();
    },

    async deleteImage(dataset, split, filename) {
        const res = await fetch(`/api/datasets/${dataset}/images/${split}/${filename}`, {
            method: 'DELETE'
        });
        return res.json();
    },

    async getLabels(dataset, split, filename) {
        const res = await fetch(`/api/datasets/${dataset}/labels/${split}/${filename}`);
        return res.json();
    },

    async saveLabels(dataset, split, filename, annotations) {
        const res = await fetch(`/api/datasets/${dataset}/labels/${split}/${filename}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(annotations)
        });
        return res.json();
    },

    async changeSplit(dataset, currentSplit, filename, newSplit) {
        const res = await fetch(`/api/datasets/${dataset}/split/${currentSplit}/${filename}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ split: newSplit })
        });
        return res.json();
    },

    async getDatasetStats(name) {
        const res = await fetch(`/api/datasets/${name}/stats`);
        return res.json();
    },

    getExportUrl(name) {
        return `/api/datasets/${name}/export`;
    },

    async importDataset(file) {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/api/datasets/import', {
            method: 'POST',
            body: formData
        });
        return res.json();
    }
};

// ============ Initialization ============
async function init() {
    await loadDatasets();
    setupEventListeners();
    setupKeyboardShortcuts();
    setupDragAndDrop();
    render();
}

async function loadDatasets() {
    state.datasets = await api.getDatasets();
    renderDatasetSelect();
}

async function selectDataset(name) {
    if (!name) {
        state.currentDataset = null;
        state.images = [];
        state.classes = {};
        state.currentImage = null;
        elements.exportBtn.disabled = true;
        render();
        return;
    }

    const dataset = await api.getDataset(name);
    state.currentDataset = name;
    state.classes = {};

    // Convert classes from {0: 'name'} to {0: {name, color}}
    const classNames = dataset.classes || {};
    Object.entries(classNames).forEach(([id, name]) => {
        state.classes[id] = {
            name,
            color: DEFAULT_COLORS[parseInt(id) % DEFAULT_COLORS.length]
        };
    });

    state.images = await api.getImages(name);
    state.currentImage = null;
    state.annotations = [];

    // Enable export button
    elements.exportBtn.disabled = false;

    render();
    setStatus(`Loaded dataset: ${name}`);
}

// ============ Event Listeners ============
function setupEventListeners() {
    // Dataset selection
    elements.datasetSelect.addEventListener('change', (e) => selectDataset(e.target.value));
    elements.newDatasetBtn.addEventListener('click', () => showModal('new-dataset'));

    // New dataset modal
    elements.createDatasetBtn.addEventListener('click', createDataset);
    elements.cancelDatasetBtn.addEventListener('click', () => hideModal());
    elements.datasetNameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') createDataset();
    });

    // Label modal
    elements.addLabelBtn.addEventListener('click', () => {
        elements.labelColorInput.value = DEFAULT_COLORS[Object.keys(state.classes).length % DEFAULT_COLORS.length];
        showModal('add-label');
    });
    elements.createLabelBtn.addEventListener('click', createLabel);
    elements.cancelLabelBtn.addEventListener('click', () => hideModal());
    elements.labelNameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') createLabel();
    });

    // Image upload
    elements.uploadBtn.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileUpload);

    // Webcam
    elements.webcamBtn.addEventListener('click', openWebcam);
    elements.captureWebcamBtn.addEventListener('click', captureWebcam);
    elements.cancelWebcamBtn.addEventListener('click', closeWebcam);

    // Export
    elements.exportBtn.addEventListener('click', openExportModal);
    elements.confirmExportBtn.addEventListener('click', confirmExport);
    elements.cancelExportBtn.addEventListener('click', () => hideModal());

    // Import
    elements.importBtn.addEventListener('click', openImportModal);
    elements.confirmImportBtn.addEventListener('click', confirmImport);
    elements.cancelImportBtn.addEventListener('click', () => {
        selectedImportFile = null;
        hideModal();
    });
    elements.importDropzone.addEventListener('click', () => elements.importFileInput.click());
    elements.importFileInput.addEventListener('change', handleImportFileSelect);
    elements.importDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.importDropzone.classList.add('drag-over');
    });
    elements.importDropzone.addEventListener('dragleave', () => {
        elements.importDropzone.classList.remove('drag-over');
    });
    elements.importDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.importDropzone.classList.remove('drag-over');
        const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.zip'));
        if (files.length > 0) {
            handleImportFile(files[0]);
        }
    });

    // Tools
    elements.toolSelect.addEventListener('click', () => setTool('select'));
    elements.toolDraw.addEventListener('click', () => setTool('draw'));
    elements.zoomIn.addEventListener('click', () => setZoom(state.zoom * 1.25));
    elements.zoomOut.addEventListener('click', () => setZoom(state.zoom / 1.25));
    elements.zoomFit.addEventListener('click', fitToView);
    elements.deleteBox.addEventListener('click', deleteSelectedAnnotation);
    elements.clearAll.addEventListener('click', clearAllAnnotations);
    elements.deleteImage.addEventListener('click', deleteCurrentImage);
    elements.saveBtn.addEventListener('click', saveAnnotations);

    // Split change
    elements.splitSelect.addEventListener('change', handleSplitChange);

    // Canvas events
    elements.canvas.addEventListener('mousedown', handleCanvasMouseDown);
    elements.canvas.addEventListener('mousemove', handleCanvasMouseMove);
    elements.canvas.addEventListener('mouseup', handleCanvasMouseUp);
    elements.canvas.addEventListener('mouseleave', handleCanvasMouseUp);
    elements.canvas.addEventListener('wheel', handleCanvasWheel);

    // Modal close buttons
    document.querySelectorAll('.modal-close').forEach(btn => {
        btn.addEventListener('click', hideModal);
    });
}

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Don't handle shortcuts when typing in inputs
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        // Number keys 1-9 for label selection (and changing selected box label)
        if (e.key >= '1' && e.key <= '9') {
            const classId = parseInt(e.key) - 1;
            if (state.classes[classId]) {
                selectClass(classId);
            }
            return;
        }

        switch (e.key.toLowerCase()) {
            case 'n':
                setTool('draw');
                break;
            case 'v':
                setTool('select');
                break;
            case 'w':
                openWebcam();
                break;
            case 'delete':
            case 'backspace':
                e.preventDefault();
                if (e.shiftKey) {
                    deleteCurrentImage();
                } else if (state.selectedAnnotation !== null) {
                    deleteSelectedAnnotation();
                }
                break;
            case 'escape':
                state.selectedAnnotation = null;
                state.isDrawing = false;
                setTool('select');
                render();
                break;
            case 's':
                if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    saveAnnotations();
                }
                break;
            case 'arrowleft':
                navigateImage(-1);
                break;
            case 'arrowright':
                navigateImage(1);
                break;
            case '=':
            case '+':
                setZoom(state.zoom * 1.25);
                break;
            case '-':
                setZoom(state.zoom / 1.25);
                break;
        }
    });
}

function setupDragAndDrop() {
    const container = elements.canvasContainer;

    container.addEventListener('dragover', (e) => {
        e.preventDefault();
        container.classList.add('drag-over');
    });

    container.addEventListener('dragleave', () => {
        container.classList.remove('drag-over');
    });

    container.addEventListener('drop', async (e) => {
        e.preventDefault();
        container.classList.remove('drag-over');

        if (!state.currentDataset) {
            setStatus('Please select a dataset first', 'warning');
            return;
        }

        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        await uploadFiles(files);
    });
}

// ============ Dataset Functions ============
async function createDataset() {
    const name = elements.datasetNameInput.value.trim();
    if (!name) return;

    try {
        await api.createDataset(name);
        await loadDatasets();
        elements.datasetSelect.value = name;
        await selectDataset(name);
        hideModal();
        elements.datasetNameInput.value = '';
        setStatus(`Created dataset: ${name}`);
    } catch (err) {
        setStatus('Failed to create dataset', 'error');
    }
}

// ============ Label Functions ============
async function createLabel() {
    const name = elements.labelNameInput.value.trim();
    const color = elements.labelColorInput.value;
    if (!name || !state.currentDataset) return;

    const newId = Object.keys(state.classes).length;
    state.classes[newId] = { name, color };

    // Save to server
    const classNames = {};
    Object.entries(state.classes).forEach(([id, data]) => {
        classNames[id] = data.name;
    });
    await api.updateDataset(state.currentDataset, { classes: classNames });

    hideModal();
    elements.labelNameInput.value = '';
    renderLabelList();
    setStatus(`Added label: ${name}`);
}

function selectClass(classId) {
    state.selectedClass = classId;

    // If there's a selected annotation, change its class
    if (state.selectedAnnotation !== null) {
        state.annotations[state.selectedAnnotation].classId = classId;
        state.isDirty = true;
        render();
    }

    renderLabelList();
}

async function deleteClass(classId) {
    if (!confirm(`Delete label "${state.classes[classId].name}"? This will not remove existing annotations.`)) {
        return;
    }

    delete state.classes[classId];

    // Re-index classes
    const newClasses = {};
    let newId = 0;
    Object.values(state.classes).forEach(data => {
        newClasses[newId] = data;
        newId++;
    });
    state.classes = newClasses;

    // Save to server
    const classNames = {};
    Object.entries(state.classes).forEach(([id, data]) => {
        classNames[id] = data.name;
    });
    await api.updateDataset(state.currentDataset, { classes: classNames });

    renderLabelList();
}

// ============ Image Functions ============
async function handleFileUpload(e) {
    const files = Array.from(e.target.files);
    await uploadFiles(files);
    e.target.value = '';
}

async function uploadFiles(files) {
    if (!state.currentDataset) {
        setStatus('Please select a dataset first', 'warning');
        return;
    }

    for (const file of files) {
        setStatus(`Uploading ${file.name}...`);
        try {
            const split = elements.splitSelect.value;
            const image = await api.uploadImage(state.currentDataset, file, split);
            state.images.push(image);
        } catch (err) {
            setStatus(`Failed to upload ${file.name}`, 'error');
        }
    }

    renderImageList();
    setStatus(`Uploaded ${files.length} image(s)`);
}

async function selectImage(image) {
    // Save current if dirty
    if (state.isDirty && state.currentImage) {
        await saveAnnotations();
    }

    state.currentImage = image;
    state.selectedAnnotation = null;
    elements.splitSelect.value = image.split;

    // Load image
    const img = new Image();
    img.onload = async () => {
        state.imageElement = img;

        // Load annotations
        state.annotations = await api.getLabels(
            state.currentDataset,
            image.split,
            image.filename
        );

        state.isDirty = false;
        fitToView();
        render();
    };
    img.src = image.path;
}

async function deleteCurrentImage() {
    if (!state.currentImage) return;
    if (!confirm('Delete this image and its annotations?')) return;

    await api.deleteImage(
        state.currentDataset,
        state.currentImage.split,
        state.currentImage.filename
    );

    state.images = state.images.filter(i =>
        !(i.filename === state.currentImage.filename && i.split === state.currentImage.split)
    );

    state.currentImage = null;
    state.imageElement = null;
    state.annotations = [];

    render();
    setStatus('Image deleted');
}

function navigateImage(direction) {
    if (!state.currentImage || state.images.length === 0) return;

    const currentIndex = state.images.findIndex(i =>
        i.filename === state.currentImage.filename && i.split === state.currentImage.split
    );

    let newIndex = currentIndex + direction;
    if (newIndex < 0) newIndex = state.images.length - 1;
    if (newIndex >= state.images.length) newIndex = 0;

    selectImage(state.images[newIndex]);
}

async function handleSplitChange() {
    if (!state.currentImage) return;

    const newSplit = elements.splitSelect.value;
    if (newSplit === state.currentImage.split) return;

    // Save annotations first
    if (state.isDirty) {
        await saveAnnotations();
    }

    const result = await api.changeSplit(
        state.currentDataset,
        state.currentImage.split,
        state.currentImage.filename,
        newSplit
    );

    // Update local state
    const imgIndex = state.images.findIndex(i =>
        i.filename === state.currentImage.filename && i.split === state.currentImage.split
    );
    if (imgIndex !== -1) {
        state.images[imgIndex] = result.image;
        state.currentImage = result.image;
    }

    renderImageList();
    setStatus(`Moved to ${newSplit}`);
}

// ============ Webcam Functions ============
let webcamStream = null;

async function openWebcam() {
    if (!state.currentDataset) {
        setStatus('Please select a dataset first', 'warning');
        return;
    }

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 }
        });
        elements.webcamVideo.srcObject = webcamStream;
        showModal('webcam');
    } catch (err) {
        setStatus('Could not access webcam', 'error');
    }
}

async function captureWebcam() {
    const video = elements.webcamVideo;
    const canvas = elements.webcamCanvas;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
        const file = new File([blob], `webcam_${Date.now()}.jpg`, { type: 'image/jpeg' });
        await uploadFiles([file]);
        closeWebcam();
    }, 'image/jpeg', 0.95);
}

function closeWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    elements.webcamVideo.srcObject = null;
    hideModal();
}

// ============ Export Functions ============
async function openExportModal() {
    if (!state.currentDataset) {
        setStatus('Please select a dataset first', 'warning');
        return;
    }

    // Show modal with loading state
    elements.exportInfo.innerHTML = '<p>Loading dataset info...</p>';
    elements.exportWarning.classList.add('hidden');
    showModal('export');

    try {
        const stats = await api.getDatasetStats(state.currentDataset);

        // Build info table
        const totalImages = stats.splitCounts.train + stats.splitCounts.val + stats.splitCounts.test;
        const isLarge = stats.totalSize > 100 * 1024 * 1024; // > 100MB

        elements.exportInfo.innerHTML = `
            <table>
                <tr>
                    <td>Dataset</td>
                    <td><strong>${stats.name}</strong></td>
                </tr>
                <tr>
                    <td>Total Size</td>
                    <td class="${isLarge ? 'export-size-large' : ''}">${stats.totalSizeFormatted}</td>
                </tr>
                <tr>
                    <td>Total Files</td>
                    <td>${stats.fileCount}</td>
                </tr>
                <tr>
                    <td>Images</td>
                    <td>${totalImages} (train: ${stats.splitCounts.train}, val: ${stats.splitCounts.val}, test: ${stats.splitCounts.test})</td>
                </tr>
            </table>
        `;

        // Show warning for large datasets
        if (isLarge) {
            elements.exportWarning.classList.remove('hidden');
        }
    } catch (err) {
        elements.exportInfo.innerHTML = '<p class="error">Failed to load dataset info</p>';
    }
}

function confirmExport() {
    if (!state.currentDataset) return;

    setStatus('Preparing download...');
    hideModal();

    // Trigger download via hidden link
    const link = document.createElement('a');
    link.href = api.getExportUrl(state.currentDataset);
    link.download = `${state.currentDataset}.zip`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    setStatus('Download started');
}

// ============ Import Functions ============
let selectedImportFile = null;

function openImportModal() {
    selectedImportFile = null;
    elements.importFileInput.value = '';
    elements.importStatus.classList.add('hidden');
    elements.importStatus.textContent = '';
    elements.confirmImportBtn.disabled = true;
    elements.importDropzone.innerHTML = `
        <p>Drag & drop a .zip file here</p>
        <p class="hint">or click to browse</p>
    `;
    showModal('import');
}

function handleImportFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleImportFile(file);
    }
}

function handleImportFile(file) {
    if (!file.name.endsWith('.zip')) {
        elements.importStatus.textContent = 'Please select a .zip file';
        elements.importStatus.classList.remove('hidden');
        elements.importStatus.classList.add('error');
        return;
    }

    selectedImportFile = file;
    elements.importDropzone.innerHTML = `
        <p class="selected-file">${file.name}</p>
        <p class="hint">${formatSize(file.size)}</p>
    `;
    elements.importStatus.classList.add('hidden');
    elements.confirmImportBtn.disabled = false;
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
}

async function confirmImport() {
    if (!selectedImportFile) return;

    elements.confirmImportBtn.disabled = true;
    elements.importStatus.textContent = 'Importing dataset...';
    elements.importStatus.classList.remove('hidden', 'error');

    try {
        const result = await api.importDataset(selectedImportFile);

        if (result.error) {
            elements.importStatus.textContent = result.error;
            elements.importStatus.classList.add('error');
            elements.confirmImportBtn.disabled = false;
            return;
        }

        // Success - reload datasets and select the new one
        await loadDatasets();
        elements.datasetSelect.value = result.name;
        await selectDataset(result.name);

        hideModal();
        selectedImportFile = null;
        setStatus(`Imported dataset: ${result.name} (${result.imageCount} images)`);
    } catch (err) {
        elements.importStatus.textContent = 'Import failed: ' + err.message;
        elements.importStatus.classList.add('error');
        elements.confirmImportBtn.disabled = false;
    }
}

// ============ Annotation Functions ============
async function saveAnnotations() {
    if (!state.currentImage || !state.currentDataset) return;

    await api.saveLabels(
        state.currentDataset,
        state.currentImage.split,
        state.currentImage.filename,
        state.annotations
    );

    state.isDirty = false;

    // Update image in list
    const imgIndex = state.images.findIndex(i =>
        i.filename === state.currentImage.filename && i.split === state.currentImage.split
    );
    if (imgIndex !== -1) {
        state.images[imgIndex].annotationCount = state.annotations.length;
        state.images[imgIndex].hasAnnotations = state.annotations.length > 0;
    }

    renderImageList();
    setStatus('Saved');
}

function deleteSelectedAnnotation() {
    if (state.selectedAnnotation === null) return;

    state.annotations.splice(state.selectedAnnotation, 1);
    state.selectedAnnotation = null;
    state.isDirty = true;
    render();
}

function clearAllAnnotations() {
    if (state.annotations.length === 0) return;
    if (!confirm('Clear all annotations?')) return;

    state.annotations = [];
    state.selectedAnnotation = null;
    state.isDirty = true;
    render();
}

// ============ Canvas Functions ============
function setTool(tool) {
    state.tool = tool;
    elements.toolSelect.classList.toggle('active', tool === 'select');
    elements.toolDraw.classList.toggle('active', tool === 'draw');
    elements.canvas.style.cursor = tool === 'draw' ? 'crosshair' : 'default';
}

function setZoom(zoom) {
    state.zoom = Math.max(0.1, Math.min(5, zoom));
    elements.zoomLevel.textContent = Math.round(state.zoom * 100) + '%';
    render();
}

function fitToView() {
    if (!state.imageElement) return;

    const containerRect = elements.canvasContainer.getBoundingClientRect();
    const padding = 40;

    const scaleX = (containerRect.width - padding) / state.imageElement.width;
    const scaleY = (containerRect.height - padding) / state.imageElement.height;

    state.zoom = Math.min(scaleX, scaleY, 1);
    state.pan = { x: 0, y: 0 };
    elements.zoomLevel.textContent = Math.round(state.zoom * 100) + '%';
    render();
}

function getCanvasCoords(e) {
    const rect = elements.canvas.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left) / state.zoom,
        y: (e.clientY - rect.top) / state.zoom
    };
}

function getNormalizedCoords(pixelX, pixelY, width, height) {
    if (!state.imageElement) return null;
    return {
        x: (pixelX + width / 2) / state.imageElement.width,
        y: (pixelY + height / 2) / state.imageElement.height,
        width: Math.abs(width) / state.imageElement.width,
        height: Math.abs(height) / state.imageElement.height
    };
}

function getPixelCoords(ann) {
    if (!state.imageElement) return null;
    const w = ann.width * state.imageElement.width;
    const h = ann.height * state.imageElement.height;
    return {
        x: ann.x * state.imageElement.width - w / 2,
        y: ann.y * state.imageElement.height - h / 2,
        width: w,
        height: h
    };
}

// Check if point is on a resize handle, returns handle name or null
function getResizeHandle(coords, px) {
    const handleSize = 12 / state.zoom; // slightly larger hit area than visual
    const handles = {
        'nw': { x: px.x, y: px.y },
        'ne': { x: px.x + px.width, y: px.y },
        'sw': { x: px.x, y: px.y + px.height },
        'se': { x: px.x + px.width, y: px.y + px.height },
        'n':  { x: px.x + px.width / 2, y: px.y },
        's':  { x: px.x + px.width / 2, y: px.y + px.height },
        'w':  { x: px.x, y: px.y + px.height / 2 },
        'e':  { x: px.x + px.width, y: px.y + px.height / 2 },
    };

    for (const [name, pos] of Object.entries(handles)) {
        if (Math.abs(coords.x - pos.x) <= handleSize / 2 &&
            Math.abs(coords.y - pos.y) <= handleSize / 2) {
            return name;
        }
    }
    return null;
}

// Check if point is inside a box (for moving)
function isInsideBox(coords, px) {
    return coords.x >= px.x && coords.x <= px.x + px.width &&
           coords.y >= px.y && coords.y <= px.y + px.height;
}

// Get cursor style based on handle
function getHandleCursor(handle) {
    const cursors = {
        'nw': 'nwse-resize', 'se': 'nwse-resize',
        'ne': 'nesw-resize', 'sw': 'nesw-resize',
        'n': 'ns-resize', 's': 'ns-resize',
        'e': 'ew-resize', 'w': 'ew-resize',
    };
    return cursors[handle] || 'default';
}

function handleCanvasMouseDown(e) {
    if (!state.imageElement) return;

    const coords = getCanvasCoords(e);

    if (state.tool === 'draw') {
        state.isDrawing = true;
        state.drawStart = coords;
        state.selectedAnnotation = null;
    } else {
        // Select mode - first check if clicking on selected box's resize handle
        if (state.selectedAnnotation !== null) {
            const ann = state.annotations[state.selectedAnnotation];
            const px = getPixelCoords(ann);
            const handle = getResizeHandle(coords, px);

            if (handle) {
                // Start resizing
                state.isResizing = true;
                state.resizeHandle = handle;
                state.dragStart = coords;
                state.dragOriginal = { ...ann };
                return;
            }

            // Check if clicking inside selected box to move it
            if (isInsideBox(coords, px)) {
                state.isDragging = true;
                state.dragStart = coords;
                state.dragOriginal = { ...ann };
                return;
            }
        }

        // Check if clicking on any box to select it
        let found = false;
        for (let i = state.annotations.length - 1; i >= 0; i--) {
            const px = getPixelCoords(state.annotations[i]);
            if (isInsideBox(coords, px)) {
                state.selectedAnnotation = i;
                found = true;

                // Start moving immediately if clicking on a box
                state.isDragging = true;
                state.dragStart = coords;
                state.dragOriginal = { ...state.annotations[i] };
                break;
            }
        }
        if (!found) {
            state.selectedAnnotation = null;
        }
        render();
    }
}

function handleCanvasMouseMove(e) {
    const coords = getCanvasCoords(e);

    // Update cursor position display
    if (state.imageElement) {
        elements.cursorPosition.textContent = `${Math.round(coords.x)}, ${Math.round(coords.y)}`;
    }

    // Handle box moving
    if (state.isDragging && state.selectedAnnotation !== null) {
        const dx = (coords.x - state.dragStart.x) / state.imageElement.width;
        const dy = (coords.y - state.dragStart.y) / state.imageElement.height;

        const ann = state.annotations[state.selectedAnnotation];
        ann.x = Math.max(ann.width / 2, Math.min(1 - ann.width / 2, state.dragOriginal.x + dx));
        ann.y = Math.max(ann.height / 2, Math.min(1 - ann.height / 2, state.dragOriginal.y + dy));

        state.isDirty = true;
        render();
        return;
    }

    // Handle box resizing
    if (state.isResizing && state.selectedAnnotation !== null) {
        const ann = state.annotations[state.selectedAnnotation];
        const orig = state.dragOriginal;
        const handle = state.resizeHandle;

        // Calculate original box edges in normalized coords
        let left = orig.x - orig.width / 2;
        let right = orig.x + orig.width / 2;
        let top = orig.y - orig.height / 2;
        let bottom = orig.y + orig.height / 2;

        // Current mouse position in normalized coords
        const mouseX = coords.x / state.imageElement.width;
        const mouseY = coords.y / state.imageElement.height;

        // Adjust edges based on handle
        if (handle.includes('w')) left = Math.min(mouseX, right - 0.01);
        if (handle.includes('e')) right = Math.max(mouseX, left + 0.01);
        if (handle.includes('n')) top = Math.min(mouseY, bottom - 0.01);
        if (handle.includes('s')) bottom = Math.max(mouseY, top + 0.01);

        // Clamp to image bounds
        left = Math.max(0, left);
        right = Math.min(1, right);
        top = Math.max(0, top);
        bottom = Math.min(1, bottom);

        // Update annotation
        ann.x = (left + right) / 2;
        ann.y = (top + bottom) / 2;
        ann.width = right - left;
        ann.height = bottom - top;

        state.isDirty = true;
        render();
        return;
    }

    // Update cursor based on hover state (when not dragging)
    if (state.tool === 'select' && state.selectedAnnotation !== null) {
        const ann = state.annotations[state.selectedAnnotation];
        const px = getPixelCoords(ann);
        const handle = getResizeHandle(coords, px);

        if (handle) {
            elements.canvas.style.cursor = getHandleCursor(handle);
        } else if (isInsideBox(coords, px)) {
            elements.canvas.style.cursor = 'move';
        } else {
            elements.canvas.style.cursor = 'default';
        }
    } else if (state.tool === 'select') {
        // Check if hovering over any box
        let overBox = false;
        for (let i = state.annotations.length - 1; i >= 0; i--) {
            const px = getPixelCoords(state.annotations[i]);
            if (isInsideBox(coords, px)) {
                overBox = true;
                break;
            }
        }
        elements.canvas.style.cursor = overBox ? 'pointer' : 'default';
    }

    if (state.isDrawing && state.drawStart) {
        render();
        // Draw preview box (apply zoom transform)
        const x = Math.min(state.drawStart.x, coords.x);
        const y = Math.min(state.drawStart.y, coords.y);
        const w = Math.abs(coords.x - state.drawStart.x);
        const h = Math.abs(coords.y - state.drawStart.y);

        ctx.save();
        ctx.scale(state.zoom, state.zoom);
        ctx.strokeStyle = state.classes[state.selectedClass]?.color || '#ff0000';
        ctx.lineWidth = 2 / state.zoom;
        ctx.setLineDash([5 / state.zoom, 5 / state.zoom]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
        ctx.restore();
    }
}

function handleCanvasMouseUp(e) {
    // End dragging
    if (state.isDragging) {
        state.isDragging = false;
        state.dragStart = null;
        state.dragOriginal = null;
        render();
        return;
    }

    // End resizing
    if (state.isResizing) {
        state.isResizing = false;
        state.resizeHandle = null;
        state.dragStart = null;
        state.dragOriginal = null;
        render();
        return;
    }

    // End drawing
    if (!state.isDrawing) return;

    const coords = getCanvasCoords(e);
    const x = Math.min(state.drawStart.x, coords.x);
    const y = Math.min(state.drawStart.y, coords.y);
    const w = Math.abs(coords.x - state.drawStart.x);
    const h = Math.abs(coords.y - state.drawStart.y);

    // Only create box if it has some size
    if (w > 5 && h > 5) {
        const normalized = getNormalizedCoords(x, y, w, h);
        state.annotations.push({
            classId: state.selectedClass,
            ...normalized
        });
        state.isDirty = true;
        state.selectedAnnotation = state.annotations.length - 1;
    }

    state.isDrawing = false;
    state.drawStart = null;
    render();
}

function handleCanvasWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(state.zoom * delta);
}

// ============ Rendering ============
function render() {
    renderImageList();
    renderLabelList();
    renderCanvas();
    renderAnnotationList();
}

function renderDatasetSelect() {
    elements.datasetSelect.innerHTML = '<option value="">Select dataset...</option>';
    state.datasets.forEach(d => {
        const option = document.createElement('option');
        option.value = d.name;
        option.textContent = `${d.name} (${d.imageCount} images)`;
        elements.datasetSelect.appendChild(option);
    });
}

function renderImageList() {
    if (state.images.length === 0) {
        elements.imageList.innerHTML = '<p class="empty-message">No images in dataset</p>';
        return;
    }

    elements.imageList.innerHTML = state.images.map(img => {
        const isActive = state.currentImage &&
            state.currentImage.filename === img.filename &&
            state.currentImage.split === img.split;

        return `
            <div class="image-item ${isActive ? 'active' : ''}"
                 data-filename="${img.filename}"
                 data-split="${img.split}">
                <img class="thumbnail" src="${img.path}" alt="">
                <div class="info">
                    <div class="filename">${img.filename}</div>
                    <div class="meta">
                        <span>${img.split}</span>
                        <span class="badge ${img.hasAnnotations ? 'badge-annotated' : 'badge-empty'}">
                            ${img.annotationCount || 0} boxes
                        </span>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Add click handlers
    elements.imageList.querySelectorAll('.image-item').forEach(item => {
        item.addEventListener('click', () => {
            const img = state.images.find(i =>
                i.filename === item.dataset.filename && i.split === item.dataset.split
            );
            if (img) selectImage(img);
        });
    });
}

function renderLabelList() {
    if (Object.keys(state.classes).length === 0) {
        elements.labelList.innerHTML = '<p class="empty-message">No labels defined</p>';
        return;
    }

    elements.labelList.innerHTML = Object.entries(state.classes).map(([id, data]) => `
        <div class="label-item ${state.selectedClass === parseInt(id) ? 'active' : ''}"
             data-id="${id}">
            <span class="color-dot" style="background: ${data.color}"></span>
            <span class="label-name">${data.name}</span>
            <span class="shortcut">${parseInt(id) + 1}</span>
            <button class="delete-label" data-id="${id}">&times;</button>
        </div>
    `).join('');

    // Add click handlers
    elements.labelList.querySelectorAll('.label-item').forEach(item => {
        item.addEventListener('click', (e) => {
            if (!e.target.classList.contains('delete-label')) {
                selectClass(parseInt(item.dataset.id));
            }
        });
    });

    elements.labelList.querySelectorAll('.delete-label').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteClass(parseInt(btn.dataset.id));
        });
    });
}

function renderAnnotationList() {
    if (state.annotations.length === 0) {
        elements.annotationList.innerHTML = '<p class="empty-message">No annotations</p>';
        return;
    }

    const classOptions = Object.entries(state.classes)
        .map(([id, data]) => `<option value="${id}">${data.name}</option>`)
        .join('');

    elements.annotationList.innerHTML = state.annotations.map((ann, i) => {
        const classData = state.classes[ann.classId] || { name: `Class ${ann.classId}`, color: '#888' };
        return `
            <div class="annotation-item ${state.selectedAnnotation === i ? 'selected' : ''}"
                 data-index="${i}">
                <span class="color-dot" style="background: ${classData.color}"></span>
                <select class="annotation-class-select" data-index="${i}">
                    ${Object.entries(state.classes).map(([id, data]) =>
                        `<option value="${id}" ${parseInt(id) === ann.classId ? 'selected' : ''}>${data.name}</option>`
                    ).join('')}
                </select>
                <button class="delete-annotation" data-index="${i}" title="Delete">&times;</button>
            </div>
        `;
    }).join('');

    // Add click handlers for selection
    elements.annotationList.querySelectorAll('.annotation-item').forEach(item => {
        item.addEventListener('click', (e) => {
            if (e.target.tagName === 'SELECT' || e.target.tagName === 'BUTTON') return;
            state.selectedAnnotation = parseInt(item.dataset.index);
            render();
        });
    });

    // Add change handlers for class dropdown
    elements.annotationList.querySelectorAll('.annotation-class-select').forEach(select => {
        select.addEventListener('change', (e) => {
            const idx = parseInt(select.dataset.index);
            state.annotations[idx].classId = parseInt(e.target.value);
            state.isDirty = true;
            render();
        });
        select.addEventListener('click', (e) => e.stopPropagation());
    });

    // Add delete handlers
    elements.annotationList.querySelectorAll('.delete-annotation').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const idx = parseInt(btn.dataset.index);
            state.annotations.splice(idx, 1);
            if (state.selectedAnnotation === idx) {
                state.selectedAnnotation = null;
            } else if (state.selectedAnnotation > idx) {
                state.selectedAnnotation--;
            }
            state.isDirty = true;
            render();
        });
    });
}

function renderCanvas() {
    if (!state.imageElement) {
        elements.noImageMessage.classList.remove('hidden');
        elements.canvas.classList.add('hidden');
        return;
    }

    elements.noImageMessage.classList.add('hidden');
    elements.canvas.classList.remove('hidden');

    // Set canvas size
    elements.canvas.width = state.imageElement.width * state.zoom;
    elements.canvas.height = state.imageElement.height * state.zoom;

    ctx.save();
    ctx.scale(state.zoom, state.zoom);

    // Draw image
    ctx.drawImage(state.imageElement, 0, 0);

    // Draw annotations
    state.annotations.forEach((ann, i) => {
        const px = getPixelCoords(ann);
        const classData = state.classes[ann.classId] || { name: `Class ${ann.classId}`, color: '#888' };
        const isSelected = state.selectedAnnotation === i;

        // Box
        ctx.strokeStyle = classData.color;
        ctx.lineWidth = (isSelected ? 3 : 2) / state.zoom;
        ctx.strokeRect(px.x, px.y, px.width, px.height);

        // Fill with transparency
        ctx.fillStyle = classData.color + '20';
        ctx.fillRect(px.x, px.y, px.width, px.height);

        // Label background
        const labelText = classData.name;
        ctx.font = `${12 / state.zoom}px sans-serif`;
        const textWidth = ctx.measureText(labelText).width;
        const labelHeight = 16 / state.zoom;
        const padding = 4 / state.zoom;

        ctx.fillStyle = classData.color;
        ctx.fillRect(px.x, px.y - labelHeight, textWidth + padding * 2, labelHeight);

        // Label text
        ctx.fillStyle = 'white';
        ctx.fillText(labelText, px.x + padding, px.y - 4 / state.zoom);

        // Selection handles
        if (isSelected) {
            const handleSize = 8 / state.zoom;
            ctx.fillStyle = classData.color;
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 1 / state.zoom;

            // Corner handles
            const corners = [
                { x: px.x, y: px.y },                           // nw
                { x: px.x + px.width, y: px.y },                // ne
                { x: px.x, y: px.y + px.height },               // sw
                { x: px.x + px.width, y: px.y + px.height },    // se
            ];
            corners.forEach(c => {
                ctx.fillRect(c.x - handleSize / 2, c.y - handleSize / 2, handleSize, handleSize);
                ctx.strokeRect(c.x - handleSize / 2, c.y - handleSize / 2, handleSize, handleSize);
            });

            // Edge handles (smaller)
            const edgeSize = 6 / state.zoom;
            const edges = [
                { x: px.x + px.width / 2, y: px.y },                    // n
                { x: px.x + px.width / 2, y: px.y + px.height },        // s
                { x: px.x, y: px.y + px.height / 2 },                   // w
                { x: px.x + px.width, y: px.y + px.height / 2 },        // e
            ];
            edges.forEach(c => {
                ctx.fillRect(c.x - edgeSize / 2, c.y - edgeSize / 2, edgeSize, edgeSize);
                ctx.strokeRect(c.x - edgeSize / 2, c.y - edgeSize / 2, edgeSize, edgeSize);
            });
        }
    });

    ctx.restore();
}

// ============ Modal Functions ============
function showModal(type) {
    elements.modalOverlay.classList.remove('hidden');
    document.querySelectorAll('.modal').forEach(m => m.classList.add('hidden'));

    if (type === 'new-dataset') {
        elements.newDatasetModal.classList.remove('hidden');
        elements.datasetNameInput.focus();
    } else if (type === 'add-label') {
        elements.addLabelModal.classList.remove('hidden');
        elements.labelNameInput.focus();
    } else if (type === 'webcam') {
        elements.webcamModal.classList.remove('hidden');
    } else if (type === 'export') {
        elements.exportModal.classList.remove('hidden');
    } else if (type === 'import') {
        elements.importModal.classList.remove('hidden');
    }
}

function hideModal() {
    elements.modalOverlay.classList.add('hidden');
    document.querySelectorAll('.modal').forEach(m => m.classList.add('hidden'));
}

// ============ Status ============
function setStatus(message, type = 'info') {
    elements.statusMessage.textContent = message;
    elements.statusMessage.style.color = type === 'error' ? 'var(--danger-color)' :
        type === 'warning' ? 'var(--warning-color)' : 'var(--text-secondary)';
}

// ============ Start ============
init();
