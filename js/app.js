/**
 * Main Application - On-Device RAG Pipeline
 * Orchestrates all components: chunking, embedding, retrieval, and generation
 */

import { chunkText, chunkByLines, getChunkStats } from './chunker.js';
import { initVectorStore, addVectors, getVectors, clearVectors, exportBinaryGenerator, importVectors, getVectorCount, getStoreStats, saveDocuments, loadDocuments, clearDocuments, removeVectorsByDocId } from './vectorStore.js';
import { initEmbedder, embed, embedBatch, isEmbedderReady, getEmbeddingModelName } from './embedder.js';
import { hybridSearch, semanticSearch, preprocessQuery } from './retriever.js';
import { initLLM, generate, isLLMReady, checkWebGPU, getModelName, AVAILABLE_MODELS, switchModel, getCurrentModelInfo } from './llm.js';
import { processFile, isPdf } from './pdfParser.js';

// DOM Elements

const dropZoneEl = document.getElementById('drop-zone');
const fileInputEl = document.getElementById('file-input');
const documentGridEl = document.getElementById('document-grid');
const corpusEl = document.getElementById('corpus');
const addTextBtn = document.getElementById('btn-add-text');
const ingestBtn = document.getElementById('btn-ingest');
const clearBtn = document.getElementById('btn-clear');
const exportBtn = document.getElementById('btn-export');
const importBtn = document.getElementById('btn-import');
const loadDemoBtn = document.getElementById('btn-load-demo');
const importFileEl = document.getElementById('import-file');

const queryEl = document.getElementById('query');
const askBtn = document.getElementById('btn-ask');
const searchModeEl = document.getElementById('search-mode');

const outputEl = document.getElementById('output');
const contextEl = document.getElementById('context');
const metricsEl = document.getElementById('metrics');
const chunkStatsEl = document.getElementById('chunk-stats');

// Modal Elements
const modalOverlayEl = document.getElementById('confirm-modal');
const modalTitleEl = document.getElementById('modal-title');
const modalMessageEl = document.getElementById('modal-message');
const modalCancelBtn = document.getElementById('btn-modal-cancel');
const modalConfirmBtn = document.getElementById('btn-modal-confirm');

// State
let llmAvailable = false;
let documents = []; // Array of { id, name, text, type, pageCount }
let documentIdCounter = 0;
let modalResolver = null; // Promise resolver for modal

// Modal Event Listeners
modalCancelBtn.addEventListener('click', () => {
    if (modalResolver) modalResolver(false);
    hideModal();
});

modalConfirmBtn.addEventListener('click', () => {
    if (modalResolver) modalResolver(true);
    hideModal();
});

function hideModal() {
    modalOverlayEl.classList.add('hidden');
    modalResolver = null;
}

/**
 * Show confirmation modal
 * @param {string} title 
 * @param {string} message 
 * @param {string} confirmText 
 * @returns {Promise<boolean>}
 */
function showConfirm(title, message, confirmText = 'Confirm') {
    modalTitleEl.textContent = title;
    modalMessageEl.textContent = message;
    modalConfirmBtn.textContent = confirmText;
    modalOverlayEl.classList.remove('hidden');
    
    return new Promise((resolve) => {
        modalResolver = resolve;
    });
}



/**
 * Initialize the application
 */
async function init() {
    showToast('Initializing...', 5, 'loading');
    
    try {
        // 1. Initialize vector store (load from IndexedDB)
        await initVectorStore();
        const existingCount = getVectorCount();
        
        // 1b. Restore saved documents BEFORE updating stats
        const savedDocs = await loadDocuments();
        if (savedDocs.length > 0) {
            documents = savedDocs;
            documentIdCounter = Math.max(...savedDocs.map(d => parseInt(d.id.replace('doc-', '')) || 0)) + 1;
            savedDocs.forEach(doc => addDocumentToGrid(doc));
        }
        
        // 1c. Update chunk stats AFTER documents are loaded
        if (existingCount > 0) {
            updateChunkStats();
        }
        
        // 2. Check WebGPU availability
        const hasWebGPU = await checkWebGPU();
        if (!hasWebGPU) {
            document.getElementById('webgpu-warning').classList.remove('hidden');
        }
        
        // 3. Initialize embedding model
        showToast('Loading embedding model...', 20);
        await initEmbedder((msg) => {
            showToast(msg, 30);
        });
        
        // 4. Initialize LLM (if WebGPU available)
        if (hasWebGPU) {
            showToast('Loading LLM model...', 50);
            llmAvailable = await initLLM((msg) => {
                // Extract progress from message
                const match = msg.match(/(\d+)%/);
                const progress = match ? 50 + parseInt(match[1]) * 0.5 : 60;
                showToast(msg, progress);
            });
        }
        
        // 5. Update footer with model info
        updateFooter();
        
        // 6. Enable UI
        enableUI();
        setupDragDrop();
        setupEventListeners();
        setupModelSelector();
        setupStatusModal();
                
        if (existingCount > 0) {
            showToast(`Ready! ${existingCount} chunks loaded.`, 100, 'ready');
        } else {
            showToast('Ready! Drop files to begin.', 100, 'ready');
        }
        setTimeout(hideToast, 2000);
        
    } catch (error) {
        showToast(`Initialization failed: ${error.message}`, 0, 'error');
        console.error('Init error:', error);
    }
}

/**
 * Update footer with current model names (clickable for model selection)
 */
function updateFooter() {
    const footerModelsEl = document.getElementById('footer-models');
    if (footerModelsEl) {
        const embModel = getEmbeddingModelName();
        if (llmAvailable) {
            const modelInfo = getCurrentModelInfo();
            footerModelsEl.innerHTML = `Embeddings: ${embModel} | LLM: <strong>${modelInfo.name}</strong> ⚙️`;
        } else {
            footerModelsEl.textContent = `Embeddings: ${embModel} | LLM: N/A`;
        }
    }
}

/**
 * Setup model selector click handler
 */
function setupModelSelector() {
    const footerModelsEl = document.getElementById('footer-models');
    const modelModal = document.getElementById('model-modal');
    const modelOptions = document.getElementById('model-options');
    const modelCancelBtn = document.getElementById('btn-model-cancel');
    
    if (!footerModelsEl || !modelModal || !modelOptions) return;
    
    // Click on footer to open model selector
    footerModelsEl.addEventListener('click', () => {
        if (!llmAvailable) {
            showConfirm('LLM Unavailable', 'WebGPU is not available. Model selection requires WebGPU support.', 'OK');
            return;
        }
        
        renderModelOptions();
        modelModal.classList.remove('hidden');
    });
    
    // Cancel button
    modelCancelBtn.addEventListener('click', () => {
        modelModal.classList.add('hidden');
    });
    
    // Click outside to close
    modelModal.addEventListener('click', (e) => {
        if (e.target === modelModal) {
            modelModal.classList.add('hidden');
        }
    });
}

/**
 * Render model options in the modal
 */
function renderModelOptions() {
    const modelOptions = document.getElementById('model-options');
    const currentModelId = getModelName();
    
    modelOptions.innerHTML = AVAILABLE_MODELS.map((model, index) => {
        const isActive = model.id === currentModelId;
        const isFirst = index === 0;
        
        return `
            <div class="model-option ${isActive ? 'active' : ''}" data-model-id="${model.id}">
                <span class="model-option-icon">${isFirst ? '🚀' : index === 1 ? '⚖️' : '🧠'}</span>
                <div class="model-option-info">
                    <div class="model-option-name">
                        ${model.name}
                        ${isFirst ? '<span class="badge recommended">Default</span>' : ''}
                        ${index === 2 ? '<span class="badge">Best</span>' : ''}
                    </div>
                    <div class="model-option-details">
                        ${model.size} • ${model.context} • ${model.description}
                    </div>
                </div>
                ${isActive ? '<span class="model-option-status">✓ Active</span>' : ''}
            </div>
        `;
    }).join('');
    
    // Add click handlers
    modelOptions.querySelectorAll('.model-option').forEach(option => {
        option.addEventListener('click', () => handleModelSelect(option.dataset.modelId));
    });
}

/**
 * Handle model selection
 */
async function handleModelSelect(modelId) {
    const currentModelId = getModelName();
    if (modelId === currentModelId) {
        document.getElementById('model-modal').classList.add('hidden');
        return;
    }
    
    const modelInfo = AVAILABLE_MODELS.find(m => m.id === modelId);
    
    // Show loading state
    const option = document.querySelector(`[data-model-id="${modelId}"]`);
    if (option) {
        option.classList.add('loading');
        option.querySelector('.model-option-status')?.remove();
        option.insertAdjacentHTML('beforeend', '<span class="model-option-status loading">⏳ Loading...</span>');
    }
    
    showToast(`Switching to ${modelInfo.name}...`, 10, 'loading');
    
    try {
        const success = await switchModel(modelId, (msg) => {
            showToast(msg, 50);
        });
        
        if (success) {
            showToast(`${modelInfo.name} ready!`, 100, 'ready');
            setTimeout(hideToast, 2000);
            updateFooter();
        } else {
            throw new Error('Model switch failed');
        }
    } catch (error) {
        showToast(`Failed to load ${modelInfo.name}`, 0, 'error');
        console.error('Model switch error:', error);
    }
    
    document.getElementById('model-modal').classList.add('hidden');
    renderModelOptions(); // Refresh to show new active state
}

/**
 * Enable UI elements after initialization
 */
function enableUI() {
    ingestBtn.disabled = false;
    clearBtn.disabled = false;
    exportBtn.disabled = false;
    importBtn.disabled = false;
    loadDemoBtn.disabled = false;
    queryEl.disabled = false;
    askBtn.disabled = false;
    searchModeEl.disabled = false;
    
    if (!llmAvailable) {
        askBtn.textContent = '🔍 Search Only';
        searchModeEl.innerHTML = '<option value="hybrid">Hybrid Search</option><option value="semantic">Semantic Only</option>';
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    addTextBtn.addEventListener('click', addTextDocument);
    ingestBtn.addEventListener('click', ingestData);
    clearBtn.addEventListener('click', handleClear);
    exportBtn.addEventListener('click', handleExport);
    importBtn.addEventListener('click', handleImportClick);
    importFileEl.addEventListener('change', handleImport);
    askBtn.addEventListener('click', askQuestion);
    loadDemoBtn.addEventListener('click', loadDemoData);
    
    // Enter key support for query
    queryEl.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !askBtn.disabled) {
            askQuestion();
        }
    });

    // Enter key for manual text
    corpusEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            addTextDocument();
        }
    });
}

/**
 * Setup drag and drop handlers
 */
function setupDragDrop() {
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZoneEl.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when dragging over
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZoneEl.addEventListener(eventName, () => {
            dropZoneEl.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZoneEl.addEventListener(eventName, () => {
            dropZoneEl.classList.remove('drag-over');
        }, false);
    });

    // Handle dropped files
    dropZoneEl.addEventListener('drop', handleDrop, false);
    
    // Handle click to browse
    dropZoneEl.addEventListener('click', () => fileInputEl.click());
    fileInputEl.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFiles(Array.from(e.target.files));
        }
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

async function handleDrop(e) {
    const files = Array.from(e.dataTransfer.files);
    await handleFiles(files);
}

/**
 * Process dropped/selected files
 */
async function handleFiles(files) {
    const ONE_MB = 1024 * 1024;
    
    for (const file of files) {
        // Warn for files larger than 1MB
        if (file.size > ONE_MB) {
            const sizeMB = (file.size / ONE_MB).toFixed(1);
            const confirmed = await showConfirm(
                'Large File Warning',
                `"${file.name}" is ${sizeMB}MB. Large files may take longer to process and use more memory. Continue?`,
                'Continue'
            );
            if (!confirmed) {
                showToast(`Skipped ${file.name}`, 100, 'warning');
                setTimeout(hideToast, 1500);
                continue;
            }
        }
        
        const docId = `doc-${documentIdCounter++}`;
        
        // Add placeholder to grid
        addDocumentToGrid({
            id: docId,
            name: file.name,
            type: getFileType(file.name),
            processing: true
        });
        
        try {
            showToast(`Processing ${file.name}...`, 10, 'loading');
            const result = await processFile(file);
            
            // Update document
            const doc = {
                id: docId,
                name: result.name,
                text: result.text,
                type: getFileType(file.name),
                pageCount: result.pageCount,
                processing: false
            };
            
            // Replace in documents array
            const existingIdx = documents.findIndex(d => d.id === docId);
            if (existingIdx >= 0) {
                documents[existingIdx] = doc;
            } else {
                documents.push(doc);
            }
            
            updateDocumentInGrid(doc);
            showToast(`Added ${file.name}`, 100, 'ready');
            setTimeout(hideToast, 1500);
            
        } catch (error) {
            console.error(`Error processing ${file.name}:`, error);
            removeDocumentFromGrid(docId);
            showToast(`Failed to process ${file.name}: ${error.message}`, 0, 'error');
        }
    }
    
    updateDocumentCount();
}

/**
 * Get file type from filename
 */
function getFileType(filename) {
    const ext = filename.toLowerCase().split('.').pop();
    if (ext === 'pdf') return 'pdf';
    if (ext === 'md') return 'md';
    return 'txt';
}

/**
 * Add document to grid (desktop icon style)
 */
function addDocumentToGrid(doc) {
    const item = document.createElement('div');
    // Include ingested class if document was already ingested (restored from DB)
    const classes = ['document-item'];
    if (doc.processing) classes.push('processing');
    if (doc.ingested) classes.push('ingested');
    item.className = classes.join(' ');
    item.id = `grid-${doc.id}`;
    item.innerHTML = `
        <button class="document-delete" title="Remove document" style="pointer-events: auto;">×</button>
        <span class="document-icon ${doc.type}">${getFileIcon(doc.type)}</span>
        <span class="document-name">${escapeHtml(doc.name)}</span>
        ${doc.pageCount ? `<span class="document-meta">${doc.pageCount} page${doc.pageCount > 1 ? 's' : ''}</span>` : ''}
    `;
    
    // Delete button handler
    const deleteBtn = item.querySelector('.document-delete');
    deleteBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        e.preventDefault();
        removeDocument(doc.id);
    });
    
    documentGridEl.appendChild(item);
    
    // Also add to documents array if not processing placeholder
    if (!doc.processing && !documents.find(d => d.id === doc.id)) {
        documents.push(doc);
    }
}

/**
 * Update document in grid
 */
function updateDocumentInGrid(doc) {
    const item = document.getElementById(`grid-${doc.id}`);
    if (!item) return;
    
    item.className = 'document-item';
    item.innerHTML = `
        <button class="document-delete" title="Remove document">×</button>
        <span class="document-icon ${doc.type}">${getFileIcon(doc.type)}</span>
        <span class="document-name">${escapeHtml(doc.name)}</span>
        ${doc.pageCount ? `<span class="document-meta">${doc.pageCount} page${doc.pageCount > 1 ? 's' : ''}</span>` : ''}
    `;
    
    item.querySelector('.document-delete').addEventListener('click', (e) => {
        e.stopPropagation();
        removeDocument(doc.id);
    });
}

/**
 * Remove document from grid
 */
function removeDocumentFromGrid(docId) {
    const item = document.getElementById(`grid-${docId}`);
    if (item) {
        item.remove();
    }
}

/**
 * Remove document completely
 */
async function removeDocument(docId) {
    // Find if document was ingested
    const doc = documents.find(d => d.id === docId);
    
    // If doc not found in array (unexpected), define fallback
    const wasIngested = doc?.ingested || false;
    const docName = doc?.name || 'Document';
    
    // Show confirmation for ingested documents (they have indexed chunks)
    if (wasIngested) {
        // Use custom modal instead of native confirm
        const confirmed = await showConfirm(
            'Remove Document?', 
            `Remove "${docName}"? This will also delete its indexed chunks.`,
            'Delete'
        );
        
        if (!confirmed) return;
    }
    
    // Remove from documents array
    documents = documents.filter(d => d.id !== docId);
    removeDocumentFromGrid(docId);
    
    // If it was ingested, also remove vectors from IndexedDB
    if (wasIngested) {
        showToast('Removing chunks...', 50);
        await removeVectorsByDocId(docId);
        updateChunkStats();
    }
    
    // Persist document list
    await saveDocuments(documents);
    
    updateDocumentCount();
    showToast('Document removed', 100);
    setTimeout(hideToast, 1500);
}

/**
 * Get file type label for modern CSS icons
 */
function getFileIcon(type) {
    // Returns a span with type label for CSS styling
    switch (type) {
        case 'pdf': return '<span>PDF</span>';
        case 'md': return '<span>MD</span>';
        default: return '<span>TXT</span>';
    }
}

// Toast elements
const toastEl = document.getElementById('toast');
const toastMessageEl = document.getElementById('toast-message');
const toastProgressEl = document.getElementById('toast-progress');
const toastIndicatorEl = document.getElementById('toast-indicator');
const toastTitleEl = document.getElementById('toast-title');
const toastCloseBtn = document.getElementById('toast-close');

// Toast close button handler
toastCloseBtn.addEventListener('click', () => {
    hideToast();
});

/**
 * Show toast notification with state indicator
 * @param {string} message - Message to display
 * @param {number} progress - Progress percentage (0-100)
 * @param {string} type - State type: 'loading', 'ready', 'warning', 'error'
 * @param {string} title - Optional title override
 */
function showToast(message, progress = 0, type = 'loading', title = null) {
    // Remove all state classes first
    toastEl.classList.remove('hidden', 'loading', 'ready', 'warning', 'error');
    toastIndicatorEl.classList.remove('loading', 'ready', 'warning', 'error');
    
    // Add current state class
    toastEl.classList.add(type);
    toastIndicatorEl.classList.add(type);
    
    // Set content
    toastMessageEl.textContent = message;
    toastProgressEl.style.width = `${progress}%`;
    
    // Set title based on type if not provided
    if (title) {
        toastTitleEl.textContent = title;
    } else {
        switch(type) {
            case 'loading': toastTitleEl.textContent = 'Processing'; break;
            case 'ready': toastTitleEl.textContent = 'Ready'; break;
            case 'warning': toastTitleEl.textContent = 'Warning'; break;
            case 'error': toastTitleEl.textContent = 'Error'; break;
            default: toastTitleEl.textContent = 'Status';
        }
    }
    
    // Show close button only for errors (user must dismiss)
    if (type === 'error') {
        toastCloseBtn.classList.remove('hidden');
    } else {
        toastCloseBtn.classList.add('hidden');
    }
}

/**
 * Hide toast notification
 */
function hideToast() {
    toastEl.classList.add('hidden');
    toastCloseBtn.classList.add('hidden');
}

/**
 * Update document count display
 * Uses vector store stats for ingested docs (since text is cleared after ingestion)
 */
function updateDocumentCount() {
    const count = documents.length;
    const ingestedCount = documents.filter(d => d.ingested).length;
    
    if (count === 0) {
        chunkStatsEl.innerHTML = '<span class="stat">No documents</span>';
        return;
    }
    
    // For ingested docs, get chars from vector store; for pending docs, sum text length
    const storeStats = getStoreStats();
    const pendingChars = documents
        .filter(d => !d.ingested)
        .reduce((sum, d) => sum + (d.text?.length || 0), 0);
    const totalChars = storeStats.totalChars + pendingChars;
    
    // If we have ingested docs, show chunk info too
    if (ingestedCount > 0 && storeStats.vectorCount > 0) {
        chunkStatsEl.innerHTML = `
            <span class="stat"><strong>${count}</strong> doc${count !== 1 ? 's' : ''}</span>
            <span class="stat"><strong>${storeStats.vectorCount}</strong> chunks</span>
            <span class="stat"><strong>${totalChars.toLocaleString()}</strong> chars</span>
        `;
    } else {
        chunkStatsEl.innerHTML = `
            <span class="stat"><strong>${count}</strong> doc${count !== 1 ? 's' : ''}</span>
            <span class="stat"><strong>${totalChars.toLocaleString()}</strong> chars</span>
        `;
    }
}

/**
 * Add text from textarea as a document
 */
function addTextDocument() {
    const text = corpusEl.value.trim();
    if (!text) {
        showToast('Please enter some text.', 0, 'warning');
        setTimeout(hideToast, 2000);
        return;
    }
    
    const docId = `doc-${documentIdCounter++}`;
    const doc = {
        id: docId,
        name: `Text ${documents.filter(d => d.name.startsWith('Text')).length + 1}`,
        text: text,
        type: 'txt',
        pageCount: 1
    };
    
    documents.push(doc);
    addDocumentToGrid(doc);
    corpusEl.value = '';
    updateDocumentCount();
    showToast('Text added as document.', 100, 'ready');
    setTimeout(hideToast, 1500);
}

/**
 * Ingest all documents
 */
/**
 * Ingest data (process documents)
 */
async function ingestData() {
    // Collect all text from non-ingested documents
    const newDocs = documents.filter(d => !d.ingested);

    if (newDocs.length === 0) {
        setStatus('No new documents to ingest.', 'info');
        showToast('No new documents to ingest', 100);
        setTimeout(hideToast, 2000);
        return;
    }
    
    // Show toast IMMEDIATELY before any processing
    showToast('Preparing documents...', 0);
    setStatus('Processing documents...', 'processing');
    ingestBtn.disabled = true;
    
    // Yield to UI so toast renders before heavy work
    await new Promise(r => setTimeout(r, 50));
    
    try {
        const allChunks = [];
        const totalDocs = newDocs.length;
        
        // Chunk each new document
        for (let i = 0; i < totalDocs; i++) {
            const doc = newDocs[i];
            const charCount = doc.text?.length || 0;
            const docName = doc.name.length > 20 ? doc.name.substring(0, 20) + '...' : doc.name;
            
            showToast(`Chunking "${docName}" (${(charCount / 1024).toFixed(0)}KB)...`, Math.round((i / totalDocs) * 20));
            
            // Use the async chunker with progress callback
            const chunks = await chunkText(doc.text, { 
                chunkSize: 512, 
                chunkOverlap: 50,
                onProgress: (completed, total) => {
                    const baseProgress = Math.round((i / totalDocs) * 20);
                    const chunkProgress = Math.round((completed / total) * (20 / totalDocs));
                    showToast(`Chunking "${docName}"... ${completed}/${total} chunks`, baseProgress + chunkProgress);
                }
            });
            
            // Prefix chunk IDs with docId so we can delete them later
            for (let j = 0; j < chunks.length; j++) {
                chunks[j].id = `${doc.id}-${chunks[j].id}`;
                chunks[j].source = doc.name;
                allChunks.push(chunks[j]);
            }
            
            // Mark as processing in grid
            const item = document.getElementById(`grid-${doc.id}`);
            if (item) item.classList.add('processing');
            
            showToast(`Chunked "${docName}" → ${chunks.length} chunks`, Math.round(((i + 1) / totalDocs) * 20));
            await new Promise(r => setTimeout(r, 10));
        }
        
        if (allChunks.length === 0) {
            setStatus('No valid chunks created.', 'warning');
            hideToast();
            return;
        }
        
        showToast(`Embedding 0/${allChunks.length} chunks...`, 5);
        
        // Embed each chunk
        const vectors = [];
        for (let i = 0; i < allChunks.length; i++) {
            const progress = 5 + Math.round((i / allChunks.length) * 90);
            showToast(`Embedding ${i + 1}/${allChunks.length} chunks...`, progress);
            
            // Small delay to allow browser to repaint the toast
            await new Promise(r => setTimeout(r, 10));
            
            const embedding = await embed(allChunks[i].text);
            vectors.push({
                id: allChunks[i].id,
                text: allChunks[i].text,
                source: allChunks[i].source,
                embedding
            });
        }
        
        showToast('Saving to database...', 98);
        
        // Store vectors
        await addVectors(vectors);
        
        // Mark all documents as ingested and CLEAR TEXT to save memory
        documents.forEach(doc => {
            if (doc.ingested) { 
                 doc.text = null; // Clear existing text too
            }
        });
        
        newDocs.forEach(doc => {
             doc.ingested = true;
             doc.text = null; // Clear text to save memory
             const item = document.getElementById(`grid-${doc.id}`);
             if (item) {
                 item.classList.remove('processing');
                 item.classList.add('ingested');
             }
        });
        
        // Persist documents to IndexedDB
        await saveDocuments(documents);
        
        showToast(`✓ Done! ${allChunks.length} chunks from ${newDocs.length} new docs`, 100, 'ready');
        setTimeout(hideToast, 2000);
        
        updateChunkStats();
        
    } catch (error) {
        showToast(`Ingest error: ${error.message}`, 0, 'error');
        console.error('Ingest error:', error);
    } finally {
        ingestBtn.disabled = false;
    }
}

/**
 * Update chunk statistics display
 */
function updateChunkStats() {
    const stats = getStoreStats();
    const ingestedDocs = documents.filter(d => d.ingested).length;
    

    
    
    if (stats.vectorCount > 0) {
        chunkStatsEl.innerHTML = `
            <span class="stat"><strong>${ingestedDocs}</strong> doc${ingestedDocs !== 1 ? 's' : ''}</span>
            <span class="stat"><strong>${stats.vectorCount}</strong> chunks</span>
            <span class="stat"><strong>${stats.totalChars.toLocaleString()}</strong> chars</span>
        `;
    } else {
        chunkStatsEl.innerHTML = '<span class="stat">No data</span>';
    }
}

/**
 * Handle question/search
 */
async function askQuestion() {
    const query = queryEl.value.trim();
    if (!query) {
        setStatus('Please enter a question.', 'warning');
        return;
    }
    
    if (getVectorCount() === 0) {
        showToast('Please ingest some documents first.', 0, 'warning');
        setTimeout(hideToast, 2000);
        return;
    }
    
    showToast('Searching...', 10, 'loading');
    askBtn.disabled = true;
    outputEl.textContent = '';
    contextEl.innerHTML = '';
    
    try {
        // Perform retrieval with progress
        const searchMode = searchModeEl.value;
        const searchResult = searchMode === 'hybrid' 
            ? await hybridSearch(query, 3, 0.7, (phase, pct) => {
                showToast(phase, pct, 'loading');
            })
            : await semanticSearch(query, 3);
        
        const { results, metrics } = searchResult;
        
        // Display retrieved context
        displayContext(results);
        displayMetrics(metrics);
        
        // Generate answer if LLM available
        if (llmAvailable) {
            showToast('Generating answer...', 50, 'loading');
            outputEl.textContent = '';
            
            await generate(query, results, (token) => {
                outputEl.textContent += token;
            });
            
            showToast('Answer generated!', 100, 'ready');
            setTimeout(hideToast, 2000);
        } else {
            outputEl.innerHTML = `<p class="no-llm-notice">LLM not available. Copy the context above to use with an external LLM like ChatGPT.</p>
<p><strong>Suggested prompt:</strong></p>
<pre>Using the following context, answer: "${query}"

${results.map((r, i) => `[${i + 1}] ${r.text}`).join('\n\n')}</pre>`;
            showToast('Search complete (LLM unavailable)', 100, 'ready');
            setTimeout(hideToast, 2000);
        }
        
    } catch (error) {
        showToast(`Search error: ${error.message}`, 0, 'error');
        console.error('Ask error:', error);
    } finally {
        askBtn.disabled = false;
    }
}

/**
 * Display retrieved context with scores
 */
function displayContext(results) {
    contextEl.innerHTML = results.map((r, i) => `
        <div class="context-chunk" style="--score: ${r.score || r.rrfScore}">
            <div class="chunk-header">
                <span class="chunk-number">[${i + 1}]${r.source ? ` <small>${escapeHtml(r.source)}</small>` : ''}</span>
                <span class="chunk-scores">
                    ${r.semanticScore !== undefined ? `<span class="score semantic">Semantic: ${(r.semanticScore * 100).toFixed(1)}%</span>` : ''}
                    ${r.bm25Score !== undefined ? `<span class="score bm25">BM25: ${r.bm25Score.toFixed(2)}</span>` : ''}
                    ${r.rrfScore !== undefined ? `<span class="score rrf">RRF: ${r.rrfScore.toFixed(4)}</span>` : ''}
                </span>
            </div>
            <div class="chunk-text">${escapeHtml(r.text)}</div>
        </div>
    `).join('');
}

/**
 * Display retrieval metrics
 */
function displayMetrics(metrics) {
    metricsEl.innerHTML = `
        <span class="metric"><strong>Time:</strong> ${metrics.retrievalTimeMs || 0}ms</span>
        <span class="metric"><strong>Top Score:</strong> ${metrics.topScore || 'N/A'}</span>
        <span class="metric"><strong>Score Gap:</strong> ${metrics.scoreGap || 'N/A'}</span>
        <span class="metric"><strong>Searched:</strong> ${metrics.totalDocuments || 0} chunks</span>
    `;
}

/**
 * Clear all stored data
 */
async function handleClear() {
    if (!confirm('Clear all documents and indexed data? This cannot be undone.')) return;
    
    await clearVectors();
    await clearDocuments();
    documents = [];
    documentGridEl.innerHTML = '';
    chunkStatsEl.innerHTML = '<span class="stat">No data</span>';
    showToast('All data cleared', 100, 'ready');
    setTimeout(hideToast, 1500);
}

/**
 * Export vectors to compressed gzip JSON file using streams
 * Uses File System Access API to stream directly to disk to avoid OOM
 */
async function handleExport() {
    showToast('Preparing export...', 10, 'loading');
    
    // Allow UI to update before work begins
    await new Promise(r => setTimeout(r, 50));
    
    try {
        // Create a ReadableStream from the generator
        // CRITICAL FIX uses pull() instead of start() loop to avoid 
        // synchronous blocking and memory explosion
        let generator = null;
        
        const stream = new ReadableStream({
            start(controller) {
                // Initialize binary generator with progress callback
                generator = exportBinaryGenerator((processed, total) => {
                    const pct = Math.round((processed / total) * 100);
                    showToast(`Exporting... ${pct}%`, pct, 'loading');
                });
            },
            
            pull(controller) {
                try {
                    const result = generator.next();
                    
                    if (result.done) {
                        controller.close();
                    } else {
                        controller.enqueue(result.value);
                    }
                } catch (err) {
                    controller.error(err);
                }
            }
        });

        // Pipeline: Binary Chunks -> GZIP Compressor (no TextEncoder needed!)
        const compressedStream = stream
            .pipeThrough(new CompressionStream('gzip'));
            
        // Use File System Access API if available (Chrome, Edge, Opera)
        // This streams directly to disk, avoiding OOM on large datasets
        if (window.showSaveFilePicker) {
            try {
                const handle = await window.showSaveFilePicker({
                    suggestedName: `rag-knowledge-base-${Date.now()}.rag.gz`,
                    types: [{
                        description: 'RAG Binary Data',
                        accept: { 'application/gzip': ['.rag.gz', '.gz'] }
                    }]
                });
                
                showToast('Exporting to disk...', 20, 'loading');
                const writable = await handle.createWritable();
                await compressedStream.pipeTo(writable);
                
                showToast('Export complete!', 100, 'ready');
                setTimeout(hideToast, 2000);
                return;
            } catch (err) {
                // If user cancels, stop
                if (err.name === 'AbortError') {
                    hideToast();
                    return;
                }
                throw err;
            }
        }

        // Fallback for browsers without File System Access API (Firefox, Safari)
        // Note: This may still OOM on very large datasets
        console.warn('File System Access API not supported. Falling back to Blob (may OOM).');
        const response = new Response(compressedStream);
        const blob = await response.blob();
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `rag-knowledge-base-${Date.now()}.json.gz`;
        a.click();
        
        URL.revokeObjectURL(url);
        
        showToast(`Exported ${(blob.size / 1024 / 1024).toFixed(2)}MB`, 100, 'ready');
        setTimeout(hideToast, 2000);
    } catch (error) {
        showToast(`Export failed: ${error.message}`, 0, 'error');
        console.error('Export error:', error);
    }
}

/**
 * Handle import file selection
 */
function handleImportClick() {
    importFileEl.click();
}

/**
 * Import vectors from JSON or gzip file
 */
async function handleImport(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    showToast('Reading file...', 20, 'loading');
    
    try {
        showToast('Decompressing...', 40, 'loading');
        const arrayBuffer = await file.arrayBuffer();
        
        let data;
        // Check if file is gzipped
        if (file.name.endsWith('.gz')) {
            const decompressedStream = new Response(arrayBuffer).body.pipeThrough(new DecompressionStream('gzip'));
            data = await new Response(decompressedStream).arrayBuffer();
        } else {
            data = arrayBuffer;
        }
        
        showToast('Importing vectors...', 60, 'loading');
        const count = await importVectors(data);
        updateChunkStats();
        showToast(`Imported ${count} chunks`, 100, 'ready');
        setTimeout(hideToast, 2000);
    } catch (error) {
        showToast(`Import failed: ${error.message}`, 0, 'error');
        console.error('Import error:', error);
    }
    
    // Reset file input
    importFileEl.value = '';
}

/**
 * Escape HTML entities
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Load default demo data (Wikipedia) - tries binary format first, then JSON
 */
async function loadDemoData() {
    if (getVectorCount() > 0) {
        const confirmed = await showConfirm(
            'Load Demo Data?', 
            'This will merge demo data with your existing data. Continue?', 
            'Load'
        );
        if (!confirmed) return;
    }

    loadDemoBtn.disabled = true;
    showToast('Fetching demo data...', 10, 'loading');

    try {
        // Load binary format only
        const response = await fetch('plain-text-wikipedia-simpleenglish.rag.gz');
        
        if (!response.ok) {
            throw new Error('plain-text-wikipedia-simpleenglish.rag.gz not found.');
        }

        // Decompress with progress by reading chunks
        showToast('Decompressing... 0MB', 30, 'loading');
        const decompressedStream = response.body.pipeThrough(new DecompressionStream('gzip'));
        const reader = decompressedStream.getReader();
        
        const chunks = [];
        let totalBytes = 0;
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            chunks.push(value);
            totalBytes += value.length;
            
            // Update progress every ~1MB
            const mb = (totalBytes / 1024 / 1024).toFixed(1);
            showToast(`Decompressing... ${mb}MB`, 30 + Math.min(25, totalBytes / 1024 / 1024), 'loading');
        }
        
        // Show progress before heavy combine step
        showToast('Preparing data...', 55, 'loading');
        await new Promise(r => setTimeout(r, 50)); // Yield to UI
        
        // Combine chunks into single ArrayBuffer
        const data = new Uint8Array(totalBytes);
        let offset = 0;
        for (const chunk of chunks) {
            data.set(chunk, offset);
            offset += chunk.length;
        }
        
        showToast('Importing vectors...', 60, 'loading');
        await new Promise(r => setTimeout(r, 50)); // Yield to UI
        
        const count = await importVectors(data.buffer, (processed, total, savingTotal) => {
            if (processed === -2) {
                // Saving progress: total = saved, savingTotal = total to save
                const pct = Math.round((total / savingTotal) * 100);
                showToast(`Saving to database... ${pct}%`, 95 + (pct * 0.04), 'loading');
            } else if (processed === -1) {
                showToast('Saving to database...', 95, 'loading');
            } else {
                const pct = Math.round((processed / total) * 100);
                showToast(`Importing vectors... ${pct}%`, 60 + (pct * 0.35), 'loading');
            }
        });
        
        updateChunkStats();
        showToast(`Loaded ${count} chunks from demo`, 100, 'ready');
        setTimeout(hideToast, 2000);

    } catch (error) {
        showToast(`Error loading demo: ${error.message}`, 0, 'error');
        console.error('Load demo error:', error);
    } finally {
        loadDemoBtn.disabled = false;
    }
}

/**
 * Setup project status modal
 */
function setupStatusModal() {
    const statusBtn = document.getElementById('btn-status');
    const statusModal = document.getElementById('status-modal');
    const closeBtn = document.getElementById('btn-status-close');
    
    if (!statusBtn || !statusModal || !closeBtn) return;
    
    // Open modal
    statusBtn.addEventListener('click', () => {
        statusModal.classList.remove('hidden');
    });
    
    // Close button
    closeBtn.addEventListener('click', () => {
        statusModal.classList.add('hidden');
    });
    
    // Click outside to close
    statusModal.addEventListener('click', (e) => {
        if (e.target === statusModal) {
            statusModal.classList.add('hidden');
        }
    });
}

// Start initialization
init();
