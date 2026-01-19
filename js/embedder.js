/**
 * Embedding Model Wrapper
 * Uses Transformers.js to run embedding models in the browser
 */

let pipeline = null;
let embedder = null;

// Model configuration
const EMBEDDING_MODEL = 'Supabase/gte-small';
const EMBEDDING_OPTIONS = { pooling: 'mean', normalize: true };

/**
 * Initialize the embedding model
 * @param {Function} progressCallback - Called with progress updates
 * @returns {Promise<void>}
 */
export async function initEmbedder(progressCallback = () => {}) {
    if (embedder) return; // Already initialized
    
    progressCallback('Loading Transformers.js library...');
    
    // Dynamic import of Transformers.js
    const transformers = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.0');
    pipeline = transformers.pipeline;
    
    // Configure to fetch from HuggingFace Hub
    transformers.env.allowLocalModels = false;
    
    progressCallback(`Loading embedding model (${EMBEDDING_MODEL})...`);
    
    embedder = await pipeline('feature-extraction', EMBEDDING_MODEL, {
        progress_callback: (progress) => {
            if (progress.status === 'downloading') {
                const pct = Math.round((progress.loaded / progress.total) * 100);
                progressCallback(`Downloading ${progress.name}: ${pct}%`);
            }
        }
    });
    
    progressCallback('Embedding model ready');
}

/**
 * Generate embedding for a single text
 * @param {string} text - Text to embed
 * @returns {Promise<Float32Array>} Embedding vector
 */
export async function embed(text) {
    if (!embedder) {
        throw new Error('Embedder not initialized. Call initEmbedder() first.');
    }
    
    const output = await embedder(text, EMBEDDING_OPTIONS);
    return output.data;
}

/**
 * Generate embeddings for multiple texts
 * @param {Array<string>} texts - Array of texts to embed
 * @param {Function} progressCallback - Progress callback
 * @returns {Promise<Array<Float32Array>>} Array of embeddings
 */
export async function embedBatch(texts, progressCallback = () => {}) {
    if (!embedder) {
        throw new Error('Embedder not initialized. Call initEmbedder() first.');
    }
    
    const embeddings = [];
    for (let i = 0; i < texts.length; i++) {
        const output = await embedder(texts[i], EMBEDDING_OPTIONS);
        embeddings.push(output.data);
        progressCallback(`Embedding ${i + 1}/${texts.length}`);
    }
    
    return embeddings;
}

/**
 * Check if embedder is ready
 * @returns {boolean}
 */
export function isEmbedderReady() {
    return embedder !== null;
}

/**
 * Get embedding dimensions
 * @returns {number}
 */
export function getEmbeddingDimensions() {
    return 384; // gte-small uses 384 dimensions
}

/**
 * Get embedding model name
 * @returns {string}
 */
export function getEmbeddingModelName() {
    return EMBEDDING_MODEL;
}
