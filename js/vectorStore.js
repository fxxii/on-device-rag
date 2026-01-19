/**
 * Vector Store with IndexedDB Persistence
 * Provides persistent storage for embeddings with in-memory caching
 */

const DB_NAME = 'clientSideRAG';
const DB_VERSION = 2; // Version 2 adds documents store
const STORE_NAME = 'vectors';
const DOC_STORE_NAME = 'documents';

let db = null;
let memoryCache = []; // In-memory cache for fast access

/**
 * Open IndexedDB connection
 * @returns {Promise<IDBDatabase>}
 */
async function openDB() {
    if (db) return db;
    
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        
        request.onerror = () => reject(request.error);
        
        request.onsuccess = () => {
            db = request.result;
            resolve(db);
        };
        
        request.onupgradeneeded = (event) => {
            const database = event.target.result;
            // Create vectors store if needed
            if (!database.objectStoreNames.contains(STORE_NAME)) {
                database.createObjectStore(STORE_NAME, { keyPath: 'id' });
            }
            // Create documents store if needed
            if (!database.objectStoreNames.contains(DOC_STORE_NAME)) {
                database.createObjectStore(DOC_STORE_NAME, { keyPath: 'id' });
            }
        };
    });
}

/**
 * Initialize vector store and load from IndexedDB
 * @returns {Promise<Array>} Loaded vectors
 */
export async function initVectorStore() {
    await openDB();
    memoryCache = await loadFromDB();
    return memoryCache;
}

/**
 * Load all vectors from IndexedDB
 * @returns {Promise<Array>}
 */
async function loadFromDB() {
    const database = await openDB();
    
    return new Promise((resolve, reject) => {
        const tx = database.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const request = store.getAll();
        
        request.onsuccess = () => {
            // Convert stored arrays back to Float32Array
            const items = request.result.map(item => ({
                ...item,
                embedding: new Float32Array(item.embedding)
            }));
            resolve(items);
        };
        request.onerror = () => reject(request.error);
    });
}

/**
 * Add vectors to the store (memory + IndexedDB)
 * @param {Array<{id: string, text: string, embedding: Float32Array}>} vectors
 * @param {Function} onProgress - Optional callback(current, total)
 */
export async function addVectors(vectors, onProgress) {
    const database = await openDB();
    const total = vectors.length;
    const BATCH_SIZE = 10000;
    
    // Process in batches to allow progress reporting
    for (let batchStart = 0; batchStart < total; batchStart += BATCH_SIZE) {
        const batchEnd = Math.min(batchStart + BATCH_SIZE, total);
        const batch = vectors.slice(batchStart, batchEnd);
        
        await new Promise((resolve, reject) => {
            const tx = database.transaction(STORE_NAME, 'readwrite');
            const store = tx.objectStore(STORE_NAME);
            
            for (const vector of batch) {
                // Store embedding as regular array for IndexedDB
                store.put({
                    ...vector,
                    embedding: Array.from(vector.embedding),
                    timestamp: Date.now()
                });
                
                // Update memory cache
                const existingIdx = memoryCache.findIndex(v => v.id === vector.id);
                if (existingIdx >= 0) {
                    memoryCache[existingIdx] = vector;
                } else {
                    memoryCache.push(vector);
                }
            }
            
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
        
        // Report progress and yield to UI
        if (onProgress) {
            onProgress(batchEnd, total);
            await new Promise(r => setTimeout(r, 0));
        }
    }
}

/**
 * Get all vectors from memory cache
 * @returns {Array}
 */
export function getVectors() {
    return memoryCache;
}

/**
 * Get vector count
 * @returns {number}
 */
export function getVectorCount() {
    return memoryCache.length;
}

/**
 * Clear all vectors from store
 */
export async function clearVectors() {
    const database = await openDB();
    
    return new Promise((resolve, reject) => {
        const tx = database.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        const request = store.clear();
        
        request.onsuccess = () => {
            memoryCache = [];
            resolve();
        };
        request.onerror = () => reject(request.error);
    });
}

// ==========================================
// Base64 Helpers for Vector Compression
// ==========================================

/**
 * Convert Float32Array to Base64 string
 */
/**
 * Convert Float32Array to Base64 string
 */
function float32ToBase64(float32Array) {
    // IMPORTANT: Use byteOffset and byteLength to handle views (subarrays) correctly
    const uint8Array = new Uint8Array(float32Array.buffer, float32Array.byteOffset, float32Array.byteLength);
    let binary = '';
    const len = uint8Array.byteLength;
    // Use a chunked approach to avoid stack overflow with spread operator on huge arrays
    // though strict loop is fine for performance here
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(uint8Array[i]);
    }
    return btoa(binary);
}

/**
 * Convert Base64 string to Float32Array
 */
function base64ToFloat32(base64) {
    const binary = atob(base64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
}

/**
 * Binary Format Constants
 */
const BINARY_MAGIC = 'RAGV';
const BINARY_VERSION = 1;

/**
 * Export vectors as binary format generator
 * Format: Header (12 bytes) + Chunks (variable)
 * @param {Function} onProgress - Callback(current, total)
 * @returns {Generator<Uint8Array>} Binary chunks
 */
export function* exportBinaryGenerator(onProgress) {
    const total = memoryCache.length;
    const dimensions = memoryCache[0]?.embedding?.length || 384;
    
    // Yield header (12 bytes)
    const header = new ArrayBuffer(12);
    const headerView = new DataView(header);
    const encoder = new TextEncoder();
    
    // Magic "RAGV" (4 bytes)
    const magic = encoder.encode(BINARY_MAGIC);
    new Uint8Array(header).set(magic, 0);
    
    // Version (2 bytes)
    headerView.setUint16(4, BINARY_VERSION, true);
    
    // Dimensions (2 bytes)
    headerView.setUint16(6, dimensions, true);
    
    // Count (4 bytes)
    headerView.setUint32(8, total, true);
    
    yield new Uint8Array(header);
    
    // Yield each chunk
    for (let i = 0; i < total; i++) {
        const v = memoryCache[i];
        
        // Encode strings
        const idBytes = encoder.encode(v.id);
        const textBytes = encoder.encode(v.text);
        
        // Get embedding as Float32Array
        let embedding = v.embedding;
        if (Array.isArray(embedding)) {
            embedding = new Float32Array(embedding);
        }
        
        // Calculate chunk size: id_len(2) + id + text_len(4) + text + embedding
        const chunkSize = 2 + idBytes.length + 4 + textBytes.length + (dimensions * 4);
        const chunk = new ArrayBuffer(chunkSize);
        const chunkView = new DataView(chunk);
        const chunkBytes = new Uint8Array(chunk);
        
        let offset = 0;
        
        // id_len (2 bytes)
        chunkView.setUint16(offset, idBytes.length, true);
        offset += 2;
        
        // id (utf8 bytes)
        chunkBytes.set(idBytes, offset);
        offset += idBytes.length;
        
        // text_len (4 bytes)
        chunkView.setUint32(offset, textBytes.length, true);
        offset += 4;
        
        // text (utf8 bytes)
        chunkBytes.set(textBytes, offset);
        offset += textBytes.length;
        
        // embedding (raw Float32 bytes)
        const embeddingBytes = new Uint8Array(embedding.buffer, embedding.byteOffset, embedding.byteLength);
        chunkBytes.set(embeddingBytes, offset);
        
        yield chunkBytes;
        
        // Progress
        if (onProgress && i % 2000 === 0) {
            onProgress(i, total);
        }
    }
    
    if (onProgress) onProgress(total, total);
}

/**
 * Import vectors from binary format
 * @param {ArrayBuffer} buffer - Binary data
 * @param {Function} onProgress - Optional callback(current, total)
 * @returns {Promise<number>} Number of imported vectors
 */
export async function importBinaryVectors(buffer, onProgress) {
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    const decoder = new TextDecoder();
    
    // Read header
    const magic = decoder.decode(bytes.slice(0, 4));
    if (magic !== BINARY_MAGIC) {
        throw new Error('Invalid file format');
    }
    
    const version = view.getUint16(4, true);
    const dimensions = view.getUint16(6, true);
    const count = view.getUint32(8, true);
    
    let offset = 12;
    const vectors = [];
    
    for (let i = 0; i < count; i++) {
        // id_len
        const idLen = view.getUint16(offset, true);
        offset += 2;
        
        // id
        const id = decoder.decode(bytes.slice(offset, offset + idLen));
        offset += idLen;
        
        // text_len
        const textLen = view.getUint32(offset, true);
        offset += 4;
        
        // text
        const text = decoder.decode(bytes.slice(offset, offset + textLen));
        offset += textLen;
        
        // embedding
        const embeddingBytes = bytes.slice(offset, offset + dimensions * 4);
        const embedding = new Float32Array(embeddingBytes.buffer.slice(embeddingBytes.byteOffset, embeddingBytes.byteOffset + embeddingBytes.byteLength));
        offset += dimensions * 4;
        
        vectors.push({ id, text, embedding });
        
        // Report progress every 5000 vectors and yield to UI
        if (onProgress && i % 5000 === 0) {
            onProgress(i, count);
            await new Promise(r => setTimeout(r, 0));
        }
    }
    
    if (onProgress) onProgress(count, count);
    
    // Signal saving phase (-1 means 'saving to database')
    if (onProgress) onProgress(-1, count);
    await new Promise(r => setTimeout(r, 50)); // Yield for UI update
    
    // Pass saving progress callback: use negative values (-2, -N) to signal saving phase
    await addVectors(vectors, (saved, total) => {
        if (onProgress) onProgress(-2, saved, total);
    });
    return vectors.length;
}

/**
 * Import vectors - auto-detects format (binary or legacy JSON)
 * @param {ArrayBuffer|string} data - Binary or JSON data
 * @param {Function} onProgress - Optional callback(current, total)
 * @returns {Promise<number>} Number of imported vectors
 */
export async function importVectors(data, onProgress) {
    // If string, it's legacy JSON
    if (typeof data === 'string') {
        return importLegacyJSON(data);
    }
    
    // Check for binary magic
    const bytes = new Uint8Array(data);
    const decoder = new TextDecoder();
    const magic = decoder.decode(bytes.slice(0, 4));
    
    if (magic === BINARY_MAGIC) {
        return importBinaryVectors(data, onProgress);
    }
    
    // Try JSON (might be ArrayBuffer from file read)
    const text = decoder.decode(bytes);
    return importLegacyJSON(text);
}

/**
 * Import legacy JSON format
 */
async function importLegacyJSON(jsonString) {
    const data = JSON.parse(jsonString);
    const vectors = data.map(item => {
        let embedding;
        if (typeof item.embedding === 'string') {
            embedding = base64ToFloat32(item.embedding);
        } else {
            embedding = new Float32Array(item.embedding);
        }
        return { ...item, embedding };
    });
    await addVectors(vectors);
    return vectors.length;
}

/**
 * Get store statistics
 * @returns {Object}
 */
export function getStoreStats() {
    return {
        vectorCount: memoryCache.length,
        totalChars: memoryCache.reduce((sum, v) => sum + v.text.length, 0),
        embeddingDimensions: memoryCache[0]?.embedding?.length || 0
    };
}

// ==========================================
// Document Persistence (separate from vectors)
// ==========================================

/**
 * Save documents to IndexedDB
 * @param {Array} documents - Array of document objects
 */
export async function saveDocuments(documents) {
    const database = await openDB();
    
    return new Promise((resolve, reject) => {
        const tx = database.transaction(DOC_STORE_NAME, 'readwrite');
        const store = tx.objectStore(DOC_STORE_NAME);
        
        // Clear existing and add all
        store.clear();
        
        for (const doc of documents) {
            // Don't store the full text in documents store (it's in vectors)
            store.put({
                id: doc.id,
                name: doc.name,
                type: doc.type,
                pageCount: doc.pageCount,
                ingested: doc.ingested || false,
                charCount: doc.text?.length || 0
            });
        }
        
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
    });
}

/**
 * Load documents from IndexedDB
 * @returns {Promise<Array>}
 */
export async function loadDocuments() {
    try {
        const database = await openDB();
        
        return new Promise((resolve, reject) => {
            const tx = database.transaction(DOC_STORE_NAME, 'readonly');
            const store = tx.objectStore(DOC_STORE_NAME);
            const request = store.getAll();
            
            request.onsuccess = () => resolve(request.result || []);
            request.onerror = () => reject(request.error);
        });
    } catch (e) {
        console.warn('Could not load documents:', e);
        return [];
    }
}

/**
 * Clear documents from IndexedDB
 */
export async function clearDocuments() {
    try {
        const database = await openDB();
        
        return new Promise((resolve, reject) => {
            const tx = database.transaction(DOC_STORE_NAME, 'readwrite');
            const store = tx.objectStore(DOC_STORE_NAME);
            const request = store.clear();
            
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    } catch (e) {
        console.warn('Could not clear documents:', e);
    }
}

/**
 * Remove vectors by document ID prefix
 * @param {string} docId - Document ID to remove vectors for
 */
export async function removeVectorsByDocId(docId) {
    const database = await openDB();
    
    // Filter out vectors that belong to this document
    const toRemove = memoryCache.filter(v => v.id.startsWith(docId));
    
    return new Promise((resolve, reject) => {
        const tx = database.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        
        for (const vector of toRemove) {
            store.delete(vector.id);
        }
        
        tx.oncomplete = () => {
            // Update memory cache
            const beforeCount = memoryCache.length;
            memoryCache = memoryCache.filter(v => !v.id.startsWith(docId));
            resolve();
        };
        tx.onerror = () => reject(tx.error);
    });
}
