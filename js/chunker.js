/**
 * RecursiveCharacterTextSplitter - Pure JavaScript Implementation
 * Matches LangChain.js behavior: recursively splits by paragraphs, newlines, sentences, then characters
 * Ensures chunks are semantically coherent while respecting size limits
 */

// Default separators in order of preference (most to least semantic)
const DEFAULT_SEPARATORS = [
    "\n\n",   // Paragraph breaks (highest priority)
    "\n",     // Line breaks
    ". ",     // Sentence endings
    "! ",     // Exclamation endings
    "? ",     // Question endings
    "; ",     // Semicolon breaks
    ", ",     // Comma breaks
    " ",      // Word breaks
    ""        // Character-level (last resort)
];

/**
 * RecursiveCharacterTextSplitter - splits text recursively by semantic boundaries
 * @param {string} text - Input text to chunk
 * @param {Object} options - Configuration options
 * @param {number} options.chunkSize - Maximum chunk size in characters (default: 500)
 * @param {number} options.chunkOverlap - Overlap between chunks (default: 50)
 * @param {string[]} options.separators - Custom separators (default: paragraphs → newlines → sentences → words → chars)
 * @param {boolean} options.keepSeparator - Whether to keep separators in output (default: true)
 * @param {Function} options.onProgress - Optional progress callback (chunksProcessed, estimatedTotal)
 * @returns {Promise<Array<{id: string, text: string, startPos: number, endPos: number}>>}
 */
export async function chunkText(text, {
    chunkSize = 500,
    chunkOverlap = 50,
    separators = DEFAULT_SEPARATORS,
    keepSeparator = true,
    onProgress = null
} = {}) {
    if (!text || text.trim().length === 0) return [];

    // Get raw splits from iterative algorithm with progress reporting
    const rawChunks = await iterativeSplit(text, separators, chunkSize, chunkOverlap, keepSeparator, onProgress);
    
    // Merge chunks that are too small while respecting overlap
    const mergedChunks = mergeSplits(rawChunks, chunkSize, chunkOverlap);
    
    // Convert to output format with positions
    return formatChunks(mergedChunks, text);
}

/**
 * Iteratively split text using separators in order of preference (non-recursive to avoid stack overflow)
 * Now async to allow UI updates during long operations
 */
async function iterativeSplit(text, separators, chunkSize, chunkOverlap, keepSeparator, onProgress) {
    const results = [];
    
    // Use a work queue instead of recursion to avoid stack overflow
    const workQueue = [{ text, separatorIndex: 0 }];
    let processedItems = 0;
    let lastProgressUpdate = Date.now();
    
    // Estimate total based on text size
    const estimatedTotal = Math.ceil(text.length / chunkSize);
    
    while (workQueue.length > 0) {
        const work = workQueue.shift();
        const currentText = work.text;
        const sepIndex = work.separatorIndex;
        processedItems++;
        
        // Yield to UI periodically (every 100ms or every 50 items)
        const now = Date.now();
        if (now - lastProgressUpdate > 100 || processedItems % 50 === 0) {
            if (onProgress) {
                onProgress(results.length, Math.max(estimatedTotal, results.length + workQueue.length));
            }
            await new Promise(r => setTimeout(r, 0)); // Yield to event loop
            lastProgressUpdate = now;
        }
        
        // If text is small enough, add to results
        if (currentText.length <= chunkSize) {
            results.push(currentText);
            continue;
        }
        
        // If we've exhausted all separators, force character split
        if (sepIndex >= separators.length) {
            const forceSplit = forceCharacterSplit(currentText, chunkSize, chunkOverlap);
            for (let k = 0; k < forceSplit.length; k++) {
                results.push(forceSplit[k]);
            }
            continue;
        }
        
        // Find the best separator that exists in the text
        let separator = null;
        let nextSepIndex = separators.length;
        
        for (let i = sepIndex; i < separators.length; i++) {
            const sep = separators[i];
            if (sep === "" || currentText.includes(sep)) {
                separator = sep;
                nextSepIndex = i + 1;
                break;
            }
        }
        
        // If no separator found, force split
        if (separator === null) {
            const forceSplit = forceCharacterSplit(currentText, chunkSize, chunkOverlap);
            for (let k = 0; k < forceSplit.length; k++) {
                results.push(forceSplit[k]);
            }
            continue;
        }
        
        // Split by the chosen separator
        const splits = splitTextWithSeparator(currentText, separator, keepSeparator);
        
        // Process each split - add small ones to results, queue large ones for further splitting
        for (const split of splits) {
            if (split.length <= chunkSize) {
                results.push(split);
            } else {
                // Queue for further splitting with remaining separators
                workQueue.push({ text: split, separatorIndex: nextSepIndex });
            }
        }
    }
    
    // Final progress update
    if (onProgress) {
        onProgress(results.length, results.length);
    }
    
    return results;
}

/**
 * Split text by separator, optionally keeping the separator
 */
function splitTextWithSeparator(text, separator, keepSeparator) {
    if (separator === "") {
        return [...text]; // Split into characters
    }
    
    const splits = text.split(separator);
    
    if (!keepSeparator) {
        return splits.filter(s => s.length > 0);
    }
    
    // Keep separator at end of each split (except last)
    const result = [];
    for (let i = 0; i < splits.length; i++) {
        if (splits[i].length === 0) continue;
        
        if (i < splits.length - 1) {
            result.push(splits[i] + separator);
        } else {
            result.push(splits[i]);
        }
    }
    
    return result;
}

/**
 * Force split text by characters when no separator works
 */
function forceCharacterSplit(text, chunkSize, overlap) {
    const chunks = [];
    let start = 0;
    
    while (start < text.length) {
        const end = Math.min(start + chunkSize, text.length);
        chunks.push(text.slice(start, end));
        start = end - overlap;
        if (start >= text.length - overlap) break;
    }
    
    return chunks;
}

/**
 * Merge small splits together while respecting chunk size and overlap
 */
function mergeSplits(splits, chunkSize, chunkOverlap) {
    if (splits.length === 0) return [];
    
    const merged = [];
    let currentChunk = [];
    let currentLength = 0;
    
    for (const split of splits) {
        const splitLength = split.length;
        
        // If adding this split exceeds chunk size, finalize current chunk
        if (currentLength + splitLength > chunkSize && currentChunk.length > 0) {
            merged.push(currentChunk.join(""));
            
            // Keep overlap from the end of current chunk
            const overlapText = getOverlapText(currentChunk, chunkOverlap);
            currentChunk = overlapText ? [overlapText] : [];
            currentLength = currentChunk.reduce((sum, s) => sum + s.length, 0);
        }
        
        currentChunk.push(split);
        currentLength += splitLength;
    }
    
    // Don't forget the last chunk
    if (currentChunk.length > 0) {
        merged.push(currentChunk.join(""));
    }
    
    return merged;
}

/**
 * Get overlap text from chunks (last N characters)
 */
function getOverlapText(chunks, overlapSize) {
    const combined = chunks.join("");
    if (combined.length <= overlapSize) {
        return combined;
    }
    return combined.slice(-overlapSize);
}

/**
 * Convert raw chunks to output format with positions
 */
function formatChunks(chunks, originalText) {
    const results = [];
    let searchStart = 0;
    
    for (let i = 0; i < chunks.length; i++) {
        const chunkText = chunks[i].trim();
        if (!chunkText) continue;
        
        // Find position in original text
        let startPos = originalText.indexOf(chunkText.substring(0, 50), searchStart);
        if (startPos === -1) startPos = searchStart;
        
        const endPos = startPos + chunkText.length;
        
        results.push({
            id: `chunk-${results.length}`,
            text: chunkText,
            startPos,
            endPos
        });
        
        // Update search start for next chunk (accounting for overlap)
        searchStart = Math.max(startPos + 1, searchStart);
    }
    
    return results;
}

/**
 * Simple line-based chunking (fallback for structured data)
 * @param {string} text - Input text
 * @returns {Array<{id: string, text: string, startPos: number, endPos: number}>}
 */
export function chunkByLines(text) {
    if (!text) return [];
    
    return text
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0)
        .map((text, index) => ({
            id: `line-${index}`,
            text,
            startPos: 0,
            endPos: text.length
        }));
}

/**
 * Get chunk statistics for display
 * @param {Array} chunks - Array of chunk objects
 * @returns {Object} Statistics about the chunks
 */
export function getChunkStats(chunks) {
    if (!chunks || chunks.length === 0) {
        return { count: 0, avgLength: 0, minLength: 0, maxLength: 0 };
    }
    
    const lengths = chunks.map(c => c.text.length);
    return {
        count: chunks.length,
        avgLength: Math.round(lengths.reduce((a, b) => a + b, 0) / lengths.length),
        minLength: Math.min(...lengths),
        maxLength: Math.max(...lengths),
        totalChars: lengths.reduce((a, b) => a + b, 0)
    };
}
