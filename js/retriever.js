/**
 * Hybrid Retriever
 * Combines BM25 (lexical) and Semantic (embedding) search using Reciprocal Rank Fusion
 */

import { embed } from './embedder.js';
import { getVectors } from './vectorStore.js';

/**
 * BM25 parameters
 */
const BM25_K1 = 1.5;
const BM25_B = 0.75;

/**
 * Preprocess text for search
 * @param {string} text - Input text
 * @returns {string} Preprocessed text
 */
export function preprocessQuery(text) {
    return text
        .toLowerCase()
        .replace(/[^\w\s]/g, ' ')  // Remove punctuation
        .replace(/\s+/g, ' ')       // Normalize whitespace
        .trim();
}

/**
 * Tokenize text into terms
 * @param {string} text - Text to tokenize
 * @returns {Array<string>} Array of terms
 */
function tokenize(text) {
    return preprocessQuery(text).split(' ').filter(t => t.length > 0);
}

/**
 * Calculate term frequency in document
 * @param {Array<string>} terms - Document terms
 * @returns {Object} Term frequency map
 */
function getTermFrequency(terms) {
    const freq = {};
    terms.forEach(t => freq[t] = (freq[t] || 0) + 1);
    return freq;
}

/**
 * Calculate BM25 score for a document
 * @param {string} query - Search query
 * @param {string} docText - Document text
 * @param {number} avgDocLength - Average document length in corpus
 * @param {number} corpusSize - Total documents in corpus
 * @param {Object} docFreq - Document frequency for each term
 * @returns {number} BM25 score
 */
function bm25Score(query, docText, avgDocLength, corpusSize, docFreq) {
    const queryTerms = tokenize(query);
    const docTerms = tokenize(docText);
    const docLength = docTerms.length;
    const termFreq = getTermFrequency(docTerms);
    
    let score = 0;
    for (const term of queryTerms) {
        const tf = termFreq[term] || 0;
        if (tf === 0) continue;
        
        // IDF calculation with smoothing
        const df = docFreq[term] || 0;
        const idf = Math.log((corpusSize - df + 0.5) / (df + 0.5) + 1);
        
        // BM25 TF component
        const tfNorm = (tf * (BM25_K1 + 1)) / 
            (tf + BM25_K1 * (1 - BM25_B + BM25_B * docLength / avgDocLength));
        
        score += idf * tfNorm;
    }
    
    return score;
}

/**
 * Calculate cosine similarity between two vectors
 * @param {Float32Array} a - First vector
 * @param {Float32Array} b - Second vector
 * @returns {number} Cosine similarity (-1 to 1)
 */
function cosineSimilarity(a, b) {
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;
    
    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        magnitudeA += a[i] * a[i];
        magnitudeB += b[i] * b[i];
    }
    
    const magnitude = Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
}

/**
 * Reciprocal Rank Fusion - combines multiple ranking lists
 * @param {Array<Array<{id: string, score: number}>>} rankings - Multiple ranked lists
 * @param {number} k - RRF parameter (default: 60)
 * @returns {Object} Map of id to RRF score
 */
function reciprocalRankFusion(rankings, k = 60) {
    const scores = {};
    
    for (const ranking of rankings) {
        ranking.forEach((item, rank) => {
            const rrfScore = 1 / (k + rank + 1);
            scores[item.id] = (scores[item.id] || 0) + rrfScore;
        });
    }
    
    return scores;
}

/**
 * Perform hybrid search combining BM25 and semantic similarity
 * @param {string} query - Search query
 * @param {number} topK - Number of results to return
 * @param {number} alpha - Weight for semantic vs BM25 (0-1, higher = more semantic)
 * @param {Function} onProgress - Optional callback(phase, percentComplete)
 * @returns {Promise<{results: Array, metrics: Object}>}
 */
export async function hybridSearch(query, topK = 3, alpha = 0.7, onProgress) {
    const vectors = getVectors();
    if (vectors.length === 0) return { results: [], metrics: {} };
    
    const startTime = performance.now();
    
    // Phase 1: Embedding query
    if (onProgress) { onProgress('Embedding query...', 10); await new Promise(r => setTimeout(r, 0)); }
    const queryEmbedding = await embed(query);
    
    // Phase 2: Semantic similarity
    if (onProgress) { onProgress('Semantic search...', 30); await new Promise(r => setTimeout(r, 0)); }
    const semanticResults = vectors.map(doc => ({
        id: doc.id,
        text: doc.text,
        semanticScore: cosineSimilarity(queryEmbedding, doc.embedding)
    }));
    
    // Phase 3: BM25 scores
    if (onProgress) { onProgress('BM25 search...', 50); await new Promise(r => setTimeout(r, 0)); }
    const avgDocLength = vectors.reduce((sum, d) => sum + tokenize(d.text).length, 0) / vectors.length;
    const corpusSize = vectors.length;
    
    // Calculate document frequency for each term
    const docFreq = {};
    vectors.forEach(doc => {
        const terms = new Set(tokenize(doc.text));
        terms.forEach(t => docFreq[t] = (docFreq[t] || 0) + 1);
    });
    
    const bm25Results = vectors.map(doc => ({
        id: doc.id,
        text: doc.text,
        bm25Score: bm25Score(query, doc.text, avgDocLength, corpusSize, docFreq)
    }));
    
    // Phase 4: Ranking
    if (onProgress) { onProgress('Ranking results...', 80); await new Promise(r => setTimeout(r, 0)); }
    const semanticRanked = [...semanticResults].sort((a, b) => b.semanticScore - a.semanticScore);
    const bm25Ranked = [...bm25Results].sort((a, b) => b.bm25Score - a.bm25Score);
    
    // 5. Apply Reciprocal Rank Fusion (only on top candidates to save computation)
    const TOP_N_FOR_RRF = 1000; // Only consider top 1000 from each ranking
    const rrfScores = reciprocalRankFusion([
        semanticRanked.slice(0, TOP_N_FOR_RRF).map(r => ({ id: r.id, score: r.semanticScore })),
        bm25Ranked.slice(0, TOP_N_FOR_RRF).map(r => ({ id: r.id, score: r.bm25Score }))
    ]);
    
    // 6. Build lookup Maps for O(1) access (critical for large datasets)
    const semanticMap = new Map(semanticResults.map(r => [r.id, r.semanticScore]));
    const bm25Map = new Map(bm25Results.map(r => [r.id, r.bm25Score]));
    
    // 7. Precompute min/max for BM25 normalization (ONCE, not per-item!)
    let bm25Max = -Infinity;
    let bm25Min = Infinity;
    for (const r of bm25Results) {
        if (r.bm25Score > bm25Max) bm25Max = r.bm25Score;
        if (r.bm25Score < bm25Min) bm25Min = r.bm25Score;
    }
    const bm25Range = bm25Max - bm25Min;
    
    // 8. Only process candidates that have RRF scores (major optimization)
    const candidateIds = Object.keys(rrfScores);
    const combinedResults = candidateIds.map(id => {
        const doc = vectors.find(v => v.id === id); // OK since candidateIds is small (~1000)
        const semanticScore = semanticMap.get(id) || 0;
        const bm25Score = bm25Map.get(id) || 0;
        const normalizedBm25 = bm25Range === 0 ? 0 : (bm25Score - bm25Min) / bm25Range;
        
        return {
            id,
            text: doc?.text || '',
            semanticScore,
            bm25Score,
            rrfScore: rrfScores[id] || 0,
            score: alpha * semanticScore + (1 - alpha) * normalizedBm25
        };
    });
    
    // Sort by RRF score and return top K
    combinedResults.sort((a, b) => b.rrfScore - a.rrfScore);
    
    const endTime = performance.now();
    const results = combinedResults.slice(0, topK);
    
    // Add retrieval metadata
    return {
        results,
        metrics: {
            retrievalTimeMs: Math.round(endTime - startTime),
            totalDocuments: vectors.length,
            topScore: results[0]?.score.toFixed(4) || 0,
            scoreGap: results.length > 1 ? 
                (results[0].rrfScore - results[1].rrfScore).toFixed(4) : 'N/A'
        }
    };
}

/**
 * Normalize score to 0-1 range
 * Using loop instead of Math.max(...) to avoid stack overflow on large arrays
 */
function normalizeScore(score, allResults) {
    if (allResults.length === 0) return 0;
    
    let max = -Infinity;
    let min = Infinity;
    for (let i = 0; i < allResults.length; i++) {
        const s = allResults[i].bm25Score;
        if (s > max) max = s;
        if (s < min) min = s;
    }
    
    return max === min ? 0 : (score - min) / (max - min);
}

/**
 * Simple semantic-only search (fallback)
 * @param {string} query - Search query
 * @param {number} topK - Number of results
 * @returns {Promise<Array>}
 */
export async function semanticSearch(query, topK = 3) {
    const vectors = getVectors();
    if (vectors.length === 0) return { results: [], metrics: {} };
    
    const queryEmbedding = await embed(query);
    
    const results = vectors
        .map(doc => ({
            id: doc.id,
            text: doc.text,
            score: cosineSimilarity(queryEmbedding, doc.embedding)
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);
    
    return {
        results,
        metrics: {
            totalDocuments: vectors.length,
            topScore: results[0]?.score.toFixed(4) || 0
        }
    };
}
