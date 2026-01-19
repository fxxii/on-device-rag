/**
 * WebLLM Wrapper with Dynamic Model Selection
 * Runs LLM inference in the browser using WebGPU
 */

let engine = null;
let webllmLib = null;
let currentModelId = null;

// Available models (ordered by size - smallest first as default)
export const AVAILABLE_MODELS = [
    {
        id: 'SmolLM2-360M-Instruct-q4f16_1-MLC',
        name: 'SmolLM2 360M',
        size: '~200MB',
        context: '2K tokens',
        description: 'Fastest, smallest'
    },
    {
        id: 'Qwen2-0.5B-Instruct-q4f16_1-MLC',
        name: 'Qwen2 0.5B',
        size: '~300MB',
        context: '32K tokens',
        description: 'Balanced'
    },
    {
        id: 'Llama-3.2-1B-Instruct-q4f16_1-MLC',
        name: 'Llama 3.2 1B',
        size: '~600MB',
        context: '128K tokens',
        description: 'Best quality'
    }
];

// Default model (smallest/fastest)
const DEFAULT_MODEL_ID = AVAILABLE_MODELS[0].id;

/**
 * Check if WebGPU is available
 * @returns {Promise<boolean>}
 */
export async function checkWebGPU() {
    if (!navigator.gpu) return false;
    
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter !== null;
    } catch {
        return false;
    }
}

/**
 * Load WebLLM library (cached)
 */
async function loadWebLLMLibrary() {
    if (!webllmLib) {
        webllmLib = await import('https://esm.run/@mlc-ai/web-llm');
    }
    return webllmLib;
}

/**
 * Initialize the LLM engine with specified model
 * @param {Function} progressCallback - Called with progress updates
 * @param {string} modelId - Model ID to load (default: SmolLM2-360M)
 * @returns {Promise<boolean>} True if initialization successful
 */
export async function initLLM(progressCallback = () => {}, modelId = DEFAULT_MODEL_ID) {
    const hasWebGPU = await checkWebGPU();
    
    if (!hasWebGPU) {
        progressCallback('WebGPU not available - LLM features disabled');
        return false;
    }
    
    progressCallback('Loading WebLLM library...');
    
    try {
        const webllm = await loadWebLLMLibrary();
        
        const modelInfo = AVAILABLE_MODELS.find(m => m.id === modelId) || AVAILABLE_MODELS[0];
        progressCallback(`Loading ${modelInfo.name} (${modelInfo.size})...`);
        
        // Use CreateMLCEngine - it handles model fetching from CORS-enabled CDN
        engine = await webllm.CreateMLCEngine(modelId, {
            initProgressCallback: (report) => {
                progressCallback(report.text);
            }
        });
        
        currentModelId = modelId;
        progressCallback('LLM ready');
        return true;
    } catch (error) {
        progressCallback(`LLM initialization failed: ${error.message}`);
        console.error('LLM init error:', error);
        return false;
    }
}

/**
 * Switch to a different model
 * @param {string} modelId - Model ID to switch to
 * @param {Function} progressCallback - Called with progress updates
 * @returns {Promise<boolean>} True if switch successful
 */
export async function switchModel(modelId, progressCallback = () => {}) {
    if (currentModelId === modelId) {
        progressCallback('Model already loaded');
        return true;
    }
    
    // Unload current engine
    if (engine) {
        try {
            await engine.unload();
        } catch (e) {
            console.warn('Error unloading model:', e);
        }
        engine = null;
    }
    
    // Load new model
    return await initLLM(progressCallback, modelId);
}

/**
 * Generate a response using RAG context
 * @param {string} question - User's question
 * @param {Array<{text: string}>} context - Retrieved context chunks
 * @param {Function} streamCallback - Called with each generated token
 * @returns {Promise<string>} Complete response
 */
export async function generate(question, context, streamCallback = () => {}) {
    if (!engine) {
        throw new Error('LLM not initialized. Call initLLM() first.');
    }
    
    // Build context string with source numbering
    const contextText = context
        .map((doc, i) => `[${i + 1}] ${doc.text}`)
        .join('\n');
    
    const systemPrompt = `You are a helpful assistant. Answer the user's question using ONLY the provided context. 
If the answer is found in the context, cite the source numbers like [1], [2] in your answer.
If the answer is NOT in the context, say "I don't have enough information to answer that."
Be concise and direct.`;

    const userPrompt = `Context:
${contextText}

Question: ${question}`;

    const messages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
    ];
    
    let fullResponse = '';
    
    const completion = await engine.chat.completions.create({
        messages,
        temperature: 0.3,
        max_tokens: 512,
        stream: true
    });
    
    for await (const chunk of completion) {
        const token = chunk.choices[0]?.delta?.content || '';
        fullResponse += token;
        streamCallback(token);
    }
    
    return fullResponse;
}

/**
 * Check if LLM is ready
 * @returns {boolean}
 */
export function isLLMReady() {
    return engine !== null;
}

/**
 * Get current model ID
 * @returns {string}
 */
export function getModelName() {
    return currentModelId || DEFAULT_MODEL_ID;
}

/**
 * Get current model info object
 * @returns {Object}
 */
export function getCurrentModelInfo() {
    const id = currentModelId || DEFAULT_MODEL_ID;
    return AVAILABLE_MODELS.find(m => m.id === id) || AVAILABLE_MODELS[0];
}
