/**
 * PDF Parser Module
 * Uses PDF.js to extract text from PDF files
 */

// PDF.js library from CDN
const PDFJS_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.0.379/pdf.min.mjs';
const PDFJS_WORKER_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.0.379/pdf.worker.min.mjs';

let pdfjsLib = null;

/**
 * Initialize PDF.js library
 */
async function initPdfJs() {
    if (pdfjsLib) return;
    
    pdfjsLib = await import(PDFJS_CDN);
    pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_CDN;
}

/**
 * Extract text from a PDF file
 * @param {File} file - PDF file object
 * @returns {Promise<{name: string, text: string, pageCount: number}>}
 */
export async function extractTextFromPdf(file) {
    await initPdfJs();
    
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    
    let fullText = '';
    const pageCount = pdf.numPages;
    
    for (let i = 1; i <= pageCount; i++) {
        const page = await pdf.getPage(i);
        const textContent = await page.getTextContent();
        const pageText = textContent.items
            .map(item => item.str)
            .join(' ');
        fullText += pageText + '\n\n';
    }
    
    return {
        name: file.name,
        text: fullText.trim(),
        pageCount
    };
}

/**
 * Check if a file is a PDF
 * @param {File} file
 * @returns {boolean}
 */
export function isPdf(file) {
    return file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
}

/**
 * Extract text from a plain text file
 * @param {File} file
 * @returns {Promise<{name: string, text: string}>}
 */
export async function extractTextFromFile(file) {
    const text = await file.text();
    return {
        name: file.name,
        text: text.trim(),
        pageCount: 1
    };
}

/**
 * Process any supported file (PDF or text)
 * @param {File} file
 * @returns {Promise<{name: string, text: string, pageCount: number}>}
 */
export async function processFile(file) {
    if (isPdf(file)) {
        return extractTextFromPdf(file);
    } else {
        return extractTextFromFile(file);
    }
}
