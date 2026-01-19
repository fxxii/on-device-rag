# 🧠 On-Device RAG — Private, Browser-Native Retrieval-Augmented Generation

> **A production-grade RAG system running entirely in your browser** — demonstrating mastery of vector search, hybrid retrieval, and on-device LLM inference. Zero server. Zero API keys. Complete privacy.

[![Demo](https://img.shields.io/badge/Demo-Live-brightgreen)]()
[![WebGPU](https://img.shields.io/badge/WebGPU-Accelerated-blue)]()
[![Zero Backend](https://img.shields.io/badge/Backend-None%20Required-orange)]()
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local-purple)]()
[![Offline](https://img.shields.io/badge/Offline-Capable-success)]()

---

## 🎯 What Makes This Different

Most RAG tutorials rely on cloud APIs and pre-built abstractions. **This project goes deeper**:

| Typical RAG Demo | This Implementation |
|------------------|---------------------|
| Uses OpenAI API | Runs **entirely in-browser** with WebGPU |
| Black-box chunking | **Custom RecursiveCharacterTextSplitter** matching LangChain.js behavior |
| Single search method | **Hybrid Search**: BM25 + Semantic + Reciprocal Rank Fusion |
| Server-side embeddings | **Client-side embeddings** via Transformers.js |
| JSON export (slow, heavy) | **Binary Streamed Export** (.rag.gz) with Gzip compression |
| API key required | **Zero dependencies**, zero costs, complete privacy |

🚀 **Live Demo**: [https://fxxii.github.io/on-device-rag/](https://fxxii.github.io/on-device-rag/)

---

## ✨ Core Features

### 📄 Document Processing
- **Multi-format support**: PDF, TXT, and Markdown files
- **Drag-and-drop interface** with visual document grid
- **PDF.js integration** for client-side PDF text extraction
- **Desktop-style file icons** with ingestion status indicators

### 🔀 Hybrid Retrieval Engine
| Component | Implementation |
|-----------|----------------|
| **BM25 (Lexical)** | Custom TF-IDF with BM25 scoring (k1=1.5, b=0.75) |
| **Semantic Search** | Cosine similarity on 384-dim embeddings |
| **Rank Fusion** | Reciprocal Rank Fusion (k=60) combining both methods |

> BM25 catches exact keyword matches that semantic search misses. RRF elegantly combines rankings without needing score calibration.

### � Intelligent Text Chunking
**RecursiveCharacterTextSplitter** — production-grade chunking that:
- Splits by **semantic boundaries**: paragraphs → newlines → sentences → words
- Maintains **configurable overlap** (default: 50 chars) to preserve context
- Uses **iterative splitting** (non-recursive) to avoid stack overflow on large documents
- Async processing with **progress callbacks** for responsive UI

```
Document → [Chunk₁|overlap] [Chunk₂|overlap] [Chunk₃] → Embeddings → Vector Store
```

### 💾 Advanced Data Management
- **Binary Export (.rag.gz)** — Custom binary format with Gzip compression (~50% smaller than JSON)
- **Streaming I/O** — Uses File System Access API to stream exports directly to disk, preventing OOM
- **IndexedDB Persistence** — Changes saved immediately; works across page reloads



### 🚀 On-Device LLM Inference
Multiple model options optimized for different use cases:

| Model | Size | Context | Best For |
|-------|------|---------|----------|
| `SmolLM2-360M` | ~200MB | 2K tokens | **Fast responses, low VRAM** (Default) |
| `Qwen2-0.5B` | ~300MB | 32K tokens | Balanced performance |
| `Llama-3.2-1B` | ~600MB | 128K tokens | Highest quality |

**🔄 Dynamic Model Selection** — Click the footer to switch models at runtime without page reload!

All models run via **WebGPU** with streaming token generation.

### 📡 Offline Support
- **Works offline after first load** — Models cached in browser storage
- **Embedding-Only Mode** — No WebGPU? Embeddings + hybrid search still work fully
- **Graceful degradation** — UI adapts to show "Search Only" when LLM unavailable

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                              Browser                                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   📄 PDF.js      │    │   Transformers   │    │     WebLLM       │  │
│  │   Text Extract   │    │   Embeddings     │    │   Generation     │  │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘  │
│           │                       │                       │            │
│           ▼                       ▼                       ▼            │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   Recursive      │    │   Vector Store   │    │  Hybrid Search   │  │
│  │   Chunker        │───▶│   (IndexedDB)    │◀───│  BM25 + Semantic │  │
│  │   (Semantic)     │    │   + Memory Cache │    │  + RRF Fusion    │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- **Chrome 113+** or **Edge 113+** (WebGPU required for LLM)
- ~4GB free VRAM for full LLM features
- Embedding-only mode works on any modern browser

### Run Locally

```bash
# Option 1: Using npx serve
npx serve .

# Option 2: Using Python
python -m http.server 8000

# Option 3: VS Code Live Server
# Right-click index.html → "Open with Live Server"
```

Then open http://localhost:8000 (or http://localhost:5000 for serve).

### First Load
1. Models download automatically (~30MB embeddings + ~200-600MB LLM)
2. **Cached after first load** — subsequent visits are instant

---

## 📁 Project Structure

```
Day3-Client Side RAG/
├── index.html             # Main UI with glassmorphism design
├── css/
│   └── styles.css         # Premium dark theme (1000+ lines)
├── js/
│   ├── app.js             # Main orchestrator (950 lines)
│   ├── chunker.js         # RecursiveCharacterTextSplitter
│   ├── embedder.js        # Transformers.js wrapper
│   ├── retriever.js       # BM25 + Semantic + RRF hybrid search
│   ├── vectorStore.js     # IndexedDB with memory cache
│   ├── pdfParser.js       # PDF.js text extraction
│   └── llm.js             # WebLLM wrapper with streaming
└── README.md
```

---

## 🔧 Technical Deep Dive

### Why Hybrid Search?

```javascript
// Semantic search alone misses exact matches
Query: "What is GPT-4?"
Document: "GPT-4 is a large language model..."
→ Semantic: high similarity ✓

// BM25 catches keyword matches that embeddings miss
Query: "Error code 0x8007"
Document: "...returns error code 0x8007..."
→ Semantic: low similarity ✗
→ BM25: exact match ✓

// RRF combines rankings without score normalization
score = Σ 1/(k + rank)  // k=60 is empirically optimal

Top 3 Chunks (by RRF score) is sent to LLM for generating Response
```

> **Why RRF?** It combines rankings without needing score normalization — BM25 scores (0-10+) and cosine similarity (0-1) have incompatible scales, but ranks are universal.

### Why Overlapping Chunks?

Without overlap, context at chunk boundaries is lost:
```
Chunk 1: "...machine learning models require"
Chunk 2: "extensive training data to perform well."
```
With 50-char overlap, the boundary text appears in both chunks.

### Why IndexedDB over localStorage?

| localStorage | IndexedDB |
|--------------|-----------|
| 5-10MB limit | No practical limit |
| Strings only | Structured data with indices |
| Synchronous | Async, non-blocking |
| No transactions | ACID transactions |

### 🧠 Prompt Engineering

**Prompt structure** (from `llm.js`):
```
System: "Answer using ONLY the provided context. Cite sources [1], [2]..."
User: "Context:
  [1] {retrieved chunk 1}
  [2] {retrieved chunk 2}
  [3] {retrieved chunk 3}
  
  Question: {user's question}"
```

> **Privacy note**: Only the 3 most relevant chunks (~1500 chars) are sent to the LLM — not your entire knowledge base. The LLM also cites sources with `[1]`, `[2]` markers so you can verify answers.

---

## 🎨 UI/UX Highlights

- **Glassmorphism cards** with backdrop blur
- **Desktop-style document icons** with file type indicators
- **Real-time ingestion progress** with toast notifications
- **Confirmation modals** for destructive actions
- **Clickable model selector** — switch LLM models from the footer
- **WebGPU status detection** with graceful degradation
- **Responsive design** for all screen sizes

### ➕ Additional Features
- **Project Status Dashboard** — Real-time metrics on vector count, memory usage, and component health
- **One-Click Demo** — Instantly load a pre-embedded Wikipedia dataset to test without bringing your own files
- **Search Modes** — Toggle between "Hybrid Search" (default) and "Semantic Only" for debugging

---

---

## 🎯 Skills Demonstrated

| Area | Implementation |
|------|----------------|
| **AI/ML** | Vector embeddings, semantic search, LLM inference |
| **Information Retrieval** | BM25, TF-IDF, Reciprocal Rank Fusion |
| **NLP** | Text chunking with semantic boundaries |
| **WebGPU** | GPU-accelerated model inference in browser |
| **Storage** | IndexedDB persistence with memory caching |
| **Modern JavaScript** | ES Modules, async/await, dynamic imports |

---

## ⚠️ Limitations

### Browser & Hardware
- **Browser Support**: Full LLM features require WebGPU (Chrome/Edge 113+)
- **Memory**: LLM inference needs ~4GB VRAM; embedding works with less
- **Initial Load**: First-time model download takes 1-2 minutes (cached after)

### Large File Processing
- **Embedding Batch Size**: Files are processed one chunk at a time to avoid browser memory exhaustion
- **Large Documents** (1MB+): Browsers may slow down during ingestion. System warns before processing >1MB files.
- **Chunk Size**: Fixed at 512 characters (with 50-char overlap) for optimal retriever performance
- **PDF Complexity**: Scanned PDFs or complex layouts may extract poorly (no OCR support)
- **IndexedDB Limits**: Practical limit ~500MB-2GB depending on browser/disk space

### LLM Response Constraints
| Model | Context Window | Max Response | Practical Limit |
|-------|----------------|--------------|-----------------|
| SmolLM2-360M | 2K tokens | 512 tokens | ~1-2 short paragraphs |
| Qwen2-0.5B | 32K tokens | 512 tokens | ~1-2 paragraphs |
| Llama-3.2-1B | 128K tokens | 512 tokens | ~1-2 paragraphs |

> **Note**: Response length is capped at **512 tokens** (`max_tokens` in `llm.js`) to balance speed and quality with small models. Smaller models (360M-1B parameters) excel at concise, factual answers but may struggle with complex reasoning or lengthy explanations.

---

## 📄 License

MIT License — feel free to use for your portfolio!

---

<p align="center">
  Built with ❤️ using 
  <a href="https://huggingface.co/docs/transformers.js">Transformers.js</a> • 
  <a href="https://webllm.mlc.ai/">WebLLM</a> • 
  <a href="https://mozilla.github.io/pdf.js/">PDF.js</a>
</p>
