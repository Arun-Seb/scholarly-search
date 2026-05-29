# 🎓 PhD Research RAG Assistant

> **Ask questions about your academic research papers using AI — runs free on CPU or GPU**

A Retrieval-Augmented Generation (RAG) system that lets you query your PhD thesis, publications, and literature review articles using natural language. Built to run entirely free on Google Colab — no GPU required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/phd-rag-assistant/blob/main/notebooks/phd_rag_assistant.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![CPU Compatible](https://img.shields.io/badge/CPU-Compatible-green)
![GPU Optional](https://img.shields.io/badge/GPU-Optional-orange)

---

## 📋 Table of Contents

- [What It Does](#-what-it-does)
- [Why RAG Instead of Fine-Tuning](#-why-rag-instead-of-fine-tuning)
- [Tech Stack](#-tech-stack)
- [CPU vs GPU](#-cpu-vs-gpu)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Detailed Setup Guide](#-detailed-setup-guide)
- [How the Code Works](#-how-the-code-works)
- [Example Questions](#-example-questions)
- [Troubleshooting](#-troubleshooting)
- [Extending the Project](#-extending-the-project)
- [Contributing](#-contributing)

---

## 🔍 What It Does

This tool allows you to:

- **Upload** your academic PDFs (thesis, papers, literature) to Google Drive
- **Index** them using semantic embeddings so the AI understands meaning, not just keywords
- **Ask questions** in plain English and get accurate answers grounded in your documents
- **See sources** — every answer tells you exactly which PDF and passage it came from
- **Save the index** to Google Drive so you never rebuild it twice

### Example interactions:
```
Q: What is the main contribution of this thesis?
Q: What methodology was used in the sleep study?
Q: What gaps in literature does this research address?
Q: Summarise the findings from the EMBC 2020 paper
Q: What are the recommendations for future research?
```

---

## 🤔 Why RAG Instead of Fine-Tuning?

| Approach | Fine-Tuning | RAG (This Project) |
|---|---|---|
| **Setup time** | 1–3 hours | 10 minutes |
| **GPU required** | Yes (strongly) | No — CPU works fine |
| **Cost** | Free tier limited | Completely free |
| **Updates docs** | Retrain required | Just re-index |
| **Cites sources** | No | Yes — shows exact source |
| **Best for** | Changing model behaviour | Q&A over documents |
| **Accuracy** | Can hallucinate | Grounded in your text |

**RAG is the right tool for document Q&A.** Fine-tuning is better when you want to change *how* a model speaks or reasons — not what it knows about specific documents.

---

## 🛠 Tech Stack

| Component | Tool | Why |
|---|---|---|
| **RAG Framework** | [LlamaIndex](https://www.llamaindex.ai/) | Best-in-class document indexing and retrieval |
| **Embeddings** | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | Fast, accurate, free, runs on CPU |
| **LLM** | [Groq API — Llama 3.1 8B](https://console.groq.com) | Free tier, very fast inference |
| **PDF Extraction** | [pdfplumber](https://github.com/jsvine/pdfplumber) | Reliable text extraction from academic PDFs |
| **Environment** | [Google Colab](https://colab.research.google.com) | Free compute, easy Drive integration |
| **Storage** | Google Drive | Persistent index across sessions |

---

## 💻 CPU vs GPU

**This project runs perfectly on CPU.** No GPU is needed.

### Why CPU works here:
- The heavy LLM inference (Llama 3.1 8B) runs on **Groq's servers** via API — not your machine
- The embedding model (`bge-small-en-v1.5`) is small enough to run fast on CPU
- Index building is a one-time cost (~3–5 mins on CPU, saved permanently after)

### Performance comparison:

| Task | CPU (Colab free) | GPU (T4) |
|---|---|---|
| Install packages | ~2 min | ~2 min |
| Copy PDFs locally | ~1 min | ~1 min |
| Build index (50 PDFs) | ~5–8 min | ~2–3 min |
| Load saved index | ~10 sec | ~5 sec |
| Answer a question | ~2–4 sec | ~1–2 sec |

**Once the index is saved to Drive, subsequent sessions load in ~10 seconds regardless of CPU or GPU.**

### To use CPU in Colab:
```
Runtime → Change runtime type → CPU
```
No other changes needed — all code works identically.

---

## 📁 Project Structure

```
phd-rag-assistant/
│
├── 📓 notebooks/
│   └── phd_rag_assistant.ipynb     # Main Colab notebook (run this)
│
├── 🐍 src/
│   ├── extract.py                  # PDF text extraction logic
│   ├── index.py                    # Index building and loading
│   ├── query.py                    # Query engine and Q&A logic
│   └── utils.py                    # Helper functions
│
├── 📄 docs/
│   ├── setup_guide.md              # Detailed setup instructions
│   ├── troubleshooting.md          # Common errors and fixes
│   └── extending.md                # How to extend the project
│
├── 📄 requirements.txt             # All Python dependencies
├── 📄 .gitignore                   # Excludes PDFs, keys, index
├── 📄 LICENSE                      # MIT License
└── 📄 README.md                    # This file
```

---

## ⚡ Quick Start

### 1. Get a Free Groq API Key
```
1. Go to https://console.groq.com
2. Sign up (free, takes 1 minute)
3. Click API Keys → Create API Key
4. Copy the key (starts with gsk_...)
```

### 2. Set Up Google Drive
```
Create this folder structure in your Google Drive:

📁 MyDrive/
└── 📁 phd_finetune/
    ├── 📁 thesis/          ← your thesis PDF
    ├── 📁 publications/    ← your published papers
    └── 📁 literature/      ← related review articles
```

### 3. Open the Notebook
Click the **Open in Colab** badge at the top of this README, or go to:
```
notebooks/phd_rag_assistant.ipynb
```

### 4. Run All Cells in Order
The notebook guides you through every step with clear output at each stage.

---

## 📖 Detailed Setup Guide

See [docs/setup_guide.md](docs/setup_guide.md) for a complete walkthrough including screenshots and expected outputs at every step.

---

## 🧠 How the Code Works

### Architecture Overview

```
Your PDFs (Google Drive)
        │
        ▼
  [PDF Extraction]
  pdfplumber reads each page,
  extracts raw text, skips
  unreadable pages gracefully
        │
        ▼
  [Chunking & Embedding]
  LlamaIndex splits text into
  overlapping chunks (~512 tokens)
  bge-small converts each chunk
  into a 384-dim vector
        │
        ▼
  [Vector Index]
  All vectors stored in an
  in-memory VectorStoreIndex,
  persisted to Google Drive
        │
        ▼
  [Query Time]
  Your question → embedded →
  top-K most similar chunks
  retrieved by cosine similarity
        │
        ▼
  [LLM Answer]
  Retrieved chunks + question
  sent to Llama 3.1 8B via Groq
  Answer generated, sources cited
```

### Key Code Components

#### 1. PDF Extraction (`src/extract.py`)
```python
def extract_pdfs(folder_path):
    """
    Reads all PDFs in a folder and returns a list of document dicts.
    
    - Handles extraction errors per-page (skips bad pages, continues)
    - Filters out empty documents
    - Tags each document with its source filename for citation
    
    Args:
        folder_path (str): Path to folder containing PDFs
    
    Returns:
        list[dict]: Each dict has 'text' and 'filename' keys
    """
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            try:
                with pdfplumber.open(filepath) as pdf:
                    full_text = ""
                    for i, page in enumerate(pdf.pages):
                        try:
                            text = page.extract_text()
                            if text:
                                full_text += text + "\n"
                        except Exception as e:
                            # Skip unreadable pages without crashing
                            print(f"  ⚠️ Skipping page {i+1}: {e}")
                            continue
                    if full_text.strip():
                        documents.append({
                            "text": full_text,
                            "filename": filename
                        })
            except Exception as e:
                print(f"  ❌ Failed: {filename} → {e}")
    return documents
```

**Why page-level error handling?** Academic PDFs often contain scanned images, figures, or corrupted pages. Without per-page try/catch, one bad page crashes the entire extraction. This approach extracts everything readable and skips what it can't handle.

---

#### 2. Index Building (`src/index.py`)
```python
def build_index(all_docs, index_path):
    """
    Converts extracted documents into a searchable vector index.
    
    LlamaIndex automatically:
    - Splits long documents into overlapping chunks
    - Runs each chunk through the embedding model
    - Stores vectors for fast similarity search
    
    The index is saved to disk so it only needs to be built once.
    
    Args:
        all_docs (list[dict]): Output from extract_pdfs()
        index_path (str): Where to save the index
    
    Returns:
        VectorStoreIndex: Ready-to-query index
    """
    index_docs = [
        Document(
            text=doc["text"],
            metadata={"filename": doc["filename"]}
        )
        for doc in all_docs
    ]
    
    index = VectorStoreIndex.from_documents(index_docs)
    index.storage_context.persist(index_path)
    return index


def load_index(index_path):
    """
    Loads a previously saved index from disk.
    Much faster than rebuilding — takes ~10 seconds vs 5 minutes.
    """
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    return load_index_from_storage(storage_context)
```

**Why save the index?** Building the index (embedding all chunks) takes 3–8 minutes and requires compute. Once saved, it loads in ~10 seconds every subsequent session. This makes the tool practical for daily use.

---

#### 3. Query Engine (`src/query.py`)
```python
def ask(index, question, top_k=8):
    """
    Queries the index and returns an answer with sources.
    
    How it works:
    1. Embeds the question into a vector
    2. Finds top_k most similar document chunks by cosine similarity
    3. Sends those chunks + the question to the LLM
    4. LLM synthesises an answer grounded in the retrieved text
    
    The 'tree_summarize' response mode is used for multi-document
    queries — it recursively summarises retrieved chunks before
    giving a final answer, improving coherence on long contexts.
    
    Args:
        index: VectorStoreIndex
        question (str): Natural language question
        top_k (int): Number of chunks to retrieve (default 8)
    
    Returns:
        response: LlamaIndex response object with .response and .source_nodes
    """
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        response_mode="tree_summarize"
    )
    return query_engine.query(question)
```

**Why `tree_summarize`?** For academic documents, answers often span multiple papers. `tree_summarize` recursively merges retrieved chunks into a coherent summary rather than just concatenating them — producing more readable and accurate answers.

---

#### 4. Settings Configuration
```python
# Embedding model — runs locally on CPU
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
    # 384-dimensional vectors
    # ~90MB model size
    # Excellent accuracy for academic text
    # No API key needed
)

# LLM — runs on Groq's free servers
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    # Free tier: 14,400 requests/day
    # ~500 tokens/sec — very fast
    # No local compute needed
)
```

**Why this combination?** The embedding model runs locally (no API cost, no latency) while the LLM runs on Groq (no local GPU needed, free tier is generous). This hybrid approach gives the best of both worlds for a zero-cost setup.

---

## 💬 Example Questions

Once your index is built, try asking:

```python
# Research overview
"What is the main research problem this thesis addresses?"
"What are the key contributions of this work?"
"How does this work differ from existing approaches?"

# Methodology
"What data collection methods were used?"
"How many participants were in the study?"
"What signal processing techniques were applied?"

# Findings
"What were the main findings of the research?"
"Were the hypotheses supported by the results?"
"What statistical methods were used to validate results?"

# Literature
"What gap in existing literature does this address?"
"Which authors are most cited in this work?"
"How does this compare to [specific paper/approach]?"

# Future work
"What are the limitations of this study?"
"What future research directions are suggested?"
"What are the clinical implications of these findings?"
```

---

## 🔧 Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md) for full details. Common issues:

| Error | Cause | Fix |
|---|---|---|
| `Transport endpoint not connected` | Google Drive disconnected | Remount drive, copy PDFs locally |
| `NameError: name 'index' is not defined` | Session restarted | Re-run cells from Cell 2 |
| `ImportError: cannot import HuggingFaceInferenceAPI` | Wrong package | `pip install llama-index-llms-huggingface-api` |
| `403 Forbidden` (HuggingFace) | Token permissions | Switch to Groq API instead |
| `No config file found` (Unsloth) | Wrong model name | Use `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` |
| Empty answers | PDF has no extractable text | Check if PDF is scanned — may need OCR |

---

## 🚀 Extending the Project

### Add a Chat Interface (Streamlit)
```python
# app.py
import streamlit as st
from src.index import load_index
from src.query import ask

st.title("🎓 PhD Research Assistant")
question = st.text_input("Ask a question about the research:")
if question:
    response = ask(index, question)
    st.write(response.response)
```

### Add OCR for Scanned PDFs
```python
# For scanned PDFs that pdfplumber can't read
!pip install pytesseract pdf2image
import pytesseract
from pdf2image import convert_from_path

def extract_scanned_pdf(filepath):
    images = convert_from_path(filepath)
    return "\n".join([pytesseract.image_to_string(img) for img in images])
```

### Fine-Tune on Top of RAG
After building the RAG system, you can also fine-tune a small model (Llama 3.2 3B) on your documents for even better domain-specific responses. See the fine-tuning notebook (coming soon).

---

## 🔒 Security Notes

- **Never commit API keys** — use Colab Secrets (left sidebar → 🔑) or `.env` files
- **Never commit your PDFs** — they may be under copyright or contain unpublished research
- The `.gitignore` in this repo excludes all PDFs, API keys, and the index folder

```python
# Safe way to use API keys in Colab
from google.colab import userdata
GROQ_API_KEY = userdata.get('GROQ_API_KEY')
```

---

## 📦 Requirements

```
llama-index>=0.10.0
llama-index-embeddings-huggingface>=0.2.0
llama-index-llms-groq>=0.1.0
llama-index-llms-huggingface-api>=0.2.0
pdfplumber>=0.10.0
sentence-transformers>=2.7.0
transformers>=4.40.0
accelerate>=0.30.0
groq>=0.5.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions welcome! Ideas for improvement:

- [ ] Add Streamlit web UI
- [ ] Support for `.docx` files
- [ ] OCR support for scanned PDFs
- [ ] Export answers to markdown/PDF report
- [ ] Multi-language support
- [ ] BibTeX citation export

Please open an issue before submitting a PR.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [LlamaIndex](https://www.llamaindex.ai/) for the RAG framework
- [Groq](https://groq.com) for free LLM inference
- [BAAI](https://huggingface.co/BAAI) for the BGE embedding models
- [Unsloth](https://unsloth.ai) for inspiration on efficient LLM workflows

---

*Built for researchers, by a researcher. If this helped your work, consider giving it a ⭐*
