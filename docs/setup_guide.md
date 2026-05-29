# 🎓 Scholarly Search — PhD Research RAG Assistant

> **Ask questions about your academic research papers using AI — runs free on CPU or GPU**

A Retrieval-Augmented Generation (RAG) system that lets you query your PhD thesis, publications, and literature review articles using natural language. Built to run entirely free on Google Colab — no GPU required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arun-Seb/scholarly-search/blob/main/notebooks/phd_rag_assistant.ipynb)
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
- [Possible Applications](#-possible-applications)
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
scholarly-search/
│
├── 📓 notebooks/
│   └── phd_rag_assistant.py     # Main Colab notebook (run this)
│
├── 🐍 src/
│   ├── extract.py               # PDF text extraction logic
│   ├── index.py                 # Index building and loading
│   ├── query.py                 # Query engine and Q&A logic
│   └── utils.py                 # Helper functions
│
├── 📄 docs/
│   ├── setup_guide.md           # Detailed setup instructions
│   └── troubleshooting.md       # Common errors and fixes
│
├── 📄 requirements.txt          # All Python dependencies
├── 📄 .gitignore                # Excludes PDFs, keys, index
├── 📄 LICENSE                   # MIT License
└── 📄 README.md                 # This file
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
notebooks/phd_rag_assistant.py
```

### 4. Run All Cells in Order
The notebook guides you through every step with clear output at each stage.

---

## 📖 Detailed Setup Guide

See [docs/setup_guide.md](docs/setup_guide.md) for a complete walkthrough including expected outputs at every step.

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

#### 2. Index Building (`src/index.py`)
```python
def build_index(all_docs, index_path):
    """
    Converts extracted documents into a searchable vector index.
    Saves to disk so it only needs to be built once.
    """
    index_docs = [
        Document(text=doc["text"], metadata={"filename": doc["filename"]})
        for doc in all_docs
    ]
    index = VectorStoreIndex.from_documents(index_docs)
    index.storage_context.persist(index_path)
    return index
```

#### 3. Query Engine (`src/query.py`)
```python
def ask(index, question, top_k=8):
    """
    Queries the index and returns an answer with sources.
    Uses tree_summarize for coherent multi-document answers.
    """
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        response_mode="tree_summarize"
    )
    return query_engine.query(question)
```

#### 4. Settings Configuration
```python
# Embedding model — runs locally on CPU, no API needed
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
    # 384-dimensional vectors, ~90MB, excellent for academic text
)

# LLM — runs on Groq's free servers, no local GPU needed
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    # Free tier: 14,400 requests/day, ~500 tokens/sec
)
```

---

## 💬 Example Questions

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

# Future work
"What are the limitations of this study?"
"What future research directions are suggested?"
"What are the clinical implications of these findings?"
```

---

## 🚀 Possible Applications

Scholarly Search is a foundation you can build on. Here are the most useful extensions for PhD researchers:

---

### 1. 🌐 Streamlit Web App
Turn it into a proper chat interface anyone can use — no Colab needed:

```python
# app.py
import streamlit as st
from src.query import ask

st.title("🎓 Scholarly Search")
question = st.text_input("Ask a question about your research:")
if question:
    response = ask(index, question)
    st.write(response.response)
    for node in response.source_nodes:
        st.caption(f"Source: {node.metadata['filename']} (score: {node.score:.3f})")
```

```bash
pip install streamlit
streamlit run app.py
```

Deploy free on **streamlit.io** — share with your supervisor via a single link.

---

### 2. 📊 Research Gap Finder
Automatically surface unanswered questions across all your literature:

```python
ask(index, "What research questions remain unanswered across all papers?")
ask(index, "What contradictions exist between different studies?")
ask(index, "What methods have not been tried yet in this field?")
```

Great for identifying your next publication angle.

---

### 3. 📝 Auto Literature Review Writer
Generate draft paragraphs for your literature review chapter:

```python
ask(index, "Write a literature review paragraph on sleep staging methods")
ask(index, "Summarise how different papers approach EEG signal processing")
ask(index, "Compare the methodologies used across all studies")
```

Use as a first draft — always review and edit before submitting.

---

### 4. 🔍 Citation Finder
Find exactly which papers support a specific claim:

```python
ask(index, "Which papers support the use of deep learning for sleep analysis?")
ask(index, "What evidence exists for [specific finding]?")
```

Saves hours of manual searching during thesis writing.

---

### 5. 🎓 Thesis Writing Assistant
Get AI help writing specific thesis sections grounded in your actual research:

```python
ask(index, "Draft an abstract based on the key findings across all papers")
ask(index, "Write a conclusion chapter based on all results")
ask(index, "Suggest a limitations section based on the methodology used")
ask(index, "Generate research questions for a future study")
```

---

### 6. 📧 Supervisor Email Drafter
Summarise your progress for supervisor meetings:

```python
ask(index, "Summarise the latest findings to report to my supervisor")
ask(index, "What have been the key milestones in this research so far?")
```

---

### 7. 📱 Mobile-Friendly App (Gradio)
Quick interface that works on any device including phones:

```python
import gradio as gr

def answer(question):
    response = ask(index, question, verbose=False)
    return response.response

gr.Interface(fn=answer, inputs="text", outputs="text",
             title="Scholarly Search").launch(share=True)
```

```bash
pip install gradio
```

Deploy free on **huggingface.co/spaces** — accessible from any browser.

---

### 8. 🤖 Slack / Teams Bot
Bring your research assistant into your team workspace:
- Share the RAG system across your research group
- Everyone in the lab can query the shared literature pool
- Useful for lab meetings and collaborative writing

---

### Application Comparison

| Application | Effort | Impact | Best For |
|---|---|---|---|
| **Streamlit web UI** | Low | ⭐⭐⭐⭐⭐ | Daily use, sharing with supervisor |
| **Literature review writer** | Low | ⭐⭐⭐⭐⭐ | Thesis writing |
| **Research gap finder** | Low | ⭐⭐⭐⭐ | Finding next publication topic |
| **Citation finder** | Low | ⭐⭐⭐⭐ | Reference management |
| **Gradio mobile app** | Low | ⭐⭐⭐⭐ | On-the-go access |
| **Thesis writing assistant** | Low | ⭐⭐⭐⭐ | Chapter drafting |
| **Slack/Teams bot** | Medium | ⭐⭐⭐ | Research group collaboration |

---

## 🔧 Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md) for full details. Common issues:

| Error | Cause | Fix |
|---|---|---|
| `Transport endpoint not connected` | Google Drive disconnected | Remount drive, copy PDFs locally |
| `NameError: name 'index' is not defined` | Session restarted | Re-run cells from Cell 2 |
| `ImportError: cannot import HuggingFaceInferenceAPI` | Wrong package | `pip install llama-index-llms-huggingface-api` |
| `403 Forbidden` (HuggingFace) | Token permissions | Switch to Groq API instead |
| Empty answers | PDF has no extractable text | Check if PDF is scanned — may need OCR |

---

## 🔩 Extending the Project

### Add OCR for Scanned PDFs
```python
!apt-get install tesseract-ocr poppler-utils
!pip install pytesseract pdf2image

from pdf2image import convert_from_path
import pytesseract

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

- [ ] Streamlit web UI
- [ ] Gradio mobile interface
- [ ] Support for `.docx` files
- [ ] OCR support for scanned PDFs
- [ ] Export answers to markdown/PDF report
- [ ] Multi-language support
- [ ] BibTeX citation export
- [ ] Slack / Teams bot integration
- [ ] Research gap auto-detection
- [ ] Supervisor email draft generator

Please open an issue before submitting a PR.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [LlamaIndex](https://www.llamaindex.ai/) for the RAG framework
- [Groq](https://groq.com) for free LLM inference
- [BAAI](https://huggingface.co/BAAI) for the BGE embedding models

---

*Built for researchers, by a researcher. If this helped your work, consider giving it a ⭐*
