# Setup Guide

Complete step-by-step setup for the PhD RAG Assistant.

---

## Prerequisites

- A Google account (for Colab + Drive)
- Your PDF files (thesis, papers, literature)
- A free Groq API key (takes 2 minutes)

No GPU, no paid subscriptions, no local installation required.

---

## Step 1: Get a Free Groq API Key

Groq provides free access to Llama 3.1 8B — the LLM that answers your questions.

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up with Google (free, instant)
3. Click **API Keys** in the left sidebar
4. Click **Create API Key**
5. Copy the key — it starts with `gsk_`

**Free tier limits:** 14,400 requests/day, 6,000 tokens/minute — more than enough for research Q&A.

---

## Step 2: Organise Your Google Drive

Create this folder structure in your Google Drive:

```
📁 My Drive/
└── 📁 phd_finetune/
    ├── 📁 thesis/          ← put your thesis PDF here
    ├── 📁 publications/    ← put your published papers here
    └── 📁 literature/      ← put related review articles here
```

**Tips:**
- PDFs can be any size — the system handles large thesis documents
- You can add more PDFs later and rebuild the index
- Folder names must match exactly (lowercase)

---

## Step 3: Open the Notebook in Colab

Option A — from GitHub:
1. Go to the notebook: `notebooks/phd_rag_assistant.ipynb`
2. Click **Open in Colab** badge

Option B — manually:
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook → select `phd_rag_assistant.ipynb`

---

## Step 4: Choose CPU or GPU

**CPU (recommended for this project):**
```
Runtime → Change runtime type → CPU
```

**GPU (optional, faster index building):**
```
Runtime → Change runtime type → T4 GPU
```

The system works identically on both. GPU only saves ~3 minutes on the one-time index build.

---

## Step 5: Add Your Groq Key Securely

In Colab, use Secrets instead of hardcoding the key:

1. Click the **🔑 key icon** in the left sidebar
2. Click **Add new secret**
3. Name: `GROQ_API_KEY`
4. Value: paste your `gsk_...` key
5. Toggle **Notebook access** to ON

Then in the notebook, load it safely:
```python
from google.colab import userdata
GROQ_API_KEY = userdata.get("GROQ_API_KEY")
```

---

## Step 6: Run the Notebook

Run cells in order — each cell has a clear description of what it does.

**Expected timeline (first run):**

| Step | Time (CPU) | Time (GPU) |
|---|---|---|
| Install packages | ~2 min | ~2 min |
| Mount Drive + copy PDFs | ~1 min | ~1 min |
| Load embedding model | ~1 min | ~30 sec |
| Build index (50 PDFs) | ~5-8 min | ~2-3 min |
| First question | ~3-5 sec | ~2-3 sec |
| **Total first run** | **~12 min** | **~8 min** |

**Every subsequent session:**

| Step | Time |
|---|---|
| Mount Drive | ~30 sec |
| Load models | ~1 min |
| Load saved index | ~10 sec |
| Ask questions | ~3-5 sec |
| **Total** | **~2 min** |

---

## Step 7: Ask Questions

Once the index is loaded, use the interactive Q&A cell:

```python
interactive_qa(index)
```

Or ask individual questions:

```python
ask(index, "What is the main contribution of this thesis?")
```

---

## After the First Run

The index is saved to:
```
📁 MyDrive/phd_finetune/rag_index/
```

Every future session, the notebook loads this saved index — no rebuilding needed.

If you add new PDFs, delete the `rag_index` folder from Drive and rebuild.
