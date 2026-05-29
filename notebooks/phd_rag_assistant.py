# ===========================================================================
# PhD RAG Assistant — Full Colab Notebook
# ===========================================================================
# Run each cell in order. After the first run, the index is saved to
# Google Drive and loads in ~10 seconds every subsequent session.
#
# CPU or GPU: both work. GPU is ~3 mins faster on the one-time index build.
# Set runtime: Runtime → Change runtime type → CPU or T4 GPU
# ===========================================================================


# ---------------------------------------------------------------------------
# CELL 1 — Install Packages
# ---------------------------------------------------------------------------
# Run this cell once. After it finishes, RESTART THE SESSION.
# Then start from Cell 2 — do NOT re-run this cell after restarting.
# ---------------------------------------------------------------------------

!pip install llama-index
!pip install llama-index-embeddings-huggingface
!pip install llama-index-llms-groq
!pip install llama-index-llms-huggingface-api
!pip install pdfplumber
!pip install sentence-transformers

print("\n✅ All packages installed!")
print("⚠️  RESTART THE SESSION NOW: Runtime → Restart Session")
print("   Then continue from Cell 2 — do NOT re-run this cell.")


# ---------------------------------------------------------------------------
# CELL 2 — Imports
# ---------------------------------------------------------------------------
# Run after restarting. Imports all required modules.
# ---------------------------------------------------------------------------

from google.colab import drive, userdata
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import pdfplumber
import shutil
import os

print("✅ All imports successful!")


# ---------------------------------------------------------------------------
# CELL 3 — Mount Google Drive + Verify Structure
# ---------------------------------------------------------------------------
# Mounts your Google Drive and checks that the PDF folders exist.
#
# Expected structure in your Drive:
#   MyDrive/phd_finetune/
#     thesis/        ← your thesis PDF
#     publications/  ← your published papers
#     literature/    ← related review articles
# ---------------------------------------------------------------------------

drive.mount('/content/drive')

DRIVE_BASE = "/content/drive/MyDrive/phd_finetune"
INDEX_PATH = f"{DRIVE_BASE}/rag_index"
folders = ["thesis", "publications", "literature"]

print(f"\n🔍 Checking folder structure at: {DRIVE_BASE}\n")
all_ok = True
for folder in folders:
    path = os.path.join(DRIVE_BASE, folder)
    if os.path.exists(path):
        pdfs = [f for f in os.listdir(path) if f.endswith(".pdf")]
        print(f"  ✅ {folder}/: {len(pdfs)} PDFs")
        for pdf in pdfs:
            print(f"     - {pdf}")
    else:
        print(f"  ❌ {folder}/: NOT FOUND")
        all_ok = False

if not all_ok:
    print("\n⚠️  Create the missing folders in Google Drive and add your PDFs.")
else:
    print("\n✅ Folder structure looks good!")


# ---------------------------------------------------------------------------
# CELL 4 — Configure LLM and Embeddings
# ---------------------------------------------------------------------------
# Sets up:
# - BAAI/bge-small-en-v1.5 for embeddings (local, CPU-friendly, ~90MB)
# - Groq Llama 3.1 8B for answer generation (free API, very fast)
#
# IMPORTANT: Add your Groq key to Colab Secrets (most secure):
#   Left sidebar → 🔑 Secrets → Add GROQ_API_KEY → paste your gsk_... key
#
# Get a free Groq key at: https://console.groq.com
# ---------------------------------------------------------------------------

# Load API key securely from Colab Secrets
try:
    GROQ_API_KEY = userdata.get("GROQ_API_KEY")
    print("✅ Groq API key loaded from Colab Secrets")
except Exception:
    # Fallback — replace with your key if not using Secrets
    GROQ_API_KEY = "gsk_your_key_here"
    print("⚠️  Using hardcoded key — consider using Colab Secrets instead")

# Embedding model — runs locally on CPU, no API needed
print("\n⏳ Loading embedding model (downloads ~90MB on first run)...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
    # Why bge-small?
    # - Only 90MB — fast to load on CPU
    # - 384-dim vectors — good accuracy for academic text
    # - No API key, no internet needed at query time
)
print("✅ Embedding model ready: BAAI/bge-small-en-v1.5")

# LLM — runs on Groq's servers, no local GPU needed
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    # Why Groq?
    # - Free: 14,400 requests/day
    # - Fast: ~500 tokens/sec
    # - Reliable: more stable than HuggingFace free API
    # - No local GPU needed
)
print("✅ LLM ready: Llama 3.1 8B (Groq)")


# ---------------------------------------------------------------------------
# CELL 5 — Load or Build Index
# ---------------------------------------------------------------------------
# Smart loader:
# - If a saved index exists in Drive → loads it in ~10 seconds
# - If not → copies PDFs locally, extracts text, builds index, saves it
#
# After the first build, this cell always takes ~10 seconds.
# ---------------------------------------------------------------------------

LOCAL_BASE = "/content/pdfs"


def extract_pdfs(folder_path):
    """
    Extract text from all PDFs in a folder.
    Handles errors per-page so one bad page doesn't lose the whole document.
    """
    documents = []
    if not os.path.exists(folder_path):
        print(f"  ⚠️  Folder not found: {folder_path}")
        return documents

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    for filename in pdf_files:
        filepath = os.path.join(folder_path, filename)
        print(f"\n  📄 Reading: {filename}")
        try:
            with pdfplumber.open(filepath) as pdf:
                full_text = ""
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            full_text += text + "\n"
                    except Exception as e:
                        print(f"    ⚠️  Skipping page {i+1}: {e}")
                        continue
                if full_text.strip():
                    documents.append({
                        "text": full_text.strip(),
                        "filename": filename
                    })
                    word_count = len(full_text.split())
                    print(f"  ✅ Done: {filename} ({len(pdf.pages)} pages, ~{word_count:,} words)")
                else:
                    print(f"  ⚠️  No text extracted (may be scanned/image-based): {filename}")
        except Exception as e:
            print(f"  ❌ Failed: {filename} → {e}")
    return documents


# ── Check for saved index ───────────────────────────────────────────────────
if os.path.exists(INDEX_PATH):
    print("✅ Saved index found! Loading from Drive (~10 seconds)...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
    index = load_index_from_storage(storage_context)
    print("✅ Index loaded! Jump to Cell 6 to ask questions.")

else:
    print("⚠️  No saved index found. Building from scratch...\n")

    # Step 1: Copy PDFs from Drive to local storage
    # (prevents OSError 107 connection drops during extraction)
    print("📁 Step 1/3: Copying PDFs from Drive to local storage...")
    for folder in folders:
        src = os.path.join(DRIVE_BASE, folder)
        dst = os.path.join(LOCAL_BASE, folder)
        os.makedirs(dst, exist_ok=True)
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            pdf_count = len([f for f in os.listdir(dst) if f.endswith(".pdf")])
            print(f"  ✅ {folder}/: {pdf_count} files copied")
        else:
            print(f"  ⚠️  {folder}/ not found in Drive — skipping")

    # Step 2: Extract text from all PDFs
    print("\n📄 Step 2/3: Extracting text from PDFs...")
    all_docs = []
    for folder in folders:
        print(f"\n📁 {folder}/")
        docs = extract_pdfs(os.path.join(LOCAL_BASE, folder))
        all_docs.extend(docs)
        print(f"   → {len(docs)} documents extracted")

    print(f"\n📊 Total documents extracted: {len(all_docs)}")

    if len(all_docs) == 0:
        print("\n❌ No documents extracted!")
        print("   Check that your PDFs are in the correct Drive folders.")
        print("   If your PDFs are scanned images, you'll need OCR — see docs/troubleshooting.md")
    else:
        # Step 3: Build and save the index
        print(f"\n⏳ Step 3/3: Building vector index...")
        print("   (This takes 3–8 mins on CPU, 2–3 mins on GPU)")
        print("   Only runs once — index saves to Drive after.\n")

        index_docs = [
            Document(
                text=doc["text"],
                metadata={"filename": doc["filename"]}
            )
            for doc in all_docs
        ]

        index = VectorStoreIndex.from_documents(index_docs, show_progress=True)

        # Save permanently to Google Drive
        os.makedirs(INDEX_PATH, exist_ok=True)
        index.storage_context.persist(INDEX_PATH)

        print(f"\n✅ Index built and saved to: {INDEX_PATH}")
        print("🎉 Ready to ask questions!")


# ---------------------------------------------------------------------------
# CELL 6 — Ask Preset Research Questions
# ---------------------------------------------------------------------------
# Runs a standard set of research questions against your documents.
# Each answer shows which PDFs were used as sources.
# ---------------------------------------------------------------------------

query_engine = index.as_query_engine(
    similarity_top_k=8,        # retrieve 8 most relevant chunks per question
    response_mode="tree_summarize",
    # tree_summarize: recursively summarises chunks before final answer
    # best for questions that span multiple documents or long contexts
)

questions = [
    "What is the main research problem or gap this work addresses?",
    "What are the key contributions and novelty of this research?",
    "What methodology and research design was used?",
    "What are the main findings and results?",
    "What are the limitations of this study?",
    "What future research directions are recommended?",
]

for q in questions:
    print(f"\n{'='*65}")
    print(f"Q: {q}")
    print(f"{'='*65}")

    response = query_engine.query(q)
    print(f"\n💡 Answer:\n{response.response}")

    print(f"\n📚 Sources used:")
    for i, node in enumerate(response.source_nodes):
        filename = node.metadata.get("filename", "unknown")
        score = node.score if node.score else 0.0
        print(f"  [{i+1}] {filename}  (relevance: {score:.3f})")


# ---------------------------------------------------------------------------
# CELL 7 — Interactive Q&A
# ---------------------------------------------------------------------------
# Ask anything about your research in a chat-like loop.
# Type 'quit' to stop.
# ---------------------------------------------------------------------------

query_engine = index.as_query_engine(
    similarity_top_k=8,
    response_mode="tree_summarize"
)

print("🎓 PhD Research Assistant — Interactive Q&A")
print("=" * 50)
print("Tips:")
print("  - Be specific: 'What EEG features were used?' > 'What features?'")
print("  - Reference specific papers: 'In the EMBC 2020 paper, what...'")
print("  - Ask about methods, findings, limitations, future work")
print("  - Type 'quit' to stop\n")

while True:
    try:
        question = input("Your question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nSession ended.")
        break

    if question.lower() in ["quit", "exit", "q", ""]:
        print("Session ended.")
        break

    print("\n⏳ Searching your documents...")
    response = query_engine.query(question)

    print(f"\n💡 Answer:\n{response.response}")

    print(f"\n📚 Sources ({len(response.source_nodes)} chunks):")
    for i, node in enumerate(response.source_nodes):
        filename = node.metadata.get("filename", "unknown")
        score = node.score if node.score else 0.0
        preview = node.text[:120].replace("\n", " ")
        print(f"  [{i+1}] {filename} (score: {score:.3f})")
        print(f"      \"{preview}...\"")

    print()
