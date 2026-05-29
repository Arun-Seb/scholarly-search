"""
utils.py
--------
Utility functions for the PhD RAG Assistant.

Helpers for:
- Setting up the LLM and embedding model
- Copying PDFs from Google Drive to local storage
- Verifying folder structure
- Safe API key loading
"""

import os
import shutil


def setup_models(groq_api_key: str, embed_model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Configure LlamaIndex global settings for embeddings and LLM.

    This must be called before building or loading the index.
    Uses:
    - HuggingFace bge-small for embeddings (local, CPU-friendly, free)
    - Groq Llama 3.1 8B for generation (API, free tier, very fast)

    Why bge-small-en-v1.5?
    - Only 90MB — loads fast on CPU
    - 384-dimensional vectors — good balance of speed vs accuracy
    - Excellent performance on academic/scientific text
    - No API key or internet needed at query time (only at index build)

    Why Groq?
    - Free tier: 14,400 requests/day, 500 tokens/sec
    - No local GPU needed
    - Much more reliable than HuggingFace Inference API free tier
    - Llama 3.1 8B is strong enough for research Q&A

    Args:
        groq_api_key (str): Your Groq API key (starts with gsk_).
                            Get one free at https://console.groq.com
        embed_model_name (str): HuggingFace embedding model to use.
                                Default: BAAI/bge-small-en-v1.5

    Example:
        >>> setup_models(groq_api_key="gsk_...")
        ✅ Embedding model ready: BAAI/bge-small-en-v1.5
        ✅ LLM ready: llama-3.1-8b-instant (Groq)
    """
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.groq import Groq

    print(f"⏳ Loading embedding model: {embed_model_name}")
    print("   (Downloads ~90MB on first run, cached after)")
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    print(f"✅ Embedding model ready: {embed_model_name}")

    Settings.llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
    )
    print("✅ LLM ready: llama-3.1-8b-instant (Groq)")


def get_groq_key() -> str:
    """
    Safely retrieve Groq API key from Colab Secrets or environment.

    Tries Colab Secrets first (most secure), falls back to
    environment variable. Never hardcode keys in notebooks.

    Returns:
        str: Groq API key.

    Raises:
        ValueError: If key not found in either location.

    Setup in Colab:
        Left sidebar → 🔑 Secrets → Add secret:
        Name: GROQ_API_KEY
        Value: gsk_your_key_here

    Example:
        >>> key = get_groq_key()
        >>> print(key[:8] + "...")  # gsk_xxxx...
    """
    # Try Colab Secrets first
    try:
        from google.colab import userdata
        key = userdata.get("GROQ_API_KEY")
        if key:
            print("✅ Groq API key loaded from Colab Secrets")
            return key
    except Exception:
        pass

    # Fall back to environment variable
    key = os.environ.get("GROQ_API_KEY")
    if key:
        print("✅ Groq API key loaded from environment variable")
        return key

    raise ValueError(
        "Groq API key not found!\n"
        "Option 1 (Recommended): Add to Colab Secrets\n"
        "  Left sidebar → 🔑 Secrets → Add GROQ_API_KEY\n"
        "Option 2: Set environment variable\n"
        "  import os; os.environ['GROQ_API_KEY'] = 'gsk_...'\n"
        "Get a free key at: https://console.groq.com"
    )


def copy_pdfs_locally(
    drive_base: str,
    local_base: str = "/content/pdfs",
    folders: list[str] = None,
    verbose: bool = True
) -> str:
    """
    Copy PDFs from Google Drive to Colab local storage.

    Why copy locally?
    - Google Drive connection can drop mid-read (OSError 107)
    - Local reads are 3-5x faster than Drive reads
    - No connection interruptions during index building

    Args:
        drive_base (str): Base path in Google Drive.
                          e.g. "/content/drive/MyDrive/phd_finetune"
        local_base (str): Where to copy files locally.
                          Default: "/content/pdfs"
        folders (list[str]): Subfolders to copy.
                             Default: ['thesis', 'publications', 'literature']
        verbose (bool): Print progress. Default True.

    Returns:
        str: Local base path (for use with extract.py).

    Example:
        >>> local_path = copy_pdfs_locally(
        ...     drive_base="/content/drive/MyDrive/phd_finetune"
        ... )
        >>> all_docs = extract_all(local_path)
    """
    if folders is None:
        folders = ["thesis", "publications", "literature"]

    if verbose:
        print("📁 Copying PDFs from Google Drive to local storage...")
        print("   (Prevents connection drop errors during extraction)\n")

    for folder in folders:
        src = os.path.join(drive_base, folder)
        dst = os.path.join(local_base, folder)
        os.makedirs(dst, exist_ok=True)

        if not os.path.exists(src):
            if verbose:
                print(f"  ⚠️  Source folder not found: {src}")
                print(f"      Create this folder in Google Drive and add your PDFs")
            continue

        shutil.copytree(src, dst, dirs_exist_ok=True)
        pdf_count = len([f for f in os.listdir(dst) if f.endswith(".pdf")])

        if verbose:
            print(f"  ✅ {folder}/: {pdf_count} PDFs copied → {dst}")

    if verbose:
        print(f"\n✅ All PDFs copied to: {local_base}")

    return local_base


def verify_drive_structure(drive_base: str, folders: list[str] = None) -> bool:
    """
    Check that the expected Google Drive folder structure exists.

    Call this before copying PDFs to catch path errors early.

    Args:
        drive_base (str): Base folder path to check.
        folders (list[str]): Expected subfolders. Default: standard 3.

    Returns:
        bool: True if all folders exist and contain PDFs, False otherwise.

    Example:
        >>> ok = verify_drive_structure("/content/drive/MyDrive/phd_finetune")
        >>> if not ok:
        ...     print("Fix your folder structure before continuing")
    """
    if folders is None:
        folders = ["thesis", "publications", "literature"]

    all_ok = True
    print(f"🔍 Checking Google Drive structure at: {drive_base}\n")

    for folder in folders:
        path = os.path.join(drive_base, folder)
        if os.path.exists(path):
            pdfs = [f for f in os.listdir(path) if f.endswith(".pdf")]
            if pdfs:
                print(f"  ✅ {folder}/: {len(pdfs)} PDFs found")
                for pdf in pdfs:
                    print(f"     - {pdf}")
            else:
                print(f"  ⚠️  {folder}/: folder exists but NO PDFs found")
                all_ok = False
        else:
            print(f"  ❌ {folder}/: folder NOT found")
            all_ok = False

    if all_ok:
        print("\n✅ Drive structure looks good!")
    else:
        print("\n⚠️  Fix the issues above before proceeding.")
        print(f"   Expected structure:")
        print(f"   {drive_base}/")
        for f in folders:
            print(f"     {f}/  ← add your PDFs here")

    return all_ok
