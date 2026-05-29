"""
index.py
--------
Vector index management for the PhD RAG Assistant.

This module handles building, saving, and loading the vector index
that powers semantic search over your academic documents.

How the index works:
1. Each document is split into overlapping chunks (~512 tokens each)
2. Each chunk is converted to a 384-dimensional vector by the embedding model
3. Vectors are stored in a VectorStoreIndex (in-memory by default)
4. At query time, your question is embedded and compared to all chunk vectors
5. The most similar chunks are retrieved and passed to the LLM

Why save the index?
- Building takes 3-8 minutes (embedding every chunk)
- Loading takes ~10 seconds
- After first build, you never wait again
- Index is saved to Google Drive so it persists across Colab sessions
"""

import os
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)


def build_index(all_docs: list[dict], save_path: str = None, verbose: bool = True) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from extracted PDF documents.

    LlamaIndex handles chunking and embedding automatically:
    - Splits each document into overlapping 512-token chunks
    - Runs each chunk through the configured Settings.embed_model
    - Stores all vectors for fast cosine similarity search

    The metadata (filename) is preserved per chunk so every answer
    can cite which PDF it came from.

    Args:
        all_docs (list[dict]): Documents from extract.py.
                               Each dict needs 'text' and 'filename'.
        save_path (str): Optional path to save index for reuse.
                         Strongly recommended — saves rebuild time.
        verbose (bool): Print progress. Default True.

    Returns:
        VectorStoreIndex: Built and optionally saved index.

    Example:
        >>> index = build_index(all_docs, save_path="/content/drive/MyDrive/phd_finetune/rag_index")
    """
    if not all_docs:
        raise ValueError(
            "No documents provided. Check that your PDFs were extracted correctly."
        )

    if verbose:
        print(f"⏳ Building index from {len(all_docs)} documents...")
        print("   (This takes 3–8 minutes on CPU, 2–3 mins on GPU)")
        print("   This only needs to run once — index will be saved.\n")

    # Convert dicts to LlamaIndex Document objects
    # The metadata dict is attached to every chunk derived from this document
    # so source citations work automatically at query time
    index_docs = [
        Document(
            text=doc["text"],
            metadata={
                "filename": doc["filename"],
                # Add more metadata here if needed, e.g.:
                # "source_type": doc.get("source_type", "unknown"),
                # "year": doc.get("year", ""),
            }
        )
        for doc in all_docs
    ]

    # Build the index — this is where embedding happens
    # Settings.embed_model must be configured before calling this
    index = VectorStoreIndex.from_documents(
        index_docs,
        show_progress=verbose,
    )

    # Save to disk if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        index.storage_context.persist(save_path)
        if verbose:
            print(f"\n✅ Index saved to: {save_path}")
            print("   Next session: use load_index() to reload in ~10 seconds")

    if verbose:
        print("🎉 Index ready!")

    return index


def load_index(index_path: str, verbose: bool = True) -> VectorStoreIndex:
    """
    Load a previously saved index from disk.

    Much faster than rebuilding — ~10 seconds vs 3-8 minutes.
    Use this at the start of every Colab session after the first build.

    Args:
        index_path (str): Path where index was saved by build_index().
        verbose (bool): Print progress. Default True.

    Returns:
        VectorStoreIndex: Loaded index ready for queries.

    Raises:
        FileNotFoundError: If index_path doesn't exist.

    Example:
        >>> index = load_index("/content/drive/MyDrive/phd_finetune/rag_index")
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No saved index found at: {index_path}\n"
            "Run build_index() first to create and save the index."
        )

    if verbose:
        print(f"⏳ Loading saved index from: {index_path}")

    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)

    if verbose:
        print("✅ Index loaded and ready!")

    return index


def load_or_build_index(
    all_docs: list[dict],
    index_path: str,
    verbose: bool = True
) -> VectorStoreIndex:
    """
    Smart loader: load saved index if it exists, otherwise build it.

    This is the recommended function to use in your notebook —
    it handles both the first run (builds and saves) and all
    subsequent runs (loads quickly from disk).

    Args:
        all_docs (list[dict]): Documents needed if building from scratch.
                               Can be empty list if index already exists.
        index_path (str): Path to save/load the index.
        verbose (bool): Print progress. Default True.

    Returns:
        VectorStoreIndex: Ready-to-query index.

    Example:
        >>> index = load_or_build_index(all_docs, "/content/drive/MyDrive/phd_finetune/rag_index")
    """
    if os.path.exists(index_path):
        if verbose:
            print("✅ Saved index found! Loading from Drive...")
        return load_index(index_path, verbose=verbose)
    else:
        if verbose:
            print("⚠️  No saved index found. Building from scratch...")
        return build_index(all_docs, save_path=index_path, verbose=verbose)
