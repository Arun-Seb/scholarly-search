"""
extract.py
----------
PDF text extraction module for the PhD RAG Assistant.

This module handles reading academic PDFs and extracting clean text.
It is designed to be robust — skipping unreadable pages or broken files
without crashing the entire pipeline.

Why pdfplumber?
- Better than PyPDF2 for academic papers (handles tables, columns)
- More reliable text extraction from research PDFs
- Active maintenance and good error messages

Limitations:
- Cannot extract text from scanned PDFs (images of text)
- Complex two-column layouts may have merged text
- For scanned PDFs, use extract_scanned_pdf() with OCR (requires tesseract)
"""

import os
import pdfplumber


def extract_pdfs(folder_path: str, verbose: bool = True) -> list[dict]:
    """
    Extract text from all PDFs in a given folder.

    Reads each PDF page by page. If a page fails (e.g. corrupted,
    image-only), it skips that page and continues with the rest.
    This prevents one bad page from losing an entire document.

    Args:
        folder_path (str): Path to folder containing PDF files.
        verbose (bool): Print progress messages. Default True.

    Returns:
        list[dict]: List of documents, each with:
            - 'text' (str): Full extracted text of the document
            - 'filename' (str): Original PDF filename (used for citations)

    Example:
        >>> docs = extract_pdfs("/content/pdfs/thesis")
        >>> print(f"Extracted {len(docs)} documents")
    """
    documents = []

    if not os.path.exists(folder_path):
        print(f"  ❌ Folder not found: {folder_path}")
        return documents

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"  ⚠️  No PDFs found in: {folder_path}")
        return documents

    for filename in pdf_files:
        filepath = os.path.join(folder_path, filename)
        if verbose:
            print(f"\n  📄 Reading: {filename}")

        try:
            with pdfplumber.open(filepath) as pdf:
                full_text = ""
                total_pages = len(pdf.pages)

                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            full_text += text + "\n"
                    except Exception as page_err:
                        # Skip this page but continue with the rest
                        if verbose:
                            print(f"    ⚠️  Skipping page {i+1}/{total_pages}: {page_err}")
                        continue

                if full_text.strip():
                    documents.append({
                        "text": full_text.strip(),
                        "filename": filename
                    })
                    if verbose:
                        word_count = len(full_text.split())
                        print(f"  ✅ Done: {filename} ({total_pages} pages, ~{word_count:,} words)")
                else:
                    if verbose:
                        print(f"  ⚠️  No text extracted from: {filename}")
                        print(f"      (PDF may be scanned/image-based — try extract_scanned_pdf())")

        except Exception as file_err:
            if verbose:
                print(f"  ❌ Failed to open: {filename}")
                print(f"     Error: {file_err}")
            continue

    return documents


def extract_all(base_path: str, folders: list[str] = None, verbose: bool = True) -> list[dict]:
    """
    Extract PDFs from multiple subfolders at once.

    Convenience function that loops over thesis/, publications/,
    and literature/ subfolders and combines all results.

    Args:
        base_path (str): Root folder containing subfolders.
        folders (list[str]): Subfolder names. Defaults to
                             ['thesis', 'publications', 'literature'].
        verbose (bool): Print progress. Default True.

    Returns:
        list[dict]: Combined list of all extracted documents.

    Example:
        >>> all_docs = extract_all("/content/pdfs")
        >>> print(f"Total: {len(all_docs)} documents")
    """
    if folders is None:
        folders = ["thesis", "publications", "literature"]

    all_docs = []
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if verbose:
            print(f"\n📁 Processing folder: {folder}/")
        docs = extract_pdfs(folder_path, verbose=verbose)
        all_docs.extend(docs)
        if verbose:
            print(f"   → {len(docs)} documents extracted from {folder}/")

    if verbose:
        print(f"\n📊 Total documents extracted: {len(all_docs)}")

    return all_docs


def extract_scanned_pdf(filepath: str) -> str:
    """
    Extract text from a scanned (image-based) PDF using OCR.

    Requires tesseract and poppler to be installed:
        !apt-get install tesseract-ocr poppler-utils
        !pip install pytesseract pdf2image

    This is slower than pdfplumber (~30 sec per page) but works
    on PDFs that are scans of printed documents.

    Args:
        filepath (str): Path to the scanned PDF file.

    Returns:
        str: Extracted text via OCR.

    Example:
        >>> text = extract_scanned_pdf("/content/pdfs/scanned_paper.pdf")
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "OCR dependencies not installed. Run:\n"
            "  !apt-get install tesseract-ocr poppler-utils\n"
            "  !pip install pytesseract pdf2image"
        )

    print(f"  🔍 Running OCR on: {os.path.basename(filepath)}")
    images = convert_from_path(filepath)
    pages_text = []

    for i, image in enumerate(images):
        print(f"    OCR page {i+1}/{len(images)}...")
        text = pytesseract.image_to_string(image)
        if text.strip():
            pages_text.append(text)

    return "\n".join(pages_text)
