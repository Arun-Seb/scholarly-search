# Troubleshooting Guide

Common errors and how to fix them.

---

## `OSError: [Errno 107] Transport endpoint is not connected`

**Cause:** Google Drive disconnected while reading a PDF.

**Fix:**
```python
# Remount Drive
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)
```
Then use `copy_pdfs_locally()` to copy files before reading — local reads never drop.

---

## `NameError: name 'index' is not defined`

**Cause:** Colab session restarted and lost all variables.

**Fix:** Re-run all cells from Cell 2 downward. The index loads from Drive in ~10 seconds — it does not need to be rebuilt.

---

## `ImportError: cannot import name 'HuggingFaceInferenceAPI'`

**Cause:** Wrong package installed.

**Fix:**
```bash
pip install llama-index-llms-huggingface-api
```
Then use the correct import:
```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
```
Or switch to Groq (recommended — more reliable):
```python
from llama_index.llms.groq import Groq
```

---

## `403 Forbidden` (HuggingFace API)

**Cause:** HuggingFace token doesn't have Inference Provider permissions.

**Fix:** Switch to Groq API instead. It's faster and more reliable on the free tier. See [setup_guide.md](setup_guide.md) Step 5.

---

## `RuntimeError: Unsloth: No config file found`

**Cause:** Using a model name that doesn't exist on HuggingFace. Note: Llama 3.2 does not have a 2B version — only 1B and 3B.

**Fix:** Use one of these verified model names:
```python
# Recommended
"unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

# Alternative
"unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit"

# Smallest option
"unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
```

---

## Empty answers / "I don't have information about that"

**Cause:** Either the PDF text wasn't extracted, or the question doesn't match the document content well.

**Fixes:**
1. Check extraction worked: `print(len(all_docs))` should be > 0
2. Increase `top_k`: `ask(index, question, top_k=12)`
3. Rephrase the question using terms that appear in the document
4. Check if the PDF is scanned — `pdfplumber` can't read image-based PDFs

---

## PDF has no extractable text (scanned document)

**Cause:** The PDF is a scan of a printed document — it's an image, not text.

**Fix:** Use OCR:
```bash
apt-get install tesseract-ocr poppler-utils
pip install pytesseract pdf2image
```
```python
from src.extract import extract_scanned_pdf
text = extract_scanned_pdf("/path/to/scanned.pdf")
```

---

## `WARNING: pip's dependency resolver...` (cuda-python conflict)

**Cause:** Minor version mismatch in CUDA packages.

**Fix:** Ignore it — this warning does not affect functionality. The packages that matter are installed correctly.

---

## Colab session keeps disconnecting

**Cause:** Free Colab disconnects after ~90 minutes of inactivity.

**Fix:**
- Save the index to Google Drive (already done if you ran `build_index` with `save_path`)
- On reconnect, just run Cells 2-5 — index loads in ~10 seconds
- For long index builds, keep the browser tab active

---

## `groq.AuthenticationError`

**Cause:** Invalid or expired Groq API key.

**Fix:**
1. Go to [console.groq.com](https://console.groq.com)
2. Regenerate your API key
3. Update in Colab Secrets
