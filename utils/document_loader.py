import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger("document_loader")

def extract_text_from_file(file_path: str, filename: str) -> Dict[str, Any]:
    """
    Extracts text content from uploaded files (.pdf, .docx, .txt, .md, .html).
    Returns dict with success, text, filename, file_type, and chunk_count.
    """
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    success = False
    error = None

    try:
        if ext == ".pdf":
            import pypdf
            reader = pypdf.PdfReader(file_path)
            page_texts = []
            for idx, page in enumerate(reader.pages):
                p_text = page.extract_text() or ""
                if p_text.strip():
                    page_texts.append(f"--- Page {idx + 1} ---\n{p_text.strip()}")
            text = "\n\n".join(page_texts)
            success = True
        elif ext == ".docx":
            import docx
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs)
            success = True
        elif ext in [".txt", ".md", ".json", ".csv", ".py", ".html"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            success = True
        else:
            error = f"Unsupported file extension: {ext}"
            logger.warning(error)
    except Exception as e:
        error = f"Error reading {filename}: {str(e)}"
        logger.error(error)

    if success and text:
        # Simple chunking by 1000 characters
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        return {
            "success": True,
            "filename": filename,
            "file_type": ext,
            "text": text,
            "chunk_count": len(chunks),
            "error": None
        }

    return {
        "success": False,
        "filename": filename,
        "file_type": ext,
        "text": "",
        "chunk_count": 0,
        "error": error or "Failed to extract text from file"
    }
