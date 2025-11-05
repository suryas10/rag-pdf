"""
Text extraction from PDF files.
Converts PDF pages to raw text with memory-efficient streaming.
"""

import pypdf
from typing import List, Dict
import io


def extract_text_from_pdf(pdf_path: str = None, pdf_bytes: bytes = None) -> List[Dict[str, any]]:
    """
    Extract text from PDF file, returning structured list of pages.
    
    Args:
        pdf_path: Path to PDF file (optional if pdf_bytes provided)
        pdf_bytes: PDF file content as bytes (optional if pdf_path provided)
    
    Returns:
        List of dicts: [{"page": n, "text": "..."}, ...]
    """
    if pdf_bytes:
        pdf_file = io.BytesIO(pdf_bytes)
    elif pdf_path:
        pdf_file = open(pdf_path, 'rb')
    else:
        raise ValueError("Either pdf_path or pdf_bytes must be provided")
    
    try:
        reader = pypdf.PdfReader(pdf_file)
        pages_data = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                pages_data.append({
                    "page": page_num,
                    "text": text
                })
            except Exception as e:
                # Skip pages that can't be extracted
                pages_data.append({
                    "page": page_num,
                    "text": f"[Error extracting page {page_num}: {str(e)}]"
                })
        
        return pages_data
    finally:
        if pdf_path and pdf_file:
            pdf_file.close()


def stream_text_pages(pdf_path: str = None, pdf_bytes: bytes = None):
    """
    Generator that yields text pages one at a time for memory efficiency.
    
    Args:
        pdf_path: Path to PDF file
        pdf_bytes: PDF file content as bytes
    
    Yields:
        Dict with page number and text: {"page": n, "text": "..."}
    """
    if pdf_bytes:
        pdf_file = io.BytesIO(pdf_bytes)
    elif pdf_path:
        pdf_file = open(pdf_path, 'rb')
    else:
        raise ValueError("Either pdf_path or pdf_bytes must be provided")
    
    try:
        reader = pypdf.PdfReader(pdf_file)
        
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                yield {
                    "page": page_num,
                    "text": text
                }
            except Exception as e:
                yield {
                    "page": page_num,
                    "text": f"[Error extracting page {page_num}: {str(e)}]"
                }
    finally:
        if pdf_path and pdf_file:
            pdf_file.close()

