"""
Image extraction from PDF files.
Extracts embedded images using PyMuPDF (fitz) with validation.
"""

from typing import List, Dict, Optional
import io
import logging

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


def extract_images_from_pdf(
    pdf_path: str = None,
    pdf_bytes: bytes = None,
) -> List[Dict[str, any]]:
    """
    Extract embedded images from PDF pages.

    Args:
        pdf_path: Path to PDF file (optional if pdf_bytes provided)
        pdf_bytes: PDF file content as bytes (optional if pdf_path provided)

    Returns:
        List of dicts: [{"page": n, "image_index": i, "image": PIL.Image}, ...]
        Returns empty list if no images exist or extraction fails.
    """
    if not pdf_bytes and not pdf_path:
        raise ValueError("Either pdf_path or pdf_bytes must be provided")

    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf") if pdf_bytes else fitz.open(pdf_path)
        images_data: List[Dict[str, any]] = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            if not image_list:
                continue

            for img_idx, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image")
                    if not image_bytes:
                        continue

                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    if pil_img.width == 0 or pil_img.height == 0:
                        continue

                    images_data.append({
                        "page": page_index + 1,
                        "image_index": img_idx,
                        "image": pil_img
                    })
                except Exception as e:
                    logger.warning("Skipping invalid image on page %s (index %s): %s", page_index + 1, img_idx, e)
                    continue

        return images_data
    except Exception as e:
        logger.error("Failed to extract images from PDF: %s", e)
        return []
    finally:
        if doc is not None:
            doc.close()


def stream_image_pages(
    pdf_path: str = None,
    pdf_bytes: bytes = None,
):
    """
    Generator that yields embedded images one at a time for memory efficiency.

    Args:
        pdf_path: Path to PDF file
        pdf_bytes: PDF file content as bytes

    Yields:
        Dict with page number, image index, and PIL Image:
        {"page": n, "image_index": i, "image": PIL.Image}
    """
    if not pdf_bytes and not pdf_path:
        raise ValueError("Either pdf_path or pdf_bytes must be provided")

    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf") if pdf_bytes else fitz.open(pdf_path)
        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            if not image_list:
                continue
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image")
                    if not image_bytes:
                        continue
                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    if pil_img.width == 0 or pil_img.height == 0:
                        continue
                    yield {
                        "page": page_index + 1,
                        "image_index": img_idx,
                        "image": pil_img
                    }
                except Exception as e:
                    logger.warning("Skipping invalid image on page %s (index %s): %s", page_index + 1, img_idx, e)
                    continue
    except Exception as e:
        logger.error("Failed to stream images from PDF: %s", e)
        return
    finally:
        if doc is not None:
            doc.close()


def save_image_page(image: Image.Image, output_path: str, fmt: str = "jpg"):
    """
    Save a PIL Image to disk.
    
    Args:
        image: PIL Image object
        output_path: Path to save the image
        fmt: Image format - "jpg" or "png"
    """
    image.save(output_path, format=fmt.upper())

