"""
Image extraction from PDF files.
Converts PDF pages to images using pdf2image.
"""

from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
from typing import List, Dict, Optional
import io


def extract_images_from_pdf(
    pdf_path: str = None,
    pdf_bytes: bytes = None,
    dpi: int = 300,
    fmt: str = "jpg"
) -> List[Dict[str, any]]:
    """
    Extract images from PDF pages.
    
    Args:
        pdf_path: Path to PDF file (optional if pdf_bytes provided)
        pdf_bytes: PDF file content as bytes (optional if pdf_path provided)
        dpi: DPI for image conversion (default: 300)
        fmt: Image format - "jpg" or "png" (default: "jpg")
    
    Returns:
        List of dicts: [{"page": n, "image": PIL.Image}, ...]
    """
    if pdf_bytes:
        images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt=fmt)
    elif pdf_path:
        images = convert_from_path(pdf_path, dpi=dpi, fmt=fmt)
    else:
        raise ValueError("Either pdf_path or pdf_bytes must be provided")
    
    pages_data = []
    for page_num, image in enumerate(images, start=1):
        pages_data.append({
            "page": page_num,
            "image": image
        })
    
    return pages_data


def stream_image_pages(
    pdf_path: str = None,
    pdf_bytes: bytes = None,
    dpi: int = 300,
    fmt: str = "jpg"
):
    """
    Generator that yields image pages one at a time for memory efficiency.
    
    Args:
        pdf_path: Path to PDF file
        pdf_bytes: PDF file content as bytes
        dpi: DPI for image conversion
        fmt: Image format - "jpg" or "png"
    
    Yields:
        Dict with page number and PIL Image: {"page": n, "image": PIL.Image}
    """
    if pdf_bytes:
        images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt=fmt)
    elif pdf_path:
        images = convert_from_path(pdf_path, dpi=dpi, fmt=fmt)
    else:
        raise ValueError("Either pdf_path or pdf_bytes must be provided")
    
    for page_num, image in enumerate(images, start=1):
        yield {
            "page": page_num,
            "image": image
        }


def save_image_page(image: Image.Image, output_path: str, fmt: str = "jpg"):
    """
    Save a PIL Image to disk.
    
    Args:
        image: PIL Image object
        output_path: Path to save the image
        fmt: Image format - "jpg" or "png"
    """
    image.save(output_path, format=fmt.upper())

