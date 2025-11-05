"""
Text cleaning and chunking with overlapping semantic chunks.
Handles headers, footers, multi-column layouts, and dynamic chunking.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    page_no: int
    chunk_id: str
    start_offset: int
    end_offset: int
    metadata: Optional[Dict] = None


class TextCleaner:
    """Cleans raw text by removing headers, footers, repeated footnotes."""
    
    @staticmethod
    def remove_headers_footers(text: str, max_header_footer_length: int = 100) -> str:
        """Remove common headers and footers."""
        lines = text.split('\n')
        if len(lines) < 3:
            return text
        
        # Remove first and last lines if they're short (likely headers/footers)
        cleaned_lines = lines.copy()
        if len(lines[0]) < max_header_footer_length:
            cleaned_lines = cleaned_lines[1:]
        if len(lines) > 1 and len(lines[-1]) < max_header_footer_length:
            cleaned_lines = cleaned_lines[:-1]
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def remove_repeated_footnotes(text: str) -> str:
        """Remove repeated footnote patterns."""
        # Remove common footnote patterns
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)  # Numbers at end of lines
        text = re.sub(r'^Page \d+$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        return text
    
    @staticmethod
    def fix_reading_order(text: str) -> str:
        """Fix reading order for multi-column layouts (basic implementation)."""
        # This is a simplified version - advanced multi-column detection
        # would require layout analysis
        lines = text.split('\n')
        # Remove excessive whitespace
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Apply all cleaning operations."""
        text = TextCleaner.remove_headers_footers(text)
        text = TextCleaner.remove_repeated_footnotes(text)
        text = TextCleaner.fix_reading_order(text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()


class Chunker:
    """Creates overlapping semantic chunks from cleaned text."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 500,
        max_chunk_size: int = 800
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        return len(text) // 4
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into sentences for better chunking."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(
        self,
        text: str,
        page_no: int,
        chunk_prefix: str = ""
    ) -> List[Chunk]:
        """
        Create overlapping chunks from text.
        
        Args:
            text: Input text to chunk
            page_no: Page number for metadata
            chunk_prefix: Prefix for chunk IDs
        
        Returns:
            List of Chunk objects
        """
        # Clean the text first
        cleaned_text = TextCleaner.clean_text(text)
        
        if not cleaned_text:
            return []
        
        # Split into sentences
        sentences = self._split_text(cleaned_text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            sentence_tokens = self._estimate_tokens(sentence)
            
            # If adding this sentence would exceed max size, save current chunk
            if current_length + sentence_tokens > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk_id = f"{chunk_prefix}page_{page_no}_chunk_{chunk_idx}"
                    end_char = start_char + len(chunk_text)
                    chunks.append(Chunk(
                        text=chunk_text,
                        page_no=page_no,
                        chunk_id=chunk_id,
                        start_offset=start_char,
                        end_offset=end_char,
                        metadata={"original_length": len(chunk_text)}
                    ))
                    chunk_idx += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Keep last N sentences for overlap
                    overlap_tokens = 0
                    overlap_sentences = []
                    for s in reversed(current_chunk):
                        s_tokens = self._estimate_tokens(s)
                        if overlap_tokens + s_tokens <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_tokens += s_tokens
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_length = overlap_tokens
                else:
                    current_chunk = []
                    current_length = 0
                    start_char = end_char if chunks else 0
            
            current_chunk.append(sentence)
            current_length += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size // 2:  # Allow smaller final chunk
                chunk_id = f"{chunk_prefix}page_{page_no}_chunk_{chunk_idx}"
                end_char = start_char + len(chunk_text)
                chunks.append(Chunk(
                    text=chunk_text,
                    page_no=page_no,
                    chunk_id=chunk_id,
                    start_offset=start_char,
                    end_offset=end_char,
                    metadata={"original_length": len(chunk_text)}
                ))
        
        return chunks
    
    def chunk_pages(
        self,
        pages: List[Dict[str, any]],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Chunk]:
        """
        Chunk multiple pages of text.
        
        Args:
            pages: List of page dicts with "page" and "text" keys
            chunk_size: Override default chunk size
            chunk_overlap: Override default overlap
        
        Returns:
            List of Chunk objects
        """
        if chunk_size:
            self.chunk_size = chunk_size
        if chunk_overlap:
            self.chunk_overlap = chunk_overlap
        
        all_chunks = []
        for page_data in pages:
            page_no = page_data["page"]
            text = page_data["text"]
            chunks = self.chunk_text(text, page_no)
            all_chunks.extend(chunks)
        
        return all_chunks


def apply_redaction(text: str, redaction_patterns: List[str]) -> str:
    """
    Apply redaction patterns to text.
    
    Args:
        text: Input text
        redaction_patterns: List of regex patterns to redact
    
    Returns:
        Redacted text
    """
    redacted_text = text
    for pattern in redaction_patterns:
        redacted_text = re.sub(pattern, '[REDACTED]', redacted_text)
    return redacted_text

