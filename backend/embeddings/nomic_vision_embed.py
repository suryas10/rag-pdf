"""
Image embeddings using nomic-embed-vision-v1.5 model.
Generates dense embeddings for images extracted from PDFs.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from typing import List, Union, Optional
import numpy as np
from tqdm import tqdm


class NomicVisionEmbedder:
    """Generate image embeddings using Nomic Embed Vision model."""
    
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-vision-v1.5",
        batch_size: int = 8,
        device: Optional[str] = None
    ):
        """
        Initialize Nomic vision embedder.
        
        Args:
            model_name: HuggingFace model identifier
            batch_size: Batch size for processing (smaller for images)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading Nomic vision embedding model: {model_name} on {self.device}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
    
    def encode(
        self,
        images: Union[Image.Image, List[Image.Image]],
        show_progress: bool = True,
        convert_to_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode images into embeddings.
        
        Args:
            images: Single PIL Image or list of PIL Images
            show_progress: Show progress bar
            convert_to_numpy: Convert to numpy array instead of torch tensor
        
        Returns:
            Embeddings tensor or array
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        all_embeddings = []
        
        for i in tqdm(
            range(0, len(images), self.batch_size),
            desc="Encoding images",
            disable=not show_progress
        ):
            batch = images[i:i + self.batch_size]
            
            # Process images
            inputs = self.processor(batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token (first token) as embedding
                img_embeddings = outputs.last_hidden_state[:, 0]
            
            # L2 normalize
            img_embeddings = F.normalize(img_embeddings, p=2, dim=1)
            
            all_embeddings.append(img_embeddings.cpu())
        
        # Concatenate all batches
        result = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return result.numpy()
        
        return result
    
    def encode_with_metadata(
        self,
        images: List[Image.Image],
        metadata: Optional[List[dict]] = None,
        show_progress: bool = True
    ) -> List[dict]:
        """
        Encode images and return with metadata.
        
        Args:
            images: List of PIL Image objects
            metadata: Optional list of metadata dicts (one per image)
            show_progress: Show progress bar
        
        Returns:
            List of dicts with 'embedding', 'image', and metadata
        """
        embeddings = self.encode(images, show_progress=show_progress, convert_to_numpy=True)
        
        results = []
        for i, (image, embedding) in enumerate(zip(images, embeddings)):
            result = {
                "embedding": embedding.tolist(),
                "image": image  # Keep reference to original image
            }
            if metadata and i < len(metadata):
                result.update(metadata[i])
            results.append(result)
        
        return results

