"""Backend API client for Streamlit UI."""

from typing import Dict, Optional, Generator, Any
import requests
import base64
import json


class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def get_health(self) -> Optional[Dict]:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    def upload_file(self, filename: str, file_bytes: bytes, chunk_size: Optional[int], chunk_overlap: Optional[int], include_images: bool) -> Optional[Dict]:
        try:
            files = {"file": (filename, file_bytes, "application/pdf")}
            data = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "include_images": str(include_images).lower()
            }
            response = requests.post(f"{self.base_url}/upload", files=files, data=data, timeout=600)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    def get_ingestion_status(self, job_id: str) -> Optional[Dict]:
        try:
            response = requests.get(f"{self.base_url}/ingestion/status/{job_id}", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    def query(self, payload: Dict) -> Optional[Dict]:
        try:
            response = requests.post(f"{self.base_url}/query", json=payload, timeout=120)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    def query_stream(self, payload: Dict) -> Generator[Dict[str, Any], None, None]:
        response = requests.post(f"{self.base_url}/query/stream", json=payload, stream=True, timeout=120)
        if response.status_code != 200:
            yield {"type": "error", "message": response.text}
            return
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

    def clear_conversation(self, file_id: Optional[str] = None) -> bool:
        try:
            payload = {"file_id": file_id} if file_id else {}
            response = requests.post(f"{self.base_url}/conversation/clear", json=payload, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def clear_index(self) -> bool:
        try:
            response = requests.post(f"{self.base_url}/index/clear", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def reset(self) -> bool:
        try:
            response = requests.post(f"{self.base_url}/reset", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    @staticmethod
    def encode_image_to_base64(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")
