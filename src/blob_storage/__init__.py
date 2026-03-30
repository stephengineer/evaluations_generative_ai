"""
Azure Blob Storage client for listing containers, directories (prefixes), and blobs.

Credentials are read from environment variables:
- AZURE_STORAGE_ACCOUNT_URL (required)
- AZURE_STORAGE_SAS_TOKEN (required)
- AZURE_STORAGE_CONTAINER (optional, default: chat-data)
"""

from src.blob_storage.client import BlobStorageService

__all__ = ["BlobStorageService"]
