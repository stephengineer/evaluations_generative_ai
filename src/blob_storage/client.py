"""
Azure Blob Storage client implementation.

Connects using account URL + SAS token from environment;
exposes container client, list_directories (virtual folders), list_blobs, and download_blob.
"""

import os

from azure.storage.blob import BlobPrefix, BlobServiceClient


class BlobStorageService:
    """
    Client for Azure Blob Storage.

    Reads AZURE_STORAGE_ACCOUNT_URL, AZURE_STORAGE_SAS_TOKEN, and optionally
    AZURE_STORAGE_CONTAINER from the environment. Raises ValueError if URL or SAS is missing.
    """

    def __init__(self) -> None:
        account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL", "").strip()
        sas_token = os.environ.get("AZURE_STORAGE_SAS_TOKEN", "").strip()
        if not account_url:
            raise ValueError(
                "AZURE_STORAGE_ACCOUNT_URL must be set (e.g. https://<account>.blob.core.windows.net/)"
            )
        if not sas_token:
            raise ValueError("AZURE_STORAGE_SAS_TOKEN must be set")
        self._account_url = account_url
        self._sas_token = sas_token
        self._default_container = os.environ.get(
            "AZURE_STORAGE_CONTAINER", "chat-data"
        ).strip()
        if not self._default_container:
            self._default_container = "chat-data"
        self._blob_service_client = BlobServiceClient(
            account_url=self._account_url,
            credential=self._sas_token,
        )

    def get_container_client(self, container_name: str | None = None):
        """Return the container client; use default container from env if name is None."""
        name = container_name if container_name is not None else self._default_container
        return self._blob_service_client.get_container_client(name)

    def list_directories(self, prefix: str = ""):
        """
        List virtual directory (prefix) names under the given prefix.

        Uses delimiter='/' so only the next level of "folders" is returned.
        Yields prefix names (e.g. "2024/", "2024/05/").
        """
        container = self.get_container_client()
        for item in container.walk_blobs(name_starts_with=prefix, delimiter="/"):
            if isinstance(item, BlobPrefix):
                yield item.name

    def list_blobs(self, prefix: str = ""):
        """List blobs under the given prefix. Yields BlobProperties for each blob."""
        container = self.get_container_client()
        yield from container.list_blobs(name_starts_with=prefix)

    def download_blob(self, blob_name: str) -> bytes:
        """Download a blob by name and return its content as bytes."""
        container = self.get_container_client()
        blob_client = container.get_blob_client(blob_name)
        return bytes(blob_client.download_blob().readall())
