"""
Tests for the Azure Blob Storage client (src.blob_storage).

Run with: pytest tests/test_blob_storage.py -v
"""

import os
import sys
from unittest import mock

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestBlobStorageServiceInit:
    """BlobStorageService construction and env validation."""

    def test_raises_when_account_url_missing(self, monkeypatch):
        """Raises a clear error when AZURE_STORAGE_ACCOUNT_URL is not set."""
        monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_URL", raising=False)
        monkeypatch.setenv("AZURE_STORAGE_SAS_TOKEN", "token")
        from src.blob_storage import BlobStorageService

        with pytest.raises(ValueError) as exc_info:
            BlobStorageService()
        assert "AZURE_STORAGE_ACCOUNT_URL" in str(exc_info.value)

    def test_raises_when_sas_token_missing(self, monkeypatch):
        """Raises a clear error when AZURE_STORAGE_SAS_TOKEN is not set."""
        monkeypatch.setenv(
            "AZURE_STORAGE_ACCOUNT_URL", "https://account.blob.core.windows.net/"
        )
        monkeypatch.delenv("AZURE_STORAGE_SAS_TOKEN", raising=False)
        from src.blob_storage import BlobStorageService

        with pytest.raises(ValueError) as exc_info:
            BlobStorageService()
        assert "AZURE_STORAGE_SAS_TOKEN" in str(exc_info.value)

    def test_uses_default_container_from_env_when_not_set(self, monkeypatch):
        """Default container is 'chat-data' when AZURE_STORAGE_CONTAINER is not set."""
        monkeypatch.setenv(
            "AZURE_STORAGE_ACCOUNT_URL", "https://account.blob.core.windows.net/"
        )
        monkeypatch.setenv("AZURE_STORAGE_SAS_TOKEN", "sas")
        monkeypatch.delenv("AZURE_STORAGE_CONTAINER", raising=False)
        with mock.patch("src.blob_storage.client.BlobServiceClient") as mock_bsc:
            from src.blob_storage import BlobStorageService

            svc = BlobStorageService()
            svc.get_container_client()
            mock_bsc.return_value.get_container_client.assert_called_once_with(
                "chat-data"
            )

    def test_uses_container_from_env_when_set(self, monkeypatch):
        """Uses AZURE_STORAGE_CONTAINER when set."""
        monkeypatch.setenv(
            "AZURE_STORAGE_ACCOUNT_URL", "https://account.blob.core.windows.net/"
        )
        monkeypatch.setenv("AZURE_STORAGE_SAS_TOKEN", "sas")
        monkeypatch.setenv("AZURE_STORAGE_CONTAINER", "my-container")
        with mock.patch("src.blob_storage.client.BlobServiceClient") as mock_bsc:
            from src.blob_storage import BlobStorageService

            svc = BlobStorageService()
            svc.get_container_client()
            mock_bsc.return_value.get_container_client.assert_called_once_with(
                "my-container"
            )


class TestBlobStorageServiceAPI:
    """BlobStorageService get_container_client, list_directories, list_blobs."""

    @pytest.fixture
    def env_and_mock(self, monkeypatch):
        """Set env and patch BlobServiceClient; yield (service, mock_container_client)."""
        monkeypatch.setenv(
            "AZURE_STORAGE_ACCOUNT_URL", "https://account.blob.core.windows.net/"
        )
        monkeypatch.setenv("AZURE_STORAGE_SAS_TOKEN", "sas")
        monkeypatch.delenv("AZURE_STORAGE_CONTAINER", raising=False)
        mock_container = mock.MagicMock()
        with mock.patch("src.blob_storage.client.BlobServiceClient") as mock_bsc:
            mock_bsc.return_value.get_container_client.return_value = mock_container
            from src.blob_storage import BlobStorageService

            svc = BlobStorageService()
            yield svc, mock_container

    def test_get_container_client_returns_container_client(self, env_and_mock):
        """get_container_client() returns the container client from BlobServiceClient."""
        svc, mock_container = env_and_mock
        result = svc.get_container_client()
        assert result is mock_container

    def test_get_container_client_with_name_uses_given_container(self, env_and_mock):
        """get_container_client(container_name) uses the given name."""
        svc, mock_container = env_and_mock
        mock_blob_svc = svc._blob_service_client
        svc.get_container_client("other-container")
        mock_blob_svc.get_container_client.assert_called_with("other-container")

    def test_list_directories_uses_delimiter_and_returns_prefix_names(
        self, env_and_mock
    ):
        """list_directories(prefix) uses delimiter='/' and returns virtual folder names."""
        svc, mock_container = env_and_mock
        # walk_blobs yields BlobPrefix and BlobProperties; we only want prefix names
        from azure.storage.blob import BlobPrefix

        prefix_a = mock.MagicMock(spec=BlobPrefix)
        prefix_a.name = "2024/"
        prefix_b = mock.MagicMock(spec=BlobPrefix)
        prefix_b.name = "2023/"
        mock_container.walk_blobs.return_value = [prefix_a, prefix_b]
        result = list(svc.list_directories(""))
        assert result == ["2024/", "2023/"]
        mock_container.walk_blobs.assert_called_once_with(
            name_starts_with="", delimiter="/"
        )

    def test_list_directories_with_prefix_passes_prefix(self, env_and_mock):
        """list_directories(prefix) passes prefix to walk_blobs."""
        svc, mock_container = env_and_mock
        mock_container.walk_blobs.return_value = []
        list(svc.list_directories("2024/05/"))
        mock_container.walk_blobs.assert_called_once_with(
            name_starts_with="2024/05/", delimiter="/"
        )

    def test_list_blobs_yields_blob_properties(self, env_and_mock):
        """list_blobs(prefix) iterates over blobs under prefix."""
        svc, mock_container = env_and_mock
        blob1 = mock.MagicMock()
        blob1.name = "2024/05/01/a.json"
        blob2 = mock.MagicMock()
        blob2.name = "2024/05/01/b.json"
        mock_container.list_blobs.return_value = [blob1, blob2]
        result = list(svc.list_blobs("2024/05/01/"))
        assert len(result) == 2
        assert result[0].name == "2024/05/01/a.json"
        assert result[1].name == "2024/05/01/b.json"
        mock_container.list_blobs.assert_called_once_with(
            name_starts_with="2024/05/01/"
        )

    def test_download_blob_returns_bytes(self, env_and_mock):
        """download_blob(blob_name) downloads and returns blob content as bytes."""
        svc, mock_container = env_and_mock
        mock_blob_client = mock.MagicMock()
        mock_blob_client.download_blob.return_value.readall.return_value = b"hello"
        mock_container.get_blob_client.return_value = mock_blob_client
        result = svc.download_blob("path/to/file.json")
        assert result == b"hello"
        mock_container.get_blob_client.assert_called_once_with("path/to/file.json")
        mock_blob_client.download_blob.return_value.readall.assert_called_once()
