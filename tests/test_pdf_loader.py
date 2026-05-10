"""Tests for pdf_loader.py — download and loading logic with mocked network."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from pdf_loader import download_pdf


class TestDownloadPdf:
    def test_cached_file_skips_download(self, tmp_path):
        """If the file already exists locally, no HTTP request should be made."""
        # Create a fake cached file
        fake_file = tmp_path / "cached.pdf"
        fake_file.write_bytes(b"%PDF-fake")

        with patch("pdf_loader.PDFS_DIR", tmp_path):
            result = download_pdf("https://example.com/cached.pdf", "cached.pdf")

        assert result == fake_file
        assert result.exists()

    @patch("pdf_loader.requests.get")
    def test_successful_download(self, mock_get, tmp_path):
        """Simulates a successful PDF download."""
        mock_response = MagicMock()
        mock_response.content = b"%PDF-1.4 fake content"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with patch("pdf_loader.PDFS_DIR", tmp_path):
            result = download_pdf("https://example.com/report.pdf", "report.pdf")

        assert result.exists()
        assert result.read_bytes() == b"%PDF-1.4 fake content"
        mock_get.assert_called_once_with("https://example.com/report.pdf", timeout=120)

    @patch("pdf_loader.requests.get")
    def test_network_error_raises_runtime(self, mock_get, tmp_path):
        """Network failures should raise RuntimeError with a clear message."""
        import requests
        mock_get.side_effect = requests.ConnectionError("No network")

        with patch("pdf_loader.PDFS_DIR", tmp_path):
            with pytest.raises(RuntimeError, match="Failed to download"):
                download_pdf("https://example.com/fail.pdf", "fail.pdf")

    @patch("pdf_loader.requests.get")
    def test_http_error_raises_runtime(self, mock_get, tmp_path):
        """HTTP 404/500 errors should raise RuntimeError."""
        import requests
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404")
        mock_get.return_value = mock_response

        with patch("pdf_loader.PDFS_DIR", tmp_path):
            with pytest.raises(RuntimeError, match="Failed to download"):
                download_pdf("https://example.com/missing.pdf", "missing.pdf")
