import unittest
from unittest.mock import patch

from pathrel.gpu_diagnostics import cuda_unavailable_message


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def device_count() -> int:
        return 0


class _FakeTorch:
    __version__ = "test-torch"
    version = type("Version", (), {"cuda": "13.0"})()
    cuda = _FakeCuda()


class CudaDiagnosticsTest(unittest.TestCase):
    def test_distinguishes_loaded_driver_without_compute_nodes(self) -> None:
        with patch("pathrel.gpu_diagnostics.glob.glob", return_value=[]), patch(
            "pathrel.gpu_diagnostics.Path.is_file", return_value=True
        ):
            message = cuda_unavailable_message(_FakeTorch)
        self.assertIn("Host evidence is present", message)
        self.assertIn("/dev/nvidia* compute nodes", message)
        self.assertIn("test-torch", message)

