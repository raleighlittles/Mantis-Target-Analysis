import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import decode_aruco_markers


class TestDecodeArucoMarkers(unittest.TestCase):
    def test_gather_images_directory_filters_extensions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "image1.jpg").write_bytes(b"")
            (tmp_path / "image2.png").write_bytes(b"")
            (tmp_path / "notes.txt").write_text("ignore")

            images = decode_aruco_markers.gather_images(tmp_path)

            self.assertEqual(
                images,
                [tmp_path / "image1.jpg", tmp_path / "image2.png"],
            )

    def test_flatten_ids_returns_empty_for_none(self):
        self.assertEqual(decode_aruco_markers._flatten_ids(None), [])

    def test_decode_images_handles_unreadable_images(self):
        if decode_aruco_markers.cv2 is None:
            self.skipTest("OpenCV not available")
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "fake.jpg"
        with patch.object(decode_aruco_markers.cv2, "imread", return_value=None):
            results = decode_aruco_markers.decode_images([image_path], "4X4_50")
        self.assertEqual(results, {str(image_path): []})


if __name__ == "__main__":
    unittest.main()
