import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools import download_bdd100k as downloader


class DownloadBDD100KTest(unittest.TestCase):
    def make_archive(self, path, include_images=True):
        with zipfile.ZipFile(path, "w") as archive:
            for split in ("train", "val"):
                if include_images:
                    archive.writestr(f"bdd100k/bdd100k/images/100k/{split}/one.jpg", b"image")
                archive.writestr(
                    f"bdd100k/labels/bdd100k_labels_images_{split}.json",
                    json.dumps([{
                        "name": "one.jpg",
                        "labels": [{
                            "category": "person",
                            "box2d": {"x1": 10, "y1": 20, "x2": 40, "y2": 60},
                        }],
                    }]),
                )
            archive.writestr("images/100k/train/../../escape.jpg", b"bad")

    def run_main(self, *arguments):
        with patch("sys.argv", ["download_bdd100k", *map(str, arguments)]):
            with contextlib.redirect_stdout(io.StringIO()):
                downloader.main()

    def test_local_archive_prepares_training_layout_and_is_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "dataset.zip"
            destination = root / "data"
            self.make_archive(archive)
            with patch.object(downloader, "download_with_progress") as download:
                self.run_main("--dest-dir", destination, "--archive", archive)
                self.run_main("--dest-dir", destination)
                download.assert_not_called()
            self.assertTrue(archive.is_file())
            self.assertFalse((destination / "images/escape.jpg").exists())
            for split in ("train", "val"):
                self.assertEqual(
                    (destination / f"images/100k/{split}/one.jpg").read_bytes(), b"image"
                )
                result = json.loads(
                    (destination / f"annotations/bdd100k_det_{split}_coco.json").read_text()
                )
                self.assertEqual(result["annotations"][0]["bbox"], [10, 20, 30, 40])
                self.assertEqual(result["annotations"][0]["category_id"], 1)

    def test_incomplete_archive_is_not_marked_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "dataset.zip"
            destination = root / "data"
            self.make_archive(archive, include_images=False)
            with self.assertRaisesRegex(SystemExit, "Missing train images"):
                self.run_main("--dest-dir", destination, "--archive", archive)
            self.assertFalse((destination / ".kaggle_bdd100k_complete").exists())
            self.assertTrue(archive.is_file())

    def test_non_zip_response_does_not_replace_existing_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dataset.zip"
            destination.write_bytes(b"existing archive")
            response = io.BytesIO(b"<html>Sign in</html>")
            response.info = lambda: {}
            with patch.object(downloader.urllib.request, "build_opener") as opener:
                opener.return_value.open.return_value = response
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertFalse(downloader.download_with_progress(
                        downloader.BDD100K_DATASET_URL, destination, "dataset"
                    ))
            self.assertEqual(destination.read_bytes(), b"existing archive")
            self.assertFalse(Path(str(destination) + ".part").exists())


if __name__ == "__main__":
    unittest.main()
