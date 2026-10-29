"""Contract tests; real-data checks skip until dataset preparation is complete."""
import unittest
import re
import shutil
import tempfile
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from src.dataset import LeafDataset, open_rgb, transform_for
from src.model import AgroVisionCNN
from src.predict import Predictor, webcam
from src.prepare_dataset import CLASSES, prepare
from src.utils import ROOT, read_json


class ModelTests(unittest.TestCase):
    def test_forward_backward(self):
        torch.set_num_threads(2)
        model = AgroVisionCNN()
        logits = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(logits.shape), (2, 3))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.features[0].weight.grad)

    def test_missing_checkpoint(self):
        with self.assertRaisesRegex(FileNotFoundError, "Train first"):
            Predictor(ROOT / "models/does-not-exist.pth")

    def test_unavailable_camera_releases_capture(self):
        with patch("src.predict.cv2.VideoCapture") as factory, patch("src.predict.cv2.destroyAllWindows"):
            capture = factory.return_value
            capture.isOpened.return_value = False
            with self.assertRaisesRegex(RuntimeError, "Cannot open camera"):
                webcam(Mock())
            capture.release.assert_called_once()

    def test_failed_camera_read_releases_capture(self):
        with patch("src.predict.cv2.VideoCapture") as factory, patch("src.predict.cv2.destroyAllWindows"):
            capture = factory.return_value
            capture.isOpened.return_value = True
            capture.read.return_value = (False, None)
            with self.assertRaisesRegex(RuntimeError, "stopped providing frames"):
                webcam(Mock())
            capture.release.assert_called_once()

    @unittest.skipUnless((ROOT / "dataset/manifest.json").is_file(), "Prepare real data first")
    def test_real_data_split_integrity(self):
        manifest = read_json(ROOT / "dataset/manifest.json")
        group_splits, pixel_splits, file_splits, named_splits = {}, {}, {}, {}
        for image in manifest["images"]:
            self.assertTrue((ROOT / "dataset" / image["destination"]).is_file())
            for key, lookup in (("group", group_splits), ("sha256_pixels", pixel_splits), ("sha256_file", file_splits)):
                self.assertEqual(lookup.setdefault(image[key], image["split"]), image["split"])
            stem = Path(image["source"]).stem.split("___", 1)[-1].lower()
            match = re.fullmatch(r"(.+?)\s+leaf\s+(\d+)(?:\.\d+)?(?:\s+day\s+\d+)?", stem)
            if match:
                identity = (image["class"], match[1], match[2])
                self.assertEqual(named_splits.setdefault(identity, image["split"]), image["split"])
        for split in ("train", "val", "test"):
            self.assertTrue(all(count > 0 for count in manifest["counts"][split].values()))

    @unittest.skipUnless((ROOT / "dataset/manifest.json").is_file(), "Prepare real data first")
    def test_validation_matches_inference(self):
        dataset = LeafDataset(ROOT / "dataset", "val")
        tensor, label = dataset[0]
        expected = transform_for()(open_rgb(dataset.samples[0][0]))
        self.assertTrue(torch.equal(tensor, expected))
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertIn(label, (0, 1, 2))

    @unittest.skipUnless((ROOT / "dataset/manifest.json").is_file(), "Prepare real data first")
    def test_preparation_reproducibility_and_invalid_images(self):
        manifest = read_json(ROOT / "dataset/manifest.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for upstream, canonical in CLASSES.items():
                folder = root / "source" / upstream
                folder.mkdir(parents=True)
                groups = set()
                for record in manifest["images"]:
                    if record["class"] == canonical and record["group"] not in groups:
                        groups.add(record["group"])
                        shutil.copy2(record["source"], folder / Path(record["source"]).name)
                        if len(groups) == 3:
                            break
            first_folder = root / "source" / next(iter(CLASSES))
            real_image = next(first_folder.iterdir())
            shutil.copy2(real_image, first_folder / "duplicate.jpg")
            (first_folder / "invalid.jpg").write_text("invalid image", encoding="utf-8")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                first = prepare(root / "source", root / "one", 42)
                second = prepare(root / "source", root / "two", 42)
            assignments = lambda result: [(r["sha256_pixels"], r["split"]) for r in result["images"]]
            self.assertEqual(assignments(first), assignments(second))
            self.assertEqual(first["duplicates_removed"], 1)
            self.assertEqual(first["invalid_skipped"], 1)
            with self.assertRaisesRegex(ValueError, "already contain"):
                prepare(root / "source", root / "one", 42)

    @unittest.skipUnless((ROOT / "models/agrovision_cnn.pth").is_file(), "Train a checkpoint first")
    def test_checkpoint_real_image(self):
        predictor = Predictor(ROOT / "models/agrovision_cnn.pth", "cpu")
        data = LeafDataset(ROOT / "dataset", "test")
        result = predictor.predict_file(data.samples[0][0])
        self.assertEqual(len(result["probabilities"]), 3)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)
        self.assertIn(result["best_match"], result["probabilities"])


if __name__ == "__main__":
    unittest.main()
