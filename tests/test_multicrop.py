"""Contracts for multi-task routing, supported classes, and real split integrity."""
import unittest
from pathlib import Path

import torch

from src.model import AgroVisionCNN
from src.predict import Predictor
from src.utils import ROOT, read_json


class RoutingTests(unittest.TestCase):
    def setUp(self):
        # Deterministic probabilities test the routing policy, not model accuracy.
        self.predictor = Predictor.__new__(Predictor)
        self.predictor.classes = ["produce", "leaf", "other"]
        self.predictor.catalog = {
            "produce": {"task": "vegetable", "crop": "Tomato", "condition": None, "display_name": "Tomato"},
            "leaf": {"task": "leaf", "crop": "Potato", "condition": "Healthy", "display_name": "Potato leaf - Healthy"},
            "other": {"task": "unknown", "crop": None, "condition": None, "display_name": "Other / unsupported"}}
        self.predictor.metadata = {"class_catalog": self.predictor.catalog, "confidence_threshold": .65, "margin_threshold": .15}

    def test_fruit_is_not_presented_as_leaf_disease(self):
        result = self.predictor.describe([.98, .01, .01], "leaf")
        self.assertEqual(result["status"], "wrong_mode")
        self.assertIsNone(result["condition"])
        self.assertIsNone(result["crop"])
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1)

    def test_known_other_is_rejected(self):
        result = self.predictor.describe([.01, .01, .98], "vegetable")
        self.assertEqual(result["status"], "unsupported")
        self.assertIsNone(result["crop"])

    def test_uncertain_input_is_not_a_diagnosis(self):
        self.assertEqual(self.predictor.describe([.4, .35, .25], "vegetable")["status"], "uncertain")
        self.assertEqual(self.predictor.describe([.67, .3, .03], "vegetable")["status"], "identified")

    def test_leaf_and_crop_are_separate_fields(self):
        result = self.predictor.describe([.01, .98, .01], "leaf")
        self.assertEqual((result["crop"], result["condition"]), ("Potato", "Healthy"))
        self.assertEqual(self.predictor.describe([.98, .01, .01], "vegetable")["condition"], None)

    def test_variable_output_forward_backward(self):
        torch.set_num_threads(2)
        model = AgroVisionCNN(19, first_stride=2)
        logits = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(logits.shape), (2, 19))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 18]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    @unittest.skipUnless((ROOT / "dataset/multicrop/manifest.json").exists(), "Prepare multicrop data first")
    def test_real_multicrop_coverage_and_split_isolation(self):
        root = ROOT / "dataset/multicrop"
        manifest, catalog = read_json(root / "manifest.json"), read_json(root / "class_catalog.json")
        mapping = read_json(root / "class_to_idx.json")
        self.assertEqual(set(catalog), set(mapping))
        self.assertEqual(sorted(mapping.values()), list(range(19)))
        self.assertEqual(len({v["crop"] for v in catalog.values() if v["task"] == "vegetable"}), 10)
        self.assertEqual({v["crop"] for v in catalog.values() if v["task"] == "leaf"}, {"Tomato", "Potato", "Bell pepper"})
        seen = {key: {} for key in ("group", "sha256_file", "sha256_pixels")}
        for image in manifest["images"]:
            self.assertTrue((root / image["destination"]).is_file())
            for key, lookup in seen.items():
                self.assertEqual(lookup.setdefault(image[key], image["split"]), image["split"])
        for split in ("train", "val", "test"):
            self.assertTrue(all(manifest["counts"][split].values()))
