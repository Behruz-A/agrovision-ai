"""General leaf-health probability semantics and real-data checks."""
import unittest

from src.predict import Predictor, WorkbenchPredictor
from src.utils import ROOT, read_json


class LeafHealthTests(unittest.TestCase):
    def setUp(self):
        self.predictor = Predictor.__new__(Predictor)
        self.predictor.classes = ["Diseased", "Healthy", "Not_leaf"]
        self.predictor.catalog = {
            "Diseased": {"task": "leaf", "crop": None, "condition": "Diseased", "health_state": "diseased", "display_name": "Diseased leaf"},
            "Healthy": {"task": "leaf", "crop": None, "condition": "Healthy", "health_state": "healthy", "display_name": "Healthy leaf"},
            "Not_leaf": {"task": "unknown", "crop": None, "condition": None, "health_state": "not_leaf", "display_name": "Not a leaf"}}
        self.predictor.metadata = {"assessment_type": "leaf_health", "confidence_threshold": .65, "margin_threshold": .15}

    def test_healthy_estimate_needs_no_species(self):
        result = self.predictor.describe([.1, .85, .05], "leaf")
        self.assertIsNone(result["crop"])
        self.assertEqual(result["condition"], "Healthy")
        self.assertAlmostEqual(result["health_scores"]["healthy"], .85)
        self.assertAlmostEqual(sum(result["health_scores"].values()), 1)
        self.assertIn("not percent of damaged", result["warning"])

    def test_diseased_estimate_has_no_invented_disease_name(self):
        result = self.predictor.describe([.85, .1, .05], "leaf")
        self.assertEqual(result["prediction"], "Diseased leaf")
        self.assertEqual(result["condition"], "Diseased")
        self.assertNotIn("blight", result["prediction"].lower())

    def test_non_leaf_is_not_a_health_diagnosis(self):
        result = self.predictor.describe([.05, .05, .9], "leaf")
        self.assertEqual(result["status"], "not_leaf")
        self.assertIsNone(result["condition"])

    def test_uncertain_still_shows_honest_estimates(self):
        result = self.predictor.describe([.45, .5, .05], "leaf")
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["condition"])
        self.assertEqual(result["health_scores"]["healthy"], .5)

    @unittest.skipUnless((ROOT / "dataset/leaf-health/manifest.json").exists(), "Prepare general health data first")
    def test_health_data_has_only_three_states_and_no_split_duplicates(self):
        folder = ROOT / "dataset/leaf-health"
        self.assertEqual(set(read_json(folder / "class_to_idx.json")), {"Healthy", "Diseased", "Not_leaf"})
        provenance = read_json(folder / "provenance.json")
        self.assertGreaterEqual(len(provenance["plant_groups"]), 10)
        seen = {key: {} for key in ("group", "sha256_file", "sha256_pixels")}
        for image in read_json(folder / "manifest.json")["images"]:
            for key, lookup in seen.items():
                self.assertEqual(lookup.setdefault(image[key], image["split"]), image["split"])

    @unittest.skipUnless((ROOT / "models/agrovision_leaf_health.pth").is_file(), "Train general health checkpoint first")
    def test_workbench_routes_leaf_to_health_checkpoint(self):
        router = WorkbenchPredictor(ROOT / "models/agrovision_multicrop.pth", "cpu")
        manifest = read_json(ROOT / "dataset/leaf-health/manifest.json")
        image = next(item for item in manifest["images"] if item["split"] == "test" and item["class"] == "Healthy")
        result = router.predict_file(ROOT / "dataset/leaf-health" / image["destination"], "leaf")
        self.assertEqual(result["assessment_type"], "leaf_health")
        self.assertEqual(set(result["health_scores"]), {"healthy", "diseased", "not_leaf"})
        self.assertEqual(len(result["probabilities"]), 3)
        self.assertIsNone(result["crop"])
