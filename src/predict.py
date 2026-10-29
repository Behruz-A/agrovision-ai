"""Checkpoint-backed image and center-region webcam classification."""
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

from .dataset import open_rgb, transform_for
from .model import AgroVisionCNN
from .utils import device_for


class Predictor:
    def __init__(self, checkpoint: Path, device: str = "auto") -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {checkpoint}. Train first: python -m src.train")
        self.device = device_for(device)
        if self.device.type == "cpu":
            torch.set_num_threads(4)
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.metadata = saved["metadata"]
        if self.metadata["architecture"] not in {"AgroVisionCNN-v1", "AgroVisionCNN-v2"}:
            raise ValueError("Unsupported checkpoint architecture.")
        mapping = self.metadata["class_to_idx"]
        if len(mapping) < 2 or sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("Checkpoint must contain contiguous class indices.")
        self.classes = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
        self.transform = transform_for(config=self.metadata["preprocessing"])
        self.catalog = self.metadata.get("class_catalog") or {
            name: {"task": "leaf", "crop": name.split("_")[0],
                   "condition": " ".join(name.split("_")[1:]),
                   "display_name": name.replace("_", " ")} for name in self.classes}
        if set(self.catalog) != set(mapping):
            raise ValueError("Checkpoint class catalog does not match its mapping.")
        self.model = AgroVisionCNN(len(mapping), first_stride=2 if self.metadata["architecture"].endswith("v2") else 1).to(self.device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image, task: str | None = None) -> dict:
        if task not in {None, "vegetable", "leaf"}:
            raise ValueError("Task must be vegetable, leaf, or None.")
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probabilities = self.model(tensor).softmax(1)[0].cpu().tolist()
        return self.describe(probabilities, task)

    def describe(self, probabilities: list[float], task: str | None = None) -> dict:
        """Apply task routing and cautious abstention without renormalizing away other classes."""
        if self.metadata.get("assessment_type") == "leaf_health":
            return self.describe_health(probabilities, task)
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        entry = self.catalog[self.classes[index]]
        ordered = sorted(probabilities, reverse=True)
        confidence, margin = probabilities[index], ordered[0] - ordered[1]
        status, warning = "identified", ""
        if entry["task"] == "unknown":
            status, warning = "unsupported", "This image does not match a supported crop. Upload one clear vegetable or a supported leaf."
        elif task and entry["task"] != task:
            status = "wrong_mode"
            warning = ("This looks like a vegetable, not a leaf. Switch to Vegetable identification."
                       if entry["task"] == "vegetable" else "This looks like a leaf. Switch to Leaf health for supported crops.")
        elif confidence < self.metadata.get("confidence_threshold", 0) or margin < self.metadata.get("margin_threshold", 0):
            status, warning = "uncertain", "The model is uncertain. Try a closer, well-lit image with one subject."
        if "class_catalog" not in self.metadata:
            warning = "Legacy tomato leaf model: vegetable recognition and unsupported-image rejection are unavailable."
            if task == "vegetable":
                status = "unsupported"
        titles = {"unsupported": "Unsupported image", "wrong_mode": "Different image type", "uncertain": "Uncertain result"}
        return {"prediction": entry["display_name"] if status == "identified" else titles[status],
                "best_match": entry["display_name"], "confidence": confidence, "margin": margin,
                "crop": entry["crop"] if status == "identified" else None,
                "condition": entry["condition"] if status == "identified" else None,
                "task": entry["task"], "requested_task": task, "status": status, "warning": warning,
                "probabilities": {self.catalog[name]["display_name"]: p for name, p in zip(self.classes, probabilities)},
                "smoke_test_model": self.metadata.get("smoke_test", False)}

    def describe_health(self, probabilities: list[float], task: str | None = None) -> dict:
        scores = {self.catalog[name]["health_state"]: probability for name, probability in zip(self.classes, probabilities)}
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        entry = self.catalog[self.classes[index]]
        ordered = sorted(probabilities, reverse=True)
        confidence, margin = ordered[0], ordered[0] - ordered[1]
        warning = "Model estimates, not percent of damaged leaf area. Unfamiliar plants and field conditions can produce errors."
        status, title = "identified", entry["display_name"]
        if task == "vegetable":
            status, title = "wrong_mode", "Leaf assessment model"
            warning = "Use the vegetable identification model for a vegetable photo."
        elif confidence < self.metadata.get("confidence_threshold", .65) or margin < self.metadata.get("margin_threshold", .15):
            status, title = "uncertain", "Uncertain leaf assessment"
            warning = "The model is uncertain. Try a close, well-lit leaf photo. Percentages are model estimates, not damaged area."
        elif entry["health_state"] == "not_leaf":
            status, title = "not_leaf", "Leaf not detected"
            warning = "This image may not show a leaf. Upload one clearly visible leaf."
        return {"assessment_type": "leaf_health", "prediction": title, "best_match": entry["display_name"],
            "confidence": confidence, "margin": margin, "crop": None,
            "condition": entry["condition"] if status == "identified" else None,
            "task": entry["task"], "requested_task": task, "status": status, "warning": warning,
            "health_scores": scores,
            "probabilities": {self.catalog[name]["display_name"]: p for name, p in zip(self.classes, probabilities)},
            "smoke_test_model": self.metadata.get("smoke_test", False)}

    def predict_file(self, path: Path, task: str | None = None) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return self.predict(open_rgb(path), task)


class WorkbenchPredictor:
    """Route leaf checks to their own species-agnostic health model."""
    def __init__(self, checkpoint: Path, device: str = "auto", health_checkpoint: Path | None = None) -> None:
        from .utils import ROOT
        self.primary = Predictor(checkpoint, device)
        self.metadata, self.catalog = self.primary.metadata, self.primary.catalog
        health_checkpoint = health_checkpoint or ROOT / "models/agrovision_leaf_health.pth"
        if self.primary.metadata.get("assessment_type") == "leaf_health":
            self.health = self.primary
        else:
            self.health = Predictor(health_checkpoint, device) if health_checkpoint.is_file() else None
        if self.health and self.health.metadata.get("assessment_type") != "leaf_health":
            raise ValueError("The selected health checkpoint is not a general leaf-health model.")

    def predict(self, image: Image.Image, task: str | None = None) -> dict:
        if task == "leaf":
            if not self.health:
                raise FileNotFoundError("General leaf-health model missing. Prepare leaf-health data and train it using README.md.")
            return self.health.predict(image, "leaf")
        result = self.primary.predict(image, task)
        if self.primary is self.health:
            return result
        if task is None and self.health and result["task"] in {"leaf", "unknown"}:
            return self.health.predict(image, "leaf")
        return result

    def predict_file(self, path: Path, task: str | None = None) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return self.predict(open_rgb(path), task)


def webcam(predictor: Predictor, camera: int = 0, interval: float = 0.5, task: str | None = None) -> None:
    if interval <= 0:
        raise ValueError("Inference interval must be positive.")
    capture = cv2.VideoCapture(camera)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera {camera}. Check connection and Windows camera permissions.")
        last, result = float("-inf"), None
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped providing frames.")
            height, width = frame.shape[:2]
            side = int(min(height, width) * 0.65)
            x, y = (width - side) // 2, (height - side) // 2
            if time.monotonic() - last >= interval:
                crop = cv2.cvtColor(frame[y:y + side, x:x + side], cv2.COLOR_BGR2RGB)
                result = predictor.predict(Image.fromarray(crop), task)
                last = time.monotonic()
            cv2.rectangle(frame, (x, y), (x + side, y + side), (80, 220, 100), 2)
            cv2.rectangle(frame, (0, 0), (width, 95), (25, 35, 25), -1)
            text = f"{result['prediction']}: {result['confidence']:.1%}" if result else "Waiting..."
            cv2.putText(frame, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "Position ONE subject in box | Q: quit", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 200), 1)
            if result and result.get("smoke_test_model"):
                cv2.putText(frame, "SMOKE TEST MODEL - not trained for practical use", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 190, 255), 1)
            cv2.imshow("AgroVision AI - center crop classification", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except cv2.error as exc:
        raise RuntimeError(f"OpenCV webcam failure: {exc}. Check camera access and desktop GUI support.") from exc
    finally:
        capture.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # A GUI-less build may have no windows to destroy.
