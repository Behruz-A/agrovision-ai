"""Desktop integration test using real held-out images and the trained checkpoint."""
import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

from src.gui import AgroVisionApp
from src.utils import ROOT, default_checkpoint, read_json


@unittest.skipUnless((ROOT / "dataset/manifest.json").is_file() and
                     (ROOT / "models/agrovision_cnn.pth").is_file(), "Real dataset and checkpoint required")
class GuiIntegrationTests(unittest.TestCase):
    def test_upload_error_recovery_and_export(self):
        root = tk.Tk()
        root.withdraw()
        app = AgroVisionApp(root, default_checkpoint(), "cpu")
        try:
            data = ROOT / ("dataset/multicrop" if app.checkpoint.name == "agrovision_multicrop.pth" else "dataset")
            image = next(record for record in read_json(data / "manifest.json")["images"] if record["split"] == "test")
            path = data / image["destination"]

            def finish():
                deadline = time.monotonic() + 60
                while app.busy and time.monotonic() < deadline:
                    root.update()
                    time.sleep(.02)
                self.assertFalse(app.busy, "Background image prediction timed out")

            with patch("src.gui.filedialog.askopenfilename", return_value=str(path)):
                app.choose_image()
            ticks = []
            root.after(0, lambda: ticks.append(True))
            finish()
            self.assertTrue(ticks, "The GUI event loop must stay responsive during prediction")
            self.assertIsNotNone(app.source_image)
            self.assertAlmostEqual(sum(app.result["probabilities"].values()), 1.0, places=5)
            with tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "result.json"
                with patch("src.gui.filedialog.asksaveasfilename", return_value=str(target)):
                    app.save_result()
                self.assertEqual(read_json(target), app.result)

            app.load_image(ROOT / "missing-image.jpg")
            self.assertIsNone(app.result, "Do not show/save a previous result for a failed upload")
            finish()
            self.assertIn("Error", app.status.cget("text"))
            self.assertEqual(str(app.save_button.cget("state")), "disabled")

            app.load_image(path)
            finish()
            self.assertIsNotNone(app.result, "Selecting a valid image must recover after an error")
            previous = app.result
            app.show_probabilities()
            report_window = app.probabilities_window
            with patch("src.gui.filedialog.askopenfilename", return_value=""):
                app.choose_image()
            self.assertIs(app.result, previous, "Cancelling the picker should preserve the current result")
            old_request = app.request_id
            app.clear_result()
            self.assertFalse(report_window.winfo_exists())
            app.events.put(("result", old_request, previous))
            app.poll_results()
            self.assertIsNone(app.result, "An old worker result must not reappear after Clear")
            self.assertIsNone(app.source_image)
            self.assertEqual(str(app.save_button.cget("state")), "disabled")
            self.assertEqual(app.filename.cget("text"), "No image selected")
            app.load_image(path)
            pending = app.future
            app.clear_result()
            if not pending.cancelled():
                pending.result(timeout=60)
            app.poll_results()
            self.assertIsNone(app.result, "Clearing during real prediction must discard its completion")
            self.assertFalse(app.busy)
            app.load_image(path)
            finish()
            app.set_task("leaf")
            finish()
            self.assertEqual(app.result["requested_task"], "leaf")
            self.assertEqual(app.result["assessment_type"], "leaf_health")
            self.assertEqual([label.cget("text") for label, _, _ in app.probability_rows],
                             ["Healthy leaf", "Diseased leaf", "Not a leaf"])
            self.assertEqual(app.image_path, path)
            self.assertEqual(app.clear_button.cget("text"), "Clear result")
            self.assertEqual(app.save_button.cget("text"), "Save report")
        finally:
            app.close()

    def test_save_and_clear_stay_inside_small_window(self):
        root = tk.Tk()
        root.withdraw()
        root.geometry("980x700")
        app = AgroVisionApp(root, default_checkpoint(), "cpu")
        try:
            root.geometry("980x700")
            root.deiconify()
            root.update()
            for button in (app.save_button, app.clear_button):
                relative_bottom = button.winfo_rooty() - root.winfo_rooty() + button.winfo_height()
                self.assertLessEqual(relative_bottom, root.winfo_height())
            self.assertGreater(app.preview.winfo_height(), 100)
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
