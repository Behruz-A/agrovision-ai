"""English desktop workbench for vegetable recognition and supported leaf health."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageOps, ImageTk

from .dataset import open_rgb
from .predict import WorkbenchPredictor
from .utils import write_json

BG, WHITE, INK, MUTED = "#F2F5F8", "#FFFFFF", "#1B303B", "#5E7280"
GREEN, TINT, LINE, AMBER = "#18674F", "#E7F2EC", "#DCE5EA", "#975A1D"


class AgroVisionApp:
    def __init__(self, root: tk.Tk, checkpoint: Path, device: str = "auto", task: str = "vegetable") -> None:
        self.root, self.checkpoint, self.device = root, checkpoint, device
        self.predictor = self.result = self.source_image = self.photo = self.image_path = None
        self.busy = self.closed = False
        self.probabilities_window = None
        self.request_id, self.future = 0, None
        self.events, self.worker = Queue(), ThreadPoolExecutor(max_workers=1)
        self.task = tk.StringVar(root, value=task)
        root.title("AgroVision AI | Crop workbench")
        root.geometry("1120x780")
        root.minsize(980, 700)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Leaf.Horizontal.TProgressbar", background=GREEN, troughcolor=TINT, borderwidth=0)
        style.configure("Disease.Horizontal.TProgressbar", background=AMBER, troughcolor="#F8EDE1", borderwidth=0)
        style.configure("Object.Horizontal.TProgressbar", background=MUTED, troughcolor=BG, borderwidth=0)
        style.configure("Treeview", rowheight=30, font=("Segoe UI", 10))
        header = tk.Frame(root, bg=WHITE, padx=28, pady=15)
        header.grid(row=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.label(header, "AgroVision AI", 22, bold=True).grid(row=0, column=0, sticky="w")
        self.label(header, "Crop recognition & leaf health", 10, color=MUTED).grid(row=1, column=0, sticky="w")
        self.model_badge = self.label(header, "Loading model...", 10, color=GREEN)
        self.model_badge.grid(row=0, column=1, rowspan=2, padx=14)
        self.button(header, "Model details", self.show_model_details).grid(row=0, column=2, rowspan=2)
        body = tk.Frame(root, bg=BG, padx=26, pady=20)
        body.grid(row=1, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, minsize=370)
        body.rowconfigure(3, weight=1)
        self.heading = self.label(body, "", 23, bold=True)
        self.heading.grid(row=0, column=0, columnspan=2, sticky="w")
        self.instructions = self.label(body, "", 10, color=MUTED)
        self.instructions.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 16))
        tabs = tk.Frame(body, bg=LINE, padx=2, pady=2)
        tabs.grid(row=2, column=0, sticky="w", pady=(0, 16))
        self.mode_buttons = {}
        for column, (name, caption) in enumerate((("vegetable", "Vegetable identification"), ("leaf", "Leaf health"))):
            control = self.button(tabs, caption, lambda value=name: self.set_task(value))
            control.grid(row=0, column=column)
            self.mode_buttons[name] = control
        self.choose_button = self.button(body, "Upload image", self.choose_image, primary=True)
        self.choose_button.grid(row=2, column=1, sticky="e", pady=(0, 16))
        image_panel = tk.Frame(body, bg=WHITE, highlightbackground=LINE, highlightthickness=1)
        image_panel.grid(row=3, column=0, sticky="nsew")
        image_panel.columnconfigure(0, weight=1)
        image_panel.rowconfigure(1, weight=1)
        self.label(image_panel, "Image preview", 11, bold=True).grid(row=0, sticky="w", padx=18, pady=13)
        self.preview = tk.Canvas(image_panel, bg="#F8FAFB", highlightthickness=0, cursor="hand2")
        self.preview.grid(row=1, sticky="nsew", padx=12)
        self.preview.bind("<Configure>", self.draw_preview)
        self.preview.bind("<Button-1>", lambda event: self.choose_image())
        self.filename = self.label(image_panel, "No image selected", 10)
        self.filename.grid(row=2, sticky="w", padx=18, pady=9)
        self.image_info = self.label(image_panel, "JPG, PNG, BMP or TIFF | Ctrl+O to browse", 9, color=MUTED)
        self.image_info.grid(row=3, sticky="w", padx=18, pady=(0, 14))
        result_shell = tk.Frame(body, bg=WHITE, padx=16, pady=16, width=370, highlightbackground=LINE, highlightthickness=1)
        result_shell.grid(row=3, column=1, sticky="nsew", padx=(18, 0))
        result_shell.grid_propagate(False)
        result_shell.columnconfigure(0, weight=1)
        result_shell.rowconfigure(0, weight=1)
        self.result_scroller = tk.Canvas(result_shell, bg=WHITE, highlightthickness=0, width=320)
        self.result_scroller.grid(row=0, column=0, sticky="nsew")
        result_scrollbar = ttk.Scrollbar(result_shell, command=self.result_scroller.yview)
        result_scrollbar.grid(row=0, column=1, sticky="ns")
        self.result_scroller.configure(yscrollcommand=result_scrollbar.set)
        panel = tk.Frame(self.result_scroller, bg=WHITE)
        panel_id = self.result_scroller.create_window((0, 0), window=panel, anchor="nw")
        self.result_scroller.bind("<Configure>", lambda event: self.result_scroller.itemconfigure(panel_id, width=event.width))
        self.result_scroller.bind("<MouseWheel>", lambda event: self.result_scroller.yview_scroll(-int(event.delta / 120), "units"))
        panel.columnconfigure(0, weight=1)
        panel.bind("<Configure>", self.resize_result)
        result_header = tk.Frame(panel, bg=WHITE)
        result_header.grid(row=0, sticky="ew")
        self.label(result_header, "Analysis", 11, bold=True).pack(side="left")
        self.state_label = self.label(result_header, "Ready", 9, color=GREEN)
        self.state_label.pack(side="right")
        self.prediction = self.label(panel, "Your result\nappears here", 25, bold=True)
        self.prediction.grid(row=1, sticky="ew", pady=(18, 5))
        self.condition = self.label(panel, "", 13, color=GREEN)
        self.condition.grid(row=2, sticky="ew")
        self.confidence = self.label(panel, "", 10, color=MUTED)
        self.confidence.grid(row=3, sticky="ew", pady=(7, 5))
        self.result_note = self.label(panel, "Upload one clear image to begin.", 10, color=MUTED)
        self.result_note.grid(row=4, sticky="ew", pady=(3, 12))
        self.matches_title = self.label(panel, "Model matches", 10, bold=True)
        self.matches_title.grid(row=5, sticky="w", pady=(3, 5))
        self.probability_rows = []
        matches = tk.Frame(panel, bg=WHITE)
        matches.grid(row=6, sticky="ew")
        matches.columnconfigure(0, weight=1)
        for index in range(3):
            line = tk.Frame(matches, bg=WHITE)
            line.grid(row=index * 2, sticky="ew", pady=(7, 3))
            line.columnconfigure(0, weight=1)
            label, value = self.label(line, "-", 9, color=MUTED), self.label(line, "", 9)
            label.grid(row=0, column=0, sticky="w")
            value.grid(row=0, column=1, sticky="e")
            bar = ttk.Progressbar(matches, maximum=1, style="Leaf.Horizontal.TProgressbar")
            bar.grid(row=index * 2 + 1, sticky="ew")
            self.probability_rows.append((label, bar, value))
        self.all_button = self.button(panel, "View all matches", self.show_probabilities)
        self.all_button.grid(row=7, sticky="w", pady=(10, 0))
        self.all_button.configure(state="disabled")
        actions = tk.Frame(result_shell, bg=WHITE)
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        actions.columnconfigure((0, 1), weight=1)
        self.save_button = self.button(actions, "Save report", self.save_result, primary=True)
        self.save_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.save_button.configure(state="disabled")
        self.clear_button = self.button(actions, "Clear result", self.clear_result)
        self.clear_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        footer = tk.Frame(root, bg=BG, padx=26, pady=8)
        footer.grid(row=2, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.status = self.label(footer, "Choose an image. Processing stays on this computer.", 9, color=MUTED)
        self.status.configure(wraplength=990)
        self.status.grid(row=0, sticky="ew")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", style="Leaf.Horizontal.TProgressbar")
        self.progress.grid(row=1, sticky="ew", pady=(7, 5))
        self.progress.grid_remove()
        self.scope = self.label(root, "Reading model coverage...", 9, color=MUTED)
        self.scope.grid(row=3, sticky="w", padx=26, pady=8)
        root.bind("<Control-o>", lambda event: self.choose_image())
        root.bind("<Control-s>", lambda event: self.save_result())
        root.bind("<Escape>", lambda event: self.clear_result())
        self.refresh_mode()
        self.worker.submit(self.initialize_model)
        self.poll_id = root.after(80, self.poll_results)

    @staticmethod
    def label(parent, text, size, bold=False, color=INK):
        return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color, font=("Segoe UI", size, "bold" if bold else "normal"), anchor="w", justify="left", wraplength=310)

    @staticmethod
    def button(parent, text, command, primary=False):
        return tk.Button(parent, text=text, command=command, bg=GREEN if primary else WHITE, fg=WHITE if primary else INK,
                         activebackground="#267D61" if primary else TINT, activeforeground=WHITE if primary else GREEN,
                         relief="flat", borderwidth=0, highlightthickness=1, highlightbackground=LINE, highlightcolor=GREEN,
                         font=("Segoe UI", 10, "bold" if primary else "normal"), padx=15, pady=9, cursor="hand2")

    def refresh_mode(self) -> None:
        leaf = self.task.get() == "leaf"
        self.heading.configure(text="Check leaf health" if leaf else "Identify your vegetable", wraplength=900)
        self.instructions.configure(text="Check one clear leaf. Plant identification is not required; estimates do not measure damaged area."
                                    if leaf else "Photograph one vegetable in good light, with the whole subject in view.", wraplength=950)
        self.matches_title.configure(text="Health estimates" if leaf else "Model matches")
        for index, (_, bar, _) in enumerate(self.probability_rows):
            bar.configure(style=("Leaf.Horizontal.TProgressbar", "Disease.Horizontal.TProgressbar", "Object.Horizontal.TProgressbar")[index] if leaf else "Leaf.Horizontal.TProgressbar")
        for mode, button in self.mode_buttons.items():
            button.configure(bg=GREEN if mode == self.task.get() else WHITE, fg=WHITE if mode == self.task.get() else INK)

    def set_task(self, task: str) -> None:
        if task != self.task.get():
            path = self.image_path
            self.task.set(task)
            self.refresh_mode()
            self.clear_result()
            if path:
                self.load_image(path)

    def initialize_model(self) -> None:
        try:
            self.predictor = WorkbenchPredictor(self.checkpoint, self.device)
            self.events.put(("model", None, self.predictor))
        except Exception as exc:
            self.events.put(("model_error", None, str(exc)))

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, title="Choose a crop image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if path:
            self.load_image(Path(path))

    def clear_result(self) -> None:
        """Clear this view, never saved files; invalidate results from old jobs."""
        self.request_id += 1
        if self.probabilities_window is not None and self.probabilities_window.winfo_exists():
            self.probabilities_window.destroy()
        self.probabilities_window = None
        self.result_scroller.yview_moveto(0)
        if self.future:
            self.future.cancel()
        self.future, self.busy = None, False
        self.result = self.source_image = self.photo = self.image_path = None
        for button in (self.save_button, self.all_button):
            button.configure(state="disabled")
        self.filename.configure(text="No image selected")
        self.image_info.configure(text="JPG, PNG, BMP or TIFF | Ctrl+O to browse")
        self.prediction.configure(text="Your result\nappears here", fg=INK)
        self.condition.configure(text="")
        self.confidence.configure(text="")
        self.result_note.configure(text="Upload one clear image to begin.", fg=MUTED)
        self.state_label.configure(text="Ready", fg=GREEN)
        for label, bar, value in self.probability_rows:
            label.configure(text="-")
            value.configure(text="")
            bar["value"] = 0
        self.progress.stop()
        self.progress["value"] = 0
        self.progress.grid_remove()
        self.status.configure(text="View cleared. Saved reports have not been deleted.", fg=MUTED)
        self.draw_preview()

    def load_image(self, path: Path) -> None:
        if self.closed:
            return
        self.clear_result()
        self.busy, self.image_path = True, path
        self.filename.configure(text=path.name[:65], wraplength=500)
        self.prediction.configure(text="Analyzing image...")
        self.result_note.configure(text="You can replace or clear the image while analysis runs.")
        self.state_label.configure(text="Analyzing", fg=GREEN)
        self.status.configure(text="Reading your image and comparing supported classes...", fg=MUTED)
        self.progress.start(15)
        self.progress.grid()
        self.future = self.worker.submit(self.analyze, path, self.task.get(), self.request_id)

    def analyze(self, path: Path, task: str, request_id: int) -> None:
        # Workers never touch Tk. Request IDs prevent stale results after clear/replace.
        try:
            if request_id != self.request_id:
                return
            image = open_rgb(path)
            self.events.put(("preview", request_id, image))
            if self.predictor is None:
                raise RuntimeError("Model is unavailable. See Model details and README.md for training instructions.")
            result = self.predictor.predict(image, task)
            result["source_image"] = path.name
            self.events.put(("result", request_id, result))
        except Exception as exc:
            self.events.put(("error", request_id, str(exc)))

    def poll_results(self) -> None:
        if self.closed:
            return
        if self.poll_id is not None:
            self.root.after_cancel(self.poll_id)
            self.poll_id = None
        while True:
            try:
                kind, request_id, payload = self.events.get_nowait()
            except Empty:
                break
            if kind == "model":
                self.update_coverage(payload)
                continue
            if kind == "model_error":
                self.model_badge.configure(text="Model unavailable", fg=AMBER)
                self.scope.configure(text="No model loaded. See Model details for training steps.", wraplength=990)
                if not self.busy:
                    self.status.configure(text=f"Model error: {payload}", fg=AMBER)
                continue
            if request_id != self.request_id:
                continue
            if kind == "preview":
                self.source_image = payload
                self.image_info.configure(text=f"{payload.width} x {payload.height} pixels | Click preview to replace")
                self.draw_preview()
                continue
            self.busy = False
            self.progress.stop()
            self.progress["value"] = 0
            self.progress.grid_remove()
            if kind == "error":
                self.prediction.configure(text="Unable to analyze", fg=AMBER)
                self.state_label.configure(text="Error", fg=AMBER)
                self.result_note.configure(text="Choose a valid image and try again.")
                self.status.configure(text=f"Error: {payload}", fg=AMBER)
            elif kind == "result":
                self.render_result(payload)
        self.poll_id = self.root.after(80, self.poll_results)

    def render_result(self, result: dict) -> None:
        self.result = result
        accepted = result["status"] == "identified"
        health = result.get("assessment_type") == "leaf_health"
        color = GREEN if accepted and result.get("condition") != "Diseased" else AMBER
        self.prediction.configure(text=(result["crop"] or result["prediction"]) if accepted else result["prediction"], fg=color)
        self.condition.configure(text=f"Leaf condition: {result['condition']}" if accepted and result["condition"] and not health else "")
        self.confidence.configure(text=f"Class probability: {result['confidence']:.1%}" if accepted else "Result withheld - please review the image")
        note = result.get("warning") or ("Vegetable identification does not assess freshness or disease." if result["task"] == "vegetable" else "A model estimate, not an agronomic diagnosis.")
        if result.get("smoke_test_model"):
            note = "Smoke-test model only. Do not use these predictions in practice."
        self.result_note.configure(text=note, fg=MUTED if accepted else AMBER)
        self.state_label.configure(text="Complete" if accepted else "Review image", fg=GREEN if accepted else AMBER)
        ranked = sorted(result["probabilities"].items(), key=lambda item: -item[1])
        if health:
            ranked = [("Healthy leaf", result["health_scores"]["healthy"]),
                      ("Diseased leaf", result["health_scores"]["diseased"]),
                      ("Not a leaf", result["health_scores"]["not_leaf"])]
        for (label, bar, value), (name, probability) in zip(self.probability_rows, ranked):
            label.configure(text=name)
            value.configure(text=f"{probability:.1%}")
            bar["value"] = probability
        self.save_button.configure(state="normal")
        self.all_button.configure(state="normal")
        self.status.configure(text="Analysis complete. Upload another image, switch mode, or save the report.", fg=MUTED)

    def update_coverage(self, predictor: WorkbenchPredictor) -> None:
        vegetables = sorted({entry["crop"] for entry in predictor.catalog.values() if entry["task"] == "vegetable"})
        self.model_badge.configure(text=f"{len(vegetables)} vegetables / General leaf check" if predictor.health else "General leaf model unavailable", fg=GREEN if predictor.health else AMBER)
        self.scope.configure(text="Leaf check: Healthy / Diseased / Not a leaf. Estimates are not damaged-area percentages or a verified diagnosis.", wraplength=1030)

    def draw_preview(self, event=None) -> None:
        self.preview.delete("all")
        width, height = max(self.preview.winfo_width(), 1), max(self.preview.winfo_height(), 1)
        if self.source_image is None:
            cx, cy = width / 2, height / 2 - 35
            self.preview.create_polygon(cx-28, cy+15, cx-25, cy-34, cx+28, cy-44, cx+27, cy+5, fill=TINT, outline=GREEN, width=2, smooth=True)
            self.preview.create_line(cx-20, cy+30, cx+16, cy-24, fill=GREEN, width=2)
            self.preview.create_text(cx, cy+66, text="Choose a crop image", fill=INK, font=("Segoe UI", 16, "bold"), width=max(width-30, 1))
            self.preview.create_text(cx, cy+98, text="Click here or use Upload image", fill=MUTED, font=("Segoe UI", 10), width=max(width-30, 1))
        else:
            image = ImageOps.contain(self.source_image, (max(width-20, 1), max(height-20, 1)), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image, master=self.root)
            self.preview.create_image(width/2, height/2, image=self.photo)

    def resize_result(self, event) -> None:
        for label in (self.prediction, self.condition, self.result_note):
            label.configure(wraplength=max(event.width-12, 100))
        self.result_scroller.configure(scrollregion=self.result_scroller.bbox("all"))

    def show_probabilities(self) -> None:
        if self.result is None:
            return
        if self.probabilities_window is not None and self.probabilities_window.winfo_exists():
            self.probabilities_window.lift()
            return
        window = tk.Toplevel(self.root)
        self.probabilities_window = window
        window.title("All model matches | AgroVision AI")
        window.geometry("620x460")
        window.transient(self.root)
        window.rowconfigure(1, weight=1)
        window.columnconfigure(0, weight=1)
        self.label(window, "Probabilities are relative to the model's known classes.", 10).grid(row=0, sticky="w", padx=15, pady=12)
        table = ttk.Treeview(window, columns=("class", "probability"), show="headings")
        table.heading("class", text="Class")
        table.heading("probability", text="Probability")
        table.column("class", width=410)
        table.column("probability", width=120, anchor="e")
        table.grid(row=1, sticky="nsew", padx=(15, 0), pady=(0, 15))
        scroll = ttk.Scrollbar(window, command=table.yview)
        scroll.grid(row=1, column=1, sticky="ns", pady=(0, 15))
        table.configure(yscrollcommand=scroll.set)
        for name, probability in sorted(self.result["probabilities"].items(), key=lambda item: -item[1]):
            table.insert("", "end", values=(name, f"{probability:.2%}"))

    def show_model_details(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("Model coverage | AgroVision AI")
        window.geometry("620x470")
        window.transient(self.root)
        text = tk.Text(window, wrap="word", font=("Segoe UI", 11), padx=20, pady=20, relief="flat")
        text.pack(fill="both", expand=True)
        if self.predictor:
            vegetables = sorted({entry["crop"] for entry in self.predictor.catalog.values() if entry["task"] == "vegetable"})
            details = "Vegetable identification\n" + (", ".join(vegetables) or "Unavailable in this checkpoint")
            details += "\n\nGeneral leaf assessment\nHealthy leaf / Diseased leaf / Not a leaf. No species identification is required and no specific disease is diagnosed."
            if self.predictor.health:
                groups = self.predictor.health.metadata.get("dataset_provenance", {}).get("plant_groups", [])
                details += "\n\nTraining plant groups\n" + ", ".join(name.replace("_", " ") for name in groups)
            else:
                details += "\n\nThe general leaf-health checkpoint is missing. See README.md for preparation and training."
            details += "\n\nLimitations\nProbabilities are not calibrated certainty. Unsupported examples and confidence rules reduce some errors, but cannot reject every unknown object. Use one subject per image. Field performance has not been validated."
            if self.predictor.metadata.get("smoke_test"):
                details += "\n\nThis is a smoke-test checkpoint."
        else:
            details = "Model is loading or unavailable.\n\nPrepare and train the multicrop checkpoint using README.md, then reopen the app.\n\nCheckpoint: " + str(self.checkpoint)
        text.insert("1.0", details)
        text.configure(state="disabled")

    def save_result(self) -> None:
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="Save analysis report", defaultextension=".json", initialfile="agrovision_report.json", filetypes=[("JSON report", "*.json")])
        if path:
            try:
                write_json(Path(path), self.result)
                self.status.configure(text=f"Report saved: {path}", fg=MUTED)
            except OSError as exc:
                self.status.configure(text=f"Could not save report: {exc}", fg=AMBER)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.request_id += 1
        if self.poll_id is not None:
            self.root.after_cancel(self.poll_id)
            self.poll_id = None
        # Bound worker methods retain this app and its Tk variables. Join the short
        # local load/inference before destroying Tk so finalizers stay on this thread.
        self.worker.shutdown(wait=True, cancel_futures=True)
        self.root.destroy()


def launch_gui(checkpoint: Path, device: str = "auto", image: Path | None = None, task: str = "vegetable") -> None:
    root = tk.Tk()
    app = AgroVisionApp(root, checkpoint, device, task)
    if image is not None:
        root.after(150, app.load_image, image)
    root.mainloop()
