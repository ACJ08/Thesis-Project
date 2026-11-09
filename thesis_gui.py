"""Launch with python thesis_gui.py."""
import json
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ttkbootstrap import Style
from ttkbootstrap.style import ThemeDefinition


ROOT = Path(__file__).resolve().parent
BACKGROUND = "#151D2B"
PANEL = "#202C3D"
INPUT = "#111A29"
ACCENT = "#397AD0"
TEXT = "#EEF4FF"
MUTED = "#B4C4DB"
SUCCESS = "#78CDA6"


def configure_theme(root):
    style = Style()
    style.register_theme(ThemeDefinition("lyric_studio", mode="dark", colors={
        "primary": ACCENT, "secondary": "#34465D", "success": SUCCESS,
        "info": "#92C2FF", "warning": "#F1C578", "danger": "#F3A0B1",
        "light": TEXT, "dark": INPUT, "bg": BACKGROUND, "fg": TEXT,
        "selectbg": "#344C6E", "selectfg": TEXT, "border": "#455872",
        "inputfg": TEXT, "inputbg": INPUT, "active": "#2C3D54",
    }))
    style.theme_use("lyric_studio")
    root.configure(background=BACKGROUND)
    root.option_add("*Font", "{Segoe UI} 10")
    style.configure("TFrame", background=PANEL)
    style.configure("Shell.TFrame", background=BACKGROUND)
    style.configure("TLabel", background=PANEL, foreground=TEXT)
    style.configure("Header.TLabel", background=BACKGROUND, font=("Segoe UI", 21, "bold"))
    style.configure("Subtitle.TLabel", background=BACKGROUND, foreground=MUTED)
    style.configure("Section.TLabel", font=("Segoe UI", 15, "bold"))
    style.configure("Hint.TLabel", foreground=MUTED)
    style.configure("Status.TLabel", foreground=SUCCESS, font=("Segoe UI", 10, "bold"))
    style.configure("TNotebook.Tab", padding=(14, 9), font=("Segoe UI", 10))
    style.map("TNotebook.Tab", background=[("selected", "#34465D"), ("!selected", BACKGROUND)],
              foreground=[("selected", TEXT), ("!selected", MUTED)])
    style.configure("TButton", padding=(12, 7))
    style.configure("primary.TButton", background=ACCENT, foreground="#FFFFFF",
                    lightcolor=ACCENT, darkcolor=ACCENT, bordercolor=ACCENT,
                    padding=(14, 11), font=("Segoe UI", 11, "bold"))
    style.map("primary.TButton",
              background=[("disabled", "#34465D"), ("pressed", "#285A9E"), ("active", "#4586DA")],
              foreground=[("disabled", MUTED)])
    style.configure("TCheckbutton", background=PANEL)
    style.configure("Treeview", rowheight=28)


def studio_text(parent, **options):
    return tk.Text(parent, background=INPUT, foreground=TEXT, insertbackground=TEXT,
                   selectbackground="#344C6E", selectforeground=TEXT, relief="flat",
                   borderwidth=0, padx=12, pady=9, **options)


MODELS = {
    "Hybrid RoBERTa–CNN–LSTM": "hybrid",
    "BERT–CNN–LSTM": "bert_cnn_lstm",
    "LSTM–CNN": "lstm_cnn",
    "RoBERTa": "roberta",
}
MODES = {**MODELS, "All baselines": "all_baselines", "All four models": "all_models",
         "Frozen model comparison": "compare"}


def run_choices(model):
    return sorted(
        (path for path in (ROOT / "experiments").glob(f"{model}_v26_*")
         if (path / "best_model.pt").is_file()),
        reverse=True,
    )


class Application:
    def __init__(self, root):
        self.root = root
        self.process = None
        self.events = queue.Queue()
        self.buttons = []
        configure_theme(root)
        root.title("Lyric Classification Studio | v26")
        root.geometry("1120x850")
        root.minsize(820, 690)
        outer = ttk.Frame(root, style="Shell.TFrame", padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Lyric Classification Studio", style="Header.TLabel").pack(anchor="w")
        ttk.Label(outer, text="v26 · Model training and explainability",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(0, 14))
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="x")
        self.training_tab()
        self.label_tab()
        self.explanation_tab()

        output_panel = ttk.Frame(outer, padding=(16, 10, 16, 14))
        output_panel.pack(fill="both", expand=True, pady=(14, 0))
        controls = ttk.Frame(output_panel)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Experiment Output", style="Section.TLabel").pack(side="left")
        self.status = tk.StringVar(value="Ready")
        self.cancel = ttk.Button(controls, text="Cancel run", command=self.stop, state="disabled")
        self.cancel.pack(side="right")
        ttk.Label(controls, textvariable=self.status, style="Status.TLabel").pack(side="right", padx=12)
        self.output = ttk.Notebook(output_panel)
        self.output.pack(fill="both", expand=True)
        self.results = ttk.Notebook(self.output)
        self.output.add(self.results, text="Results")
        log_frame = ttk.Frame(self.output)
        self.output.add(log_frame, text="Run log")
        self.log = studio_text(log_frame, height=8, wrap="word", font=("Consolas", 10))
        self.log.insert("1.0", "[SYSTEM] Ready to initialize model…")
        self.log.configure(state="disabled")
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)
        self.output.select(1)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(80, self.poll)

    def page(self, title, heading):
        page = ttk.Frame(self.tabs, padding=(16, 10, 16, 14))
        self.tabs.add(page, text=title)
        ttk.Label(page, text=heading, style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        return page

    def field(self, parent, title, *, values=(), browse=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=title, width=25).pack(side="left")
        value = tk.StringVar()
        if values:
            control = ttk.Combobox(row, textvariable=value, values=values, state="readonly")
            value.set(values[0])
        else:
            control = ttk.Entry(row, textvariable=value)
        control.pack(side="left", fill="x", expand=True)
        if browse:
            def choose():
                path = browse()
                if path:
                    value.set(path)
            ttk.Button(row, text="Browse…", command=choose).pack(side="left", padx=(6, 0))
        return value

    def csv_picker(self):
        return filedialog.askopenfilename(parent=self.root, filetypes=[("CSV", "*.csv"), ("All files", "*")])

    def folder_picker(self):
        return filedialog.askdirectory(parent=self.root, initialdir=ROOT / "experiments")

    def run_button(self, parent, title, callback):
        button = ttk.Button(parent, text=title, command=callback, style="primary.TButton")
        button.pack(fill="x", pady=(10, 0))
        self.buttons.append(button)

    def training_tab(self):
        page = self.page("Training", "Model Configuration")
        mode = self.field(page, "Architecture / operation", values=tuple(MODES))
        smoke = tk.BooleanVar(value=True)
        toggle = ttk.Checkbutton(page, text="Smoke test (one epoch, reduced data)", variable=smoke)
        toggle.pack(anchor="w", pady=6)
        ttk.Label(page, text="Datasets · Using default cleaned CSVs unless overridden below",
                  style="Hint.TLabel").pack(anchor="w", pady=(2, 4))
        splits = [self.field(page, title, browse=self.csv_picker)
                  for title in ("Training CSV", "Validation CSV", "Test CSV")]
        frozen = ttk.LabelFrame(page, text="Frozen runs (blank uses the configured v26 run)", padding=6)
        runs = {key: self.field(frozen, name, browse=self.folder_picker) for name, key in MODELS.items()}

        def show_comparison(*_):
            if MODES[mode.get()] == "compare":
                toggle.state(["disabled"])
                frozen.pack(fill="x", before=button)
                button.configure(text="Run Frozen Comparison")
            else:
                toggle.state(["!disabled"])
                frozen.pack_forget()
                button.configure(text="Run Training")

        self.run_button(page, "Run Training", lambda: self.start({
            "action": "train", "mode": MODES[mode.get()],
            "smoke": smoke.get() and MODES[mode.get()] != "compare",
            "datasets": [field.get().strip() for field in splits],
            "runs": {key: field.get().strip() for key, field in runs.items()},
        }))
        button = self.buttons[-1]
        mode.trace_add("write", show_comparison)

    def model_fields(self, page):
        name = self.field(page, "Model", values=tuple(MODELS))
        folder = self.field(page, "Run folder", browse=self.folder_picker)
        ttk.Label(page, text="Blank run folder selects the newest local checkpoint for this model.").pack(anchor="w")
        return name, folder

    def model_job(self, fields):
        name, folder = fields
        model = MODELS[name.get()]
        run = folder.get().strip()
        if not run:
            choices = run_choices(model)
            if not choices:
                raise ValueError(f"No local checkpoint found for {name.get()}. Select a run folder.")
            run = str(choices[0])
        return {"model": model, "run": run}

    def lyrics_field(self, page):
        ttk.Label(page, text="Song lyrics").pack(anchor="w")
        field = studio_text(page, height=4, wrap="word", font=("Segoe UI", 10))
        field.pack(fill="x", pady=(2, 4))
        return field

    def label_tab(self):
        page = self.page("Classification", "Label Songs")
        model = self.model_fields(page)
        source = self.field(page, "Input CSV (optional)", browse=self.csv_picker)
        column = self.field(page, "Lyrics column (optional)")
        lyrics = self.lyrics_field(page)

        def run():
            try:
                job = self.model_job(model)
                path = source.get().strip()
                text = lyrics.get("1.0", "end-1c").strip()
                if not path and not text:
                    raise ValueError("Paste lyrics or choose a CSV. A selected CSV takes priority.")
                output = ""
                if path:
                    output = filedialog.asksaveasfilename(
                        parent=self.root, defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                        initialfile=f"{Path(path).stem}_labeled.csv",
                    )
                    if not output:
                        return
                self.start({"action": "label", "source": path, "output": output, "lyrics": text,
                            "column": column.get().strip(), **job})
            except ValueError as error:
                messagebox.showerror("Label songs", str(error), parent=self.root)

        self.run_button(page, "Predict labels", run)

    def explanation_tab(self):
        page = self.page("LIME / SHAP", "Explain Predictions")
        model = self.model_fields(page)
        method = self.field(page, "Explanation method", values=("LIME", "SHAP", "LIME + SHAP"))
        lyrics = self.lyrics_field(page)
        validation = self.field(page, "Validation CSV (optional)", browse=self.csv_picker)
        samples = self.field(page, "Validation sample (0 = all)")
        samples.set("30")
        ttk.Label(page, text="Validation uses a stratified sample and the project validation CSV by default.").pack(anchor="w")
        actions = ttk.Frame(page)
        actions.pack(anchor="e")

        def run(batch):
            try:
                job = self.model_job(model)
                text = lyrics.get("1.0", "end-1c").strip()
                if not batch and not text:
                    raise ValueError("Paste lyrics for a single-song explanation.")
                count = int(samples.get()) if batch else 0
                if count < 0:
                    raise ValueError("Validation sample size must be zero or a positive integer.")
                selected = "Both" if method.get() == "LIME + SHAP" else method.get()
                self.start({"action": "explain", "method": selected, "lyrics": text,
                            "validation": (validation.get().strip() or "default") if batch else "",
                            "samples": count, **job})
            except ValueError as error:
                messagebox.showerror("LIME & SHAP", str(error), parent=self.root)

        for title, batch in (("Explain song + erasure", False), ("Run validation erasure", True)):
            button = ttk.Button(actions, text=title, command=lambda batch=batch: run(batch),
                                style="TButton" if batch else "primary.TButton")
            button.pack(side="left", padx=(6, 0), pady=(6, 0))
            self.buttons.append(button)

    def start(self, job):
        if self.process is not None:
            return
        try:
            self.temp = tempfile.TemporaryDirectory(prefix="thesis_gui_")
            request = Path(self.temp.name) / "request.json"
            result = request.with_name("result.json")
            request.write_text(json.dumps(job), encoding="utf-8")
            self.process = subprocess.Popen(
                [sys.executable, "-B", "-X", "utf8", "-u", str(ROOT / "thesis_gui_worker.py"),
                 str(request), str(result)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as error:
            if hasattr(self, "temp"):
                self.temp.cleanup()
            messagebox.showerror("Could not start worker", str(error), parent=self.root)
            return
        self.cancelled = False
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.output.select(1)
        self.status.set("Running…")
        self.cancel.configure(state="normal")
        for button in self.buttons:
            button.configure(state="disabled")

        def collect(process):
            with process.stdout:
                for line in process.stdout:
                    self.events.put(("log", line))
            code = process.wait()
            payload = result.read_text(encoding="utf-8") if code == 0 and result.is_file() else None
            self.events.put(("done", payload))

        threading.Thread(target=collect, args=(self.process,), daemon=True).start()

    def poll(self):
        for _ in range(150):
            try:
                event, value = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "log":
                if value.startswith("THESIS_PROGRESS "):
                    value = value.removeprefix("THESIS_PROGRESS ")
                    if not self.cancelled:
                        self.status.set(value.strip())
                self.log.configure(state="normal")
                self.log.insert("end", value)
                self.log.see("end")
                self.log.configure(state="disabled")
            else:
                self.process = None
                self.temp.cleanup()
                self.cancel.configure(state="disabled")
                for button in self.buttons:
                    button.configure(state="normal")
                if self.cancelled:
                    self.status.set("Cancelled. Partial experiment files may exist.")
                elif value is None:
                    self.status.set("Run failed. See the run log for details.")
                else:
                    self.display(json.loads(value))
                    self.status.set("Completed")
        self.root.after(80, self.poll)

    def display(self, result):
        for tab in self.results.tabs():
            self.results.nametowidget(tab).destroy()
        for method, features in result.get("highlights", {}).items():
            self.display_highlights(method, features)
        if result.get("message"):
            body = studio_text(self.results, wrap="word", height=7)
            body.insert("1.0", result["message"])
            body.configure(state="disabled")
            self.results.add(body, text="Details")
        for title, records in result.get("tables", {}).items():
            if not records:
                continue
            frame = ttk.Frame(self.results)
            self.results.add(frame, text=title)
            columns = list(records[0])
            contributions = title.endswith("Contributions")
            if contributions:
                columns.append("Relative strength")
                peak = max(abs(record["Contribution"]) for record in records) or 1
            table = ttk.Treeview(frame, columns=columns, show="headings")
            vertical = ttk.Scrollbar(frame, command=table.yview)
            horizontal = ttk.Scrollbar(frame, orient="horizontal", command=table.xview)
            table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
            vertical.pack(side="right", fill="y")
            horizontal.pack(side="bottom", fill="x")
            table.pack(fill="both", expand=True)
            for column in columns:
                table.heading(column, text=column)
                table.column(column, width=max(95, 9 * len(column)), minwidth=65)
            table.tag_configure("supports", foreground="#9AE4B8")
            table.tag_configure("opposes", foreground="#FFADC1")
            for record in records:
                values = [f"{value:.4f}" if isinstance(value, float) else value for value in record.values()]
                tags = ()
                if contributions:
                    weight = record["Contribution"]
                    values.append(("+" if weight > 0 else "−") + "█" * round(abs(weight) / peak * 20) if weight else "")
                    tags = ("supports" if weight > 0 else "opposes",) if weight else ()
                table.insert("", "end", values=values, tags=tags)
        self.output.select(0)

    def display_highlights(self, method, features):
        frame = ttk.Frame(self.results, padding=8)
        self.results.add(frame, text=f"{method} highlights")
        ttk.Label(frame, text="Evidence window · Green supports the local target · Rose opposes it · "
                             "Darker color means stronger contribution").pack(anchor="w", pady=(0, 5))
        lyrics = tk.Text(frame, wrap="word", font=("Segoe UI", 13), height=6,
                         background="#F4F7FD", foreground="#151D2B", relief="flat",
                         borderwidth=0, padx=14, pady=10)
        scroll = ttk.Scrollbar(frame, command=lyrics.yview)
        lyrics.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        lyrics.pack(fill="both", expand=True)
        palettes = {"supports": ("#CDEDD8", "#AEE3C1", "#8AD7AA", "#63C68D"),
                    "opposes": ("#F9D6E1", "#F7BDCF", "#F4A4BC", "#ED88A6")}
        for sign, colors in palettes.items():
            for level, color in enumerate(colors):
                lyrics.tag_configure(f"{sign}{level}", background=color, foreground="#161616")
        peak = max((abs(row["Contribution"]) for row in features), default=0) or 1
        for index, row in enumerate(features):
            if index:
                lyrics.insert("end", " ")
            weight = row["Contribution"]
            tags = ()
            if weight:
                sign = "supports" if weight > 0 else "opposes"
                level = min(3, int(4 * abs(weight) / peak))
                tags = (f"{sign}{level}",)
            lyrics.insert("end", row["Feature"], tags)
        lyrics.configure(state="disabled")

    def stop(self):
        if self.process and self.process.poll() is None:
            self.cancelled = True
            self.process.terminate()
            self.status.set("Cancelling…")

    def close(self):
        if self.process:
            if not messagebox.askyesno("Exit", "Cancel the active run and close?", parent=self.root):
                return
            self.stop()
            self.process.wait()
            self.temp.cleanup()
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    Application(window)
    window.mainloop()
