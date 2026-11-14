"""Launch with python thesis_gui.py."""
import json
import csv
import os
import re
from datetime import datetime
import multiprocessing
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from thesis_gui_schema import SCHEMA_VERSION, validate_job, read_json, checkpoint

from ttkbootstrap import Style
from ttkbootstrap.style import ThemeDefinition


FROZEN = getattr(sys, "frozen", False)


def project_root():
    if not FROZEN:
        return Path(__file__).resolve().parent
    app_folder = Path(sys.executable).resolve().parent
    for folder in (app_folder, *app_folder.parents):
        if (folder / "data").is_dir() or (folder / "experiments").is_dir():
            return folder
    return app_folder


ROOT = project_root()
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
    choices = []
    for checkpoint_path in (ROOT / 'experiments').rglob('best_model.pt'):
        path = checkpoint_path.parent
        meta = read_json(path / 'manifest.json')
        if isinstance(meta, dict) and ((meta.get('architecture') == model and meta.get('pipeline_version') == 'v26')
                                      or (not meta and path.name.startswith(f'{model}_v26_'))):
            choices.append(path)
    return sorted(choices, key=lambda path: (path / 'best_model.pt').stat().st_mtime, reverse=True)


class Application:
    def __init__(self, root):
        self.root = root
        self.process = None
        self.events = queue.Queue()
        self.buttons = []
        self.settings_path = ROOT / '.lyric_studio' / 'settings.json'
        self.settings = read_json(self.settings_path)
        if not isinstance(self.settings, dict):
            self.settings = {}
        self.variables = {}
        self.current_page = ''
        self.history_dir = ROOT / '.lyric_studio' / 'history'
        self.active_record = None
        self.preflighting = False
        self.started_at = None
        self.selected_record = None
        configure_theme(root)
        root.title("Lyric Classification Studio | v26")
        root.geometry("1320x900")
        root.minsize(900, 650)
        outer = ttk.Frame(root, style="Shell.TFrame", padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Lyric Classification Studio", style="Header.TLabel").pack(anchor="w")
        ttk.Label(outer, text="v26 · Model training and explainability",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(0, 14))
        self.split = ttk.Panedwindow(outer, orient='vertical')
        self.split.pack(fill='both', expand=True)
        self.config_panel = ttk.Frame(self.split)
        self.split.add(self.config_panel, weight=1)
        self.tabs = ttk.Notebook(self.config_panel)
        self.tabs.pack(fill='both', expand=True)
        self.training_tab()
        self.comparison_tab()
        self.label_tab()
        self.explanation_tab()

        output_panel = ttk.Frame(self.split, padding=(12, 8, 12, 10))
        self.split.add(output_panel, weight=4)
        controls = ttk.Frame(output_panel)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Experiment Output", style="Section.TLabel").pack(side="left")
        self.status = tk.StringVar(value="Ready")
        self.cancel = ttk.Button(controls, text="Cancel run", command=self.stop, state="disabled")
        self.cancel.pack(side="right")
        status_area = ttk.Frame(output_panel)
        status_area.pack(fill='x', pady=(0, 4))
        self.status_label = ttk.Label(status_area, textvariable=self.status, style='Status.TLabel', wraplength=1000)
        self.status_label.pack(anchor='w')
        status_area.bind('<Configure>', lambda e: self.status_label.configure(wraplength=max(200, e.width-10)))
        ttk.Button(controls, text='Expand output', command=self.expand_output).pack(side='right', padx=6)
        progress_row = ttk.Frame(output_panel)
        progress_row.pack(fill='x', pady=(0, 5))
        self.progress = ttk.Progressbar(progress_row, mode='indeterminate')
        self.progress.pack(side='left', fill='x', expand=True)
        self.elapsed = ttk.Label(progress_row, text='No active run')
        self.elapsed.pack(side='right', padx=8)
        toolbar = ttk.Frame(output_panel)
        toolbar.pack(fill='x', pady=(0, 5))
        for title, command in (('Open output folder', self.open_output), ('Export results', self.export_results),
                               ('View configuration', self.view_configuration)):
            ttk.Button(toolbar, text=title, command=command).pack(side='left', padx=(0, 6))
        self.output = ttk.Notebook(output_panel)
        self.output.pack(fill="both", expand=True)
        self.results = ttk.Notebook(self.output)
        self.output.add(self.results, text="Results")
        log_frame = ttk.Frame(self.output)
        self.output.add(log_frame, text="Run log")
        self.log = studio_text(log_frame, height=8, wrap="word", font=("Consolas", 10))
        self.log.insert("1.0", "Choose a workflow above, select a checkpoint and input, then run.\n"
                        "No model is loaded until you start a run. Logs and results appear here.\n"
                        "Drag the divider above Experiment Output to resize, or use Expand output.")
        self.log.configure(state="disabled")
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)
        self.history_list = tk.Listbox(self.output, background=INPUT, foreground=TEXT,
                                       selectbackground=ACCENT, relief='flat', font=('Segoe UI', 11))
        self.output.add(self.history_list, text='Run history')
        self.history_list.bind('<<ListboxSelect>>', self.load_history)
        self.refresh_history()
        welcome = studio_text(self.results, wrap='word')
        welcome.insert('1.0', 'Ready to explore your models\n\n'
                       'Classification: label one song or a CSV batch.\n'
                       'LIME / SHAP: explain a prediction or evaluate erasure.\n'
                       'Training and Comparison: run controlled model experiments.\n\n'
                       'Select an explicit checkpoint before inference. Previous runs are available in Run history.')
        welcome.configure(state='disabled')
        self.results.add(welcome, text='Getting started')
        self.output.select(0)
        self._split_initialized = False
        self.split.bind('<Configure>', self.initialize_split, add='+')
        root.after_idle(self.initialize_split)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(80, self.poll)

    def initialize_split(self, event=None):
        # A hidden Windows launch can leave the initial widget height at 1.
        # Wait for the real mapped size before allocating either pane.
        height = self.split.winfo_height()
        if not self._split_initialized and height >= 300:
            self._split_initialized = True
            self.root.after_idle(lambda: self.split.sashpos(0, max(150, int(self.split.winfo_height() * .30))))

    def page(self, title, heading):
        self.current_page = title
        shell = ttk.Frame(self.tabs)
        self.tabs.add(shell, text=title)
        canvas = tk.Canvas(shell, background=PANEL, highlightthickness=0)
        scroll = ttk.Scrollbar(shell, command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        canvas.pack(fill='both', expand=True)
        page = ttk.Frame(canvas, padding=(12, 8))
        item = canvas.create_window(0, 0, anchor='nw', window=page)
        page.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(item, width=e.width))
        def wheel(event):
            widget = self.root.winfo_containing(event.x_root, event.y_root)
            if widget and str(widget).startswith(str(shell)):
                canvas.yview_scroll(-int(event.delta / 120), 'units')
        self.root.bind('<MouseWheel>', wheel, add='+')
        ttk.Label(page, text=heading, style='Section.TLabel').pack(anchor='w', pady=(0, 8))
        return page

    def remember(self, key, default='', values=()):
        saved = self.settings.get(key, default)
        if values and saved not in values:
            saved = default
        value = tk.StringVar(value=saved)
        self.variables[key] = value
        value.trace_add('write', lambda *_: self.save_settings())
        return value

    def save_settings(self):
        self.settings.update({key: value.get() for key, value in self.variables.items()})
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.settings_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.settings, indent=2), encoding='utf-8')
            temporary.replace(self.settings_path)
        except OSError as error:
            self.settings_error = str(error)

    def field(self, parent, title, *, values=(), browse=None, default="", key=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=title, width=25).pack(side="left")
        value = self.remember(key or f'{self.current_page}.{title}', default or (values[0] if values else ''), values)
        if values:
            control = ttk.Combobox(row, textvariable=value, values=values, state="readonly")
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
        path = filedialog.askopenfilename(parent=self.root, initialdir=self.settings.get('csv_folder', str(ROOT / 'data')),
                                          filetypes=[('CSV', '*.csv'), ('All files', '*')])
        if path:
            self.settings['csv_folder'] = str(Path(path).parent)
            self.save_settings()
        return path

    def folder_picker(self):
        path = filedialog.askdirectory(parent=self.root, initialdir=self.settings.get('checkpoint_folder', str(ROOT / 'experiments')))
        if path:
            self.settings['checkpoint_folder'] = path
            self.save_settings()
        return path

    def run_button(self, parent, title, callback):
        button = ttk.Button(parent, text=title, command=callback, style="primary.TButton")
        button.pack(fill="x", pady=(10, 0))
        self.buttons.append(button)

    def training_tab(self):
        page = self.page('Training', 'Train Models')
        modes = {name: value for name, value in MODES.items() if value != 'compare'}
        mode = self.field(page, 'Architecture', values=tuple(modes))
        smoke = self.remember('Training.smoke', '1')
        ttk.Checkbutton(page, text='Smoke test (one epoch, reduced data)', variable=smoke,
                        onvalue='1', offvalue='0').pack(anchor='w', pady=6)
        ttk.Label(page, text='Blank dataset fields use the default cleaned CSVs.', style='Hint.TLabel').pack(anchor='w')
        splits = [self.field(page, title, browse=self.csv_picker) for title in ('Training CSV', 'Validation CSV', 'Test CSV')]
        self.run_button(page, 'Run training', lambda: self.start({
            'action': 'train', 'mode': modes[mode.get()], 'smoke': smoke.get() == '1',
            'datasets': [v.get().strip() for v in splits], 'runs': {}}))

    def comparison_tab(self):
        page = self.page('Comparison', 'Frozen Held-out Comparison')
        ttk.Label(page, text='Select four full checkpoints. Preflight checks model identity and dataset hashes.',
                  style='Hint.TLabel', wraplength=800).pack(anchor='w')
        runs = {}
        for name, model in MODELS.items():
            runs[model] = self.checkpoint_field(page, model, name)
        splits = [self.field(page, title, browse=self.csv_picker) for title in ('Training CSV', 'Validation CSV', 'Test CSV')]
        self.run_button(page, 'Validate and compare models', lambda: self.start({
            'action': 'train', 'mode': 'compare', 'smoke': False,
            'datasets': [v.get().strip() for v in splits], 'runs': {k: v.get().strip() for k, v in runs.items()}}))

    def checkpoint_field(self, page, model, title='Checkpoint', key=None):
        row = ttk.Frame(page)
        row.pack(fill='x', pady=3)
        ttk.Label(row, text=title, width=25).pack(side='left')
        paths = run_choices(model)
        value = self.remember(key or f'{self.current_page}.{model}.checkpoint', str(paths[0]) if paths else '')
        labels = {}
        for path in paths:
            meta = read_json(path / 'manifest.json')
            kind = 'SMOKE' if meta.get('smoke_test') else ('Full' if meta.get('smoke_test') is False else 'Unverified')
            date = datetime.fromtimestamp((path / 'best_model.pt').stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            labels[f'{model} | {date} | {kind} | {path.relative_to(ROOT / "experiments")}'] = str(path)
        selection = tk.StringVar()
        combo = ttk.Combobox(row, textvariable=selection, values=tuple(labels), state='readonly')
        combo.pack(side='left', fill='x', expand=True)
        combo.bind('<<ComboboxSelected>>', lambda e: value.set(labels[selection.get()]))
        def browse():
            path = self.folder_picker()
            if path:
                value.set(path)
        ttk.Button(row, text='Browse…', command=browse).pack(side='left', padx=6)
        info = ttk.Label(page, style='Hint.TLabel', wraplength=1000)
        info.pack(anchor='w', fill='x')
        def update(*_):
            path = value.get()
            meta = read_json(Path(path) / 'manifest.json') if path else {}
            kind = 'SMOKE TEST' if meta.get('smoke_test') else ('Full run' if meta.get('smoke_test') is False else 'Unverified manifest')
            label = next((label for label, stored in labels.items() if stored == path), Path(path).name if path else 'No checkpoint available')
            selection.set(label)
            info.configure(text=f'{kind} · {meta.get("architecture", model)} · {meta.get("run_id", "")}\n{path or "Browse to a checkpoint folder."}')
        value.trace_add('write', update)
        update()
        return value

    def model_fields(self, page):
        section = self.current_page
        name = self.field(page, 'Model', values=tuple(MODELS))
        container = ttk.Frame(page)
        container.pack(fill='x')
        holder = []
        def update(*_):
            for child in container.winfo_children():
                child.destroy()
            model = MODELS[name.get()]
            holder[:] = [self.checkpoint_field(container, model, key=f'{section}.{model}.checkpoint')]
        name.trace_add('write', update)
        update()
        return name, holder

    def model_job(self, fields):
        name, holder = fields
        folder = holder[0]
        model = MODELS[name.get()]
        run = folder.get().strip()
        if not run:
            raise ValueError('Select a checkpoint before running.')
        return {"model": model, "run": run}

    def lyrics_field(self, page):
        ttk.Label(page, text="Song lyrics").pack(anchor="w")
        field = studio_text(page, height=4, wrap="word", font=("Segoe UI", 10))
        field.pack(fill="x", pady=(2, 4))
        return field

    def label_tab(self):
        page = self.page("Classification", "Label Songs")
        model = self.model_fields(page)
        input_mode = self.field(page, 'Input mode', values=('Single song', 'CSV batch'))
        single = ttk.Frame(page)
        batch = ttk.Frame(page)
        lyrics = self.lyrics_field(single)
        source = self.field(batch, 'Input CSV', browse=self.csv_picker)
        column = self.field(batch, 'Lyrics column (auto if blank)')
        def toggle(*_):
            single.pack_forget()
            batch.pack_forget()
            (batch if input_mode.get() == 'CSV batch' else single).pack(fill='x', before=button)
        input_mode.trace_add('write', toggle)

        def run():
            try:
                job = self.model_job(model)
                path = source.get().strip() if input_mode.get() == 'CSV batch' else ''
                text = lyrics.get("1.0", "end-1c").strip()
                if not path and not text:
                    raise ValueError("Paste lyrics or choose a CSV for the selected input mode.")
                output = ""
                if path:
                    output = filedialog.asksaveasfilename(
                        parent=self.root, defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                        initialfile=f"{Path(path).stem}_labeled.csv",
                        initialdir=self.settings.get('output_folder', str(ROOT)),
                    )
                    if not output:
                        return
                    self.settings['output_folder'] = str(Path(output).parent)
                    self.save_settings()
                self.start({"action": "label", "source": path, "output": output, "lyrics": text,
                            "column": column.get().strip(), **job})
            except ValueError as error:
                self.set_status('Validation failed', 'failure')
                messagebox.showerror("Label songs", str(error), parent=self.root)

        self.run_button(page, "Predict labels", run)
        button = self.buttons[-1]
        toggle()
        self.classification_fields = model
        self.classification_lyrics = lyrics
        self.classification_mode = input_mode
        self.predict_song = run

    def explanation_tab(self):
        page = self.page("LIME / SHAP", "Explain Predictions")
        model = self.model_fields(page)
        method = self.field(page, "Explanation method", values=("LIME", "SHAP", "LIME + SHAP"))
        lyrics = self.lyrics_field(page)
        validation = self.field(page, "Validation CSV (optional)", browse=self.csv_picker)
        samples = self.field(page, "Validation sample (0 = all)", default="30")
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
                self.set_status('Validation failed', 'failure')
                messagebox.showerror("LIME & SHAP", str(error), parent=self.root)

        for title, batch in (("Explain song + erasure", False), ("Run validation erasure", True)):
            button = ttk.Button(actions, text=title, command=lambda batch=batch: run(batch),
                                style="TButton" if batch else "primary.TButton")
            button.pack(side="left", padx=(6, 0), pady=(6, 0))
            self.buttons.append(button)

    def start(self, job, validated=False):
        if self.process is not None or (self.preflighting and not validated):
            return
        if not validated:
            self.preflighting = True
            self.set_status('Validating inputs and checkpoint compatibility…', 'running')
            self.progress.start(12)
            for button in self.buttons:
                button.configure(state='disabled')
            def preflight():
                try:
                    normalized = validate_job({'schema_version': SCHEMA_VERSION, **job}, ROOT)
                    self.events.put(('validated', normalized))
                except Exception as error:
                    self.events.put(('invalid', str(error)))
            threading.Thread(target=preflight, daemon=True).start()
            return
        self.preflighting = False
        self.save_settings()
        try:
            self.history_dir.mkdir(parents=True, exist_ok=True)
            self.active_record = self.history_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.active_record.mkdir()
            (self.active_record / 'request.json').write_text(json.dumps(job, indent=2), encoding='utf-8')
            self.temp = tempfile.TemporaryDirectory(prefix="thesis_gui_")
            request = Path(self.temp.name) / "request.json"
            result = request.with_name("result.json")
            log_path = request.with_name("worker.log")
            request.write_text(json.dumps(job), encoding="utf-8")
            launcher = [sys.executable] if FROZEN else [sys.executable, "-B", str(Path(__file__).resolve())]
            self.process = subprocess.Popen(
                [*launcher, "--worker", str(request), str(result), str(log_path)],
                cwd=ROOT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as error:
            if hasattr(self, "temp"):
                self.temp.cleanup()
            self.progress.stop()
            self.set_status(f'Could not start worker: {error}', 'failure')
            for button in self.buttons:
                button.configure(state='normal')
            return
        self.cancelled = False
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.output.select(1)
        self.selected_record = self.active_record
        self.started_at = time.monotonic()
        self.set_status('Starting worker…', 'running')
        self.expand_output()
        self.cancel.configure(state="normal")
        for button in self.buttons:
            button.configure(state="disabled")

        def collect(process):
            try:
                while not log_path.is_file() and process.poll() is None:
                    time.sleep(0.08)
                if log_path.is_file():
                    with log_path.open(encoding="utf-8", errors="replace") as stream:
                        while True:
                            code = process.poll()
                            for line in stream:
                                self.events.put(("log", line))
                            if code is not None:
                                break
                            time.sleep(0.08)
                code = process.wait()
            except OSError as error:
                self.events.put(("log", f"{error}\n"))
                code = process.wait()
            try:
                payload = result.read_text(encoding='utf-8') if code == 0 and result.is_file() else None
                if payload:
                    parsed = json.loads(payload)
                    if not isinstance(parsed, dict):
                        raise ValueError('Worker returned an invalid result.')
            except (OSError, ValueError) as error:
                self.events.put(('log', f'Could not read worker result: {error}\n'))
                payload = None
            self.events.put(('done', payload))

        threading.Thread(target=collect, args=(self.process,), daemon=True).start()

    def poll(self):
        for _ in range(150):
            try:
                event, value = self.events.get_nowait()
            except queue.Empty:
                break
            if event == 'validated':
                self.start(value, validated=True)
                continue
            if event == 'invalid':
                self.preflighting = False
                self.progress.stop()
                self.set_status('Validation failed', 'failure')
                for button in self.buttons:
                    button.configure(state='normal')
                messagebox.showerror('Input validation', value, parent=self.root)
                continue
            if event == "log":
                if value.startswith("THESIS_PROGRESS "):
                    value = value.removeprefix("THESIS_PROGRESS ")
                    if not self.cancelled:
                        self.set_status(value.strip(), 'running')
                if not self.cancelled and re.search(r'epoch\s+\d+', value, re.IGNORECASE):
                    self.set_status(value.strip()[:160], 'running')
                self.log.configure(state="normal")
                self.log.insert("end", value)
                self.log.see("end")
                self.log.configure(state="disabled")
            else:
                self.process = None
                self.temp.cleanup()
                self.progress.stop()
                duration = time.monotonic() - self.started_at if self.started_at else 0
                self.started_at = None
                self.elapsed.configure(text=f'Elapsed {duration:.1f}s')
                self.cancel.configure(state="disabled")
                for button in self.buttons:
                    button.configure(state="normal")
                if self.cancelled:
                    self.set_status("Cancelled. Partial experiment files may exist.", 'cancelled')
                elif value is None:
                    self.set_status("Run failed. See the run log for details.", 'failure')
                else:
                    self.display(json.loads(value))
                    self.set_status("Completed", 'success')
                self.archive_run(value, duration)
        if self.started_at:
            seconds = int(time.monotonic() - self.started_at)
            self.elapsed.configure(text=f'Elapsed {seconds // 60:02d}:{seconds % 60:02d}')
        self.root.after(80, self.poll)

    def set_status(self, text, kind='running'):
        self.status.set(text)
        self.status_label.configure(foreground={'failure': '#FF6B78', 'cancelled': '#F1C578',
                                               'success': SUCCESS, 'running': '#92C2FF'}.get(kind, MUTED))

    def expand_output(self):
        self.split.sashpos(0, max(100, int(self.split.winfo_height() * .23)))

    def archive_run(self, payload, duration):
        if not self.active_record:
            return
        try:
            (self.active_record / 'worker.log').write_text(self.log.get('1.0', 'end-1c'), encoding='utf-8')
            if payload:
                (self.active_record / 'result.json').write_text(payload, encoding='utf-8')
            (self.active_record / 'summary.json').write_text(json.dumps({
                'status': 'Cancelled' if self.cancelled else ('Completed' if payload else 'Failed'),
                'seconds': duration}, indent=2), encoding='utf-8')
            self.selected_record = self.active_record
            self.refresh_history()
        except OSError as error:
            self.set_status(f'Could not save run history: {error}', 'failure')

    def refresh_history(self):
        self.history_paths = sorted(self.history_dir.glob('*/summary.json'), reverse=True)
        self.history_list.delete(0, 'end')
        for path in self.history_paths:
            summary = read_json(path)
            request = read_json(path.parent / 'request.json')
            self.history_list.insert('end', f'{path.parent.name}  |  {request.get("mode", request.get("action", ""))}  |  '
                                     f'{summary.get("status", "")}  |  {summary.get("seconds", 0):.1f}s')

    def load_history(self, event=None):
        selected = self.history_list.curselection()
        if not selected or self.process or self.preflighting:
            return
        folder = self.history_paths[selected[0]].parent
        self.selected_record = folder
        result = read_json(folder / 'result.json')
        if result:
            self.display(result)
        else:
            for tab in self.results.tabs():
                self.results.nametowidget(tab).destroy()
            self.output.select(1)
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.insert('1.0', (folder / 'worker.log').read_text(encoding='utf-8') if (folder / 'worker.log').exists() else '')
        self.log.configure(state='disabled')
        summary = read_json(folder / 'summary.json')
        state = summary.get('status', 'Completed')
        self.set_status(f'History · {state}', {'Failed': 'failure', 'Cancelled': 'cancelled'}.get(state, 'success'))

    def open_output(self):
        record = self.selected_record
        if not record:
            messagebox.showinfo('Output', 'Complete a run or select one from Run history.', parent=self.root)
            return
        result = read_json(record / 'result.json')
        folders = result.get('output_paths', [])
        path = Path(folders[0]) if folders else record
        if path.is_file():
            path = path.parent
        if not path.is_dir():
            path = record
        os.startfile(str(path))

    def export_results(self):
        if not self.selected_record:
            messagebox.showinfo('Export', 'Complete a run or select one from Run history.', parent=self.root)
            return
        folder = filedialog.askdirectory(parent=self.root, title='Export run files',
                                         initialdir=self.settings.get('output_folder', str(ROOT)))
        if not folder:
            return
        import shutil
        target = Path(folder) / self.selected_record.name
        if target.resolve() == self.selected_record.resolve() or target.exists():
            messagebox.showerror('Export', 'Choose a folder without an existing export of this run.', parent=self.root)
            return
        try:
            shutil.copytree(self.selected_record, target)
            result = read_json(target / 'result.json')
            for index, (title, rows) in enumerate(result.get('tables', {}).items(), 1):
                if rows:
                    with (target / f'table_{index}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
            self.settings['output_folder'] = folder
            self.save_settings()
            messagebox.showinfo('Exported', str(target), parent=self.root)
        except OSError as error:
            messagebox.showerror('Export failed', str(error), parent=self.root)

    def view_configuration(self):
        record = self.selected_record or self.active_record
        if not record:
            messagebox.showinfo('Configuration', 'Start a run or select one from Run history.', parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title('Run configuration')
        window.geometry('800x550')
        text = studio_text(window, wrap='word')
        scroll = ttk.Scrollbar(window, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        text.pack(fill='both', expand=True)
        text.insert('1.0', json.dumps(read_json(record / 'request.json'), indent=2))
        text.configure(state='disabled')

    def display_charts(self, result):
        charts = result.get('charts', [])
        if not charts:
            return
        frame = ttk.Frame(self.results)
        self.results.add(frame, text='Charts')
        selector = ttk.Combobox(frame, values=[chart['title'] for chart in charts], state='readonly')
        selector.pack(fill='x', padx=8, pady=6)
        canvas = tk.Canvas(frame, background=INPUT, highlightthickness=0)
        canvas.pack(fill='both', expand=True)
        def draw(event=None):
            if not canvas.winfo_exists():
                return
            canvas.delete('all')
            index = max(0, selector.current())
            chart = charts[index]
            width, height = max(400, canvas.winfo_width()), max(200, canvas.winfo_height())
            left, top, right, bottom = 75, 35, width - 35, height - 65
            canvas.create_text(width / 2, 15, text=chart['title'], fill=TEXT, font=('Segoe UI', 12, 'bold'))
            if chart.get('matrix') is not None:
                matrix = chart['matrix']
                peak = max(max(row) for row in matrix) or 1
                size = min((right-left)/3, (bottom-top)/3)
                for r, row in enumerate(matrix):
                    for c, value in enumerate(row):
                        shade = int(40 + 150 * value / peak)
                        x, y = left + c * size, top + r * size
                        canvas.create_rectangle(x, y, x+size, y+size, fill=f'#{30:02x}{shade:02x}{150:02x}', outline=PANEL)
                        canvas.create_text(x+size/2, y+size/2, text=str(value), fill=TEXT)
                for i, label in enumerate(('Safe', 'Mild', 'Explicit')):
                    canvas.create_text(left+i*size+size/2, top+3*size+15, text=label, fill=MUTED)
                    canvas.create_text(left-8, top+i*size+size/2, text=label, anchor='e', fill=MUTED)
                canvas.create_text(left+size*1.5, top+3*size+40, text='Predicted label · rows = actual label', fill=MUTED)
                return
            series = chart.get('series', [])
            values = [float(y) for item in series for y in item['y'] if y is not None]
            if not values:
                return
            low, high = min(0, min(values)), max(values)
            if low == high:
                high = low + 1
            high += (high-low)*.08
            for tick in range(5):
                value = low+(high-low)*tick/4
                y = bottom-(bottom-top)*tick/4
                canvas.create_line(left, y, right, y, fill=PANEL)
                canvas.create_text(left-8, y, text=f'{value:.3g}', fill=MUTED, anchor='e')
            colors = ('#92C2FF', '#78CDA6', '#F1C578', '#F3A0B1')
            labels = chart.get('labels', [])
            n = max((len(item['y']) for item in series), default=1)
            for j, item in enumerate(series):
                color = colors[j % len(colors)]
                canvas.create_text(left+j*180, height-15, text=item['name'], fill=color, anchor='w')
                points = []
                for i, value in enumerate(item['y']):
                    if value is None:
                        continue
                    x = left+(right-left)*(i+.5)/max(n, 1)
                    y = bottom-(float(value)-low)/(high-low)*(bottom-top)
                    if chart.get('kind') == 'bar':
                        bw = (right-left)/max(n,1)/max(len(series),1)*.7
                        bx = x+(j-(len(series)-1)/2)*bw
                        zero = bottom-(0-low)/(high-low)*(bottom-top)
                        canvas.create_rectangle(bx-bw/2, y, bx+bw/2, zero, fill=color, outline='')
                    else:
                        points.extend((x, y))
                        canvas.create_oval(x-3, y-3, x+3, y+3, fill=color, outline='')
                if len(points) >= 4:
                    canvas.create_line(*points, fill=color, width=2)
            for i in range(n):
                label = str(labels[i]) if i < len(labels) else str(i+1)
                canvas.create_text(left+(right-left)*(i+.5)/max(n,1), bottom+18, text=label, fill=MUTED, width=130)
        selector.current(0)
        selector.bind('<<ComboboxSelected>>', draw)
        canvas.bind('<Configure>', draw)
        self.root.after_idle(draw)

    def display(self, result):
        for tab in self.results.tabs():
            self.results.nametowidget(tab).destroy()
        prediction = result.get('tables', {}).get('Song label', [])
        if prediction:
            record = prediction[0]
            card = ttk.Frame(self.results, padding=24)
            self.results.add(card, text='Prediction')
            label = record.get('v26_label') or 'No usable lyrics'
            ttk.Label(card, text=label, font=('Segoe UI', 34, 'bold'),
                      foreground={'Safe': SUCCESS, 'Mild': '#F1C578', 'Explicit': '#FF6B78'}.get(label, MUTED)).pack(anchor='w')
            probability = record.get('v26_evidence_probability')
            if probability is not None:
                ttk.Label(card, text=f'Evidence-window probability: {probability:.1%}', font=('Segoe UI', 16)).pack(anchor='w', pady=10)
                ttk.Label(card, text=f'Evidence window {record.get("v26_evidence_window")} of {record.get("v26_window_count")}').pack(anchor='w')
                ttk.Label(card, text='This probability describes the selected evidence window, not a calibrated song-level confidence.',
                          style='Hint.TLabel', wraplength=900).pack(anchor='w', pady=12)
        self.display_charts(result)
        for method, features in result.get("highlights", {}).items():
            self.display_highlights(method, features, result.get('contexts', {}).get(method, ''))
        if result.get("message"):
            details = ttk.Frame(self.results)
            body = studio_text(details, wrap='word', height=7)
            scrollbar = ttk.Scrollbar(details, command=body.yview)
            body.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side='right', fill='y')
            body.pack(fill='both', expand=True)
            body.insert("1.0", result["message"])
            body.configure(state="disabled")
            self.results.add(details, text="Details")
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

    def display_highlights(self, method, features, context=''):
        frame = ttk.Frame(self.results, padding=8)
        self.results.add(frame, text=f"{method} highlights")
        if context:
            ttk.Label(frame, text=context, wraplength=1000, font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=(0, 8))
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
            self.set_status("Cancelling…", 'cancelled')

    def close(self):
        self.save_settings()
        if self.preflighting:
            messagebox.showinfo('Validating', 'Please wait for input validation to finish.', parent=self.root)
            return
        if self.process:
            if not messagebox.askyesno("Exit", "Cancel the active run and close?", parent=self.root):
                return
            self.stop()
            self.process.wait()
            self.temp.cleanup()
        self.root.destroy()


def run_worker(request, result, log_path):
    # PyInstaller's windowed executable has no stdout/stderr streams.
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                print('THESIS_PROGRESS Loading Python and model libraries…', flush=True)
                from thesis_gui_worker import run_job

                run_job(Path(request), Path(result), ROOT)
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if sys.argv[1:2] == ["--worker"]:
        sys.exit(run_worker(*sys.argv[2:]))
    window = tk.Tk()
    Application(window)
    window.mainloop()
