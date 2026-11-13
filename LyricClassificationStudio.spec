# Windows onedir bundle. Keep the executable and its _internal folder together.
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

packages = ("transformers.models.bert", "transformers.models.roberta",
            "transformers.models.auto", "shap.explainers", "shap.maskers")
hiddenimports = [name for package in packages for name in collect_submodules(package)
                 if ".modeling_tf_" not in name and ".modeling_flax_" not in name]
hiddenimports += ["accelerate", "safetensors.torch", "tokenizers", "scipy.special.cython_special"]

a = Analysis(
    ["thesis_gui.py"],
    datas=collect_data_files("ttkbootstrap") + collect_data_files("shap") + copy_metadata("ttkbootstrap"),
    hiddenimports=hiddenimports,
    excludes=["IPython", "pytest", "notebook", "tensorflow", "jax", "torchvision", "torchaudio"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="LyricClassificationStudio", console=False, upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="LyricClassificationStudio", upx=False)
