"""Run this FIRST on every machine / Colab / Kaggle session:  python scripts/check_env.py"""
import importlib
import platform
import sys

REQUIRED = [
    ("torch", "torch"), ("torchvision", "torchvision"), ("timm", "timm"),
    ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
    ("sklearn", "scikit-learn"), ("pydicom", "pydicom"),
    ("pytorch_grad_cam", "grad-cam"), ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"), ("streamlit", "streamlit"), ("yaml", "pyyaml"),
]

print(f"Python {sys.version.split()[0]} on {platform.platform()}")
if sys.version_info < (3, 10):
    print("  [X] Python >= 3.10 required")
missing = []
for mod, pip_name in REQUIRED:
    try:
        m = importlib.import_module(mod)
        print(f"  [OK] {pip_name:<14} {getattr(m, '__version__', '')}")
    except ImportError:
        print(f"  [X]  {pip_name:<14} MISSING")
        missing.append(pip_name)

try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name}, {p.total_memory / 1e9:.1f} GB")
    else:
        print("  GPU: none (CPU only). Use Colab/Kaggle for training.")
except ImportError:
    pass

if missing:
    print("\nInstall missing:  pip install " + " ".join(missing))
    sys.exit(1)
print("\nEnvironment OK")
