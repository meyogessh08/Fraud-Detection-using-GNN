"""
SYSTEM VERIFICATION SCRIPT - Windows Compatible
Run this first to ensure everything is installed correctly
"""

print("="*80)
print(" "*20 + "FRAUD DETECTION PROJECT SETUP VERIFICATION")
print("="*80)

import sys
print(f"\n[OK] Python Executable: {sys.executable}")
print(f"[OK] Python Version: {sys.version.split()[0]}")

# Track failures
failed_packages = []
passed_packages = []

# Test imports
print("\n" + "-"*80)
print("Testing Package Imports...")
print("-"*80)

try:
    import pandas as pd
    print(f"[PASS] Pandas: {pd.__version__}")
    passed_packages.append("pandas")
except ImportError as e:
    print(f"[FAIL] Pandas: {e}")
    failed_packages.append("pandas")

try:
    import numpy as np
    print(f"[PASS] NumPy: {np.__version__}")
    passed_packages.append("numpy")
except ImportError as e:
    print(f"[FAIL] NumPy: {e}")
    failed_packages.append("numpy")

try:
    import scipy
    print(f"[PASS] SciPy: {scipy.__version__}")
    passed_packages.append("scipy")
except ImportError as e:
    print(f"[FAIL] SciPy: {e}")
    failed_packages.append("scipy")

try:
    import sklearn
    print(f"[PASS] Scikit-learn: {sklearn.__version__}")
    passed_packages.append("scikit-learn")
except ImportError as e:
    print(f"[FAIL] Scikit-learn: {e}")
    failed_packages.append("scikit-learn")

try:
    import matplotlib
    print(f"[PASS] Matplotlib: {matplotlib.__version__}")
    passed_packages.append("matplotlib")
except ImportError as e:
    print(f"[FAIL] Matplotlib: {e}")
    failed_packages.append("matplotlib")

try:
    import seaborn as sns
    print(f"[PASS] Seaborn: {sns.__version__}")
    passed_packages.append("seaborn")
except ImportError as e:
    print(f"[FAIL] Seaborn: {e}")
    failed_packages.append("seaborn")

try:
    import torch
    print(f"[PASS] PyTorch: {torch.__version__}")
    print(f"       CUDA Available: {torch.cuda.is_available()}")
    print(f"       Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    passed_packages.append("torch")
except ImportError as e:
    print(f"[FAIL] PyTorch: {e}")
    failed_packages.append("torch")

try:
    import torch_geometric
    print(f"[PASS] PyTorch Geometric: {torch_geometric.__version__}")
    passed_packages.append("torch_geometric")
except ImportError as e:
    print(f"[FAIL] PyTorch Geometric: {e}")
    failed_packages.append("torch_geometric")

try:
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.data import Data
    from torch_geometric.utils import add_self_loops
    print(f"[PASS] PyTorch Geometric Components: All Available")
    print(f"       - GCNConv, SAGEConv, GATConv")
    print(f"       - Data, add_self_loops")
    passed_packages.append("torch_geometric_components")
except ImportError as e:
    print(f"[FAIL] PyTorch Geometric Components: {e}")
    failed_packages.append("torch_geometric_components")

try:
    import networkx as nx
    print(f"[PASS] NetworkX: {nx.__version__}")
    passed_packages.append("networkx")
except ImportError as e:
    print(f"[FAIL] NetworkX: {e}")
    failed_packages.append("networkx")

try:
    import shap
    print(f"[PASS] SHAP: {shap.__version__}")
    passed_packages.append("shap")
except ImportError as e:
    print(f"[FAIL] SHAP: {e}")
    failed_packages.append("shap")

try:
    import imblearn
    print(f"[PASS] Imbalanced-learn: {imblearn.__version__}")
    passed_packages.append("imbalanced-learn")
except ImportError as e:
    print(f"[FAIL] Imbalanced-learn: {e}")
    failed_packages.append("imbalanced-learn")

try:
    import tqdm
    print(f"[PASS] tqdm: {tqdm.__version__}")
    passed_packages.append("tqdm")
except ImportError as e:
    print(f"[FAIL] tqdm: {e}")
    failed_packages.append("tqdm")

# Summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)
print(f"[PASS] Passed: {len(passed_packages)}/{len(passed_packages) + len(failed_packages)}")
print(f"[FAIL] Failed: {len(failed_packages)}/{len(passed_packages) + len(failed_packages)}")

if failed_packages:
    print(f"\n[FAIL] FAILED PACKAGES:")
    for pkg in failed_packages:
        print(f"       - {pkg}")
    print(f"\n[FIX] TO FIX: Run this command in terminal:")
    print(f"      pip install {' '.join(failed_packages)}")
else:
    print(f"\n[SUCCESS] ALL PACKAGES INSTALLED SUCCESSFULLY!")
    print(f"[SUCCESS] SYSTEM IS READY FOR FRAUD DETECTION PROJECT!")

print("="*80)