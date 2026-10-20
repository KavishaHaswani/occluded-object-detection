import json
from pathlib import Path

OUT = Path("kaggle_notebook/diag")
OUT.mkdir(parents=True, exist_ok=True)

cells = []

env = (
    "import os\n\n"
    "def _pick(*cands):\n"
    "    for c in cands:\n"
    "        if os.path.isdir(c):\n"
    "            return c\n"
    "    return cands[0]\n\n"
    'CODE_DIR = _pick("/kaggle/input/datasets/kavishahaswani/occluded-object-detection-code",\n'
    '                "/kaggle/input/kavishahaswani/occluded-object-detection-code")\n'
    'CAD = _pick("/kaggle/input/datasets/tanushkalokegaonkar/camouflaged-animal-dataset/CamouflagedAnimalDataset",\n'
    '            "/kaggle/input/tanushkalokegaonkar/camouflaged-animal-dataset/CamouflagedAnimalDataset")\n'
    'print("CODE:", CODE_DIR)\n'
    'print("CAD:", CAD)\n'
    "assert os.path.isdir(CODE_DIR)\n"
    "print('code files:', sorted(os.listdir(CODE_DIR)))\n"
    "print('CAD top:', sorted(os.listdir(CAD))[:20])\n"
)
cells.append(env)

split = (
    "import subprocess, sys, os, json\n"
    "os.makedirs('/kaggle/working/diag_splits', exist_ok=True)\n"
    "subprocess.run([sys.executable, os.path.join(CODE_DIR, 'make_split.py'),\n"
    "                '--cad_root', CAD, '--out', '/kaggle/working/diag_splits'], check=True)\n"
    "print('split files:', sorted(os.listdir('/kaggle/working/diag_splits')))\n"
    "print(open('/kaggle/working/diag_splits/split_summary.json').read())\n"
)
cells.append(split)

pre = (
    "import subprocess, sys, os\n"
    "os.makedirs('/kaggle/working/diag_cache', exist_ok=True)\n"
    "subprocess.run([sys.executable, os.path.join(CODE_DIR, 'preprocess_amplify.py'),\n"
    "                '--data_roots', 'cad=' + CAD,\n"
    "                '--cache_root', '/kaggle/working/diag_cache',\n"
    "                '--fps', '24', '--window_size', '40', '--overlap', '20',\n"
    "                '--motion_size', '256'], check=True)\n"
)
cells.append(pre)

check = (
    "import os, glob\n"
    "import numpy as np\n"
    "files = sorted(glob.glob('/kaggle/working/diag_cache/**/motion/*.npy', recursive=True))\n"
    "print('n npy files:', len(files))\n"
    "print('first 8 paths:', files[:8])\n"
    "print('first 8 sizes (bytes):', [os.path.getsize(f) for f in files[:8]])\n"
    "a = np.load(files[0])\n"
    "print('loaded shape:', a.shape, 'dtype:', a.dtype, 'nbytes:', a.nbytes)\n"
    "print('min/max:', float(a.min()), float(a.max()))\n"
    "assert a.shape == (256, 256), f'EXPECTED 256x256 GOT {a.shape}'\n"
    "print('DIAG OK')\n"
)
cells.append(check)

nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": c.splitlines(keepends=True),
        }
        for c in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(OUT / "diag.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

meta = {
    "id": "kavishahaswani/occluded-object-detection-diag",
    "title": "occluded-object-detection-diag",
    "code_file": "diag.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [
        "ivanomelchenkoim11/moca-dataset",
        "tanushkalokegaonkar/camouflaged-animal-dataset",
        "kavishahaswani/occluded-object-detection-code",
    ],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("diag notebook written to", OUT)
