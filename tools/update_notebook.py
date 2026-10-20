import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / 'kaggle_notebook' / 'occluded-object-detection.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))
cells = []

_cell0 = (
'import os\n'
'\n'
'def _pick(*cands):\n'
'    for c in cands:\n'
'        if os.path.isdir(c):\n'
'            return c\n'
'    return cands[0]\n'
'\n'
'CODE_DIR = _pick(\n'
'    "/kaggle/input/datasets/kavishahaswani/occluded-object-detection-code",\n'
'    "/kaggle/input/kavishahaswani/occluded-object-detection-code",\n'
')\n'
'MOCA = _pick(\n'
'    "/kaggle/input/datasets/ivanomelchenkoim11/moca-dataset/MoCA",\n'
'    "/kaggle/input/ivanomelchenkoim11/moca-dataset/MoCA",\n'
')\n'
'CAD = _pick(\n'
'    "/kaggle/input/datasets/tanushkalokegaonkar/camouflaged-animal-dataset/CamouflagedAnimalDataset",\n'
'    "/kaggle/input/tanushkalokegaonkar/camouflaged-animal-dataset/CamouflagedAnimalDataset",\n'
')\n'
'YUV = _pick(\n'
'    "/kaggle/input/datasets/tanushkalokegaonkar/yuv20k/YUV20K",\n'
'    "/kaggle/input/tanushkalokegaonkar/yuv20k/YUV20K",\n'
')\n'
'print("CODE:", CODE_DIR)\n'
'print("MOCA:", MOCA)\n'
'print("CAD:", CAD)\n'
'print("YUV:", YUV)\n'
'assert os.path.isdir(CODE_DIR), f"code dataset not found under {CODE_DIR}"\n'
'print("code files:", sorted(os.listdir(CODE_DIR)))\n'
)
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': [_cell0]})

_cell1 = (
'import os, sys, subprocess\n'
'\n'
'def run(*args):\n'
'    """Run a python script with check=True so any failure STOPS the notebook\n'
'    (unlike `!python ...`, which silently continues on error)."""\n'
'    args = [str(a) for a in args]\n'
'    print("$ python", " ".join(args), flush=True)\n'
'    subprocess.run([sys.executable, *args], check=True)\n'
'    print("OK:", args[0].split("/")[-1], flush=True)\n'
'\n'
'NUM_WORKERS = os.cpu_count() or 1\n'
'print("CPU workers for preprocessing:", NUM_WORKERS)\n'
)
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': [_cell1]})

_cell2 = (
'# 1) Convert MoCA CSV annotations -> GT masks (MUST run before make_split.py,\n'
'#    which enumerates sequences from the converted tree).\n'
'run(f"{CODE_DIR}/convert_moca_csv_to_masks.py",\n'
'    "--moca_root", MOCA,\n'
'    "--output_root", "/kaggle/working/moca_converted")\n'
'assert os.path.isdir("/kaggle/working/moca_converted")\n'
'print("moca_converted subdirs:", sorted(os.listdir("/kaggle/working/moca_converted"))[:10])\n'
)
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': [_cell2]})

_cell3 = (
'# 2) Build the deterministic sequence-level splits (seed 42).\n'
'run(f"{CODE_DIR}/make_split.py",\n'
'    "--moca_root", "/kaggle/working/moca_converted",\n'
'    "--cad_root", CAD,\n'
'    "--yuv_root", YUV,\n'
'    "--out", "/kaggle/working/splits")\n'
'print(open("/kaggle/working/splits/split_summary.json").read())\n'
'for name in ("moca", "cad", "yuv20k"):\n'
'    for split in ("train", "val", "test"):\n'
'        p = f"/kaggle/working/splits/{name}_{split}.json"\n'
'        assert os.path.exists(p), f"missing split file {p}"\n'
'print("split files OK:", sorted(os.listdir("/kaggle/working/splits")))\n'
)
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': [_cell3]})

_cell4 = (
'# 3) Arm 1: RAW (unamplified motion cache + train)\n'
'os.makedirs("/kaggle/working/cache_raw", exist_ok=True)\n'
'os.makedirs("/kaggle/working/runs_raw", exist_ok=True)\n'
'run(f"{CODE_DIR}/preprocess_amplify.py",\n'
'    "--data_roots", "moca=/kaggle/working/moca_converted", f"cad={CAD}", f"yuv20k={YUV}",\n'
'    "--cache_root", "/kaggle/working/cache_raw",\n'
'    "--fps", "24", "--window_size", "40", "--overlap", "20", "--motion_size", "256", "--no_amplify",\n'
'    "--num_workers", str(NUM_WORKERS))\n'
'run(f"{CODE_DIR}/train.py",\n'
'    "--data_roots", "moca=/kaggle/working/moca_converted", f"cad={CAD}", f"yuv20k={YUV}",\n'
'    "--adapters", "moca=moca_mask", "cad=cad", "yuv20k=imgs_gt",\n'
'    "--cache_root", "/kaggle/working/cache_raw",\n'
'    "--split_dir", "/kaggle/working/splits", "--split", "train",\n'
'    "--checkpoint_dir", "/kaggle/working/runs_raw", "--arm_name", "raw",\n'
'    "--epochs", "10", "--batch_size", "8", "--image_size", "256", "--base_ch", "32", "--seed", "42")\n'
'assert os.path.exists("/kaggle/working/runs_raw/best_by_val.pth")\n'
'print("ARM 1 (RAW) DONE")\n'
)
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': [_cell4]})

_cell5 = (
'# 4) Arm 2: AMP_OLD (alpha 15, low 0.2, high 3.0, 4 levels)\n'
'os.makedirs("/kaggle/working/cache_amp_old", exist_ok=True)\n'
'os.makedirs("/kaggle/working/runs_amp_old", exist_ok=True)\n'
'run(f"{CODE_DIR}/preprocess_amplify.py",\n'
'    "--data_roots", "moca=/kaggle/working/moca_converted", f"cad={CAD}", f"yuv20k={YUV}",\n'
'    "--cache_root", "/kaggle/working/cache_amp_old",\n'
'    "--fps", "24", "--window_size", "40", "--overlap", "20", "--motion_size", "256",\n'
'    "--alpha", "15.0", "--low_hz", "0.2", "--high_hz", "3.0", "--levels", "4",\n'
'    "--num_workers", str(NUM_WORKERS))\n'
'run(f"{CODE_DIR}/train.py",\n'
'    "--data_roots", "moca=/kaggle/working/moca_converted", f"cad={CAD}", f"yuv20k={YUV}",\n'
'    "--adapters", "moca=moca_mask", "cad=cad", "yuv20k=imgs_gt",\n'
'    "--cache_root", "/kaggle/working/cache_amp_old",\n'
'    "--split_dir", "/kaggle/working/splits", "--split", "train",\n'
'    "--checkpoint_dir", "/kaggle/working/runs_amp_old", "--arm_name", "amp_old",\n'
'    "--epochs", "10", "--batch_size", "8", "--image_size", "256", "--base_ch", "32", "--seed", "42")\n'
'assert os.path.exists("/kaggle/working/runs_amp_old/best_by_val.pth")\n'
'print("ARM 2 (AMP_OLD) DONE")\n'
)
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': [_cell5]})

_cell6 = (
'# 5) Arm 3: AMP_V2 (alpha 5, low 0.5, high 2.5, brightness-only, coarse-only, median-sub, percentile norm)\n'
'os.makedirs("/kaggle/working/cache_amp_v2", exist_ok=True)\n'
'os.makedirs("/kaggle/working/runs_amp_v2", exist_ok=True)\n'
'run(f"{CODE_DIR}/preprocess_amplify.py",\n'
'    "--data_roots", "moca=/kaggle/working/moca_converted", f"cad={CAD}", f"yuv20k={YUV}",\n'
'    "--cache_root", "/kaggle/working/cache_amp_v2",\n'
'    "--fps", "24", "--window_size", "40", "--overlap", "20", "--motion_size", "256",\n'
'    "--alpha", "5.0", "--low_hz", "0.5", "--high_hz", "2.5", "--levels", "4",\n'
'    "--brightness_only", "--coarse_only", "--median_sub", "--norm_mode", "percentile",\n'
'    "--num_workers", str(NUM_WORKERS))\n'
'run(f"{CODE_DIR}/train.py",\n'
'    "--data_roots", "moca=/kaggle/working/moca_converted", f"cad={CAD}", f"yuv20k={YUV}",\n'
'    "--adapters", "moca=moca_mask", "cad=cad", "yuv20k=imgs_gt",\n'
'    "--cache_root", "/kaggle/working/cache_amp_v2",\n'
'    "--split_dir", "/kaggle/working/splits", "--split", "train",\n'
'    "--checkpoint_dir", "/kaggle/working/runs_amp_v2", "--arm_name", "amp_v2",\n'
'    "--epochs", "10", "--batch_size", "8", "--image_size", "256", "--base_ch", "32", "--seed", "42")\n'
'assert os.path.exists("/kaggle/working/runs_amp_v2/best_by_val.pth")\n'
'print("ARM 3 (AMP_V2) DONE")\n'
)
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': [_cell6]})

_cell7 = (
'# 6) Summary Comparison Table across all 3 arms on Test Split\n'
'import json\n'
'arms = ["raw", "amp_old", "amp_v2"]\n'
'metrics = ["IoU", "Dice", "MAE", "S_measure", "weighted_F"]\n'
'header = f"{\'Arm\':<12} | " + " | ".join(f"{m:>10}" for m in metrics)\n'
'print("=" * len(header))\n'
'print("3-ARM TEST SPLIT BENCHMARK COMPARISON")\n'
'print("=" * len(header))\n'
'print(header)\n'
'print("-" * len(header))\n'
'for arm in arms:\n'
'    p = f"/kaggle/working/runs_{arm}/test_report.json"\n'
'    if os.path.exists(p):\n'
'        rep = json.loads(open(p).read())\n'
'        ov = rep.get("overall", {})\n'
'        vals = " | ".join(f"{ov.get(m, 0.0):>10.4f}" for m in metrics)\n'
'        print(f"{arm:<12} | {vals}")\n'
'    else:\n'
'        print(f"{arm:<12} | [NOT FOUND]")\n'
'print("-" * len(header))\n'
)
cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': [_cell7]})

nb['cells'] = cells
json.dump(nb, open(nb_path, 'w', encoding='utf-8'), indent=2)
print('Updated', nb_path, 'successfully with 3 arms.')
