import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT/'kaggle_notebook'/'occluded-object-detection.ipynb'
nb = json.load(open(nb_path,encoding='utf-8'))
cells=[]

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
cells.append({'cell_type':'code','metadata':{},'outputs':[],'execution_count':None,'source':[_cell0]})

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
cells.append({'cell_type':'code','metadata':{},'outputs':[],'execution_count':None,'source':[_cell1]})

_cell2 = (
'# 1) Convert MoCA CSV annotations -> GT masks (MUST run before make_split.py,\n'
'#    which enumerates sequences from the converted tree).\n'
'run(f"{CODE_DIR}/convert_moca_csv_to_masks.py",\n'
'    "--moca_root", MOCA,\n'
'    "--output_root", "/kaggle/working/moca_converted")\n'
'assert os.path.isdir("/kaggle/working/moca_converted")\n'
'print("moca_converted subdirs:", sorted(os.listdir("/kaggle/working/moca_converted"))[:10])\n'
)
cells.append({'cell_type':'code','metadata':{},'outputs':[],'execution_count':None,'source':[_cell2]})

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
cells.append({'cell_type':'code','metadata':{},'outputs':[],'execution_count':None,'source':[_cell3]})

_cell4 = (
'# 3) Run A: amplified motion cache + train (best checkpoint chosen by val IoU).\n'
'os.makedirs("/kaggle/working/cache_amp", exist_ok=True)\n'
'os.makedirs("/kaggle/working/runs_amp", exist_ok=True)\n'
'run(f"{CODE_DIR}/preprocess_amplify.py",\n'
'    "--data_roots", "moca=/kaggle/working/moca_converted", f"cad={CAD}", f"yuv20k={YUV}",\n'
'    "--cache_root", "/kaggle/working/cache_amp",\n'
'    "--fps", "24", "--window_size", "40", "--overlap", "20", "--motion_size", "256",\n'
'    "--num_workers", str(NUM_WORKERS))\n'
'run(f"{CODE_DIR}/train.py",\n'
'    "--data_roots", "moca=/kaggle/working/moca_converted", f"cad={CAD}", f"yuv20k={YUV}",\n'
'    "--adapters", "moca=moca_mask", "cad=cad", "yuv20k=imgs_gt",\n'
'    "--cache_root", "/kaggle/working/cache_amp",\n'
'    "--split_dir", "/kaggle/working/splits", "--split", "train",\n'
'    "--checkpoint_dir", "/kaggle/working/runs_amp",\n'
'    "--epochs", "10", "--batch_size", "8", "--image_size", "256", "--base_ch", "32", "--seed", "42")\n'
'assert os.path.exists("/kaggle/working/runs_amp/best_by_val.pth")\n'
'print("RUN A DONE")\n'
)
cells.append({'cell_type':'code','metadata':{},'outputs':[],'execution_count':None,'source':[_cell4]})

_cell5 = (
'# 4) Run B: raw (unamplified) motion cache + train, identical everything else.\n'
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
'    "--checkpoint_dir", "/kaggle/working/runs_raw",\n'
'    "--epochs", "10", "--batch_size", "8", "--image_size", "256", "--base_ch", "32", "--seed", "42")\n'
'assert os.path.exists("/kaggle/working/runs_raw/best_by_val.pth")\n'
'print("RUN B DONE")\n'
)
cells.append({'cell_type':'code','metadata':{},'outputs':[],'execution_count':None,'source':[_cell5]})

nb['cells']=cells
json.dump(nb, open(nb_path,'w',encoding='utf-8'), indent=2)
print('ok')
