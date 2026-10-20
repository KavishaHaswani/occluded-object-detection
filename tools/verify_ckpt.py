import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'kaggle_code'))
from model import TwoStreamCOD  # noqa: E402

base = ROOT / 'kaggle_notebook' / 'reports'
for run in ['runs_amp', 'runs_raw']:
    for name in ['best_by_val.pth', 'last_model.pth']:
        p = base / run / name
        ck = torch.load(p, map_location='cpu', weights_only=False)
        if isinstance(ck, dict) and 'model_state' in ck:
            sd = ck['model_state']
            meta = {k: v for k, v in ck.items() if k != 'model_state'}
        else:
            sd, meta = ck, {}
        m = TwoStreamCOD(base_ch=32)
        m.load_state_dict(sd)
        m.eval()
        print('%s/%s: params=%s meta=%s -> load OK' % (
            run, name, f'{sum(x.numel() for x in m.parameters()):,}',
            {k: meta[k] for k in list(meta)[:6]}))
