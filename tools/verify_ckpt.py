import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer_demo import load_checkpoint  # noqa: E402

base = ROOT / 'kaggle_notebook' / 'reports'
for run in ['runs_amp', 'runs_raw']:
    for name in ['best_by_val.pth', 'last_model.pth']:
        p = base / run / name
        m = load_checkpoint(p, device='cpu', base_ch=32)
        meta = getattr(m, 'checkpoint_meta', {})
        print('%s/%s: params=%s meta=%s -> load OK' % (
            run, name, f'{sum(x.numel() for x in m.parameters()):,}',
            {k: meta[k] for k in list(meta)[:6]}))

