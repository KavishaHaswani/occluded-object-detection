import json
from pathlib import Path

base = Path(__file__).resolve().parents[1] / 'kaggle_notebook' / 'reports'
for run in ['runs_amp', 'runs_raw']:
    r = json.loads((base / run / 'test_report.json').read_text(encoding='utf-8'))
    print('=' * 20, run, '=' * 20)
    print('keys:', list(r.keys()))
    print('overall:', json.dumps(r.get('overall'), indent=2))
    bysrc = r.get('by_source') or {}
    for k, v in sorted(bysrc.items()):
        print('  %-8s IoU=%.4f MAE=%.4f Dice=%.4f S=%.4f Fw=%.4f n=%d' % (
            k, v['IoU'], v['MAE'], v['Dice'], v['S_measure'], v['weighted_F'], v['n_frames']))
    print('selection:', r.get('selection'))
    byseq = r.get('by_sequence') or {}
    print('n sequences:', len(byseq))
