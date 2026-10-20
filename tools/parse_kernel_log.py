import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
log_path = ROOT / 'kaggle_notebook' / 'out_tmp' / 'v12.log'
raw = log_path.read_text(encoding='utf-8-sig').strip()
try:
    arr = json.loads(raw)
    print('entries:', len(arr))
except Exception as e:
    print('json parse failed:', e)
    arr = []

txt = ''.join(e.get('data', '') for e in arr if e.get('stream_name') == 'stdout')
out = log_path.with_name('v12.stdout.txt')
out.write_text(txt, encoding='utf-8')
print('total stdout chars:', len(txt))
for pat in ['Test overall', 'Final test', 'Best by val', 'split files OK',
            'CPU workers', 'found', 'RUN A DONE', 'RUN B DONE']:
    print(f'  {pat!r} -> {txt.count(pat)}')
