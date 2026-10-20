"""Download training artifacts from the kernel output by paging the
alphabetical listing with the maximum page size, with 429 backoff.

Artifacts live under runs_amp/ and runs_raw/, which sort after the huge
cache_amp/ and cache_raw/ sections, so this walks the whole listing once.
"""
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for line in (ROOT / '.env').read_text(encoding='utf-8').splitlines():
    if '=' in line and not line.strip().startswith('#'):
        k, v = line.split('=', 1)
        os.environ[k.strip()] = v.strip()

# The new SDK authenticates with KAGGLE_API_TOKEN; mirror the CLI preamble.
if not os.environ.get('KAGGLE_API_TOKEN') and os.environ.get('KAGGLE_KEY'):
    os.environ['KAGGLE_API_TOKEN'] = os.environ['KAGGLE_KEY']

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

KERNEL = 'kavishahaswani/occluded-object-detection'
OUTDIR = ROOT / 'kaggle_notebook' / 'reports'
PATTERN = r'runs_(amp|raw)/(best_by_val|last_model)\.pth|runs_(amp|raw)/(test_report|history)\.json|split_summary\.json|SPLIT_README\.md'
WANTED = {
    'runs_amp/best_by_val.pth',
    'runs_raw/best_by_val.pth',
    'runs_amp/last_model.pth',
    'runs_raw/last_model.pth',
    'runs_amp/test_report.json',
    'runs_raw/test_report.json',
    'runs_amp/history.json',
    'runs_raw/history.json',
    'split_summary.json',
    'SPLIT_README.md',
}
PAGE = 200
SLEEP = 3.0


def main(start_path=None, max_pages=400):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    token = None
    found = set()
    for page in range(max_pages):
        for attempt in range(6):
            try:
                outfiles, token = api.kernels_output(
                    KERNEL, str(OUTDIR), file_pattern=PATTERN, force=True,
                    quiet=True, page_token=token, page_size=PAGE,
                )
                break
            except Exception as e:
                if '429' in str(e):
                    wait = min(120, 15 * (attempt + 1))
                    print(f'  429 on page {page}; backing off {wait}s', flush=True)
                    time.sleep(wait)
                else:
                    raise
        else:
            print('giving up: persistent 429', flush=True)
            return

        for f in outfiles:
            found.add(str(Path(f).relative_to(OUTDIR)).replace('\\', '/'))
        # outfiles keep original relative paths
        missing = sorted(WANTED - found)
        print(f'page {page}: matched={len(outfiles)} have={len(found)} missing={missing[:4]}', flush=True)
        if not token or not missing:
            break
        time.sleep(SLEEP)

    print('\nDONE. files on disk:')
    for f in sorted(OUTDIR.rglob('*')):
        if f.is_file():
            print(f'  {f.relative_to(OUTDIR)}  {f.stat().st_size / 1024:.1f} KB')
    print('still missing:', sorted(WANTED - found))


if __name__ == '__main__':
    main()
