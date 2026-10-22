"""Bundle all small result files (JSON summaries, per-context JSON, logs) into one zip."""
import argparse, zipfile, time
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('--output', default='revision_outputs'); a = p.parse_args()
o = Path(a.output); name = o / f"revision_results_{time.strftime('%Y%m%d_%H%M')}.zip"; n = 0
with zipfile.ZipFile(name, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(o.rglob('*')):
        if f.is_file() and '/work/' not in str(f) and f.suffix in {'.json', '.log', '.txt'} and f != name:
            z.write(f, f.relative_to(o)); n += 1
print('wrote', name, n, 'files')
