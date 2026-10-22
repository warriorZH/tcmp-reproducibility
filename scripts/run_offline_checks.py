"""Run every offline check in sequence (CPU only; no images, features or GPU).

    python scripts/run_offline_checks.py

Steps
  1  verify_release.py        manifests, 336 locked context indices, 350 + 476 result records, file checksums
  2  smoke_test.py            all 7 classifiers x 3 gates on synthetic features (functional test)
  3  analyze_formal.py        original TCMP closed/open summaries from 476 per-run records
  4  analyze_results.py       contamination, ablation, acceptance and pseudo-label analyses (Tables 1-2, Figs 4-6)
  5  compare_analysis.py      step 4 against the released reference statistics (894 values)
  6  verify_revision.py       released summaries of the additional experiments from their per-configuration files
  7  check_paper_numbers.py   manuscript values of Sections 3.3-3.7 (Tables 3-7)
  8  make_revision_tables.py  Additional file 1 Tables S16-S26, compared with the reference copy

A report is written to outputs/offline_check_report.txt. Exit status 0 means every step passed.
"""
import subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STEPS = [
    ['scripts/verify_release.py'], ['scripts/smoke_test.py'], ['scripts/analyze_formal.py'],
    ['scripts/analyze_results.py', '--input', 'results/reference', '--output', 'outputs/analysis'],
    ['scripts/compare_analysis.py'], ['scripts/verify_revision.py'], ['scripts/check_paper_numbers.py'],
    ['scripts/make_revision_tables.py', '--check'],
]


def main():
    (REPO / 'outputs').mkdir(exist_ok=True)
    report, failed = [], 0
    for i, step in enumerate(STEPS, 1):
        t0 = time.time()
        r = subprocess.run([sys.executable, *step], cwd=REPO, capture_output=True, text=True)
        ok = r.returncode == 0; failed += not ok
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        line = f"[{i}/{len(STEPS)}] {'PASS' if ok else 'FAIL'}  {' '.join(step)}  ({time.time() - t0:.0f} s)"
        print(line, flush=True); report += [line] + ['      ' + t for t in tail]
    report.append('ALL STEPS PASSED' if not failed else f'{failed} STEP(S) FAILED')
    print(report[-1])
    (REPO / 'outputs' / 'offline_check_report.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
