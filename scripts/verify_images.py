"""Check downloaded images against the released byte hashes before feature extraction."""
from pathlib import Path
import argparse,gzip,json,hashlib
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['tcmp','indian'],required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('outputs/image_audit.json'));a=p.parse_args()
with gzip.open(R/f'data/manifests/{a.dataset}.json.gz','rt',encoding='utf-8') as f:d=json.load(f)
missing=[];changed=[];n=0
for r in d['train']+d['query']:
 f=a.data_root/Path(r['path'])
 if not f.is_file():missing.append(r['path']);continue
 h=hashlib.sha256(f.read_bytes()).hexdigest()
 if h!=r['sha256']:changed.append(dict(path=r['path'],expected=r['sha256'],actual=h))
 else:n+=1
report=dict(dataset=a.dataset,matched=n,total=len(d['train'])+len(d['query']),missing=missing,changed=changed,passed=not(missing or changed));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print('Matched',n,'Missing',len(missing),'Changed',len(changed));raise SystemExit(0 if report['passed'] else 1)
