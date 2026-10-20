"""Download only the locked checkpoint; optionally select a Hugging Face mirror."""
from pathlib import Path
import argparse,json,os,hashlib
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='checkpoints/bioclip-2');p.add_argument('--endpoint');a=p.parse_args()
if a.endpoint:os.environ['HF_ENDPOINT']=a.endpoint
from huggingface_hub import snapshot_download
m=json.loads((R/'provenance/model.json').read_text());folder=snapshot_download(m['repository'],revision=m['revision'],local_dir=a.output,allow_patterns=[m['weights_file'],'open_clip_config.json','README.md','LICENSE*'])
h=hashlib.sha256()
with open(Path(folder)/m['weights_file'],'rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
assert h.hexdigest()==m['sha256'],'Downloaded checkpoint SHA-256 differs';print('Verified checkpoint:',folder)
