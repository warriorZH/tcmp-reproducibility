import json,os,hashlib,gzip
from pathlib import Path

def read(path):
 p=Path(path)
 if p.suffix=='.gz':
  with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
 return json.loads(p.read_text(encoding='utf-8'))

def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(tmp,p)

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
