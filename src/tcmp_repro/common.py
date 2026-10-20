import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parent
def write(name,value):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def read(name):return json.loads((R/name).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while x:=f.read(8388608):h.update(x)
 return h.hexdigest()
