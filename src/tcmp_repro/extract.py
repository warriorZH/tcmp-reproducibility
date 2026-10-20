"""Extract frozen BioCLIP 2 features using locked image manifests and checkpoint."""
from pathlib import Path
import argparse,json,time
from .io import read,write,sha
REPO=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['tcmp','indian'],required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int);a=p.parse_args()
 import numpy as np,torch,open_clip
 from PIL import Image,ImageOps
 from torchvision import transforms
 md=a.model_dir;meta=read(REPO/'provenance/model.json');weights=md/meta['weights_file'];assert sha(weights)==meta['sha256'],'Checkpoint SHA-256 mismatch'
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 open_clip.add_model_config(REPO/'configs/bio2.json');model=open_clip.create_model('bio2',pretrained=str(weights),precision='fp32',device='cpu');visual=model.visual.eval().requires_grad_(False).to(a.device);del model
 pp=read(REPO/'configs/open_clip_config.json')['preprocess_cfg'];tf=transforms.Compose([transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(pp['mean'],pp['std'])])
 d=read(REPO/f'data/manifests/{a.dataset}.json.gz');index={r['id']:r for r in d['train']+d['query']};ids=d['ids'];o=a.output;o.mkdir(parents=True,exist_ok=True);write(o/'ids.json',ids)
 stamp=dict(dataset=a.dataset,checkpoint_sha256=meta['sha256'],manifest_sha256=sha(REPO/f'data/manifests/{a.dataset}.json.gz'),device=a.device,batch_size=a.batch_size or (4 if a.dataset=='tcmp' else 2))
 cp=o/'checkpoint.json';start=0
 if cp.exists():
  saved=read(cp);assert saved['inputs']==stamp,'Feature inputs changed; use a new output directory';start=saved['completed']
 f=np.lib.format.open_memmap(o/'features.npy',mode='r+' if cp.exists() else 'w+',dtype=np.float32,shape=(len(ids),768));bs=stamp['batch_size'];beg=time.time()
 for first in range(start,len(ids),bs):
  batch=ids[first:first+bs];xs=[]
  for ident in batch:
   r=index[ident];path=a.data_root/Path(r['path']);assert sha(path)==r['sha256'],f'Image mismatch: {r["path"]}'
   with Image.open(path) as im:xs.append(tf(ImageOps.exif_transpose(im).convert('RGB')))
  x=torch.stack(xs).to(a.device)
  with torch.inference_mode():z=visual(x).cpu().numpy()
  z/=np.linalg.norm(z,axis=1,keepdims=True);assert np.isfinite(z).all();f[first:first+len(batch)]=z
  if first%128==0 or first+len(batch)==len(ids):
   f.flush();write(cp,dict(completed=first+len(batch),inputs=stamp));print(first+len(batch),'/',len(ids),flush=True)
 assert np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5);f.flush();write(o/'features_metadata.json',dict(**stamp,count=len(ids),sha256=sha(o/'features.npy'),seconds=time.time()-beg))
if __name__=='__main__':main()
