"""Run the original 240-class TCMP closed/full-known-pool protocol."""
import argparse,os,shutil
from pathlib import Path
from . import common,learner,original_experiment
from .io import read,write,sha
REPO=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,required=True);p.add_argument('--ids',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 d=read(REPO/'data/manifests/tcmp.json.gz');ids=read(a.ids);assert set(ids)==set(d['ids']) and len(ids)==len(set(ids))
 o=a.output;o.mkdir(parents=True,exist_ok=True)
 for name in ['cache','artifacts','results']:(o/name).mkdir(exist_ok=True)
 stamp=dict(features_sha256=sha(a.features),ids_sha256=sha(a.ids),config_sha256=sha(REPO/'configs/original_formal.json'))
 if (o/'run_inputs.json').exists():assert read(o/'run_inputs.json')==stamp
 write(o/'run_inputs.json',stamp);target=o/'cache/features.npy'
 if not target.exists():
  try:os.link(a.features.resolve(),target)
  except OSError:shutil.copy2(a.features,target)
 else:assert sha(target)==stamp['features_sha256']
 import numpy as np
 f=np.load(target,mmap_mode='r');assert f.shape==(len(ids),768) and np.isfinite(f).all() and np.allclose(np.linalg.norm(f,axis=1),1,atol=1e-5)
 cfg=read(REPO/'configs/original_formal.json');inputs={'protocol.json':cfg,'private_data.json':d,'cache/ids.json':ids}
 common.R=learner.R=original_experiment.R=o;original_experiment.read=lambda name:inputs[name]
 original_experiment.main()
 assert len(list((o/'results').glob('*.json')))==476
 print('Original TCMP protocol complete: 476 records')
if __name__=='__main__':main()
