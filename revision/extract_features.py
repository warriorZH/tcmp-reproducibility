"""Frozen feature extraction for the revision (BioCLIP 2, DINOv2, CLIP).

Uses the released preprocessing (EXIF transpose, RGB, bicubic short side 224,
224 centre crop, encoder mean/SD, float32, L2 normalisation). Image order is the
manifest 'ids' order, so outputs plug into the locked contexts. The 'tcmp_dev'
dataset enumerates the 60 TCMP development classes (folders absent from the
evaluation manifest) for the development-class sensitivity grid.
"""
import argparse, json, time, hashlib, os
from pathlib import Path
import numpy as np
import rcommon as rc

MEANS = {'clip': ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
         'dinov2': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}
EXCLUDED = []
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG'}


def dev_records(root):
    d = rc.read(rc.REPO / 'data/manifests/tcmp.json.gz'); used = set(d['classes']); recs = []
    for split, sub in [('train', 'train'), ('query', 'val')]:
        base = Path(root) / sub
        for cls in sorted(p.name for p in base.iterdir() if p.is_dir()):
            if cls in used:
                continue
            for f in sorted((base / cls).iterdir()):
                if f.suffix in IMG_EXT:
                    rel = f'{sub}/{cls}/{f.name}'
                    recs.append(dict(id=hashlib.sha1(rel.encode()).hexdigest()[:20], path=rel, split=split, cls=cls))
    # development images were never audited: drop unreadable / truncated files and record them
    from PIL import Image
    good, bad = [], []
    for r in recs:
        try:
            with Image.open(Path(root) / r['path']) as im:
                im.load()
            good.append(r)
        except Exception as e:
            bad.append(dict(path=r['path'], error=str(e)[:200]))
    print('dev images readable', len(good), 'excluded', len(bad), flush=True)
    EXCLUDED.extend(bad)
    return good


def records(dataset, root):
    if dataset == 'tcmp_dev':
        return dev_records(root)
    d = rc.read(rc.REPO / f'data/manifests/{dataset}.json.gz'); idx = {r['id']: r for r in d['train'] + d['query']}
    return [dict(id=i, path=idx[i]['path']) for i in d['ids']]


def build(encoder, checkpoint, device):
    import torch
    if encoder == 'bioclip2':
        import open_clip
        open_clip.add_model_config(rc.REPO / 'configs/bio2.json')
        m = open_clip.create_model('bio2', pretrained=str(checkpoint), precision='fp32', device='cpu')
        pp = rc.read(rc.REPO / 'configs/open_clip_config.json')['preprocess_cfg']
        vis = m.visual.eval().requires_grad_(False).to(device); del m
        return (lambda x: vis(x)), (pp['mean'], pp['std']), 'imageomics/bioclip-2 visual (frozen)'
    if encoder in ('clip', 'clip-quickgelu'):
        # 'clip' is the configuration used for the published results. open_clip warns that the
        # OpenAI weights were trained with QuickGELU while 'ViT-L-14' uses GELU; 'clip-quickgelu'
        # uses the matching activation (see README, Known limitations).
        import open_clip
        arch = 'ViT-L-14-quickgelu' if encoder == 'clip-quickgelu' else 'ViT-L-14'
        try:
            m, _, _ = open_clip.create_model_and_transforms(arch, pretrained='openai', precision='fp32')
            src = f'open_clip {arch} openai'
        except Exception as e:  # mirror fallback
            print('CLIP openai download failed, trying HF hub:', e)
            m, _, _ = open_clip.create_model_and_transforms('hf-hub:timm/vit_large_patch14_clip_224.openai', precision='fp32')
            src = 'hf-hub:timm/vit_large_patch14_clip_224.openai'
        vis = m.visual.eval().requires_grad_(False).to(device); del m
        return (lambda x: vis(x)), MEANS['clip'], src
    if encoder == 'dinov2':
        try:
            from transformers import AutoModel
            m = AutoModel.from_pretrained('facebook/dinov2-large').eval().requires_grad_(False).to(device)
            return (lambda x: m(pixel_values=x).pooler_output), MEANS['dinov2'], 'HF facebook/dinov2-large CLS (pooler_output)'
        except Exception as e:
            print('HF DINOv2 failed, trying torch.hub:', e)
            m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').eval().requires_grad_(False).to(device)
            return (lambda x: m(x)), MEANS['dinov2'], 'torch.hub dinov2_vitl14 CLS'
    raise ValueError(encoder)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--encoder', required=True, choices=['bioclip2', 'dinov2', 'clip', 'clip-quickgelu'])
    p.add_argument('--dataset', required=True, choices=['tcmp', 'indian', 'tcmp_dev']); p.add_argument('--data-root', required=True)
    p.add_argument('--checkpoint'); p.add_argument('--output', required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=32); p.add_argument('--workers', type=int, default=8); a = p.parse_args()
    import torch
    from PIL import Image, ImageOps
    from torchvision import transforms
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    if (out / 'features_metadata.json').exists():
        print('features exist:', out); return
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    recs = records(a.dataset, a.data_root); fwd, (mean, std), src = build(a.encoder, a.checkpoint, a.device)
    tf = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224),
                             transforms.ToTensor(), transforms.Normalize(mean, std)])

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(recs)
        def __getitem__(self, i):
            with Image.open(Path(a.data_root) / recs[i]['path']) as im:
                return tf(ImageOps.exif_transpose(im).convert('RGB'))

    dl = torch.utils.data.DataLoader(DS(), batch_size=a.batch_size, num_workers=a.workers, shuffle=False)
    feats = []; t0 = time.time()
    if a.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i, x in enumerate(dl):
            z = fwd(x.to(a.device)).float().cpu().numpy(); z /= np.linalg.norm(z, axis=1, keepdims=True); feats.append(z)
            if i % 50 == 0:
                print(a.encoder, a.dataset, (i + 1) * a.batch_size, '/', len(recs), '%.0fs' % (time.time() - t0), flush=True)
    f = np.concatenate(feats).astype(np.float32); assert np.isfinite(f).all() and len(f) == len(recs)
    sec = time.time() - t0
    np.save(out / 'features.npy', f); rc.write(out / 'ids.json', [r['id'] for r in recs])
    if a.dataset == 'tcmp_dev':
        rc.write(out / 'records.json', recs); rc.write(out / 'excluded_images.json', EXCLUDED)
    meta = dict(excluded_unreadable=len(EXCLUDED), encoder=a.encoder, source=src, dataset=a.dataset, count=len(recs), dim=int(f.shape[1]), seconds=sec, images_per_second=len(recs) / sec,
                device=a.device, gpu=torch.cuda.get_device_name(0) if a.device.startswith('cuda') else 'cpu',
                peak_gpu_memory_gb=(torch.cuda.max_memory_allocated() / 1e9) if a.device.startswith('cuda') else None, batch_size=a.batch_size,
                features_sha256=rc.sha(out / 'features.npy'))
    rc.write(out / 'features_metadata.json', meta); print('done', meta)


if __name__ == '__main__':
    main()
