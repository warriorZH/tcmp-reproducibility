"""W3 / Q3 (part 2): end-to-end fine-tuning on the clean closed 4-shot support set.

Backbones: BioCLIP 2 ViT-L/14 visual tower (same checkpoint) and ImageNet ResNet-50.
Recipe fixed in advance (no query-based tuning, final epoch reported): AdamW,
30 epochs, 2 warm-up epochs + cosine decay, backbone LR 1e-5 (ViT) / 1e-4 (ResNet),
head LR 1e-3, weight decay 0.05, label smoothing 0.1, batch 16, RandomResizedCrop
(0.5-1) + horizontal flip; head initialised from normalised support class means
(prototype init); float32 (P40 has no fast FP16). Wall time and peak GPU memory
are recorded as the computational cost.
"""
import argparse, json, time, math
from pathlib import Path
import numpy as np
import rcommon as rc


def backbone(name, checkpoint, device):
    import torch, torchvision
    if name == 'bioclip2_vitl14':
        import open_clip
        open_clip.add_model_config(rc.REPO / 'configs/bio2.json')
        m = open_clip.create_model('bio2', pretrained=str(checkpoint), precision='fp32', device='cpu'); vis = m.visual; del m
        try:
            vis.set_grad_checkpointing(True)
        except Exception:
            pass
        pp = rc.read(rc.REPO / 'configs/open_clip_config.json')['preprocess_cfg']
        return vis.to(device), 768, (pp['mean'], pp['std']), 1e-5
    if name == 'resnet50_imagenet':
        try:
            net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception as e:
            print('torchvision weights download failed, trying timm/HF:', e)
            import timm
            net = timm.create_model('resnet50.tv2_in1k', pretrained=True)
        net.fc = torch.nn.Identity()
        return net.to(device), 2048, ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 1e-4
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset', required=True); p.add_argument('--data-root', required=True)
    p.add_argument('--checkpoint', required=True); p.add_argument('--backbone', required=True); p.add_argument('--seeds', default='')
    p.add_argument('--output', required=True); p.add_argument('--device', default='cuda:0'); p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=16); p.add_argument('--workers', type=int, default=8); a = p.parse_args()
    import torch, torch.nn.functional as F
    from PIL import Image, ImageOps
    from torchvision import transforms
    d = rc.read(rc.REPO / f'data/manifests/{a.dataset}.json.gz'); classes = d['classes']; ci = {c: i for i, c in enumerate(classes)}
    cname = lambda r: r.get('class', r.get('class_name')); tr = {r['id']: r for r in d['train']}; qrec = d['query']
    seeds = [int(s) for s in a.seeds.split(',')] if a.seeds else rc.SEEDS[a.dataset][:3]
    res_dir = Path(a.output) / 'w3'; res_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        target = res_dir / f'finetune_{a.backbone}_{a.dataset}_s{seed}.json'
        if target.exists():
            continue
        torch.manual_seed(seed); np.random.seed(seed)
        plan = next(pp for pp in d['plans'] if pp['seed'] == seed and pp['shot'] == 4)
        sup = [tr[i] for i in plan['support']]
        net, dim, (mean, std), lr_bb = backbone(a.backbone, a.checkpoint, a.device)
        norm = transforms.Normalize(mean, std)
        tf_train = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                                       transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
        tf_eval = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(), norm])

        class DS(torch.utils.data.Dataset):
            def __init__(s, recs, tf): s.r, s.tf = recs, tf
            def __len__(s): return len(s.r)
            def __getitem__(s, i):
                r = s.r[i]
                with Image.open(Path(a.data_root) / r['path']) as im:
                    return s.tf(ImageOps.exif_transpose(im).convert('RGB')), ci[cname(r)]

        def embed(recs):
            net.eval(); out, ys = [], []
            with torch.inference_mode():
                for x, y in torch.utils.data.DataLoader(DS(recs, tf_eval), batch_size=64, num_workers=a.workers):
                    out.append(net(x.to(a.device)).float()); ys.append(y)
            return torch.cat(out), torch.cat(ys).to(a.device)

        cuda = a.device.startswith('cuda')
        if cuda: torch.cuda.reset_peak_memory_stats()
        t_start = time.time()
        zs, ys = embed(sup); zs = F.normalize(zs, dim=-1)
        head = torch.nn.Linear(dim, len(classes)).to(a.device)
        with torch.no_grad():
            proto = torch.stack([zs[ys == c].mean(0) for c in range(len(classes))]); head.weight.copy_(F.normalize(proto, dim=-1) * 10.0); head.bias.zero_()
        params = [dict(params=net.parameters(), lr=lr_bb), dict(params=head.parameters(), lr=1e-3)]
        opt = torch.optim.AdamW(params, weight_decay=0.05)
        dl = torch.utils.data.DataLoader(DS(sup, tf_train), batch_size=a.batch_size, shuffle=True, num_workers=a.workers, drop_last=False)
        steps = a.epochs * len(dl); warm = 2 * len(dl); base = [g['lr'] for g in opt.param_groups]; step = 0; log = []
        for ep in range(a.epochs):
            net.train(); tot = 0.
            for x, y in dl:
                f = 0.5 * (1 + math.cos(math.pi * max(0, step - warm) / max(1, steps - warm))) if step >= warm else (step + 1) / warm
                for g, b0 in zip(opt.param_groups, base): g['lr'] = b0 * f
                x, y = x.to(a.device), y.to(a.device); z = F.normalize(net(x).float(), dim=-1)
                loss = F.cross_entropy(head(z), y, label_smoothing=0.1); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += float(loss); step += 1
            log.append(dict(epoch=ep, loss=tot / len(dl))); print(a.backbone, a.dataset, seed, 'epoch', ep, 'loss %.4f' % (tot / len(dl)), '%.0fs' % (time.time() - t_start), flush=True)
        train_seconds = time.time() - t_start; t_eval = time.time()
        zq, yq = embed(qrec); pred = head(F.normalize(zq, dim=-1)).argmax(1)
        acc = float(100 * (pred == yq).float().mean()); per = [float((pred[yq == c] == c).float().mean()) for c in range(len(classes))]
        rec = dict(dataset=a.dataset, backbone=a.backbone, seed=seed, shot=4, classes=len(classes), n_support=len(sup), n_query=len(qrec),
                   accuracy=acc, macro_recall=100 * float(np.mean(per)), worst_class_recall=100 * float(np.min(per)), train_seconds=train_seconds,
                   eval_seconds=time.time() - t_eval, peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / 1e9 if cuda else None, gpu=torch.cuda.get_device_name(0) if cuda else 'cpu',
                   epochs=a.epochs, batch_size=a.batch_size, backbone_lr=lr_bb, head_lr=1e-3, trainable_parameters=int(sum(p.numel() for p in net.parameters()) + sum(p.numel() for p in head.parameters())),
                   loss_log=log, recipe='fixed in advance; no query-based tuning; final epoch reported')
        rc.write(target, rec); print('FINETUNE', a.backbone, a.dataset, seed, 'acc %.2f' % acc, 'train %.0fs' % train_seconds, flush=True)
        del net, head, opt
        if cuda: torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
