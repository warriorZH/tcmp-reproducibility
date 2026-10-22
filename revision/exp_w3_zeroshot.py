"""W3 / Q3 (part 1): BioCLIP 2 zero-shot classification with class names as text prompts.

Image embeddings are the released frozen BioCLIP 2 features (identical to
open_clip encode_image + L2 norm); only the text tower is added. No labels are
used except support images for the acceptance threshold (10% quantile of their
zero-shot MSP, mirroring the released support-calibrated gates).
Prompts: TCMP uses Latin binomials (folder names minus index/authority), with and
without authority; Indian uses the folder common names.
"""
import argparse, re, time
from pathlib import Path
import numpy as np
from scipy.special import softmax
import rcommon as rc


def names(dataset, classes):
    out = {}
    if dataset == 'tcmp':
        full = [re.sub(r'^\d+\.', '', c).strip() for c in classes]
        binom = [' '.join(f.split()[:2]) for f in full]
        out['binomial'] = [[f'a photo of {b}.'] for b in binom]
        out['binomial_authority'] = [[f'a photo of {f}.'] for f in full]
        out['ensemble'] = [[f'a photo of {b}.', f'a photo of {f}.', f'a photo of {b}, a medicinal plant.', f'{b}'] for b, f in zip(binom, full)]
    else:
        common = [c.replace('_', ' ').replace('-', ' ').strip() for c in classes]
        out['common_name'] = [[f'a photo of {c}.'] for c in common]
        out['common_name_leaf'] = [[f'a photo of a {c} leaf.'] for c in common]
        out['ensemble'] = [[f'a photo of {c}.', f'a photo of a {c} leaf.', f'a photo of {c}, a medicinal plant.', f'{c}'] for c in common]
    return out


def text_embed(checkpoint, prompts, device):
    import torch, open_clip
    open_clip.add_model_config(rc.REPO / 'configs/bio2.json')
    m = open_clip.create_model('bio2', pretrained=str(checkpoint), precision='fp32', device=device).eval()
    tok = open_clip.get_tokenizer('bio2') if hasattr(open_clip, 'get_tokenizer') else open_clip.tokenize
    W = []
    with torch.inference_mode():
        for ps in prompts:
            z = m.encode_text(tok(ps).to(device)).float(); z = z / z.norm(dim=-1, keepdim=True); z = z.mean(0); W.append((z / z.norm()).cpu().numpy())
    return np.stack(W), float(m.logit_scale.exp().item())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset', required=True); p.add_argument('--features', required=True); p.add_argument('--ids', required=True)
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True); p.add_argument('--device', default='cuda:0'); a = p.parse_args()
    out = rc.set_out(Path(a.output) / 'work' / a.dataset); d, tx, qx = rc.load(a.dataset, a.features, a.ids)
    classes = d['classes']; ci = {c: i for i, c in enumerate(classes)}; cname = lambda r: r.get('class', r.get('class_name'))
    ty = np.array([ci[cname(r)] for r in d['train']]); qt = np.array([ci[cname(r)] for r in d['query']]); ti = {r['id']: i for i, r in enumerate(d['train'])}
    res = {}; t0 = time.time()
    for variant, prompts in names(a.dataset, classes).items():
        tt = time.time(); W, scale = text_embed(a.checkpoint, prompts, a.device); text_seconds = time.time() - tt
        logits_q = scale * qx @ W.T; logits_t = scale * tx @ W.T
        closed = float(100 * np.mean(logits_q.argmax(1) == qt))
        per = [float(np.mean(logits_q[qt == c].argmax(1) == c)) for c in range(len(classes))]
        rows = []
        for role in d['roles']:
            known = np.array(role['known']); mapping = {v: j for j, v in enumerate(known)}; t = np.array([mapping.get(v, -1) for v in qt])
            groups = {'pool_seen': np.isin(qt, role['pool_unknown']), 'query_only': np.isin(qt, role['query_unknown'])}
            pq = softmax(logits_q[:, known], axis=1); pred = pq.argmax(1); s = pq.max(1)
            for plan in [pp for pp in d['plans'] if pp['shot'] == 4]:
                sup = np.array([ti[v] for v in plan['support'] if ty[ti[v]] in mapping])
                cut = float(np.quantile(softmax(logits_t[sup][:, known], axis=1).max(1), .1))
                m0 = rc.metrics(None, pred, t, None, groups); m1 = rc.metrics(s, pred, t, cut, groups)
                rows.append(dict(role=role['seed'], seed=plan['seed'], accuracy=m0['accuracy'], macro_recall=m0['macro_recall'],
                                 gate={k: m1[k] for k in ['known_coverage', 'unknown_acceptance', 'accepted_known_risk', 'overall_accepted_error', 'oscr']}))
        res[variant] = dict(closed_accuracy=closed, closed_macro_recall=100 * float(np.mean(per)), logit_scale=scale, text_seconds=text_seconds,
                            open_known_accuracy=float(np.mean([r['accuracy'] for r in rows])), open_macro_recall=float(np.mean([r['macro_recall'] for r in rows])),
                            open_gate={k: float(np.mean([r['gate'][k] for r in rows if r['gate'][k] is not None])) for k in rows[0]['gate']}, per_context=rows)
        rc.log('W3 zero-shot', a.dataset, variant, 'closed %.2f open-known %.2f' % (closed, res[variant]['open_known_accuracy']))
    res['seconds'] = time.time() - t0; res['note'] = 'Open known accuracy is independent of pool contamination (no pool use); gate threshold from support images.'
    Path(a.output, 'w3').mkdir(parents=True, exist_ok=True); rc.write(Path(a.output) / 'w3' / f'zeroshot_{a.dataset}.json', res)


if __name__ == '__main__':
    main()
