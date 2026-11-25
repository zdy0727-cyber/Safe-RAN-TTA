# -*- coding: utf-8 -*-
"""开放集新异分数对比：能量 / MSP / 原型余弦 / Mahalanobis / 残差PCA / kNN / 双分数
用 Xinjiang-only 训练的 MTD-Net，在清华 test 上把 Water（class 2）当 unknown。
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/root/rsc_local/code')
torch.set_num_threads(2)


class DS(Dataset):
    def __init__(self, df, root, label_col):
        self.df = df.reset_index(drop=True); self.root = root; self.label_col = label_col
        self.tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                             T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(self.root, r['path'])).convert('RGB')
        return self.tf(img), int(r[self.label_col])


def load_mt(ckpt, num_classes, device):
    from train_mt import MTDNet
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    st = ck['model']; hm = ck.get('head_mode', 'full')
    n_dom = 2
    for k, v in st.items():
        if k.startswith('dom_head') and k.endswith('weight') and v.dim() == 2:
            n_dom = int(v.shape[0])
    he = any(k.startswith('cls_linear') for k in st)
    m = MTDNet(ck.get('backbone', 'resnet50'), num_classes, hm, 256, ck.get('tasks') or [],
               n_domains=n_dom, pretrained=False, head_extra=he)
    m.load_state_dict(st)
    return m.to(device).eval(), ck


@torch.no_grad()
def extract(model, df, root, label_col, device, bs=128, workers=4):
    loader = DataLoader(DS(df, root, label_col), batch_size=bs, num_workers=workers)
    Z, F_, L, R = [], [], [], []
    ys, preds = [], []
    has_content = hasattr(model, 'content')
    for imgs, y in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        feat = out['feat'].float()
        if has_content:
            z = F.normalize(model.content(feat), dim=1)
        else:
            z = F.normalize(feat, dim=1)
        Z.append(z.cpu()); F_.append(feat.cpu()); L.append(out['cls'].float().cpu())
        R.append(out['r'].float().cpu() if out.get('r') is not None else torch.zeros(len(y), 1))
        preds.extend(out['cls'].float().argmax(1).cpu().tolist()); ys.extend(y.tolist())
    return torch.cat(Z), torch.cat(F_), torch.cat(L), torch.cat(R), np.array(ys), np.array(preds)


def zs(a):
    a = np.asarray(a, dtype=float)
    return (a - a.mean()) / (a.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/private/zdy/deepseek_road')
    ap.add_argument('--ckpt', default='/private/zdy/deepseek_road/runs/mt_xj_only/best.pt')
    ap.add_argument('--num-classes', type=int, default=6)
    ap.add_argument('--unknown-class', type=int, default=2)
    ap.add_argument('--out', default='/private/zdy/deepseek_road/runs/openset_scores.json')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--bs', type=int, default=128)
    args = ap.parse_args()
    device = torch.device(args.device)
    model, ck = load_mt(args.ckpt, args.num_classes, device)

    df = pd.read_csv(os.path.join(args.root, 'manifests/multitask_all.csv'), low_memory=False)
    df = df[df['track'] == 'track2']
    src = df[(df['split'] == 'val') & (df['dataset'] == 'xinjiang')].reset_index(drop=True)
    tgt = df[(df['split'] == 'test') & (df['dataset'] == 'tsinghua')].reset_index(drop=True)
    print('源域(新疆 val) %d, 目标域(清华 test) %d' % (len(src), len(tgt)))

    zs_, fs_, ls_, rs_, ys_, ps_ = extract(model, src, args.root, 'label6', device)
    zn = zs_.numpy()
    means = []
    for c in range(args.num_classes):
        m = ys_ == c
        means.append(zn[m].mean(0) if m.sum() >= 5 else None)
    # 池化协方差（收缩估计）
    covs = []
    for c in range(args.num_classes):
        m = ys_ == c
        if m.sum() >= 5:
            covs.append(np.cov(zn[m], rowvar=False))
    cov_pool = np.mean(covs, axis=0) if covs else np.eye(zn.shape[1])
    cov_pool = cov_pool + np.eye(zn.shape[1]) * 1e-3
    icov = np.linalg.pinv(cov_pool)

    zt, ft, lt, rt, yt, pt = extract(model, tgt, args.root, 'label6', device)
    ztn = zt.numpy()
    P = model.prototypes.detach().float().cpu()
    Pn = F.normalize(P, dim=1).numpy()
    cos = ztn @ Pn.T
    maxcos = cos.max(1)
    proto_nov = 1.0 - maxcos

    d2 = np.full(len(ztn), np.inf)
    for mu in means:
        if mu is None:
            continue
        diff = ztn - mu
        dist = np.einsum('ij,jk,ik->i', diff, icov, diff)
        d2 = np.minimum(d2, dist)
    maha_nov = d2

    logits = lt.numpy()
    mx = logits.max(1, keepdims=True)
    energy = -np.log(np.exp(logits - mx).sum(1)) - mx[:, 0]
    msp = F.softmax(lt, 1).max(1).values.numpy()

    R = rs_.numpy()
    Rc = R - R.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
    basis = Vt[:8].T
    proj = R @ basis @ basis.T
    resid_nov = np.linalg.norm(R - proj, axis=1) / (np.linalg.norm(R, axis=1) + 1e-9)

    src_bank = zn
    sim = ztn @ src_bank.T
    knn_nov = 1.0 - np.sort(sim, axis=1)[:, -5:].mean(1)

    is_u = (yt == args.unknown_class).astype(int)
    print('unknown 比例: %.3f' % is_u.mean())
    scores = {
        'energy': energy, 'msp': msp, 'proto_nov': proto_nov, 'maha_nov': maha_nov,
        'resid_nov': resid_nov, 'knn_nov': knn_nov,
        'dual_energy_proto': zs(energy) + zs(proto_nov),
        'dual_energy_maha': zs(energy) + zs(maha_nov),
        'dual_proto_maha': zs(proto_nov) + zs(maha_nov),
    }
    res = {}
    print('\n%-20s %s' % ('分数', 'AUROC'))
    for k, v in scores.items():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        a = None
        if len(np.unique(is_u)) > 1:
            try:
                a = roc_auc_score(is_u, v)
            except Exception:
                a = None
        res[k] = a
        print('%-20s %s' % (k, ('%.4f' % a) if a is not None else 'None'))
    known = is_u == 0
    acc_known = float((pt[known] == yt[known]).mean())
    print('\n已知类准确率 (排除 Water): %.4f' % acc_known)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(dict(auroc=res, known_acc=acc_known, n=len(yt),
                       unknown_rate=float(is_u.mean())), f, ensure_ascii=False, indent=1)
    print('保存:', args.out)


if __name__ == '__main__':
    main()
