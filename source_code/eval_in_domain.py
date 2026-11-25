# -*- coding: utf-8 -*-
"""在域内测试集（roadsaw+roadsc）上公平评测：macro-F1 只统计出现过的类别
支持两种 checkpoint：MTDNet（train_mt）与普通分类模型（train_cls）
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from sklearn.metrics import f1_score, confusion_matrix

sys.path.insert(0, '/root/rsc_local/code')
torch.set_num_threads(2)


class DS(Dataset):
    def __init__(self, df, root, transform, label_col, crop=False):
        self.df = df.reset_index(drop=True); self.root = root
        self.transform = transform; self.label_col = label_col; self.crop = crop

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(self.root, r['path'])).convert('RGB')
        if self.crop and np.isfinite(r['x']):
            x, y, w, h = int(r['x']), int(r['y']), int(r['w']), int(r['h'])
            img = img.crop((x, y, x + w, y + h))
        return self.transform(img), int(r[self.label_col])


def tf(img_size):
    return T.Compose([T.Resize(img_size + 32, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(img_size),
                      T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


@torch.no_grad()
def predict(model, loader, device):
    ys, ps = [], []
    for imgs, y in loader:
        logits = model(imgs.to(device))
        if isinstance(logits, dict):
            logits = logits['cls']
        ps.extend(logits.float().argmax(1).cpu().numpy().tolist()); ys.extend(y.tolist())
    return np.array(ys), np.array(ps)


def metrics(y, p, num_classes):
    present = sorted(set(y.tolist()) | set(p.tolist()))
    cm = confusion_matrix(y, p, labels=list(range(num_classes)))
    recalls = [float(cm[c, c] / cm[c].sum()) if cm[c].sum() else None for c in range(num_classes)]
    return dict(n=int(len(y)), acc=float((y == p).mean()),
                macro_f1_present=float(f1_score(y, p, average='macro', labels=present, zero_division=0)),
                macro_f1_all15=float(f1_score(y, p, average='macro', labels=list(range(num_classes)), zero_division=0)),
                per_class_recall=recalls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/private/zdy/deepseek_road')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--kind', default='mt', choices=['mt', 'plain'])
    ap.add_argument('--manifest', default='manifests/multitask_all.csv')
    ap.add_argument('--label-col', default='label15')
    ap.add_argument('--num-classes', type=int, default=15)
    ap.add_argument('--datasets', default='roadsaw,roadsc')
    ap.add_argument('--domains', default='')
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', default='')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    man = os.path.join(args.root, args.manifest)
    df = pd.read_csv(man, low_memory=False)
    if 'track' in df.columns:
        df = df[df['track'] == 'track1']
    df = df[df['split'] == args.split]
    ds_list = [x for x in args.datasets.split(',') if x]
    df = df[df['dataset'].isin(ds_list)]
    dom_list = [x for x in args.domains.split(',') if x]
    if dom_list and 'domain' in df.columns:
        df = df[df['domain'].isin(dom_list)]
    df = df[pd.to_numeric(df[args.label_col], errors='coerce').notna()]
    if 'x' not in df.columns:
        df['x'] = np.nan
    crop = df['x'].notna().any()
    print('评测样本 %d 条: %s' % (len(df), df['dataset'].value_counts().to_dict()))

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    backbone = ck.get('backbone', 'resnet50')
    if args.kind == 'mt':
        from train_mt import MTDNet
        st = ck['model']
        head_mode = ck['head_mode'] if 'head_mode' in ck else ck.get('args', {}).get('head_mode', 'full')
        n_dom = 2
        for k, v in st.items():
            if k.startswith('dom_head') and k.endswith('weight') and v.dim() == 2:
                n_dom = int(v.shape[0])
        head_extra = any(k.startswith('cls_linear') for k in st)
        model = MTDNet(backbone, args.num_classes, head_mode, 256, ck.get('tasks') or [],
                       n_domains=n_dom, pretrained=False, head_extra=head_extra)
        model.load_state_dict(st)
    else:
        model = timm.create_model(backbone, pretrained=False, num_classes=args.num_classes)
        model.load_state_dict(ck['model'])
    device = torch.device(args.device)
    model.to(device).eval()

    ds = DS(df, args.root, tf(224), args.label_col, crop=crop)
    loader = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, shuffle=False)
    y, p = predict(model, loader, device)
    res = metrics(y, p, args.num_classes)
    print('结果: n=%d acc=%.4f macroF1(出现类)=%.4f macroF1(15类)=%.4f' % (
        res['n'], res['acc'], res['macro_f1_present'], res['macro_f1_all15']))
    print('各类召回:', [round(x, 3) if x is not None else None for x in res['per_class_recall']])
    # 分数据集
    for d in ds_list:
        m = df['dataset'].values == d
        if m.sum() == 0:
            continue
        r2 = metrics(y[m], p[m], args.num_classes)
        print('  %-8s n=%5d acc=%.4f F1(出现类)=%.4f' % (d, r2['n'], r2['acc'], r2['macro_f1_present']))
        res['by_' + d] = r2
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
