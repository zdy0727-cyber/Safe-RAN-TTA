# -*- coding: utf-8 -*-
"""
评测套件：
  - 源模型直接评测（source）
  - TTA 基线：BN-Adapt / TENT / CoTTA(简化) / SAR(简化)
  - 指标：Accuracy、macro-F1、各类召回、混淆矩阵、风险加权代价、ECE、开放集 AUROC

用法示例（CPU 冒烟）：
  CUDA_VISIBLE_DEVICES= env/bin/python evaluate_tta.py \
     --ckpt /private/.../runs/roadsaw12_r50_domain/best.pt --num-classes 12 --label-col label15 \
     --manifest /private/.../manifests/track1_ctd.csv --split test --domain ctd1 \
     --root /private/zdy/deepseek_road --max-samples 200 --methods source,bn,tent --device cpu
"""
import os, sys, json, time, argparse
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

def now():
    return datetime.now().strftime('%H:%M:%S')

def log(msg):
    print('[%s] %s' % (now(), msg), flush=True)

class CsvDataset(Dataset):
    def __init__(self, df, root, transform, label_col, crop=False):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.label_col = label_col
        self.crop = crop

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(self.root, row['path'])).convert('RGB')
        if self.crop:
            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
            img = img.crop((x, y, x + w, y + h))
        return self.transform(img), int(row[self.label_col])

def eval_transform(img_size):
    return T.Compose([
        T.Resize(img_size + 32, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def tent_transform(img_size):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def norm_params(model):
    params = []
    for name, p in model.named_parameters():
        n = name.lower()
        if ('bn' in n or 'norm' in n or 'running' in n) and p.requires_grad is not None:
            if p.dim() <= 4:
                params.append(p)
    return params

def set_params_grad(model, value=True):
    for p in model.parameters():
        p.requires_grad = value
    for p in norm_params(model):
        p.requires_grad = True

def softmax_np(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def compute_metrics(y, pred, prob, num_classes, risk_weights=None, unknown_classes=None):
    y = np.asarray(y); pred = np.asarray(pred); prob = np.asarray(prob)
    out = {}
    out['n'] = int(len(y))
    out['acc'] = float((y == pred).mean())
    out['macro_f1'] = float(f1_score(y, pred, average='macro', labels=list(range(num_classes)), zero_division=0))
    cm = confusion_matrix(y, pred, labels=list(range(num_classes)))
    recalls = []
    for c in range(num_classes):
        tot = cm[c].sum()
        recalls.append(float(cm[c, c] / tot) if tot > 0 else None)
    out['per_class_recall'] = recalls
    out['confusion'] = cm.tolist()
    if risk_weights is not None:
        w = np.asarray(risk_weights, dtype=np.float64)
        cost = np.zeros((num_classes, num_classes))
        for t in range(num_classes):
            for p in range(num_classes):
                if t != p:
                    cost[t, p] = w[t]   # 漏检危险类的代价
        idx = (y * num_classes + pred)
        mean_cost = float(cost.ravel()[idx].mean())
        out['risk_cost_mean'] = mean_cost
        out['risk_cost_norm'] = mean_cost / float(w.max())
    # ECE
    conf = prob.max(axis=1)
    correct = (pred == y).astype(np.float64)
    n_bins = 15
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += (m.sum() / len(y)) * abs(correct[m].mean() - conf[m].mean())
    out['ece'] = float(ece)
    # 开放集（能量分数）
    if unknown_classes:
        logits = np.log(np.clip(prob, 1e-12, 1.0))
        energy = -np.log(np.exp(logits).sum(axis=1) + 1e-12)  # 等价 -logsumexp
        is_unknown = np.isin(y, list(unknown_classes)).astype(int)
        out['unknown_ratio'] = float(is_unknown.mean())
        if 0 < is_unknown.sum() < len(is_unknown):
            out['auroc_unknown_vs_known'] = float(roc_auc_score(is_unknown, energy))
            known_mask = is_unknown == 0
            out['known_acc'] = float((pred[known_mask] == y[known_mask]).mean())
        else:
            out['auroc_unknown_vs_known'] = None
    return out

@torch.no_grad()
def forward_logits(model, x):
    return model(x)

def run_baseline(model, loader, device, num_classes, risk_weights, unknown_classes):
    model.eval()
    ys, preds, probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = forward_logits(model, imgs)
        prob = F.softmax(logits, dim=1).cpu().numpy()
        probs.append(prob); preds.extend(prob.argmax(1).tolist()); ys.extend(labels.tolist())
    return compute_metrics(ys, preds, np.concatenate(probs), num_classes, risk_weights, unknown_classes)

def run_bn_adapt(model, loader, device):
    model.train()
    with torch.no_grad():
        for imgs, _ in loader:
            model(imgs.to(device))
    model.eval()

def run_tent(model, loader, device, num_classes, risk_weights, unknown_classes, steps=1, lr=1e-4, use_sar=False, rho=0.05):
    set_params_grad(model, False)
    params = norm_params(model)
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    model.eval()
    ys, preds, probs = [], [], []
    for bi, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        for _ in range(steps):
            # SAM 近似：先算梯度、扰动后再算一次
            logits = model(imgs)
            prob = F.softmax(logits, dim=1)
            ent = -(prob * torch.log(prob + 1e-12)).sum(1)
            if use_sar:
                # 只用低熵（高置信）样本
                k = max(1, int(0.3 * len(ent)))
                thresh = torch.sort(ent)[0][k - 1]
                mask = (ent <= thresh).float()
            else:
                mask = torch.ones_like(ent)
            loss = (ent * mask).sum() / (mask.sum() + 1e-9)
            opt.zero_grad()
            loss.backward()
            if use_sar:
                grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                norm = torch.sqrt(sum((g ** 2).sum() for g in grads)) + 1e-12
                with torch.no_grad():
                    for p, g in zip(params, grads):
                        p.add_(rho * g / norm)
                opt.zero_grad()
                logits2 = model(imgs)
                prob2 = F.softmax(logits2, dim=1)
                ent2 = -(prob2 * torch.log(prob2 + 1e-12)).sum(1)
                loss2 = (ent2 * mask).sum() / (mask.sum() + 1e-9)
                loss2.backward()
                with torch.no_grad():
                    for p, g in zip(params, grads):
                        p.sub_(rho * g / norm)
            opt.step()
        with torch.no_grad():
            logits = model(imgs)
        prob = F.softmax(logits, dim=1).cpu().numpy()
        probs.append(prob); preds.extend(prob.argmax(1).tolist()); ys.extend(labels.tolist())
    return compute_metrics(ys, preds, np.concatenate(probs), num_classes, risk_weights, unknown_classes)

def run_cotta(model, source_state, loader, device, num_classes, risk_weights, unknown_classes, lr=1e-4, alpha=0.999):
    import copy
    teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    set_params_grad(model, True)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    aug = T.Compose([T.ColorJitter(0.3, 0.3, 0.2, 0.02), T.RandomHorizontalFlip()])
    ys, preds, probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            t_logits = teacher(imgs)
            t_prob = F.softmax(t_logits, dim=1)
            conf, pseudo = t_prob.max(1)
            mask = (conf > 0.6).float()
        aug_imgs = torch.stack([aug(x.cpu()) for x in imgs]).to(device)
        logits = model(aug_imgs)
        loss = (F.cross_entropy(logits, pseudo, reduction='none') * mask).sum() / (mask.sum() + 1e-9)
        opt.zero_grad(); loss.backward(); opt.step()
        # EMA 教师更新
        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), model.parameters()):
                tp.mul_(alpha).add_(sp.detach(), alpha=1 - alpha)
            # 随机恢复（抗遗忘）
            if np.random.rand() < 0.01:
                model.load_state_dict(source_state)
        with torch.no_grad():
            logits = teacher(imgs)
        prob = F.softmax(logits, dim=1).cpu().numpy()
        probs.append(prob); preds.extend(prob.argmax(1).tolist()); ys.extend(labels.tolist())
    return compute_metrics(ys, preds, np.concatenate(probs), num_classes, risk_weights, unknown_classes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--backbone', default=None)
    ap.add_argument('--num-classes', type=int, required=True)
    ap.add_argument('--label-col', default='label6')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--domain', default=None)
    ap.add_argument('--root', default='/private/zdy/deepseek_road')
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', required=True)
    ap.add_argument('--methods', default='source,bn,tent,cotta,sar')
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--max-samples', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--risk', default='')
    ap.add_argument('--unknown-classes', default='')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log('参数: ' + json.dumps(vars(args), ensure_ascii=False))
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    backbone = args.backbone or ckpt.get('backbone', 'resnet50')
    model = timm.create_model(backbone, pretrained=False, num_classes=args.num_classes)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    log('加载模型: %s, ckpt epoch=%s' % (backbone, ckpt.get('epoch')))

    df = pd.read_csv(args.manifest, low_memory=False)
    df = df[df['split'] == args.split].copy()
    if args.domain:
        df = df[df['domain'] == args.domain].copy()
    if len(df) == 0:
        raise SystemExit('没有样本')
    if args.max_samples:
        df = df.sample(min(args.max_samples, len(df)), random_state=0).reset_index(drop=True)
    log('评测数据: %d 条 (%s, split=%s, domain=%s)' % (len(df), os.path.basename(args.manifest), args.split, args.domain))

    crop = ('x' in df.columns) and df['x'].notna().any() and (df['w'].astype(float) > 0).any()
    ds = CsvDataset(df, args.root, eval_transform(args.img_size), args.label_col, crop=crop)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    risk = None
    if args.risk:
        risk = [float(x) for x in args.risk.split(',')]
    unknown = [int(x) for x in args.unknown_classes.split(',')] if args.unknown_classes else None
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    results = {}
    source_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    order = ['source', 'bn', 'tent', 'sar', 'cotta']
    for m in [x for x in order if x in methods]:
        log('>>> 方法: %s' % m)
        # 每个方法前恢复源模型
        model.load_state_dict(source_state)
        model.to(device)
        t0 = time.time()
        if m == 'source':
            res = run_baseline(model, loader, device, args.num_classes, risk, unknown)
        elif m == 'bn':
            run_bn_adapt(model, loader, device)
            res = run_baseline(model, loader, device, args.num_classes, risk, unknown)
        elif m == 'tent':
            res = run_tent(model, loader, device, args.num_classes, risk, unknown, steps=args.steps, lr=args.lr)
        elif m == 'sar':
            res = run_tent(model, loader, device, args.num_classes, risk, unknown, steps=args.steps, lr=args.lr, use_sar=True)
        elif m == 'cotta':
            res = run_cotta(model, source_state, loader, device, args.num_classes, risk, unknown, lr=args.lr)
        else:
            log('未知方法 %s，跳过' % m); continue
        res['time_sec'] = round(time.time() - t0, 1)
        results[m] = res
        log('  结果: acc=%.4f macroF1=%.4f%s%s' % (
            res['acc'], res['macro_f1'],
            (' risk=%.4f' % res['risk_cost_norm']) if 'risk_cost_norm' in res else '',
            (' auroc=%.4f' % res['auroc_unknown_vs_known']) if res.get('auroc_unknown_vs_known') else ''))

    summary = dict(args=vars(args), results=results)
    with open(os.path.join(args.out, 'eval_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log('完成，结果已保存: %s' % os.path.join(args.out, 'eval_summary.json'))
    log('==== 汇总 ====')
    for m, r in results.items():
        log('%-6s acc=%.4f F1=%.4f%s' % (m, r['acc'], r['macro_f1'],
            (' ECE=%.4f' % r['ece']) if 'ece' in r else ''))

if __name__ == '__main__':
    main()
