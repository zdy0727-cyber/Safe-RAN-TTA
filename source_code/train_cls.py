# -*- coding: utf-8 -*-
"""
通用图像分类训练/评测脚本（路面状态识别）

用法示例：
  # 域内训练（新疆 6 类）
  python train_cls.py --manifest manifests/track2_xinjiang.csv --label-col label6 --num-classes 6 \
      --train-split train --val-split val --test-split test --out runs/xj_r50 --backbone resnet50 --epochs 30

  # 跨域评测（清华训练的模型 → 新疆测试）
  python train_cls.py --eval-only --ckpt runs/ts_r50/best.pt --eval-manifest manifests/track2_xinjiang.csv \
      --eval-split test --label-col label6 --num-classes 6 --out runs/eval_ts2xj
"""
import os, sys, csv, json, time, argparse, random
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from sklearn.metrics import f1_score, confusion_matrix

ROOT = '/private/zdy/deepseek_road'

def now():
    return datetime.now().strftime('%H:%M:%S')

def log(msg, fh=None):
    line = '[%s] %s' % (now(), msg)
    print(line, flush=True)
    if fh:
        fh.write(line + '\n'); fh.flush()

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
        path = os.path.join(self.root, row['path'])
        img = Image.open(path).convert('RGB')
        if self.crop:
            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
            img = img.crop((x, y, x + w, y + h))
        if self.transform:
            img = self.transform(img)
        return img, int(row[self.label_col])

def build_transforms(img_size, train):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.55, 1.0), ratio=(3/4, 4/3)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize(img_size + 32, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

def _worker_init(_worker_id):
    # 关键：容器 cgroup 只有 8 核，必须把每个 dataloader worker 限制为 1 线程，
    # 否则 torch 默认 64 线程会造成严重线程超配（transform 慢 20 倍以上）
    torch.set_num_threads(1)

def make_loader(df, root, img_size, train, bs, workers, label_col, crop=False):
    ds = CsvDataset(df, root, build_transforms(img_size, train), label_col, crop=crop)
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=workers,
                      pin_memory=True, persistent_workers=(workers > 0), drop_last=train,
                      worker_init_fn=_worker_init if workers > 0 else None)

def load_manifest(path):
    df = pd.read_csv(path)
    df['path'] = df['path'].astype(str)
    return df

def filter_split(df, split):
    out = df[df['split'] == split].copy()
    if len(out) == 0:
        raise ValueError('split %s 为空' % split)
    return out

def compute_class_weights(counts, num_classes):
    n = np.zeros(num_classes, dtype=np.float64)
    for c, v in counts.items():
        if 0 <= int(c) < num_classes:
            n[int(c)] = v
    n = np.maximum(n, 1)
    w = (n.sum() / (num_classes * n)) ** 0.5
    return torch.tensor(w / w.mean(), dtype=torch.float32)

@torch.no_grad()
def evaluate(model, loader, device, num_classes, fh=None):
    model.eval()
    ys, ps = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            out = model(imgs)
        ps.extend(out.argmax(1).detach().cpu().numpy().tolist())
        ys.extend(labels.numpy().tolist())
    ys = np.array(ys); ps = np.array(ps)
    acc = float((ys == ps).mean())
    macro_f1 = float(f1_score(ys, ps, average='macro', labels=list(range(num_classes)), zero_division=0))
    cm = confusion_matrix(ys, ps, labels=list(range(num_classes)))
    per_class_recall = []
    for c in range(num_classes):
        tot = cm[c].sum()
        per_class_recall.append(float(cm[c, c] / tot) if tot > 0 else None)
    return dict(acc=acc, macro_f1=macro_f1, per_class_recall=per_class_recall,
                confusion=cm.tolist(), n=len(ys))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--label-col', default='label6')
    ap.add_argument('--num-classes', type=int, default=6)
    ap.add_argument('--train-split', default='train')
    ap.add_argument('--val-split', default='val')
    ap.add_argument('--test-split', default='test')
    ap.add_argument('--eval-manifest', default=None)
    ap.add_argument('--eval-split', default='test')
    ap.add_argument('--eval-domain', default=None, help='只评测某个 domain（如 ctd1/ctd2/ctd3）')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--backbone', default='resnet50')
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--eval-bs', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--warmup-epochs', type=float, default=3)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--class-weights', action='store_true')
    ap.add_argument('--max-train', type=int, default=0)
    ap.add_argument('--max-eval', type=int, default=0)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '2')))
    os.makedirs(args.out, exist_ok=True)
    fh = open(os.path.join(args.out, 'train.log'), 'a', encoding='utf-8')
    log('参数: ' + json.dumps(vars(args), ensure_ascii=False), fh)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log('device=%s, gpus=%d, torch=%s' % (device, torch.cuda.device_count(), torch.__version__), fh)

    # ---------------- 评测模式 ----------------
    if args.eval_only:
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        model = timm.create_model(ckpt['backbone'], pretrained=False, num_classes=args.num_classes)
        model.load_state_dict(ckpt['model'])
        model.to(device)
        df = load_manifest(args.eval_manifest)
        df = filter_split(df, args.eval_split)
        if getattr(args, 'eval_domain', None):
            df = df[df['domain'] == args.eval_domain].copy()
            if len(df) == 0:
                raise ValueError('domain %s 为空' % args.eval_domain)
        if args.max_eval: df = df.sample(min(args.max_eval, len(df)), random_state=args.seed)
        crop = 'x' in df.columns and df['x'].notna().any() and (df['w'].astype(float) > 0).any()
        loader = make_loader(df, args.root, args.img_size, False, args.eval_bs, args.workers, args.label_col, crop=crop)
        res = evaluate(model, loader, device, args.num_classes, fh)
        log('评测结果: acc=%.4f macroF1=%.4f n=%d' % (res['acc'], res['macro_f1'], res['n']), fh)
        log('各类召回: ' + str(['%.3f' % r if r is not None else 'None' for r in res['per_class_recall']]), fh)
        log('混淆矩阵:\n' + str(res['confusion']), fh)
        json.dump(res, open(os.path.join(args.out, 'eval_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        return

    # ---------------- 训练模式 ----------------
    df = load_manifest(args.manifest)
    tr = filter_split(df, args.train_split)
    va = filter_split(df, args.val_split)
    if args.max_train: tr = tr.sample(min(args.max_train, len(tr)), random_state=args.seed).reset_index(drop=True)
    if args.max_eval:  va = va.sample(min(args.max_eval, len(va)), random_state=args.seed).reset_index(drop=True)
    log('数据: train=%d val=%d (manifest=%s)' % (len(tr), len(va), args.manifest), fh)
    log('train 标签分布: ' + str(Counter(tr[args.label_col].tolist())), fh)

    train_loader = make_loader(tr, args.root, args.img_size, True, args.bs, args.workers, args.label_col)
    val_loader = make_loader(va, args.root, args.img_size, False, args.eval_bs, args.workers, args.label_col)

    model = timm.create_model(args.backbone, pretrained=True, num_classes=args.num_classes)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log('模型: %s, params=%.2fM' % (args.backbone, n_params / 1e6), fh)

    weight = None
    if args.class_weights:
        weight = compute_class_weights(Counter(tr[args.label_col].tolist()), args.num_classes).to(device)
        log('类别权重: ' + str([round(float(x), 3) for x in weight]), fh)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_f1 = -1.0
    history = []
    global_step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0; nb = 0
        te = time.time()
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ep_loss += float(loss.detach()); nb += 1; global_step += 1
            if nb % 50 == 0:
                log('  ep%d it%d/%d loss=%.4f lr=%.2e' % (epoch, nb, steps_per_epoch, ep_loss / nb,
                                                          optimizer.param_groups[0]['lr']), fh)
        val = evaluate(model, val_loader, device, args.num_classes, fh)
        dt = time.time() - te
        log('epoch %d/%d  用时%.0fs  train_loss=%.4f  val_acc=%.4f  val_macroF1=%.4f' % (
            epoch, args.epochs, dt, ep_loss / max(1, nb), val['acc'], val['macro_f1']), fh)
        history.append(dict(epoch=epoch, train_loss=ep_loss / max(1, nb), **{k: v for k, v in val.items() if k != 'confusion'}))
        if val['macro_f1'] > best_f1:
            best_f1 = val['macro_f1']
            torch.save(dict(model=model.state_dict(), backbone=args.backbone, args=vars(args),
                            epoch=epoch, val=val), os.path.join(args.out, 'best.pt'))
            log('  [best] macroF1=%.4f (epoch %d)' % (best_f1, epoch), fh)
        # last.pt 减少写盘频率（NFS 共享存储写大文件容易出问题）
        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(dict(model=model.state_dict(), backbone=args.backbone, args=vars(args),
                            epoch=epoch, val=val), os.path.join(args.out, 'last.pt'))
        json.dump(history, open(os.path.join(args.out, 'history.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    log('训练结束，总用时 %.1f 分钟，best val macroF1=%.4f' % ((time.time() - t0) / 60, best_f1), fh)
    json.dump(history, open(os.path.join(args.out, 'history.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    # 训练后在 test split 上评测（同分布）
    try:
        best = torch.load(os.path.join(args.out, 'best.pt'), map_location='cpu', weights_only=False)
        model.load_state_dict(best['model'])
        te_df = filter_split(df, args.test_split)
        if args.max_eval: te_df = te_df.sample(min(args.max_eval, len(te_df)), random_state=args.seed)
        te_loader = make_loader(te_df, args.root, args.img_size, False, args.eval_bs, args.workers, args.label_col)
        res = evaluate(model, te_loader, device, args.num_classes, fh)
        log('TEST: acc=%.4f macroF1=%.4f n=%d' % (res['acc'], res['macro_f1'], res['n']), fh)
        log('TEST 各类召回: ' + str(['%.3f' % r if r is not None else 'None' for r in res['per_class_recall']]), fh)
        json.dump(res, open(os.path.join(args.out, 'test_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    except Exception as e:
        log('测试集评测失败: %r' % (e,), fh)

if __name__ == '__main__':
    main()
