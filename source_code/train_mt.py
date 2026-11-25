# -*- coding: utf-8 -*-
"""
MTD-Net: 多任务解耦网络（方案 C 核心实现）
  - 共享 backbone（timm, ResNet-50 / ConvNeXt-T）
  - 解耦头：content 子空间（类原型余弦分类）+ residual 子空间（域信息）
  - 多任务头：分类（层级 masked CE + 类权重）+ 湿滑等级/雪覆盖/粗糙度/材质序数分类 + 水膜高度回归
  - 域对抗：residual 预测域（捕捉域偏移），content 经 GRL 做域不变
  - 多任务不确定性加权（Kendall & Gal）
  - head-mode: none（普通线性头，基线）/ proto（仅原型）/ full（解耦+域对抗，本文）

用法示例：
  python train_mt.py --root /private/zdy/deepseek_road --manifest manifests/multitask_all.csv \
      --track track1 --label-col label15 --num-classes 15 --out runs/mt_track1_full --head-mode full
"""
import os, sys, json, time, csv, random, argparse
from collections import Counter, defaultdict
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
from sklearn.metrics import f1_score, confusion_matrix

torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '2')))


def now():
    return datetime.now().strftime('%H:%M:%S')


def log(*a):
    print('[%s]' % now(), *a, flush=True)


# ---------------- 数据 ----------------
class MTDataset(Dataset):
    """返回 (img, dict_targets)；缺失标签用 -1 表示（分类 ignore_index / 回归 mask）"""
    TASKS = ['cls', 'wet', 'snow', 'rough', 'mat', 'water', 'domain']

    def __init__(self, df, root, transform, label_col, crop=False):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.label_col = label_col
        self.crop = crop

        def num(col):
            if col not in self.df.columns:
                return np.full(len(self.df), -1)
            return pd.to_numeric(self.df[col], errors='coerce').fillna(-1).values

        self.cls = pd.to_numeric(self.df[label_col], errors='coerce').fillna(-1).astype(np.int64).values
        self.wet = num('wetness_level')
        self.snow = num('snow_cover')
        self.rough = num('roughness')
        self.mat = num('material')
        self.water = pd.to_numeric(self.df['water_um'], errors='coerce').values if 'water_um' in self.df else np.full(len(self.df), np.nan)
        # 域 id（按数据集名）
        domains = sorted(self.df['dataset'].astype(str).unique().tolist())
        self.dom_map = {d: i for i, d in enumerate(domains)}
        self.dom = self.df['dataset'].astype(str).map(self.dom_map).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(self.root, row['path'])).convert('RGB')
        if self.crop and np.isfinite(row['x']):
            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
            img = img.crop((x, y, x + w, y + h))
        img = self.transform(img)
        water = self.water[i]
        water = -1.0 if (water is None or not np.isfinite(water)) else float(np.log1p(water))
        return img, (int(self.cls[i]), int(self.wet[i]), int(self.snow[i]), int(self.rough[i]),
                     int(self.mat[i]), float(water), int(self.dom[i]))


# ---------------- 模型 ----------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.alpha * g, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


class MLPHead(nn.Module):
    def __init__(self, dim, out, hidden=256, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Dropout(drop), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


class MTDNet(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=15, head_mode='full', proto_dim=256,
                 tasks=(), n_domains=2, grl_alpha=1.0, pretrained=True, head_extra=False, detach_residual=False):
        super().__init__()
        self.head_mode = head_mode
        self.detach_residual = detach_residual
        self.tasks = [x for x in tasks if x != 'cls']
        self.all_task_keys = ['cls'] + self.tasks
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.n_domains = n_domains
        self.grl_alpha = grl_alpha

        if head_mode == 'none':
            self.cls_head = nn.Linear(feat_dim, num_classes)
            aux_dim = feat_dim
        elif head_mode == 'linear_res':
            self.cls_head = nn.Linear(feat_dim, num_classes)
            self.residual = nn.Linear(feat_dim, proto_dim)
            aux_dim = feat_dim
        else:
            self.content = nn.Linear(feat_dim, proto_dim)
            self.prototypes = nn.Parameter(torch.randn(num_classes, proto_dim) * 0.01)
            self.scale = nn.Parameter(torch.tensor(16.0))
            self.residual = nn.Linear(feat_dim, proto_dim) if head_mode == 'full' else None
            self.cls_linear = nn.Linear(proto_dim, num_classes, bias=False) if head_extra else None
            aux_dim = proto_dim * (2 if head_mode == 'full' else 1)

        self.aux = nn.ModuleDict()
        dims = {'wet': 4, 'snow': 3, 'rough': 3, 'mat': 4}
        for t in self.tasks:
            if t in dims:
                self.aux[t] = MLPHead(aux_dim, dims[t])
        if 'water' in self.tasks:
            self.aux['water'] = MLPHead(aux_dim, 1)
        if head_mode in ('full', 'linear_res'):
            self.dom_head = MLPHead(proto_dim, n_domains)
        if head_mode == 'full':
            self.adv_head = MLPHead(proto_dim, n_domains)
        # 多任务不确定性权重（log sigma^2）
        self.log_vars = nn.ParameterDict({t: nn.Parameter(torch.zeros(1)) for t in self.all_task_keys})

    def forward(self, x):
        feat = self.backbone(x)
        if self.head_mode == 'none':
            cls_logits = self.cls_head(feat)
            h = feat
            r = None
        elif self.head_mode == 'linear_res':
            cls_logits = self.cls_head(feat)
            h = feat
            r = self.residual(feat.detach() if self.detach_residual else feat)
        else:
            z = F.normalize(self.content(feat), dim=1)
            cls_logits = self.scale.clamp(1.0, 50.0) * (z @ F.normalize(self.prototypes, dim=1).t())
            if self.cls_linear is not None:
                cls_logits = cls_logits + self.cls_linear(z)
            if self.head_mode == 'full':
                r = self.residual(feat)
                h = torch.cat([z, r], dim=1)
            else:
                h = z
                r = None
        out = {'cls': cls_logits, 'feat': feat, 'r': r}
        if self.head_mode in ('full', 'linear_res'):
            out['dom'] = self.dom_head(r)
        if self.head_mode == 'full':
            out['adv'] = self.adv_head(grad_reverse(z, self.grl_alpha))
        for t in self.tasks:
            if t == 'water':
                out['water'] = self.aux['water'](h).squeeze(1)
            else:
                out[t] = self.aux[t](h)
        return out


# ---------------- 数据变换 ----------------
def build_transforms(img_size):
    tr = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.25, 0.05),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    ev = T.Compose([
        T.Resize(img_size + 32, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return tr, ev


def worker_init(_):
    torch.set_num_threads(1)


def load_df(args):
    df = pd.read_csv(args.manifest, low_memory=False)
    if 'track' in df.columns:
        df = df[df['track'] == args.track].copy()
    # 裁剪坐标（清华）
    if 'x' not in df.columns:
        df['x'] = np.nan
    crop = df['x'].notna().any()
    if not crop:
        df['x'] = df['y'] = df['w'] = df['h'] = np.nan
    return df, crop


def make_loader(df, root, transform, label_col, crop, bs, workers, shuffle):
    ds = MTDataset(df, root, transform, label_col, crop=crop)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, pin_memory=True,
                      drop_last=shuffle, worker_init_fn=worker_init if workers > 0 else None), ds


# ---------------- 评价 ----------------
@torch.no_grad()
def evaluate(model, loader, device, num_classes, tasks, class_names=None):
    model.eval()
    ys, preds = [], []
    water_pred, water_true = [], []
    ord_stats = {t: [[], []] for t in tasks if t in ('wet', 'snow', 'rough', 'mat')}
    per_ds = defaultdict(lambda: {'y': [], 'p': []})
    ds_names = []
    for imgs, (cls, wet, snow, rough, mat, water, dom) in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        logits = out['cls'].float()
        pred = logits.argmax(1).cpu().numpy()
        cls = cls.numpy()
        mask = cls >= 0
        if mask.any():
            ys.extend(cls[mask].tolist()); preds.extend(pred[mask].tolist())
        # 回归/序数
        for t, arr in (('wet', wet), ('snow', snow), ('rough', rough), ('mat', mat)):
            if t in ord_stats:
                arr = arr.numpy(); m = arr >= 0
                if m.any():
                    ord_stats[t][0].extend(arr[m].tolist())
                    ord_stats[t][1].extend(out[t].argmax(1).cpu().numpy()[m].tolist())
        if 'water' in tasks:
            w = water.numpy(); m = w >= 0
            if m.any():
                water_true.extend(w[m].tolist()); water_pred.extend(out['water'].float().cpu().numpy()[m].tolist())
    res = {}
    if ys:
        y = np.array(ys); p = np.array(preds)
        res['acc'] = float((y == p).mean())
        res['macro_f1'] = float(f1_score(y, p, average='macro', labels=list(range(num_classes)), zero_division=0))
        cm = confusion_matrix(y, p, labels=list(range(num_classes)))
        res['per_class_recall'] = [float(cm[c, c] / cm[c].sum()) if cm[c].sum() else None for c in range(num_classes)]
        res['n'] = int(len(y))
    for t, (a, b) in ord_stats.items():
        if a:
            a = np.array(a); b = np.array(b)
            res[t + '_acc'] = float((a == b).mean())
            res[t + '_n'] = int(len(a))
    if water_true:
        wt = np.array(water_true); wp = np.array(water_pred)
        res['water_mae_log'] = float(np.abs(wt - wp).mean())
        res['water_mae_um'] = float(np.abs(np.expm1(wt) - np.expm1(wp)).mean())
        res['water_n'] = int(len(wt))
    return res


@torch.no_grad()
def evaluate_by_group(model, loader, device, num_classes, tasks, groups):
    """按 dataset/domain 分组评测（loader 顺序与 groups 对齐）"""
    model.eval()
    idx_groups = []
    for g in groups:
        idx_groups.append(g)
    out_all = []
    start = 0
    # 简化：重新遍历，累积每条预测
    ys, preds, gs = [], [], []
    for imgs, (cls, wet, snow, rough, mat, water, dom) in loader:
        imgs = imgs.to(device, non_blocking=True)
        pred = model(imgs)['cls'].float().argmax(1).cpu().numpy()
        ys.extend(cls.numpy().tolist()); preds.extend(pred.tolist())
        gs.extend(dom.numpy().tolist())
    y = np.array(ys); p = np.array(preds); g = np.array(gs)
    res = {}
    for gi, gname in enumerate(groups):
        m = (g == gi) & (y >= 0)
        if m.sum() == 0:
            continue
        res[gname] = dict(n=int(m.sum()), acc=float((y[m] == p[m]).mean()),
                          macro_f1=float(f1_score(y[m], p[m], average='macro', labels=list(range(num_classes)), zero_division=0)))
    return res


# ---------------- 训练 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/private/zdy/deepseek_road')
    ap.add_argument('--manifest', default='manifests/multitask_all.csv')
    ap.add_argument('--track', default='track1', choices=['track1', 'track2'])
    ap.add_argument('--datasets', default='')
    ap.add_argument('--label-col', default='label15')
    ap.add_argument('--num-classes', type=int, default=15)
    ap.add_argument('--out', required=True)
    ap.add_argument('--backbone', default='resnet50')
    ap.add_argument('--head-mode', default='full', choices=['none', 'proto', 'full', 'linear_res'])
    ap.add_argument('--tasks', default='wet,snow,rough,mat,water')
    ap.add_argument('--proto-dim', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--uncertainty', type=int, default=1)
    ap.add_argument('--adv-weight', type=float, default=0.1)
    ap.add_argument('--aux-weight', type=float, default=1.0)
    ap.add_argument('--max-train', type=int, default=0)
    ap.add_argument('--max-val', type=int, default=30000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--head-extra', type=int, default=0)
    ap.add_argument('--staged', type=int, default=0)
    ap.add_argument('--aux-scale', type=float, default=1.0)
    ap.add_argument('--head-lr', type=float, default=0.0)
    ap.add_argument('--detach-residual', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tasks = [t for t in args.tasks.split(',') if t]
    log('参数:', json.dumps(vars(args), ensure_ascii=False))

    root = args.root
    man = args.manifest if os.path.isabs(args.manifest) else os.path.join(root, args.manifest)
    # 重新加载（load_df 依赖 args）
    df = pd.read_csv(man, low_memory=False)
    if 'track' in df.columns:
        df = df[df['track'] == args.track].copy()
    if 'x' not in df.columns:
        df['x'] = df['y'] = df['w'] = df['h'] = np.nan
    crop = df['x'].notna().any()

    if args.datasets:
        df = df[df['dataset'].isin(args.datasets.split(','))].copy()
    tr_df = df[df['split'].isin(['train'])].copy()
    va_df = df[df['split'].isin(['val', 'validation'])].copy()
    if args.max_train and len(tr_df) > args.max_train:
        tr_df = tr_df.sample(args.max_train, random_state=args.seed)
    if args.max_val and len(va_df) > args.max_val:
        va_df = va_df.sample(args.max_val, random_state=args.seed)
    # 去掉分类标签缺失的行
    tr_df = tr_df[pd.to_numeric(tr_df[args.label_col], errors='coerce').notna()]
    va_df = va_df[pd.to_numeric(va_df[args.label_col], errors='coerce').notna()]
    log('数据: train=%d val=%d (track=%s, 类别=%d, crop=%s)' % (len(tr_df), len(va_df), args.track, args.num_classes, crop))

    # 类别权重
    cnt = Counter(pd.to_numeric(tr_df[args.label_col]).astype(int).tolist())
    cw = torch.zeros(args.num_classes)
    total = sum(cnt.values())
    for c in range(args.num_classes):
        cw[c] = total / (max(1, len(cnt)) * max(1, cnt.get(c, 1)))
    cw = cw / cw.mean()
    log('类别权重:', [round(float(x), 3) for x in cw])

    tr_t, va_t = build_transforms(args.img_size)
    tr_loader, _ = make_loader(tr_df, root, tr_t, args.label_col, crop, args.bs, args.workers, True)
    va_loader, _ = make_loader(va_df, root, va_t, args.label_col, crop, 256, max(2, args.workers // 2), False)
    n_dom = len(sorted(df['dataset'].astype(str).unique()))
    domain_names = sorted(df['dataset'].astype(str).unique())
    log('域:', domain_names)

    model = MTDNet(args.backbone, args.num_classes, args.head_mode, args.proto_dim, tasks, n_dom,
                   head_extra=bool(args.head_extra)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log('模型: %s head_mode=%s 参数=%.1fM' % (args.backbone, args.head_mode, n_params))

    if args.head_lr > 0:
        head_names = ('content', 'prototypes', 'scale', 'residual', 'aux', 'log_vars', 'cls_linear')
        head_params = [p for n, p in model.named_parameters() if n.startswith(head_names)]
        head_ids = {id(p) for p in head_params}
        base_params = [p for p in model.parameters() if id(p) not in head_ids]
        opt = torch.optim.AdamW([{'params': base_params, 'lr': args.lr},
                                 {'params': head_params, 'lr': args.lr * args.head_lr}],
                                weight_decay=args.wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(tr_loader))
    total_steps = args.epochs * steps_per_epoch
    warm_steps = args.warmup * steps_per_epoch

    def lr_at(step):
        if step < warm_steps:
            return args.lr * step / max(1, warm_steps)
        p = (step - warm_steps) / max(1, total_steps - warm_steps)
        return args.lr * 0.5 * (1 + np.cos(np.pi * p))

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    cw = cw.to(device)
    history = []
    best = -1
    step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        model.grl_alpha = min(1.0, ep / max(1, args.warmup))
        t0 = time.time()
        losses = Counter()
        for imgs, (cls, wet, snow, rough, mat, water, dom) in tr_loader:
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            imgs = imgs.to(device, non_blocking=True)
            cls = cls.to(device); dom = dom.to(device)
            wet = wet.to(device); snow = snow.to(device); rough = rough.to(device)
            mat = mat.to(device); water = water.to(device).float()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out = model(imgs)
                staged_now = (args.staged > 0 and ep <= args.staged)
                task_losses = {}
                if (cls >= 0).any():
                    task_losses['cls'] = F.cross_entropy(out['cls'].float(), cls, weight=cw, ignore_index=-1)
                tgt_map = {'wet': wet, 'snow': snow, 'rough': rough, 'mat': mat}
                if not staged_now:
                    for t in ('wet', 'snow', 'rough', 'mat'):
                        if t in tasks and (tgt_map[t] >= 0).any():
                            task_losses[t] = args.aux_scale * F.cross_entropy(out[t].float(), tgt_map[t], ignore_index=-1)
                    if 'water' in tasks:
                        m = water >= 0
                        if m.any():
                            task_losses['water'] = args.aux_scale * F.mse_loss(out['water'].float()[m], water[m])
                if not task_losses:
                    opt.zero_grad(); continue
                total = None
                if args.uncertainty:
                    for t, l in task_losses.items():
                        lv = model.log_vars[t].clamp(-5.0, 5.0)
                        term = torch.exp(-lv) * l + lv
                        total = term if total is None else total + term
                else:
                    for t, l in task_losses.items():
                        total = l if total is None else total + l
                if args.head_mode in ('full', 'linear_res') and not staged_now:
                    L_dom = F.cross_entropy(out['dom'].float(), dom)
                    total = total + 0.5 * L_dom
                    losses['dom'] += float(L_dom)
                    if args.head_mode == 'full':
                        L_adv = F.cross_entropy(out['adv'].float(), dom)
                        total = total + args.adv_weight * L_adv
                        losses['adv'] += float(L_adv)
                if not torch.isfinite(total):
                    opt.zero_grad(); continue
            opt.zero_grad()
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses['total'] += float(total)
            if 'cls' in task_losses:
                losses['cls'] += float(task_losses['cls'])
            step += 1
            if step % 50 == 0:
                log('  ep%d it%d/%d loss=%.4f lr=%.2e' % (ep, step % steps_per_epoch or steps_per_epoch,
                                                          steps_per_epoch, float(total), opt.param_groups[0]['lr']))
        val = evaluate(model, va_loader, device, args.num_classes, tasks)
        rec = dict(epoch=ep, time=round(time.time() - t0, 1), loss=round(losses['total'] / max(1, steps_per_epoch), 4), **val)
        history.append(rec)
        log('epoch %d/%d 用时%ds loss=%.4f val_acc=%.4f val_macroF1=%.4f%s' % (
            ep, args.epochs, rec['time'], rec['loss'], val.get('acc', 0), val.get('macro_f1', 0),
            (' 水膜MAE=%.2fµm' % val['water_mae_um']) if 'water_mae_um' in val else ''))
        if val.get('macro_f1', -1) > best:
            best = val['macro_f1']
            torch.save({'model': model.state_dict(), 'backbone': args.backbone, 'num_classes': args.num_classes,
                        'head_mode': args.head_mode, 'tasks': tasks, 'args': vars(args), 'epoch': ep}, os.path.join(args.out, 'best.pt'))
            log('  [best] macroF1=%.4f (epoch %d)' % (best, ep))
        with open(os.path.join(args.out, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=1)

    # 测试
    te_df = df[df['split'].isin(['test'])].copy()
    te_df = te_df[pd.to_numeric(te_df[args.label_col], errors='coerce').notna()]
    te_loader, te_ds = make_loader(te_df, root, va_t, args.label_col, crop, 256, max(2, args.workers // 2), False)
    ck = torch.load(os.path.join(args.out, 'best.pt'), map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    test = evaluate(model, te_loader, device, args.num_classes, tasks)
    groups = list(te_ds.dom_map.keys())
    by_ds = evaluate_by_group(model, te_loader, device, args.num_classes, tasks, groups)
    log('TEST: acc=%.4f macroF1=%.4f n=%d' % (test.get('acc', 0), test.get('macro_f1', 0), test.get('n', 0)))
    log('TEST 各类召回:', [round(x, 3) if x is not None else None for x in test.get('per_class_recall', [])])
    log('TEST 分组:', json.dumps(by_ds, ensure_ascii=False))
    with open(os.path.join(args.out, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(test=test, by_dataset=by_ds, args=vars(args)), f, ensure_ascii=False, indent=1)
    log('完成: %s' % args.out)


if __name__ == '__main__':
    main()
