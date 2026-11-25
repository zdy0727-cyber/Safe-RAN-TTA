# -*- coding: utf-8 -*-
"""
RAN-TTA：风险感知、新异解耦的测试时自适应评测（方案 C 的 TTA 阶段）
方法：
  source   源模型直接评测
  bn       BN-Adapt
  tent     熵最小化（论文原始设定，风险中性）
  sar      SAM + 选择性熵最小化
  tent_risk 风险加权熵最小化（对照：只加风险权重、无解耦门控）
  ran_tta  本文完整方法：残差新异解耦 + 安全门控 + 风险加权 + 受保护记忆（T3A 式记忆增强预测）
  消融变体：ran_nogate / ran_nomem / ran_nores / ran_norisk
指标：acc、macro-F1（出现类）、风险加权代价、ECE、开放集 AUROC（能量 / 新异 / 双分数）
"""
import os, sys, json, time, argparse
from datetime import datetime
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

sys.path.insert(0, '/root/rsc_local/code')
torch.set_num_threads(2)


RISK_LAM = 0.2


def log(*a):
    print('[%s]' % datetime.now().strftime('%H:%M:%S'), *a, flush=True)


def norm_params(model):
    ps = []
    for n, p in model.named_parameters():
        ln = n.lower()
        if ('bn' in ln or 'norm' in ln) and p.dim() <= 2:
            ps.append(p)
    return ps


class DS(Dataset):
    def __init__(self, df, root, transform, label_col):
        self.df = df.reset_index(drop=True)
        self.root = root; self.transform = transform; self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(self.root, r['path'])).convert('RGB')
        y = r[self.label_col]
        return self.transform(img), int(y) if pd.notna(y) else -1


def tf_eval(sz):
    return T.Compose([T.Resize(sz + 32, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(sz),
                      T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def load_model(ckpt_path, kind, num_classes, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    backbone = ck.get('backbone', 'resnet50')
    if kind == 'mt':
        from train_mt import MTDNet
        st = ck['model']
        head_mode = ck.get('head_mode', 'full')
        n_dom = 2
        for k, v in st.items():
            if k.startswith('dom_head') and k.endswith('weight') and v.dim() == 2:
                n_dom = int(v.shape[0])
        head_extra = any(k.startswith('cls_linear') for k in st)
        model = MTDNet(backbone, num_classes, head_mode, 256, ck.get('tasks') or [],
                       n_domains=n_dom, pretrained=False, head_extra=head_extra)
        model.load_state_dict(st)
    else:
        model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
        model.load_state_dict(ck['model'])
    return model.to(device).eval(), ck


def logits_of(out):
    return out['cls'] if isinstance(out, dict) else out


def fit_basis(R, k=8):
    R = R - R.mean(0, keepdim=True)
    try:
        U, S, V = torch.pca_lowrank(R.float(), q=min(k, min(R.shape) - 1))
        return V[:, :k]
    except Exception:
        return None


def compute_metrics(y, pred, prob, num_classes, risk=None, unknown=None, novelty=None):
    y = np.asarray(y); pred = np.asarray(pred)
    prob = np.nan_to_num(np.asarray(prob, dtype=np.float64)); prob = np.clip(prob, 0, 1)
    s = prob.sum(1, keepdims=True); s[s <= 0] = 1; prob = prob / s
    present = sorted(set(y.tolist()) | set(pred.tolist()))
    out = dict(n=int(len(y)), acc=float((y == pred).mean()),
               macro_f1=float(f1_score(y, pred, average='macro', labels=present, zero_division=0)))
    cm = confusion_matrix(y, pred, labels=list(range(num_classes)))
    out['per_class_recall'] = [float(cm[c, c] / cm[c].sum()) if cm[c].sum() else None for c in range(num_classes)]
    if risk:
        w = np.asarray(risk, dtype=float)
        # 1) 常规风险代价：漏检危险类的代价（C[t,p]=w_t, t!=p）
        out['risk_cost'] = float(np.mean([w[t] if t != p else 0.0 for t, p in zip(y, pred)]) / w.max())
        # 2) 升级代价：低估危险等级重罚、高估轻罚（lam=0.2）
        lam = RISK_LAM
        C2 = np.where(w[:, None] > w[None, :], (w[:, None] - w[None, :]),
                      lam * (w[None, :] - w[:, None]))
        np.fill_diagonal(C2, 0.0)
        C2 = C2 / w.max()
        out['esc_cost'] = float(np.mean([C2[t, p] for t, p in zip(y, pred)]))
        # 3) 危险类低估率（真实为危险类、预测为更低危险等级）
        haz = w >= 0.7 * w.max()
        _v = [1.0 if (haz[t] and w[p] < w[t]) else 0.0 for t, p in zip(y, pred) if haz[t]]
        out['hazard_under'] = float(np.mean(_v)) if _v else 0.0
        out['n_hazard'] = int(sum(1 for t in y if haz[t]))
        # 4) 风险决策：在“升级代价”上做 Bayes 决策
        exp_cost = prob @ C2            # (N, num_classes)
        pred_rd = exp_cost.argmin(1)
        present_rd = sorted(set(y.tolist()) | set(pred_rd.tolist()))
        out['acc_rd'] = float((y == pred_rd).mean())
        out['macro_f1_rd'] = float(f1_score(y, pred_rd, average='macro', labels=present_rd, zero_division=0))
        out['risk_cost_rd'] = float(np.mean([w[t] if t != p else 0.0 for t, p in zip(y, pred_rd)]) / w.max())
        out['esc_cost_rd'] = float(np.mean([C2[t, p] for t, p in zip(y, pred_rd)]))
        _v2 = [1.0 if (haz[t] and w[p] < w[t]) else 0.0 for t, p in zip(y, pred_rd) if haz[t]]
        out['hazard_under_rd'] = float(np.mean(_v2)) if _v2 else 0.0
    conf = prob.max(1); correct = (pred == y).astype(float)
    ece = 0.0
    for b in range(15):
        m = (conf > b / 15) & (conf <= (b + 1) / 15)
        if m.sum():
            ece += m.sum() / len(y) * abs(correct[m].mean() - conf[m].mean())
    out['ece'] = float(ece)
    energy = -np.log(np.clip(prob, 1e-12, 1).sum(1) + 1e-12)
    if unknown:
        is_u = np.isin(y, list(unknown)).astype(int)
        out['unknown_ratio'] = float(is_u.mean())
        if 0 < is_u.sum() < len(is_u):
            known = is_u == 0
            out['known_acc'] = float((pred[known] == y[known]).mean())
            if np.isfinite(energy).all():
                try:
                    out['auroc_energy'] = float(roc_auc_score(is_u, energy))
                except Exception:
                    pass
            if novelty is not None:
                nv = np.asarray(novelty, dtype=float)
                if nv.shape[0] == len(y) and np.isfinite(nv).all():
                    try:
                        out['auroc_novelty'] = float(roc_auc_score(is_u, nv))
                    except Exception:
                        pass
                    if np.isfinite(energy).all():
                        e2 = (energy - energy.mean()) / (energy.std() + 1e-9)
                        n2 = (nv - nv.mean()) / (nv.std() + 1e-9)
                        try:
                            out['auroc_dual'] = float(roc_auc_score(is_u, e2 + n2))
                        except Exception:
                            pass
    for _k, _v in list(out.items()):
        if isinstance(_v, float) and not np.isfinite(_v):
            out[_k] = 0.0
    return out


def run_source(model, loader, device, num_classes, risk, unknown):
    ys, preds, probs, novs = [], [], [], []
    R_buf = []
    for imgs, y in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            out = model(imgs)
        logits = logits_of(out).float()
        prob = F.softmax(logits, 1)
        probs.append(prob.cpu().numpy()); preds.extend(prob.argmax(1).cpu().tolist()); ys.extend(y.tolist())
        if isinstance(out, dict) and out.get('r') is not None:
            R_buf.append(out['r'].cpu())
    nv = None
    if R_buf:
        R = torch.cat(R_buf); basis = fit_basis(R[: min(4096, len(R))])
        if basis is not None:
            proj = R @ basis @ basis.t()
            nv = ((R - proj).norm(dim=1) / (R.norm(dim=1) + 1e-9)).numpy()
    return compute_metrics(ys, preds, np.concatenate(probs), num_classes, risk, unknown, nv)


def run_bn(model, loader, device):
    model.train()
    with torch.no_grad():
        for imgs, _ in loader:
            model(imgs.to(device))
    model.eval()


def run_tent(model, loader, device, num_classes, risk, unknown, lr=1e-4, use_sar=False,
             rho=0.05, use_risk=False, risk_w=None):
    for p in model.parameters():
        p.requires_grad = False
    params = norm_params(model)
    for p in params:
        p.requires_grad = True
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    model.eval()
    ys, preds, probs, novs = [], [], [], []
    R_buf = []
    for imgs, y in loader:
        imgs = imgs.to(device)
        for _ in range(1):
            logits = logits_of(model(imgs)).float().clamp(-30, 30)
            prob = F.softmax(logits, 1)
            ent = -(prob * torch.log(prob + 1e-12)).sum(1)
            if use_risk and risk_w is not None:
                ent = ent * risk_w[prob.argmax(1)]
            if use_sar:
                kk = max(1, int(0.3 * len(ent)))
                th = torch.sort(ent)[0][kk - 1]
                mask = (ent <= th).float()
            else:
                mask = torch.ones_like(ent)
            loss = (ent * mask).sum() / (mask.sum() + 1e-9)
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                if use_sar:
                    grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                    norm = torch.sqrt(sum((g ** 2).sum() for g in grads)) + 1e-12
                    with torch.no_grad():
                        for p, g in zip(params, grads):
                            p.add_(rho * g / norm)
                    opt.zero_grad()
                    logits2 = logits_of(model(imgs)).float().clamp(-30, 30)
                    prob2 = F.softmax(logits2, 1)
                    ent2 = -(prob2 * torch.log(prob2 + 1e-12)).sum(1)
                    if use_risk and risk_w is not None:
                        ent2 = ent2 * risk_w[prob2.argmax(1)]
                    (ent2 * mask).sum().div(mask.sum() + 1e-9).backward()
                    with torch.no_grad():
                        for p, g in zip(params, grads):
                            p.sub_(rho * g / norm)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
        with torch.no_grad():
            out = model(imgs)
        logits = logits_of(out).float()
        prob = F.softmax(logits, 1)
        probs.append(prob.cpu().numpy()); preds.extend(prob.argmax(1).cpu().tolist()); ys.extend(y.tolist())
        if isinstance(out, dict) and out.get('r') is not None:
            R_buf.append(out['r'].cpu())
    nv = None
    if R_buf:
        R = torch.cat(R_buf); basis = fit_basis(R[: min(4096, len(R))])
        if basis is not None:
            proj = R @ basis @ basis.t()
            nv = ((R - proj).norm(dim=1) / (R.norm(dim=1) + 1e-9)).numpy()
    return compute_metrics(ys, preds, np.concatenate(probs), num_classes, risk, unknown, nv)


# ---------------- BBSE 类先验校正 ----------------
def source_confusion(model, src_loader, num_classes, device):
    """源域混淆矩阵 C_cond[pred,true] = P(pred|true) 与源预测先验 q_src"""
    C = np.zeros((num_classes, num_classes), dtype=np.float64)
    with torch.no_grad():
        for imgs, y in src_loader:
            out = logits_of(model(imgs.to(device))).float().argmax(1).cpu().numpy()
            for t, pr in zip(y.numpy(), out):
                if 0 <= t < num_classes:
                    C[int(pr), int(t)] += 1
    col = np.maximum(C.sum(0, keepdims=True), 1.0)
    C_cond = C / col
    q_src = C.sum(1) / max(1.0, C.sum())
    return C_cond, q_src


def estimate_target_prior(probs, C_cond):
    q_pred = np.asarray(probs, dtype=np.float64).mean(0)
    q_true = np.linalg.pinv(C_cond) @ q_pred
    q_true = np.clip(q_true, 1e-6, None)
    return q_true / q_true.sum()


def collect_probs(model, loader, device, num_classes):
    ys, probs, novs = [], [], []
    R_buf = []
    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            prob = F.softmax(logits_of(out).float(), 1)
            probs.append(prob.cpu().numpy()); ys.extend(y.tolist())
            if isinstance(out, dict) and out.get('r') is not None:
                R_buf.append(out['r'].cpu())
    nv = None
    if R_buf:
        R = torch.cat(R_buf)
        basis = fit_basis(R[: min(4096, len(R))])
        if basis is not None:
            proj = R @ basis @ basis.t()
            nv = ((R - proj).norm(dim=1) / (R.norm(dim=1) + 1e-9)).numpy()
    return np.array(ys), np.concatenate(probs), nv


def apply_prior_correction(probs, q_true, q_src, alpha=1.0):
    logits = np.log(np.clip(probs, 1e-12, 1.0)) + alpha * np.log(np.clip(q_true / np.maximum(q_src, 1e-9), 1e-9, 1e9))
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def run_bn_pc(model, loader, device, num_classes, risk, unknown, C_cond=None, q_src=None, alpha=1.0):
    """BN 统计量重估 + BBSE 类先验校正"""
    model.train()
    with torch.no_grad():
        for imgs, _ in loader:
            model(imgs.to(device))
    model.eval()
    ys, probs, nv = collect_probs(model, loader, device, num_classes)
    if C_cond is not None and q_src is not None:
        q_true = estimate_target_prior(probs, C_cond)
        probs2 = apply_prior_correction(probs, q_true, q_src, alpha)
    else:
        probs2 = probs
    preds = probs2.argmax(1).tolist()
    res = compute_metrics(ys, preds, probs2, num_classes, risk, unknown, nv)
    if C_cond is not None:
        res['prior_src'] = [round(float(x), 4) for x in q_src]
        res['prior_est'] = [round(float(x), 4) for x in q_true]
    return res


def run_tent_pc(model, loader, device, num_classes, risk, unknown, C_cond=None, q_src=None,
                lr=1e-4, alpha=1.0):
    """TENT 适配后再做类先验校正"""
    for p in model.parameters():
        p.requires_grad = False
    params = norm_params(model)
    for p in params:
        p.requires_grad = True
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    model.eval()
    for imgs, y in loader:
        imgs = imgs.to(device)
        logits = logits_of(model(imgs)).float().clamp(-30, 30)
        prob = F.softmax(logits, 1)
        ent = -(prob * torch.log(prob + 1e-12)).sum(1)
        loss = ent.mean()
        opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
    ys, probs, nv = collect_probs(model, loader, device, num_classes)
    if C_cond is not None and q_src is not None:
        q_true = estimate_target_prior(probs, C_cond)
        probs2 = apply_prior_correction(probs, q_true, q_src, alpha)
    else:
        probs2 = probs
    preds = probs2.argmax(1).tolist()
    res = compute_metrics(ys, preds, probs2, num_classes, risk, unknown, nv)
    if C_cond is not None:
        res['prior_est'] = [round(float(x), 4) for x in q_true]
    return res


def run_adapt2(model, ds, device, num_classes, risk, unknown, calib_per_class=60,
               alpha=1.0, C_cond=None, q_src=None, bs=64, workers=4, rest_loader=None, seed=0):
    """安全适配 v2：在较大校准集上比较 BN 校准 + 先验校正 vs 源模型，择优采用；其余样本评测"""
    import torch.utils.data as tud
    labels = np.array([int(v) if pd.notna(v) else -1 for v in ds.df[ds.label_col].values])
    rng = np.random.RandomState(seed)
    calib_idx, rest_idx = [], []
    for c in range(num_classes):
        idx_c = np.nonzero(labels == c)[0]
        n_take = min(calib_per_class, len(idx_c))
        take = rng.choice(idx_c, n_take, replace=False).tolist()
        calib_idx += take
        rest_idx += np.setdiff1d(idx_c, take).tolist()
    calib_eval = DataLoader(tud.Subset(ds, calib_idx), batch_size=64, shuffle=False, num_workers=0)
    if rest_loader is None:
        rest_loader = DataLoader(tud.Subset(ds, rest_idx), batch_size=bs, shuffle=False, num_workers=workers)
    log('  [ADAPT2] 校准集 %d / 评测集 %d' % (len(calib_idx), len(rest_idx)))

    # 源模型校准集指标
    ys_c, prob_c, _ = collect_probs(model, calib_eval, device, num_classes)
    src_acc = float((prob_c.argmax(1) == ys_c).mean())
    src_f1 = float(f1_score(ys_c, prob_c.argmax(1), average='macro', labels=sorted(set(ys_c.tolist())), zero_division=0))

    # 候选 1：BN 校准（统计量重估）
    for p in model.parameters():
        p.requires_grad = False
    params = norm_params(model)
    snap_full = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.train()
    with torch.no_grad():
        for imgs, _ in calib_eval:
            model(imgs.to(device))
    model.eval()
    ys_c2, prob_c2, _ = collect_probs(model, calib_eval, device, num_classes)
    if C_cond is not None and q_src is not None:
        q_true_c = estimate_target_prior(prob_c2, C_cond)
        prob_c2 = apply_prior_correction(prob_c2, q_true_c, q_src, alpha)
    bn_acc = float((prob_c2.argmax(1) == ys_c2).mean())
    bn_f1 = float(f1_score(ys_c2, prob_c2.argmax(1), average='macro', labels=sorted(set(ys_c2.tolist())), zero_division=0))
    log('  [ADAPT2] 校准集: 源 acc=%.4f F1=%.4f | BN+PC acc=%.4f F1=%.4f' % (src_acc, src_f1, bn_acc, bn_f1))

    # 采用判据：校准集准确率不降（+1%）且 F1 不降，才认为适配有益
    adopted = (bn_acc > src_acc + 0.01) and (bn_f1 >= src_f1 - 0.005)
    if not adopted:
        model.load_state_dict(snap_full)
    log('  [ADAPT2] 决策: %s' % ('采用 BN+先验校正' if adopted else '放弃适配（源模型）'))

    ys, probs, nv = collect_probs(model, rest_loader, device, num_classes)
    if adopted and C_cond is not None and q_src is not None:
        q_true = estimate_target_prior(probs, C_cond)
        probs = apply_prior_correction(probs, q_true, q_src, alpha)
    res = compute_metrics(ys, probs.argmax(1).tolist(), probs, num_classes, risk, unknown, nv)
    res['calib_src_f1'] = src_f1
    res['calib_bn_f1'] = bn_f1
    res['adopted'] = int(adopted)
    res['n_calib'] = len(calib_idx)
    return res


def run_ran_safe(model, loader, device, num_classes, risk, unknown, lr=1e-4, k_basis=8,
                 warmup_batches=8, gate_quantile=0.7, kl_thresh=0.6, ent_thresh=1.02,
                 warmup_epochs_monitor=4):
    """安全版：先在目标流上做少量自适应并监控（预测分布 KL + 熵），一旦恶化立即回滚到源模型。"""
    for p in model.parameters():
        p.requires_grad = False
    params = norm_params(model)
    for p in params:
        p.requires_grad = True
    src_bn = [p.detach().clone() for p in params]
    rw = torch.tensor(risk, dtype=torch.float32, device=device) if risk else None

    # ---- 监控阶段：收集源预测分布 & 熵 ----
    R_buf, hist_q, ents0 = [], torch.zeros(num_classes), []
    it = iter(loader)
    for _ in range(warmup_batches):
        try:
            imgs, y = next(it)
        except StopIteration:
            break
        imgs = imgs.to(device)
        with torch.no_grad():
            out = model(imgs)
        logits = logits_of(out).float()
        prob = F.softmax(logits, 1)
        hist_q += prob.mean(0).cpu()
        ents0.append(float(-(prob * torch.log(prob + 1e-12)).sum(1).mean()))
        if isinstance(out, dict) and out.get('r') is not None:
            R_buf.append(out['r'].cpu())
    hist_q = (hist_q / max(1, len(ents0))).clamp_min(1e-6)
    e0 = float(np.mean(ents0)) if ents0 else 1.0
    basis = fit_basis(torch.cat(R_buf), k_basis) if R_buf else None
    log('  [SAFE] 源熵=%.3f, 偏移子空间 k=%d' % (e0, basis.shape[1] if basis is not None else 0))

    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    model.eval()
    M = None; cnt = None
    hist_p = hist_q.clone()
    aborted = False
    n_steps = 0
    for imgs, y in loader:
        if aborted:
            break
        imgs = imgs.to(device)
        out = model(imgs)
        logits = logits_of(out).float().clamp(-30, 30)
        prob = F.softmax(logits, 1)
        z = out['feat'].detach() if isinstance(out, dict) else None
        nov = None
        if isinstance(out, dict) and out.get('r') is not None and basis is not None:
            R = out['r'].detach().cpu()
            proj = R @ basis @ basis.t()
            nov = ((R - proj).norm(dim=1) / (R.norm(dim=1) + 1e-9))
        gate = (nov.to(device) <= float(torch.quantile(nov, gate_quantile))).float() if nov is not None else \
               torch.ones(imgs.size(0), device=device)
        if use_mem := True:
            if M is not None and z is not None:
                zn = F.normalize(z, dim=1); Mn = F.normalize(M, dim=1).to(device)
                prob_f = F.softmax(logits + 1.0 * (zn @ Mn.t()), 1)
            else:
                prob_f = prob
        else:
            prob_f = prob
        ent = -(prob * torch.log(prob + 1e-12)).sum(1)
        if rw is not None:
            ent_w = ent * rw[prob.argmax(1)]
        else:
            ent_w = ent
        loss = (ent_w * gate).sum() / (gate.sum() + 1e-9)
        if loss.requires_grad and torch.isfinite(loss) and float(gate.sum()) > 0:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        n_steps += 1
        # ---- 监控：预测分布 KL / 熵 ----
        with torch.no_grad():
            hist_p = 0.7 * hist_p + 0.3 * prob.detach().mean(0).cpu()
        kl = float((hist_p * (hist_p / hist_q).log()).sum())
        if n_steps >= warmup_epochs_monitor:
            with torch.no_grad():
                ent_now = float(ent.mean())
            if kl > kl_thresh or ent_now > e0 * ent_thresh:
                log('  [SAFE] 触发回退: KL=%.3f 熵 %.3f→%.3f' % (kl, e0, ent_now))
                aborted = True
                break
        # 收集记忆
        if z is not None:
            zc = z.cpu()
            if M is None:
                M = torch.zeros(num_classes, zc.shape[1]); cnt = torch.zeros(num_classes)
            conf, pseudo = prob.detach().max(1)
            sel = ((gate > 0) & (conf > 0.9)).cpu()
            for i in torch.nonzero(sel).flatten().tolist():
                c = int(pseudo[i]); cnt[c] += 1
                M[c] += (zc[i] - M[c]) / cnt[c]

    if aborted:
        # 回滚到源参数，按 source 方式完整评测一遍
        with torch.no_grad():
            for p_, s_ in zip(params, src_bn):
                p_.copy_(s_)
        log('  [SAFE] 已回滚，按源模型评测')
        return run_source(model, loader, device, num_classes, risk, unknown)

    # 正常适配：完整跑一遍评测（与 ran_tta 一致）
    opt = torch.optim.SGD(params, lr=0.0)  # 冻结
    ys, preds, probs, novs = [], [], [], []
    for imgs, y in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        logits = logits_of(out).float().clamp(-30, 30)
        z = out['feat'].detach() if isinstance(out, dict) else None
        prob = F.softmax(logits, 1)
        if M is not None and z is not None:
            zn = F.normalize(z, dim=1); Mn = F.normalize(M, dim=1).to(device)
            prob_f = F.softmax(logits + 1.0 * (zn @ Mn.t()), 1)
        else:
            prob_f = prob
        if isinstance(out, dict) and out.get('r') is not None and basis is not None:
            R = out['r'].detach().cpu(); proj = R @ basis @ basis.t()
            novs.append(((R - proj).norm(dim=1) / (R.norm(dim=1) + 1e-9)))
        probs.append(prob_f.detach().cpu().numpy()); preds.extend(prob_f.argmax(1).cpu().tolist()); ys.extend(y.tolist())
    nv = torch.cat(novs).numpy() if novs else None
    return compute_metrics(ys, preds, np.concatenate(probs), num_classes, risk, unknown, nv)


def _bn_snapshot(params):
    return [p.detach().clone() for p in params]


def _bn_restore(params, snap):
    with torch.no_grad():
        for p, s_ in zip(params, snap):
            p.copy_(s_)


def run_cotta(model, loader, device, num_classes, risk, unknown, lr=1e-4, alpha=0.999, conf_thresh=0.6):
    """CoTTA（简化版）：EMA 教师 + 置信伪标签 CE + 随机水平翻转/颜色增强一致性"""
    import copy
    teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    aug = T.Compose([T.ColorJitter(0.3, 0.3, 0.2, 0.02), T.RandomHorizontalFlip()])
    model.eval()
    ys, preds, probs, novs = [], [], [], []
    R_buf = []
    for imgs, y in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            t_logits = logits_of(teacher(imgs)).float()
            t_prob = F.softmax(t_logits, 1)
            conf, pseudo = t_prob.max(1)
            mask = (conf > conf_thresh).float()
        aug_imgs = torch.stack([aug(x.cpu()) for x in imgs]).to(device)
        logits = logits_of(model(aug_imgs)).float().clamp(-30, 30)
        loss = (F.cross_entropy(logits, pseudo, reduction='none') * mask).sum() / (mask.sum() + 1e-9)
        opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), model.parameters()):
                tp.mul_(alpha).add_(sp.detach(), alpha=1 - alpha)
            out = teacher(imgs)
        prob = F.softmax(logits_of(out).float(), 1)
        probs.append(prob.cpu().numpy()); preds.extend(prob.argmax(1).cpu().tolist()); ys.extend(y.tolist())
        if isinstance(out, dict) and out.get('r') is not None:
            R_buf.append(out['r'].cpu())
    nv = None
    if R_buf:
        R = torch.cat(R_buf); basis = fit_basis(R[: min(4096, len(R))])
        if basis is not None:
            proj = R @ basis @ basis.t()
            nv = ((R - proj).norm(dim=1) / (R.norm(dim=1) + 1e-9)).numpy()
    return compute_metrics(ys, preds, np.concatenate(probs), num_classes, risk, unknown, nv)


def run_eata(model, loader, device, num_classes, risk, unknown, lr=1e-4, fisher=1.0):
    """EATA-lite：熵最小化 + 低熵样本筛选 + 向源参数 L2 正则（防遗忘）"""
    for p in model.parameters():
        p.requires_grad = False
    params = norm_params(model)
    for p in params:
        p.requires_grad = True
    src = [p.detach().clone() for p in params]
    n_par = sum(p.numel() for p in params)
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    model.eval()
    ys, preds, probs, novs = [], [], [], []
    R_buf = []
    for imgs, y in loader:
        imgs = imgs.to(device)
        logits = logits_of(model(imgs)).float().clamp(-30, 30)
        prob = F.softmax(logits, 1)
        ent = -(prob * torch.log(prob + 1e-12)).sum(1)
        # 低熵样本（可信）参与适配
        kk = max(1, int(0.5 * len(ent)))
        th = torch.sort(ent)[0][kk - 1]
        sel = ent <= th
        loss = ent[sel].mean() if sel.sum() else ent.mean()
        # 防遗忘：L2 正则
        reg = sum(((p - s_) ** 2).sum() for p, s_ in zip(params, src)) / max(1, n_par)
        loss = loss + fisher * 1e-3 * reg
        opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        with torch.no_grad():
            out = model(imgs)
        prob = F.softmax(logits_of(out).float(), 1)
        probs.append(prob.cpu().numpy()); preds.extend(prob.argmax(1).cpu().tolist()); ys.extend(y.tolist())
        if isinstance(out, dict) and out.get('r') is not None:
            R_buf.append(out['r'].cpu())
    nv = None
    if R_buf:
        R = torch.cat(R_buf); basis = fit_basis(R[: min(4096, len(R))])
        if basis is not None:
            proj = R @ basis @ basis.t()
            nv = ((R - proj).norm(dim=1) / (R.norm(dim=1) + 1e-9)).numpy()
    return compute_metrics(ys, preds, np.concatenate(probs), num_classes, risk, unknown, nv)


def run_deyo(model, loader, device, num_classes, risk, unknown, lr=1e-4, beta=0.9):
    """DeYO-lite：类平衡熵 + 熵阈值样本筛选 + 类别计数动量（防止塌到多数类）"""
    for p in model.parameters():
        p.requires_grad = False
    params = norm_params(model)
    for p in params:
        p.requires_grad = True
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    model.eval()
    cls_count = torch.ones(num_classes)
    ent_ema = None
    ys, preds, probs, novs = [], [], [], []
    R_buf = []
    for imgs, y in loader:
        imgs = imgs.to(device)
        logits = logits_of(model(imgs)).float().clamp(-30, 30)
        prob = F.softmax(logits, 1)
        conf, pseudo = prob.max(1)
        ent = -(prob * torch.log(prob + 1e-12)).sum(1)
        # 熵阈值筛选（DeYO 的 entropy-based filtering）
        cur = float(ent.mean())
        ent_ema = cur if ent_ema is None else beta * ent_ema + (1 - beta) * cur
        sel = ent <= ent_ema
        # 类平衡（逆频率）
        cls_count = beta * cls_count + (1 - beta) * torch.bincount(pseudo.cpu(), minlength=num_classes).float()
        freq = (cls_count + 1e-3) / (cls_count + 1e-3).sum()
        cw = (1.0 / (freq * num_classes)).clamp(0.2, 5.0).to(device)
        loss = (ent * cw[pseudo])
        loss = loss[sel].mean() if sel.sum() else loss.mean()
        opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        with torch.no_grad():
            out = model(imgs)
        prob = F.softmax(logits_of(out).float(), 1)
        probs.append(prob.cpu().numpy()); preds.extend(prob.argmax(1).cpu().tolist()); ys.extend(y.tolist())
        if isinstance(out, dict) and out.get('r') is not None:
            R_buf.append(out['r'].cpu())
    nv = None
    if R_buf:
        R = torch.cat(R_buf); basis = fit_basis(R[: min(4096, len(R))])
        if basis is not None:
            proj = R @ basis @ basis.t()
            nv = ((R - proj).norm(dim=1) / (R.norm(dim=1) + 1e-9)).numpy()
    return compute_metrics(ys, preds, np.concatenate(probs), num_classes, risk, unknown, nv)


def run_tent_pro(model, loader, device, num_classes, risk, unknown, lr=2.5e-4,
                 conf_thresh=0.5, use_prior=True):
    """强适配：置信度过滤 + 类平衡熵 + 目标先验重加权（相当于 EATA/DeYO 的轻量版）。"""
    for p in model.parameters():
        p.requires_grad = False
    params = norm_params(model)
    for p in params:
        p.requires_grad = True
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    model.eval()
    cls_count = torch.zeros(num_classes)
    ys, preds, probs, novs = [], [], [], []
    R_buf = []
    for imgs, y in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        logits = logits_of(out).float().clamp(-30, 30)
        prob = F.softmax(logits, 1)
        conf, pseudo = prob.max(1)
        # 类平衡权重（逆频率，防止塌到少数类）
        cls_count = 0.95 * cls_count + 0.05 * torch.bincount(pseudo.cpu(), minlength=num_classes).float()
        freq = (cls_count + 1.0) / (cls_count + 1.0).sum()
        cw = (1.0 / (freq * num_classes)).clamp(0.2, 5.0).to(device)
        ent = -(prob * torch.log(prob + 1e-12)).sum(1) * cw[pseudo]
        sel = conf > conf_thresh
        if sel.sum() == 0:
            sel = torch.ones_like(conf, dtype=torch.bool)
        loss = ent[sel].mean()
        opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        with torch.no_grad():
            out = model(imgs)
        logits = logits_of(out).float()
        prob = F.softmax(logits, 1)
        probs.append(prob.cpu().numpy()); preds.extend(prob.argmax(1).cpu().tolist()); ys.extend(y.tolist())
        if isinstance(out, dict) and out.get('r') is not None:
            R_buf.append(out['r'].cpu())
    nv = None
    if R_buf:
        R = torch.cat(R_buf); basis = fit_basis(R[: min(4096, len(R))])
        if basis is not None:
            proj = R @ basis @ basis.t()
            nv = ((R - proj).norm(dim=1) / (R.norm(dim=1) + 1e-9)).numpy()
    return compute_metrics(ys, preds, np.concatenate(probs), num_classes, risk, unknown, nv)


def run_adapt_abstain(model, ds, device, num_classes, risk, unknown, calib_per_class=20,
                      lr=2.5e-4, calib_rounds=2, delta=0.01, bs=64, workers=4):
    """先用少量目标域标注校准：适配后若在校准集上提升才采用，否则放弃适配（源模型）。"""
    import torch.utils.data as tud
    labels = np.array([int(v) if pd.notna(v) else -1 for v in ds.df[ds.label_col].values])
    calib_idx, rest_idx = [], []
    for c in range(num_classes):
        idx_c = np.nonzero(labels == c)[0]
        take = idx_c[:calib_per_class].tolist()
        calib_idx += take
        rest_idx += np.setdiff1d(idx_c, idx_c[:calib_per_class]).tolist()
    calib_loader = DataLoader(tud.Subset(ds, calib_idx), batch_size=32, shuffle=True, num_workers=workers)
    calib_eval = DataLoader(tud.Subset(ds, calib_idx), batch_size=64, shuffle=False, num_workers=0)
    if rest_loader is None:
        rest_loader = DataLoader(tud.Subset(ds, rest_idx), batch_size=bs, shuffle=False, num_workers=workers)
    log('  [ABSTAIN] 校准集 %d 条 / 待评测 %d 条' % (len(calib_idx), len(rest_idx)))

    for p in model.parameters():
        p.requires_grad = False
    params = norm_params(model)
    for p in params:
        p.requires_grad = True
    snap = _bn_snapshot(params)

    # 源模型在校准集上的准确率
    def eval_acc(loader):
        ys, ps = [], []
        with torch.no_grad():
            for imgs, y in loader:
                out = model(imgs.to(device))
                ps.extend(logits_of(out).float().argmax(1).cpu().tolist()); ys.extend(y.tolist())
        return float(np.mean(np.array(ys) == np.array(ps))), ys, ps

    acc_src, ys_c, _ = eval_acc(calib_eval)

    # 在校准集上做 tent_pro 适配
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    model.eval()
    cls_count = torch.zeros(num_classes)
    for rnd in range(calib_rounds):
        for imgs, y in calib_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            logits = logits_of(out).float().clamp(-30, 30)
            prob = F.softmax(logits, 1)
            conf, pseudo = prob.max(1)
            cls_count = 0.9 * cls_count + 0.1 * torch.bincount(pseudo.cpu(), minlength=num_classes).float()
            freq = (cls_count + 1.0) / (cls_count + 1.0).sum()
            cw = (1.0 / (freq * num_classes)).clamp(0.2, 5.0).to(device)
            ent = -(prob * torch.log(prob + 1e-12)).sum(1) * cw[pseudo]
            sel = conf > 0.5
            loss = ent[sel].mean() if sel.sum() else ent.mean()
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()

    acc_ad, _, _ = eval_acc(calib_eval)
    adopted = acc_ad > acc_src + delta
    log('  [ABSTAIN] 校准集 acc: 源=%.4f 适配后=%.4f → %s' % (acc_src, acc_ad, '采用适配' if adopted else '放弃适配'))
    if not adopted:
        _bn_restore(params, snap)

    res = run_source(model, rest_loader, device, num_classes, risk, unknown)
    res['calib_acc_src'] = acc_src
    res['calib_acc_adapted'] = acc_ad
    res['adopted'] = int(adopted)
    res['n_calib'] = len(calib_idx)
    return res


def run_ran_tta(model, loader, device, num_classes, risk, unknown, lr=1e-4, k_basis=8,
                warmup_batches=8, gate_quantile=0.7, use_mem=True, use_res=True,
                use_risk=True, beta=1.0):
    for p in model.parameters():
        p.requires_grad = False
    params = norm_params(model)
    for p in params:
        p.requires_grad = True
    rw = torch.tensor(risk, dtype=torch.float32, device=device) if (use_risk and risk) else None
    # ---- warmup：估计"域偏移子空间" ----
    R_buf = []
    it = iter(loader)
    for _ in range(warmup_batches):
        try:
            imgs, y = next(it)
        except StopIteration:
            break
        imgs = imgs.to(device)
        with torch.no_grad():
            out = model(imgs)
        if isinstance(out, dict) and out.get('r') is not None and use_res:
            R_buf.append(out['r'].cpu())
    basis = fit_basis(torch.cat(R_buf), k_basis) if R_buf else None
    log('  [RAN] 偏移子空间 k=%d（%d 个 warmup batch）' % (basis.shape[1] if basis is not None else 0, len(R_buf)))

    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    model.eval()
    # --- 防塌缩：源 BN 参数锚定 + 熵监控回滚 ---
    src_bn = [p.detach().clone() for p in params]
    ent_ema = None
    restore_ratio = 0.0
    M = None; cnt = None
    ys, preds, probs, novs = [], [], [], []
    tau = None
    for imgs, y in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        logits = logits_of(out).float().clamp(-30, 30)
        prob = F.softmax(logits, 1)
        z = out['feat'].detach() if isinstance(out, dict) else None
        nov = None
        if isinstance(out, dict) and out.get('r') is not None and basis is not None:
            R = out['r'].detach().cpu()
            proj = R @ basis @ basis.t()
            nov = ((R - proj).norm(dim=1) / (R.norm(dim=1) + 1e-9))
            novs.append(nov)
            if tau is None:
                tau = float(torch.quantile(nov, gate_quantile))
        gate = (nov.to(device) <= tau).float() if nov is not None else torch.ones(imgs.size(0), device=device)
        # ---- 受保护记忆增强预测（T3A 式，只使用历史低新异高置信样本）----
        if use_mem and M is not None and z is not None:
            zn = F.normalize(z, dim=1); Mn = F.normalize(M, dim=1).to(device)
            logits_f = logits + beta * (zn @ Mn.t())
            prob_f = F.softmax(logits_f, 1)
        else:
            prob_f = prob
        # ---- 风险加权熵最小化（门控后）----
        ent = -(prob * torch.log(prob + 1e-12)).sum(1)
        if rw is not None:
            ent = ent * rw[prob.argmax(1)]
        loss = (ent * gate).sum() / (gate.sum() + 1e-9)
        if loss.requires_grad and torch.isfinite(loss) and float(gate.sum()) > 0:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            # 防塌缩：若整批平均熵持续上升，则向源参数回滚一部分
            with torch.no_grad():
                e_now = float(ent.mean())
                if ent_ema is None:
                    ent_ema = e_now
                if e_now > ent_ema * 1.05:
                    restore_ratio = min(0.8, restore_ratio + 0.2)
                else:
                    restore_ratio = max(0.0, restore_ratio - 0.1)
                    ent_ema = 0.9 * ent_ema + 0.1 * e_now
                if restore_ratio > 0:
                    for p_, s_ in zip(params, src_bn):
                        p_.mul_(1 - restore_ratio).add_(s_, alpha=restore_ratio)
        probs.append(prob_f.detach().cpu().numpy()); preds.extend(prob_f.argmax(1).cpu().tolist()); ys.extend(y.tolist())
        # ---- 更新记忆 ----
        if use_mem and z is not None:
            zc = z.cpu()
            if M is None:
                M = torch.zeros(num_classes, zc.shape[1]); cnt = torch.zeros(num_classes)
            conf, pseudo = prob.detach().max(1)
            sel = ((gate > 0) & (conf > 0.9)).cpu()
            for i in torch.nonzero(sel).flatten().tolist():
                c = int(pseudo[i]); cnt[c] += 1
                M[c] += (zc[i] - M[c]) / cnt[c]
    nv = torch.cat(novs).numpy() if novs else None
    return compute_metrics(ys, preds, np.concatenate(probs), num_classes, risk, unknown, nv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/private/zdy/deepseek_road')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--kind', default='mt', choices=['mt', 'plain'])
    ap.add_argument('--manifest', default='manifests/multitask_all.csv')
    ap.add_argument('--label-col', default='label15')
    ap.add_argument('--num-classes', type=int, default=15)
    ap.add_argument('--track', default='track1')
    ap.add_argument('--split', default='test')
    ap.add_argument('--domains', default='')
    ap.add_argument('--datasets', default='')
    ap.add_argument('--methods', default='source,ran_tta')
    ap.add_argument('--risk', default='')
    ap.add_argument('--unknown-classes', default='')
    ap.add_argument('--calib-per-class', type=int, default=60)
    ap.add_argument('--source-datasets', default='')
    ap.add_argument('--pc-alpha', type=float, default=1.0)
    ap.add_argument('--calib-split', type=int, default=0)
    ap.add_argument('--calib-seed', type=int, default=0)
    ap.add_argument('--risk-lam', type=float, default=0.2)
    ap.add_argument('--max-samples', type=int, default=0)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    global RISK_LAM
    RISK_LAM = args.risk_lam
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')

    man = os.path.join(args.root, args.manifest)
    df = pd.read_csv(man, low_memory=False)
    if 'track' in df.columns:
        df = df[df['track'] == args.track]
    df = df[df['split'] == args.split]
    if args.domains:
        df = df[df['domain'].isin(args.domains.split(','))]
    if args.datasets:
        df = df[df['dataset'].isin(args.datasets.split(','))]
    df = df[pd.to_numeric(df[args.label_col], errors='coerce').notna()].reset_index(drop=True)
    if args.max_samples:
        df = df.sample(min(args.max_samples, len(df)), random_state=0).reset_index(drop=True)
    log('评测 %d 条 (domains=%s datasets=%s)' % (len(df), args.domains, args.datasets))

    model, ck = load_model(args.ckpt, args.kind, args.num_classes, device)
    C_cond = q_src = None
    need_pc = any(x in args.methods for x in ('bn_pc', 'tent_pc', 'adapt2'))
    if need_pc:
        src_df = pd.read_csv(man, low_memory=False)
        if 'track' in src_df.columns:
            src_df = src_df[src_df['track'] == args.track]
        src_df = src_df[src_df['split'].isin(['val', 'validation'])]
        if args.source_datasets:
            src_df = src_df[src_df['dataset'].isin(args.source_datasets.split(','))]
        src_df = src_df[pd.to_numeric(src_df[args.label_col], errors='coerce').notna()].reset_index(drop=True)
        log('源域校准数据 %d 条（datasets=%s）' % (len(src_df), args.source_datasets or 'val 全部'))
        src_loader = DataLoader(DS(src_df, args.root, tf_eval(224), args.label_col),
                                batch_size=128, num_workers=args.workers)
        C_cond, q_src = source_confusion(model, src_loader, args.num_classes, device)
        log('源混淆矩阵已估计，源先验=%s' % [round(float(x), 3) for x in q_src])
    risk = [float(x) for x in args.risk.split(',')] if args.risk else None
    unknown = [int(x) for x in args.unknown_classes.split(',')] if args.unknown_classes else None
    risk_w = None
    if risk:
        rw0 = torch.tensor(risk, dtype=torch.float32, device=device)
        risk_w = (rw0 / rw0.max()).clamp(0.2, 1.0)

    ds = DS(df, args.root, tf_eval(224), args.label_col)
    if args.calib_split:
        import torch.utils.data as tud
        labels = np.array([int(v) if pd.notna(v) else -1 for v in ds.df[args.label_col].values])
        rng = np.random.RandomState(args.calib_seed)
        calib_idx, rest_idx = [], []
        for c in range(args.num_classes):
            idx_c = np.nonzero(labels == c)[0]
            n_take = min(args.calib_per_class, len(idx_c))
            take = rng.choice(idx_c, n_take, replace=False).tolist()
            calib_idx += take
            rest_idx += np.setdiff1d(idx_c, take).tolist()
        rest_loader = DataLoader(tud.Subset(ds, rest_idx), batch_size=args.bs, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
        calib_loader = DataLoader(tud.Subset(ds, calib_idx), batch_size=args.bs, shuffle=False,
                                  num_workers=0)
        loader = rest_loader
        log('校准/测试划分: 校准 %d 条, 测试 %d 条（公平比较用测试集）' % (len(calib_idx), len(rest_idx)))
    else:
        loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
    results = {}
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for m in [x.strip() for x in args.methods.split(',') if x.strip()]:
        model.load_state_dict(base_state)
        log('>>> 方法: %s' % m)
        t0 = time.time()
        if m == 'source':
            res = run_source(model, loader, device, args.num_classes, risk, unknown)
        elif m == 'bn':
            run_bn(model, loader, device)
            res = run_source(model, loader, device, args.num_classes, risk, unknown)
        elif m == 'tent':
            res = run_tent(model, loader, device, args.num_classes, risk, unknown, lr=args.lr)
        elif m in ('bn_pc', 'tent_pc', 'adapt2'):
            if m == 'bn_pc':
                res = run_bn_pc(model, loader, device, args.num_classes, risk, unknown, C_cond, q_src, args.pc_alpha)
            elif m == 'tent_pc':
                res = run_tent_pc(model, loader, device, args.num_classes, risk, unknown, C_cond, q_src, args.lr, args.pc_alpha)
            else:
                if args.calib_split:
                    res = run_adapt2(model, ds, device, args.num_classes, risk, unknown,
                                     calib_per_class=args.calib_per_class, alpha=args.pc_alpha,
                                     C_cond=C_cond, q_src=q_src, rest_loader=rest_loader, seed=args.calib_seed)
                else:
                    res = run_adapt2(model, ds, device, args.num_classes, risk, unknown,
                                     calib_per_class=args.calib_per_class, alpha=args.pc_alpha,
                                     C_cond=C_cond, q_src=q_src, seed=args.calib_seed)
        elif m == 'cotta':
            res = run_cotta(model, loader, device, args.num_classes, risk, unknown, lr=args.lr)
        elif m == 'eata':
            res = run_eata(model, loader, device, args.num_classes, risk, unknown, lr=args.lr)
        elif m == 'deyo':
            res = run_deyo(model, loader, device, args.num_classes, risk, unknown, lr=args.lr)
        elif m == 'tent_pro':
            res = run_tent_pro(model, loader, device, args.num_classes, risk, unknown, lr=2.5e-4)
        elif m == 'adapt_abstain':
            res = run_adapt_abstain(model, ds, device, args.num_classes, risk, unknown,
                                    calib_per_class=getattr(args, 'calib_per_class', 20), lr=2.5e-4)
        elif m == 'tent_risk':
            res = run_tent(model, loader, device, args.num_classes, risk, unknown, lr=args.lr,
                           use_risk=True, risk_w=risk_w)
        elif m == 'sar':
            res = run_tent(model, loader, device, args.num_classes, risk, unknown, lr=args.lr, use_sar=True)
        elif m == 'ran_safe':
            res = run_ran_safe(model, loader, device, args.num_classes, risk, unknown, lr=args.lr)
        elif m.startswith('ran'):
            res = run_ran_tta(model, loader, device, args.num_classes, risk, unknown, lr=args.lr,
                              gate_quantile=(1.0 if m == 'ran_nogate' else 0.7),
                              use_mem=(m != 'ran_nomem'), use_res=(m != 'ran_nores'),
                              use_risk=(m != 'ran_norisk'))
        else:
            log('未知方法 %s' % m); continue
        res['time'] = round(time.time() - t0, 1)
        results[m] = res
        log('  %-10s acc=%.4f F1=%.4f risk=%.4f ECE=%.4f%s%s' % (
            m, res['acc'], res['macro_f1'], res.get('risk_cost', 0), res.get('ece', 0),
            (' AUROC_e=%.3f' % res['auroc_energy']) if res.get('auroc_energy') else '',
            (' AUROC_nov=%.3f' % res['auroc_novelty']) if res.get('auroc_novelty') else ''))
    with open(os.path.join(args.out, 'ran_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(args=vars(args), results=results), f, ensure_ascii=False, indent=1)
    log('完成 -> %s' % args.out)


if __name__ == '__main__':
    main()
