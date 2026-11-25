# -*- coding: utf-8 -*-
"""
构建统一数据清单（manifests）：
  Track-1（欧洲 BEV）：RoadSaW-150_l(12类) + RoadSC(3雪类) = 15 类；CtD1-3 由 wfh.dat 推导 12 类标签
  Track-2（中国前视）：新疆天山 6 类 + 清华北京 27→6 类

输出到 /private/zdy/deepseek_road/manifests/：
  track1_roadsaw.csv / track1_roadsc.csv / track1_ctd.csv
  track2_xinjiang.csv / track2_tsinghua_train.csv / track2_tsinghua_testseq.csv
  label_map.json / stats.json
"""
import os, sys, csv, json, glob, re
from collections import Counter, defaultdict

ROOT = os.environ.get('RSC_ROOT', '/private/zdy/deepseek_road')
EX = os.path.join(ROOT, 'data', 'extracted')
OUT = os.path.join(ROOT, 'manifests')
os.makedirs(OUT, exist_ok=True)

# ---------------- 统一标签空间 ----------------
U6_NAMES = ['Dry', 'Wet', 'Water', 'Snow', 'Slush', 'Ice']
U6 = {n: i for i, n in enumerate(U6_NAMES)}
RISK = {'Dry': 1, 'Wet': 3, 'Water': 7, 'Snow': 9, 'Slush': 8, 'Ice': 10}

# RoadSaW12 类（按"材质×湿度"编码为 track1 的 0-11）
RSC15_NAMES = [
    'asphalt_dry', 'asphalt_damp', 'asphalt_wet', 'asphalt_verywet',
    'cobble_dry', 'cobble_damp', 'cobble_wet', 'cobble_verywet',
    'concrete_dry', 'concrete_damp', 'concrete_wet', 'concrete_verywet',
    'snow_fresh', 'snow_fully_packed', 'snow_partially_covered',
]
RSC15 = {n: i for i, n in enumerate(RSC15_NAMES)}

# RoadSaW 文件夹名 -> (surface, wetness)
RS_FOLDER = {}
for surf in ['asphalt', 'cobble', 'concrete']:
    for wet in ['dry', 'damp', 'wet', 'verywet']:
        RS_FOLDER[f'{surf}_{wet}'] = (surf, wet)

# 水膜高度阈值（µm，RoadSaW 论文 Table 2）
THRESH = {
    'asphalt':   [(10, 'dry'), (25, 'damp'), (50, 'wet'), (10**9, 'verywet')],
    'cobblestone': [(10, 'dry'), (25, 'damp'), (75, 'wet'), (10**9, 'verywet')],
    'concrete':  [(10, 'dry'), (60, 'damp'), (200, 'wet'), (10**9, 'verywet')],
}

def wetness_from_wfh(surface, h):
    for th, name in THRESH[surface]:
        if h < th:
            return name
    return 'verywet'

def wetness_to_u6(wet):
    return {'dry': 'Dry', 'damp': 'Wet', 'wet': 'Wet', 'verywet': 'Water'}[wet]

def rs_folder_to_15(folder):
    surf, wet = RS_FOLDER[folder]
    key = f'{surf}_{wet}'
    return RSC15[key]

def rs_folder_to_6(folder):
    surf, wet = RS_FOLDER[folder]
    return wetness_to_u6(wet)

# 清华 27 类 -> 统一 6 类
def tsinghua_to_u6(name):
    n = name.lower().replace('-', '_')
    if n.startswith('dry'):    return 'Dry'
    if n.startswith('wet'):    return 'Wet'
    if n.startswith('water'):  return 'Water'
    if n.startswith('fresh'):  return 'Snow'
    if n.startswith('melted') or n.startswith('melt'): return 'Slush'
    if n.startswith('ice'):    return 'Ice'
    raise ValueError('unknown tsinghua class: %s' % name)

# 新疆 6 类 -> 统一 6 类（风吹雪单列为 Snow 的子类，open-set 时作为新类）
XJ_MAP = {
    'dry_road': ('Dry', None),
    'wet_road': ('Wet', None),
    'icy_road': ('Ice', None),
    'fully_snowy_road': ('Snow', None),
    'snow_melting_road': ('Slush', None),
    'snow_blowing_road': ('Snow', 'blowing'),   # 子类标记
}

def rel(p):
    return os.path.relpath(p, ROOT)

def write_csv(name, fieldnames, rows):
    p = os.path.join(OUT, name)
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('[CSV] %-32s %8d 行 -> %s' % (name, len(rows), p))
    return rows

stats = {}

# ================================================================ Track-1: RoadSaW
def build_roadsaw():
    base = os.path.join(EX, 'roadsaw', 'RoadSaW-150_l')
    rows = []
    for split in ['train', 'validation', 'test']:
        d = os.path.join(base, split)
        if not os.path.isdir(d):
            print('  [WARN] missing', d); continue
        for folder in sorted(os.listdir(d)):
            fd = os.path.join(d, folder)
            if not os.path.isdir(fd): continue
            if folder not in RS_FOLDER:
                print('  [WARN] unknown RoadSaW class folder:', folder); continue
            l15 = rs_folder_to_15(folder)
            u6 = rs_folder_to_6(folder)
            surf, wet = RS_FOLDER[folder]
            for fn in sorted(os.listdir(fd)):
                if not fn.lower().endswith('.png'): continue
                rows.append(dict(
                    path=rel(os.path.join(fd, fn)), dataset='roadsaw', domain='roadsaw',
                    split=split, orig_label=folder, label15=l15, label6=U6[u6], label6_name=u6,
                    surface=surf, wetness=wet, seq='', extra='',
                ))
    write_csv('track1_roadsaw.csv',
              ['path','dataset','domain','split','orig_label','label15','label6','label6_name','surface','wetness','seq','extra'],
              rows)
    stats['roadsaw'] = dict(total=len(rows),
                            by_split=dict(Counter(r['split'] for r in rows)),
                            by_label15=dict(Counter(str(r['label15']) for r in rows)),
                            by_label6=dict(Counter(r['label6_name'] for r in rows)))

def build_roadsc():
    base = find_base(os.path.join(EX, 'roadsc'))
    if base is None:
        print('  [WARN] 未找到 RoadSC 目录'); return
    name_map = {'fresh_snow': 'snow_fresh', 'fully_packed': 'snow_fully_packed',
                'partially_covered': 'snow_partially_covered'}
    rows = []
    for split in ['train', 'validation', 'test']:
        d = os.path.join(base, split)
        if not os.path.isdir(d):
            print('  [WARN] missing', d); continue
        for folder in sorted(os.listdir(d)):
            fd = os.path.join(d, folder)
            if not os.path.isdir(fd): continue
            if folder not in name_map:
                print('  [WARN] unknown RoadSC class folder:', folder); continue
            l15 = RSC15[name_map[folder]]
            for fn in sorted(os.listdir(fd)):
                if not fn.lower().endswith('.png'): continue
                rows.append(dict(
                    path=rel(os.path.join(fd, fn)), dataset='roadsc', domain='roadsc',
                    split=split, orig_label=folder, label15=l15, label6=U6['Snow'], label6_name='Snow',
                    surface='', wetness='snow', seq='', extra='',
                ))
    write_csv('track1_roadsc.csv',
              ['path','dataset','domain','split','orig_label','label15','label6','label6_name','surface','wetness','seq','extra'],
              rows)
    stats['roadsc'] = dict(total=len(rows),
                           by_split=dict(Counter(r['split'] for r in rows)),
                           by_label15=dict(Counter(str(r['label15']) for r in rows)))

# ================================================================ Track-1: CtD
def build_ctd():
    rows = []
    for ctd in ['CtD1', 'CtD2', 'CtD3']:
        base = os.path.join(EX, 'roadsaw', f'RoadSaW-{ctd}')
        if not os.path.isdir(base):
            print('  [WARN] missing', base); continue
        for seq in sorted(os.listdir(base)):
            seqdir = os.path.join(base, seq)
            if not os.path.isdir(seqdir): continue
            # 表面类型：surface-*.txt
            surf = None
            for fn in os.listdir(seqdir):
                m = re.match(r'surface-(asphalt|cobblestone|concrete)\.txt$', fn)
                if m: surf = m.group(1)
            if surf is None:
                print('  [WARN] no surface file in', seqdir); continue
            # wfh.dat：每行 "water_film_height[µm] speed[m/s]"
            wfh_path = os.path.join(seqdir, 'wfh.dat')
            wfhs = {}
            with open(wfh_path, encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f, start=1):   # i 为 1-based 行号
                    parts = line.split()
                    if len(parts) < 2: continue
                    try:
                        wfhs[i] = (float(parts[0]), float(parts[1]))
                    except ValueError:
                        continue
            imgdir = os.path.join(seqdir, 'test')
            if not os.path.isdir(imgdir):
                print('  [WARN] no test dir in', seqdir); continue
            for fn in sorted(os.listdir(imgdir)):
                if not fn.lower().endswith('.jpg'): continue
                m = re.match(r'top_view_(\d+)\.jpg$', fn)
                if not m:
                    print('  [WARN] unexpected filename', fn); continue
                idx = int(m.group(1))
                if idx not in wfhs:
                    continue   # 该帧没有有效测量
                h, v = wfhs[idx]
                wet = wetness_from_wfh(surf, h)
                key = '%s_%s' % ('cobble' if surf == 'cobblestone' else surf, wet)
                l15 = RSC15[key]
                u6 = wetness_to_u6(wet)
                rows.append(dict(
                    path=rel(os.path.join(imgdir, fn)), dataset='ctd', domain=ctd.lower(),
                    split='test', orig_label='%s_%s' % (surf, wet), label15=l15, label6=U6[u6], label6_name=u6,
                    surface=surf, wetness=wet, seq=seq, extra='wfh=%.2f;speed=%.2f' % (h, v),
                ))
    write_csv('track1_ctd.csv',
              ['path','dataset','domain','split','orig_label','label15','label6','label6_name','surface','wetness','seq','extra'],
              rows)
    stats['ctd'] = dict(total=len(rows),
                        by_domain=dict(Counter(r['domain'] for r in rows)),
                        by_label15=dict(Counter(str(r['label15']) for r in rows)),
                        by_label6=dict(Counter(r['label6_name'] for r in rows)))

# ================================================================ Track-2: 新疆
def find_base(root):
    """在目录下递归找包含 train 子目录的数据集根目录（兼容多层嵌套）"""
    for dirpath, dirnames, _ in os.walk(root):
        if 'train' in dirnames and ('val' in dirnames or 'validation' in dirnames):
            return dirpath
    return None

def build_xinjiang():
    base = find_base(os.path.join(EX, 'xinjiang'))
    if base is None:
        print('  [WARN] 未找到新疆数据集目录'); return
    print('  新疆数据根目录:', base)
    rows = []
    for split in ['train', 'val', 'test']:
        d = os.path.join(base, split)
        if not os.path.isdir(d):
            print('  [WARN] missing', d); continue
        for folder in sorted(os.listdir(d)):
            fd = os.path.join(d, folder)
            if not os.path.isdir(fd): continue
            if folder not in XJ_MAP:
                print('  [WARN] unknown Xinjiang class:', folder); continue
            u6, sub = XJ_MAP[folder]
            for fn in sorted(os.listdir(fd)):
                if not fn.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                rows.append(dict(
                    path=rel(os.path.join(fd, fn)), dataset='xinjiang', domain='xinjiang',
                    split=split, orig_label=folder, label6=U6[u6], label6_name=u6,
                    risk=RISK[u6], subclass=sub or '', extra='',
                ))
    write_csv('track2_xinjiang.csv',
              ['path','dataset','domain','split','orig_label','label6','label6_name','risk','subclass','extra'],
              rows)
    stats['xinjiang'] = dict(total=len(rows),
                             by_split=dict(Counter(r['split'] for r in rows)),
                             by_label6=dict(Counter(r['label6_name'] for r in rows)),
                             by_orig=dict(Counter(r['orig_label'] for r in rows)))

# ================================================================ Track-2: 清华
TS_PATCH_COORDS = [(480, 200), (720, 200), (960, 200), (1200, 200),
                   (480, 560), (720, 560), (960, 560), (1200, 560)]
PATCH_W, PATCH_H = 240, 360

def build_tsinghua():
    base = os.path.join(EX, 'tsinghua', 'dataset')
    train_dir = os.path.join(base, 'train-set')
    test_dir = os.path.join(base, 'test-set')

    # ---- train-set：按"分钟组"哈希切 train/val/test（70/15/15），避免相邻帧泄漏
    rows = []
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print('  清华 train-set 类别数:', len(classes))
    for cls in classes:
        u6 = tsinghua_to_u6(cls)
        fd = os.path.join(train_dir, cls)
        for fn in sorted(os.listdir(fd)):
            if not fn.lower().endswith('.jpg'): continue
            m = re.match(r'(\d{15})', fn)
            group = (m.group(1)[:12] if m else 'unknown')   # 精确到分钟
            rows.append(dict(
                path=rel(os.path.join(fd, fn)), dataset='tsinghua', domain='tsinghua',
                split='', orig_label=cls, label6=U6[u6], label6_name=u6, risk=RISK[u6],
                date=group, subclass='', extra='',
            ))

    import hashlib
    groups = sorted(set(r['date'] for r in rows))
    gsplit = {}
    for g in groups:
        h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % 100
        gsplit[g] = 'val' if h < 15 else ('test' if h < 30 else 'train')
    for r in rows:
        r['split'] = gsplit[r['date']]

    # 保证每个类在 val/test 至少各 1 个组（稀有类兜底：从 train 移组）
    def class_groups(cls):
        out = defaultdict(set)
        for r in rows:
            if r['orig_label'] == cls:
                out[r['split']].add(r['date'])
        return out
    for cls in classes:
        cg = class_groups(cls)
        for target in ['val', 'test']:
            if len(cg[target]) == 0:
                move = sorted(cg['train'])[0]
                gsplit[move] = target
                for r in rows:
                    if r['date'] == move: r['split'] = target
                print('  [FIX] %s: 补充 1 组到 %s' % (cls, target))
                cg = class_groups(cls)

    write_csv('track2_tsinghua_train.csv',
              ['path','dataset','domain','split','orig_label','label6','label6_name','risk','date','subclass','extra'],
              rows)
    by_cls_split = defaultdict(lambda: Counter())
    for r in rows: by_cls_split[r['orig_label']][r['split']] += 1
    stats['tsinghua_train'] = dict(total=len(rows),
                                   n_classes=len(classes), classes=classes,
                                   n_groups=len(groups),
                                   by_split=dict(Counter(r['split'] for r in rows)),
                                   by_label6=dict(Counter(r['label6_name'] for r in rows)),
                                   by_class_split={k: dict(v) for k, v in by_cls_split.items()})

    # ---- test-set：12 序列，每帧 8 个 patch，标签在同名 txt
    rows = []
    if os.path.isdir(test_dir):
        for seq in sorted(os.listdir(test_dir)):
            seqdir = os.path.join(test_dir, seq)
            if not os.path.isdir(seqdir): continue
            jpgs = sorted([f for f in os.listdir(seqdir) if f.lower().endswith('.jpg')])
            for fn in jpgs:
                stem = os.path.splitext(fn)[0]
                txt = os.path.join(seqdir, stem + '.txt')
                if not os.path.isfile(txt):
                    print('  [WARN] missing label txt:', txt); continue
                with open(txt, encoding='utf-8', errors='ignore') as f:
                    labels = f.read().strip().split()
                if len(labels) != 8:
                    print('  [WARN] %s labels=%d' % (txt, len(labels))); continue
                for pi, (raw, (x, y)) in enumerate(zip(labels, TS_PATCH_COORDS)):
                    cls = raw.replace('-', '_')
                    u6 = tsinghua_to_u6(cls)
                    rows.append(dict(
                        path=rel(os.path.join(seqdir, fn)), dataset='tsinghua', domain='tsinghua_testseq',
                        split='testseq', seq=seq, frame=stem, patch=pi,
                        x=x, y=y, w=PATCH_W, h=PATCH_H,
                        orig_label=raw, label6=U6[u6], label6_name=u6, risk=RISK[u6], extra='',
                    ))
    write_csv('track2_tsinghua_testseq.csv',
              ['path','dataset','domain','split','seq','frame','patch','x','y','w','h','orig_label','label6','label6_name','risk','extra'],
              rows)
    stats['tsinghua_testseq'] = dict(total=len(rows),
                                     by_label6=dict(Counter(r['label6_name'] for r in rows)),
                                     by_seq=dict(Counter(r['seq'] for r in rows)))

# ================================================================ 输出
def main():
    print('=' * 70)
    print('构建数据清单，输出目录:', OUT)
    print('=' * 70)
    for fn, f in [('RoadSaW', build_roadsaw), ('RoadSC', build_roadsc), ('CtD', build_ctd),
                  ('新疆', build_xinjiang), ('清华', build_tsinghua)]:
        print('\n--- %s ---' % fn)
        try:
            f()
        except Exception as e:
            print('  [ERROR]', fn, repr(e))
    with open(os.path.join(OUT, 'label_map.json'), 'w', encoding='utf-8') as fp:
        json.dump(dict(
            U6=U6_NAMES, risk=RISK, RSC15=RSC15_NAMES, ctd_wetness_thresholds_um=THRESH,
            rs_folder_to_15={k: RSC15[f'{v[0]}_{v[1]}'] for k, v in RS_FOLDER.items()},
            xinjiang_map=XJ_MAP,
            tsinghua_rule='dry->Dry, wet->Wet, water->Water, fresh*->Snow, melted*->Slush, ice->Ice',
            tsinghua_test_patch_coords=TS_PATCH_COORDS, patch_size=[PATCH_W, PATCH_H],
        ), fp, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, 'stats.json'), 'w', encoding='utf-8') as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)
    print('\n' + '=' * 70)
    print('完成。stats.json / label_map.json 已写出')

if __name__ == '__main__':
    main()
