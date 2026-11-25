# -*- coding: utf-8 -*-
"""构建多任务清单：在原有分类清单上, 增加连续/序数回归监督列（方案 C 的基础）

输出: manifests/multitask_all.csv
列:
  path,dataset,domain,split,track,
  label15,label6,cls_valid,            # 分类监督 + 有效掩码
  material,material_valid,             # 0=asphalt 1=cobble 2=concrete / 0=asphalt 1=concrete 2=gravel 3=mud
  wetness_level,wetness_valid,         # 0=dry 1=damp 2=wet 3=verywet
  water_um,water_valid,                # 水膜高度(µm), 仅 CtD 有真值
  snow_cover,snow_valid,               # 0=fresh 1=partial/melted 2=fully packed/blowing
  roughness,roughness_valid,           # 0=smooth 1=slight 2=severe (仅清华沥青/水泥)
  risk
"""
import os, json, csv, re

ROOT = os.environ.get('RSC_ROOT', '/private/zdy/deepseek_road')
MAN = os.path.join(ROOT, 'manifests')
OUT = os.path.join(MAN, 'multitask_all.csv')

RISK = {'Dry': 1, 'Wet': 3, 'Water': 7, 'Snow': 9, 'Slush': 8, 'Ice': 10}
WET_ORD = {'dry': 0, 'damp': 1, 'wet': 2, 'verywet': 3}
SURF_ORD = {'asphalt': 0, 'cobble': 1, 'cobblestone': 1, 'concrete': 2}
TS_MAT = {'asphalt': 0, 'concrete': 1, 'gravel': 2, 'mud': 3}
TS_ROUGH = {'smooth': 0, 'slight': 1, 'severe': 2}
SNOW_ORD = {'fresh_snow': 0, 'partially_covered': 1, 'fully_packed': 2,
            'melted_snow': 1, 'snow_melting_road': 1, 'fully_snowy_road': 2, 'snow_blowing_road': 2}


def blank(x):
    return '' if x is None or x == '' else x


def build():
    rows = []
    # ---------- Track-1 ----------
    for fn, track in [('track1_roadsaw.csv', 'track1'), ('track1_roadsc.csv', 'track1'), ('track1_ctd.csv', 'track1')]:
        p = os.path.join(MAN, fn)
        if not os.path.exists(p):
            print('  [WARN] 缺少', p); continue
        n = 0
        with open(p, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                ds = r.get('dataset', '')
                lbl15 = r.get('label15', '')
                lbl6 = r.get('label6', '')
                l6n = r.get('label6_name', '')
                surf = (r.get('surface') or '').lower()
                wet = (r.get('wetness') or '').lower()
                orig = (r.get('orig_label') or '').lower()
                extra = r.get('extra') or ''
                water_um = None
                speed = None
                if ds == 'ctd':
                    m = re.search(r'wfh=([0-9.]+)', extra)
                    if m:
                        try:
                            water_um = float(m.group(1))
                        except ValueError:
                            water_um = None
                    m2 = re.search(r'speed=([0-9.]+)', extra)
                    if m2:
                        try:
                            speed = float(m2.group(1))
                        except ValueError:
                            speed = None
                out = dict(
                    path=r['path'], dataset=ds, domain=r.get('domain', ''), split=r.get('split', ''), track=track,
                    label15=blank(lbl15), label6=blank(lbl6), cls_valid=1 if (lbl15 != '' or lbl6 != '') else 0,
                    material=blank(SURF_ORD.get(surf) if surf in SURF_ORD else None),
                    material_valid=1 if surf in SURF_ORD else 0,
                    wetness_level=blank(WET_ORD.get(wet) if wet in WET_ORD else None),
                    wetness_valid=1 if wet in WET_ORD else 0,
                    water_um=blank(water_um), water_valid=1 if water_um is not None else 0,
                    snow_cover='', snow_valid=0, roughness='', roughness_valid=0,
                    risk=RISK.get(l6n, ''),
                    speed=blank(speed),
                )
                if ds == 'roadsc':
                    sc = SNOW_ORD.get(orig)
                    out['snow_cover'] = blank(sc); out['snow_valid'] = 1 if sc is not None else 0
                rows.append(out); n += 1
        print('  %-28s %6d 行' % (fn, n))

    # ---------- Track-2 ----------
    for fn in ['track2_tsinghua_train.csv', 'track2_tsinghua_testseq.csv', 'track2_xinjiang.csv']:
        p = os.path.join(MAN, fn)
        if not os.path.exists(p):
            print('  [WARN] 缺少', p); continue
        n = 0
        with open(p, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                ds = r.get('dataset', '')
                orig = (r.get('orig_label') or '').lower()
                l6n = r.get('label6_name', '')
                out = dict(
                    path=r['path'], dataset=ds, domain=r.get('domain', ''), split=r.get('split', ''), track='track2',
                    label15='', label6=blank(r.get('label6', '')), cls_valid=1,
                    material='', material_valid=0,
                    wetness_level='', wetness_valid=0,
                    water_um='', water_valid=0,
                    snow_cover='', snow_valid=0,
                    roughness='', roughness_valid=0,
                    risk=RISK.get(l6n, ''),
                    speed='',
                )
                if ds == 'tsinghua':
                    parts = orig.rsplit('_', 1)
                    cond = parts[0]
                    base, _, tail = cond.partition('_')
                    mat = tail
                    if orig in ('fresh_snow', 'melted_snow', 'ice'):
                        ob = orig; mat = ''
                    else:
                        ob = base
                    if mat in TS_MAT:
                        out['material'] = TS_MAT[mat]; out['material_valid'] = 1
                    # 粗糙度只对沥青/水泥定义
                    if mat in ('asphalt', 'concrete') and len(parts) == 2 and parts[1] in TS_ROUGH:
                        out['roughness'] = TS_ROUGH[parts[1]]; out['roughness_valid'] = 1
                    if ob in ('dry', 'wet', 'water'):
                        out['wetness_level'] = {'dry': 0, 'wet': 1, 'water': 2}[ob]
                        out['wetness_valid'] = 1
                    sc = SNOW_ORD.get(orig)
                    if sc is not None:
                        out['snow_cover'] = sc; out['snow_valid'] = 1
                if ds == 'xinjiang':
                    wl = {'dry_road': 0, 'wet_road': 1}.get(orig)
                    if wl is not None:
                        out['wetness_level'] = wl; out['wetness_valid'] = 1
                    sc = SNOW_ORD.get(orig)
                    if sc is not None:
                        out['snow_cover'] = sc; out['snow_valid'] = 1
                rows.append(out); n += 1
        print('  %-28s %6d 行' % (fn, n))

    cols = ['path', 'dataset', 'domain', 'split', 'track', 'label15', 'label6', 'cls_valid',
            'material', 'material_valid', 'wetness_level', 'wetness_valid',
            'water_um', 'water_valid', 'snow_cover', 'snow_valid',
            'roughness', 'roughness_valid', 'risk', 'speed']
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 统计
    import collections
    st = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        st[r['dataset']]['total'] += 1
        for k in ['cls_valid', 'material_valid', 'wetness_valid', 'water_valid', 'snow_valid', 'roughness_valid']:
            st[r['dataset']][k] += int(bool(r[k]))
    print('\n[输出]', OUT, '总行数', len(rows))
    for ds, c in st.items():
        print('  %-12s total=%6d cls=%6d material=%6d wet=%6d water=%6d snow=%6d rough=%6d' % (
            ds, c['total'], c['cls_valid'], c['material_valid'], c['wetness_valid'],
            c['water_valid'], c['snow_valid'], c['roughness_valid']))
    with open(os.path.join(MAN, 'multitask_stats.json'), 'w', encoding='utf-8') as f:
        json.dump({k: dict(v) for k, v in st.items()}, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    build()
