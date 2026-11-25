# -*- coding: utf-8 -*-
"""汇总所有 spike 结果为论文草稿表格（markdown），保存到 runs/paper_results_draft.md"""
import json, os, glob

ROOT = '/private/zdy/deepseek_road'
OUT = os.path.join(ROOT, 'runs', 'paper_results_draft.md')


def load(d):
    p = os.path.join(ROOT, 'runs', d, 'ran_summary.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding='utf-8'))['results']


def table(title, dirs, note=''):
    lines = [f'### {title}', '']
    if note:
        lines += [f'> {note}', '']
    lines.append('| 方法 | acc | F1 | risk_cost | ECE | esc_cost | hazard_under | rd_acc | rd_esc | rd_haz |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for name, d in dirs:
        res = load(d)
        if not res:
            lines.append(f'| {name} | — | — | — | — | — | — | — | — | — |')
            continue
        best = max(res.values(), key=lambda v: v.get('macro_f1', 0))
        for m, v in res.items():
            tag = '**' if v is best else ''
            lines.append('| %s%s%s | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f | %.4f | %.4f | %.3f |' % (
                tag, m, tag, v.get('acc', 0), v.get('macro_f1', 0), v.get('risk_cost', 0), v.get('ece', 0),
                v.get('esc_cost', 0), v.get('hazard_under', 0), v.get('acc_rd', 0),
                v.get('esc_cost_rd', 0), v.get('hazard_under_rd', 0)))
    lines.append('')
    return '\n'.join(lines)


def auroc_table(title, d):
    res = load(d)
    if not res:
        return f'### {title}\n\n（缺数据）\n'
    lines = [f'### {title}', '', '| 方法 | known_acc | AUROC_energy | AUROC_novelty |', '|---|---|---|---|']
    for m, v in res.items():
        lines.append('| %s | %.4f | %s | %s |' % (
            m, v.get('known_acc', 0),
            ('%.4f' % v['auroc_energy']) if v.get('auroc_energy') else '—',
            ('%.4f' % v['auroc_novelty']) if v.get('auroc_novelty') else '—'))
    lines.append('')
    return '\n'.join(lines)


out = ['# 论文实验结果草稿（自动生成）', '', '数据源：/private/zdy/deepseek_road/runs/spike*/ran_summary.json', '']

out.append('## 表 1：Track-1 跨域（CtD）最新 TTA 对比（v4 模型，校准/测试公平划分）')
out.append(table('CtD1（强偏移）', [('source', 'spike8_ctd1')]))
out.append(table('CtD2（中等偏移）', [('source', 'spike7_ctd2')]))
out.append(table('CtD3（极强偏移）', [('source', 'spike8_ctd3')]))

out.append('## 表 2：Track-2 跨区域')
out.append(table('清华→新疆（跨区域）', [('source', 'spike7_ts2xj')]))
out.append(auroc_table('新疆→清华（跨区域 + 开放集，Water 未知）', 'spike8_xj2ts'))

out.append('## 表 3：校准集大小敏感性（CtD2）')
sens = []
for cp in [20, 60, 100]:
    d = f'spike4_calib{cp}' if cp != 60 else 'spike3_ctd2'
    sens.append((f'calib={cp}', d))
out.append(table('CtD2 校准集大小', sens))

out.append('## 表 4：校准集随机种子稳定性（CtD2, calib=60）')
seeds = [('seed0', 'spike3_ctd2'), ('seed1', 'spike4_seed1'), ('seed2', 'spike4_seed2')]
out.append(table('CtD2 随机种子', seeds))

out.append('## 表 5：风险权重 λ 扫描（清华→新疆）')
lam_dirs = [(f'λ={l}', f'spike6_lam{l}_ts2xj') for l in ['0.1', '0.2', '0.5', '1.0']]
out.append(table('风险决策 λ', lam_dirs))

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('已写出:', OUT)
print('\n'.join(out)[:2000])
