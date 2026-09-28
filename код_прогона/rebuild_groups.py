"""Пересборка групповых таблиц с новыми спортсменами.

Состав колонок наследуется от зафиксированной версии (там уже вычищены утечки),
цель Результативность пересчитывается как PC1 результатов на обновлённой выборке.
"""
import os, sys, shutil, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/sessions/lucid-ecstatic-cannon/mnt/outputs/_cache/prog')
import spearman_core as sc, pleiad_engine as pe

B = '/sessions/lucid-ecstatic-cannon/mnt/Батя'
STAMP = '2026-08-14'
FULL = f'{B}/Полная_таблица_признаков_{STAMP}.xlsx'
OLD_R, OLD_W = f'{B}/ГР_Результативность', f'{B}/ГР_Работоспособность'
NEW_R, NEW_W = f'{OLD_R}_{STAMP}', f'{OLD_W}_{STAMP}'
for p in (NEW_R, NEW_W):
    os.makedirs(p, exist_ok=True)

xs = pd.ExcelFile(FULL)
SP = {'Лыжники': ('Лыжники', 'ski', 'low'), 'Баскет': ('Баскетболисты', 'basket', 'high')}
report = []

for pref, (sheet, preset, orient) in SP.items():
    full = xs.parse(sheet)
    key = full.columns[0]
    full[key] = full[key].astype(str).str.strip()

    # ---------- Результативность ----------
    old = pd.read_excel(f'{OLD_R}/{pref}_Все.xlsx')
    cols = [c for c in old.columns if c not in ('Результативность', 'кластер')]
    miss = [c for c in cols if c not in full.columns and c != old.columns[0]]
    assert not miss, f'{pref}: нет колонок {miss[:5]}'
    d = full[[key] + [c for c in cols if c != old.columns[0]]].copy()
    d = d.rename(columns={key: old.columns[0]})
    tmp = f'/tmp/_{pref}_res.xlsx'
    d.to_excel(tmp, index=False)
    DF, N = sc.load_and_clean(tmp)
    sc.configure_results(sc.detect_results(DF, preset), orient)
    T, expl = pe.build_targets(DF)
    tgt = T['PC1 (общий результат)']
    d.insert(1, 'Результативность', tgt.values)
    d.insert(2, 'кластер', pe.make_clusters(tgt).values)
    for sub, sel in [('Все', slice(None)),
                     ('Сильные', d['кластер'] == 'сильные'),
                     ('Слабые', d['кластер'] == 'слабые')]:
        part = d if sub == 'Все' else d[sel]
        part.to_excel(f'{NEW_R}/{pref}_{sub}.xlsx', index=False)
        report.append(('Результативность', f'{pref}_{sub}', len(part),
                       len(pd.read_excel(f'{OLD_R}/{pref}_{sub}.xlsx'))))

    # ---------- Работоспособность ----------
    oldw = pd.read_excel(f'{OLD_W}/{pref}_Все.xlsx')
    TN = 'Пик мощности (Вт)'
    colsw = [c for c in oldw.columns if c not in (TN, 'кластер')]
    dw = full[[key] + [c for c in colsw if c != oldw.columns[0]]].copy()
    dw = dw.rename(columns={key: oldw.columns[0]})
    pcol = next(c for c in full.columns
                if str(c).lower().replace('  ', ' ').strip() in
                ('пик нагрузки (вт)', 'пик нагрузки, вт'))
    y = pd.to_numeric(full[pcol], errors='coerce')
    dw.insert(1, TN, y.values)
    dw = dw[dw[TN].notna()].reset_index(drop=True)
    med = dw[TN].median()
    dw.insert(2, 'кластер', np.where(dw[TN] >= med, 'сильные', 'слабые'))
    for sub, sel in [('Все', slice(None)),
                     ('Сильные', dw['кластер'] == 'сильные'),
                     ('Слабые', dw['кластер'] == 'слабые')]:
        part = dw if sub == 'Все' else dw[sel]
        part.to_excel(f'{NEW_W}/{pref}_{sub}.xlsx', index=False)
        report.append(('Работоспособность', f'{pref}_{sub}', len(part),
                       len(pd.read_excel(f'{OLD_W}/{pref}_{sub}.xlsx'))))

shutil.copy(f'{OLD_W}/Контроль.xlsx', f'{NEW_W}/Контроль.xlsx')
report.append(('Работоспособность', 'Контроль', 16, 16))

print(f"{'папка':20s}{'группа':20s}{'было':>6s}{'стало':>7s}")
for f, g, n, o in report:
    mark = '' if n == o else '   <-- изменилось'
    print(f'{f:20s}{g:20s}{o:6d}{n:7d}{mark}')
