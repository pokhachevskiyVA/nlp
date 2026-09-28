# -*- coding: utf-8 -*-
"""Влить ЧСС<125, коноо и счёт КИ в полную таблицу; убрать дубль ЧСС_пик."""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

B = '/sessions/lucid-ecstatic-cannon/mnt/Батя'
C = '/sessions/lucid-ecstatic-cannon/mnt/outputs/_cache'
SRC = f'{B}/Полная_таблица_признаков_2026-08-14.xlsx'
DST = f'{B}/Полная_таблица_признаков_2026-08-24.xlsx'

R = pd.DataFrame([json.loads(l) for l in open(f'{C}/feat125.jsonl')])
R = R[R.get('error').isna()] if 'error' in R.columns else R
R['k'] = R['ФИО'].astype(str).str.strip()

# служебные поля разметки тоже кладём в таблицу — их не хватало
NEW = [c for c in R.columns if c.startswith('газ_') or
       c in ('_load_start', '_rec_start', '_offset_s', '_hr_corr',
             'КИ_всего_до_восст', 'КИ_за_нагрузку', 'КИ_сумма_мс_до_восст')]
REN = {'_load_start': 'разметка_load_start_с', '_rec_start': 'разметка_rec_start_с',
       '_offset_s': 'разметка_сдвиг_газ_RR_с', '_hr_corr': 'разметка_качество_синхр'}

xs = pd.ExcelFile(SRC)
sheets = {}
log = []
for sh in xs.sheet_names:
    d = xs.parse(sh)
    key = d.columns[0]
    d['k'] = d[key].astype(str).str.strip()

    # --- А4: убрать дубль ЧСС_пик / газ_пик_ЧСС ---
    if 'ЧСС_пик' in d.columns and 'газ_пик_ЧСС' in d.columns:
        a = pd.to_numeric(d['ЧСС_пик'], errors='coerce')
        b = pd.to_numeric(d['газ_пик_ЧСС'], errors='coerce')
        ok = a.notna() & b.notna()
        same = int((np.abs(a[ok] - b[ok]) < 1e-6).sum())
        log.append(f'{sh}: ЧСС_пик и газ_пик_ЧСС совпадают у {same} из {int(ok.sum())} '
                   f'-> убран газ_пик_ЧСС')
        d = d.drop(columns=['газ_пик_ЧСС'])

    add = R[['k'] + NEW].drop_duplicates('k')
    before = d.shape[1]
    d = d.merge(add, on='k', how='left').rename(columns=REN)
    d = d.drop(columns=['k'])
    log.append(f'{sh}: колонок {before - 1} -> {d.shape[1]}, '
               f'строк с новым порогом {int(d["газ_восст_ЧСС<125_t"].notna().sum())}/{len(d)}')
    sheets[sh] = d

with pd.ExcelWriter(DST, engine='openpyxl') as w:
    for sh, d in sheets.items():
        d.to_excel(w, sheet_name=sh, index=False)

print('записан', DST)
for l in log:
    print(' ', l)
