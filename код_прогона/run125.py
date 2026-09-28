# -*- coding: utf-8 -*-
"""Прогон всех пар газ+RR с порогом ЧСС<125 и точкой конца ДОО-фазы.

Считает ТОЛЬКО новые семейства признаков (восст_ЧСС<125_*, коноо_*),
плюс служебные метки. Остальные признаки не трогаются.
"""
import sys, json, io, os, contextlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/sessions/lucid-ecstatic-cannon/mnt/outputs')
import gas
os.chdir('/sessions/lucid-ecstatic-cannon/mnt/Батя')
C = '/sessions/lucid-ecstatic-cannon/mnt/outputs/_cache'

CURVES = ['VO2', 'VCO2', 'RER', 'VE', 'VE/VO2', 'VE/VCO2', 'RR', 'O2pulse']
start, cnt = int(sys.argv[1]), int(sys.argv[2])
tasks = json.load(open(C + '/run125_list.json'))
out = open(C + '/feat125.jsonl', 'a')


def load_rr(p):
    v = []
    for ln in open(p, encoding='utf-8', errors='ignore'):
        try:
            v.append(float(ln.strip()))
        except ValueError:
            pass
    return np.array(v)


for fio, rr, xl, sport in tasks[start:start + cnt]:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = gas.make(rr_path=rr, gas_path=xl, report_only=True,
                         prestart_s=30, start_s=30,
                         hr_thresholds=(100, 125), doo_s=180.0)
        det = r.get('detail') or {}
        f = {'ФИО': fio, 'вид_спорта': sport,
             '_load_start': r.get('load_start'), '_rec_start': r.get('rec_start'),
             '_offset_s': r.get('offset_s'), '_hr_corr': r.get('hr_corr')}
        for src, pref in (('восст_ЧСС<125', 'газ_восст_ЧСС<125'),
                          ('восст_ЧСС<100', 'газ_восст_ЧСС<100_нов'),
                          ('коноо', 'газ_коноо')):
            q = det.get(src) or {}
            f[f'{pref}_t'] = q.get('t')
            f[f'{pref}_%МПК'] = q.get('pct')
            for c in CURVES:
                f[f'{pref}_{c}'] = q.get(c)
            f[f'{pref}_ЧСС'] = q.get('ЧСС')
            if 'HRR' in q:
                f[f'{pref}_HRR'] = q.get('HRR')
        # сумма кардиоинтервалов от начала записи RR до rec_start (в шкале RR)
        try:
            ov = load_rr(rr)
            T = np.cumsum(ov) / 1000.0
            off = r.get('offset_s') or 0.0
            rec_rr = r['rec_start'] - off
            ls_rr = r['load_start'] - off
            f['КИ_всего_до_восст'] = int((T <= rec_rr).sum())
            f['КИ_за_нагрузку'] = int(((T >= ls_rr) & (T <= rec_rr)).sum())
            f['КИ_сумма_мс_до_восст'] = float(ov[T <= rec_rr].sum())
        except Exception:
            pass
        out.write(json.dumps(f, ensure_ascii=False, default=float) + '\n')
        print('ok', fio)
    except Exception as e:
        out.write(json.dumps({'ФИО': fio, 'вид_спорта': sport, 'error': str(e)[:120]},
                             ensure_ascii=False) + '\n')
        print('ERR', fio, str(e)[:90])
out.close()
