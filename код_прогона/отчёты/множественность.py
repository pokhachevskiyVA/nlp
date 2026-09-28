# -*- coding: utf-8 -*-
"""Поправка на множественность для прогонов с большим числом признаков.

При 555 признаках и пороге p<0,05 примерно 28 связей окажутся значимыми
случайно. Считаем, сколько переживает контроль ложных открытий (Бенджамини —
Хохберг) и сколько — Бонферрони.
"""
import sys, os, glob
sys.path.insert(0,'/Users/mac-os/Documents/Батя/outputs/_cache')
sys.path.insert(0,'/Users/mac-os/Documents/Батя/outputs/_cache/prog')
import numpy as np, pandas as pd
import cpet

def бх(p, q=0.05):
    """Бенджамини — Хохберг: порог p и число выживших."""
    p = np.sort(np.asarray(p, float)); m = len(p)
    if m == 0: return 0.0, 0
    порог = q * np.arange(1, m + 1) / m
    ниже = np.where(p <= порог)[0]
    if len(ниже) == 0: return 0.0, 0
    k = ниже[-1]
    return float(p[k]), int(k + 1)

def разобрать(корень, наборы):
    стр = []
    for пап_н, kind in наборы:
        for пап_ц in sorted(os.listdir(f'{корень}/{пап_н}')):
            for sp in ('Лыжники', 'Баскетболисты', 'Контроль'):
                for под in ('Все', 'Сильные', 'Слабые'):
                    d = f'{корень}/{пап_н}/{пап_ц}/{sp}/{под}'
                    зп = f'{d}/Значимые_признаки.xlsx'; вх = f'{d}/_вход.xlsx'
                    if not (os.path.exists(зп) and os.path.exists(вх)): continue
                    z = pd.read_excel(зп, sheet_name='Все значимые')
                    v = pd.read_excel(вх)
                    # всего признаков — как их отобрал pick_features
                    n_пр = len([c for c in v.columns if c not in (v.columns[0], v.columns[1], 'кластер')])
                    p = pd.to_numeric(z['p'], errors='coerce').dropna().values
                    порог_бх, n_бх = бх(p)
                    бонф = 0.05 / max(1, n_пр)
                    стр.append(dict(набор=пап_н, цель=пап_ц, группа=sp, подгруппа=под,
                                    n=len(v), признаков=n_пр, значимых=len(z),
                                    ожидаемо_случайных=round(n_пр * 0.05, 1),
                                    доля_случайных=round(100 * n_пр * 0.05 / max(1, len(z))),
                                    порог_БХ=round(порог_бх, 5), выжило_БХ=n_бх,
                                    порог_Бонферрони=round(бонф, 6),
                                    выжило_Бонферрони=int((p <= бонф).sum())))
    return pd.DataFrame(стр)

if __name__ == '__main__':
    Y = cpet.Y
    Т = разобрать(f'{Y}/Кардиопризнаки_2026-09-28', [('RR', 'rr'), ('Все_признаки', 'все')])
    # газовые прогоны для сравнения
    for пап_ц in ('Прирост', 'ЭкономичностьПрироста', 'ХронотропнаяЦена', 'Мощность'):
        for sp in ('Лыжники', 'Баскетболисты'):
            for под in ('Все', 'Сильные', 'Слабые'):
                d = f'{Y}/ГазоваяЭргометрия_2026-08-24/{пап_ц}/{sp}/{под}'
                if not os.path.exists(f'{d}/Значимые_признаки.xlsx'): continue
                z = pd.read_excel(f'{d}/Значимые_признаки.xlsx', sheet_name='Все значимые')
                v = pd.read_excel(f'{d}/_вход.xlsx')
                n_пр = len([c for c in v.columns if c not in (v.columns[0], v.columns[1], 'кластер')])
                p = pd.to_numeric(z['p'], errors='coerce').dropna().values
                пб, nб = бх(p); бонф = 0.05 / max(1, n_пр)
                Т = pd.concat([Т, pd.DataFrame([dict(набор='Газ', цель=пап_ц, группа=sp, подгруппа=под,
                    n=len(v), признаков=n_пр, значимых=len(z), ожидаемо_случайных=round(n_пр*0.05,1),
                    доля_случайных=round(100*n_пр*0.05/max(1,len(z))), порог_БХ=round(пб,5), выжило_БХ=nб,
                    порог_Бонферрони=round(бонф,6), выжило_Бонферрони=int((p<=бонф).sum()))])], ignore_index=True)
    out = f'{Y}/Кардиопризнаки_2026-09-28'
    os.makedirs(out, exist_ok=True)
    with pd.ExcelWriter(f'{out}/Множественность.xlsx', engine='openpyxl') as w:
        Т.to_excel(w, sheet_name='Поправки', index=False)
    pd.set_option('display.width', 250)
    print(Т[Т['подгруппа']=='Все'][['набор','цель','группа','n','признаков','значимых',
        'ожидаемо_случайных','выжило_БХ','выжило_Бонферрони']].to_string(index=False))
    print('\nсохранено:', f'{out}/Множественность.xlsx')
