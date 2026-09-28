# -*- coding: utf-8 -*-
"""Файл длительность_и_нагрузка.xlsx по запросу из Вопросов_прежнему_агенту.md."""
import json, os, unicodedata, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

B = '/sessions/lucid-ecstatic-cannon/mnt/Батя'
C = '/sessions/lucid-ecstatic-cannon/mnt/outputs/_cache'
OUT = f'{B}/длительность_и_нагрузка.xlsx'
nf = lambda s: unicodedata.normalize('NFC', str(s)).strip()

det = {nf(r['surname']): r for r in (json.loads(l) for l in open(f'{C}/detail.jsonl'))}
# досчитанные позже — кладём по полному ФИО, они имеют приоритет
if os.path.exists(f'{C}/detail2.jsonl'):
    for l in open(f'{C}/detail2.jsonl'):
        r = json.loads(l)
        if 'error' not in r:
            det[nf(r['surname'])] = r
x = pd.ExcelFile(f'{B}/Полная_таблица_признаков_2026-08-14.xlsx')

rows = []
for sh, sport in [('Лыжники', 'лыжник'), ('Баскетболисты', 'баскетболист')]:
    d = x.parse(sh)
    key = d.columns[0]
    pcol = next(c for c in d.columns if 'ик нагрузки' in str(c))
    for _, r in d.iterrows():
        fio = nf(r[key])
        # detail.jsonl хранит фамилию, в таблице бывает ФИО с уточнениями
        rec = det.get(fio) or next((v for k, v in det.items()
                                    if k and fio.split()[0] == k.split()[0]
                                    and v['sport'] == sport), None)
        mn = pd.to_numeric(pd.Series([r.get('мин_нагрузки')]), errors='coerce').iloc[0]
        W = pd.to_numeric(pd.Series([r.get(pcol)]), errors='coerce').iloc[0]
        row = {
            'ФИО': r[key], 'вид_спорта': sport,
            'источник_RR': r.get('источник_RR'),
            't_начало_записи_с': 0.0,
            't_конец_ПС_СТ_с': 60.0,
            't_начало_нагрузки_с': rec['load_start'] if rec else np.nan,
            't_конец_ДОО_с': (rec['load_start'] + 180.0) if rec else np.nan,
            't_начало_восстановления_с': rec['rec_start'] if rec else np.nan,
            'сдвиг_газ_RR_с': rec['offset_s'] if rec else np.nan,
            'длительность_нагрузки_с': (rec['rec_start'] - rec['load_start']) if rec else
                                       (mn * 60 if pd.notna(mn) else np.nan),
            'мин_нагрузки_из_таблицы': mn,
            'минут_разгона': ((rec['rec_start'] - rec['load_start']) / 60.0 - 3.0) if rec else
                             (mn - 3.0 if pd.notna(mn) else np.nan),
            'пик_нагрузки_Вт_таблица': W,
        }
        # доля точки 4 — правильная и ошибочная
        if rec:
            t4 = (rec['detail'].get('т4') or {}).get('t')
            ls, rc = rec['load_start'], rec['rec_start']
            row['доля_т4_правильно_%'] = 100 * (t4 - ls) / (rc - ls) if t4 else np.nan
            row['доля_т4_от_ПС_СТ_%'] = 100 * (t4 - 60) / (rc - ls) if t4 else np.nan
        row['дельта_нагрузки_Вт_30втмин'] = (30 * row['минут_разгона']
                                             if pd.notna(row['минут_разгона']) else np.nan)
        row['базовая_нагрузка_Вт_расчёт'] = (W - row['дельта_нагрузки_Вт_30втмин']
                                             if pd.notna(W) and pd.notna(row['дельта_нагрузки_Вт_30втмин'])
                                             else np.nan)
        # возраст
        if sport == 'лыжник':
            row['возраст'] = pd.to_numeric(pd.Series([r.get('Возраст')]), errors='coerce').iloc[0]
            row['возраст_источник'] = 'столбец Возраст'
            row['ДОО_ккал'] = pd.to_numeric(pd.Series([r.get('Основной обмен')]), errors='coerce').iloc[0]
        else:
            mx = pd.to_numeric(pd.Series([r.get('Макс.ЧСС 1/мин')]), errors='coerce').iloc[0]
            row['возраст'] = 220 - mx if pd.notna(mx) else np.nan
            row['возраст_источник'] = '220 − Макс.ЧСС (восстановлено)'
            row['ДОО_ккал'] = np.nan
        rows.append(row)

T = pd.DataFrame(rows)

# --- лист пояснений ---
NOTE = pd.DataFrame({'поле': [
    't_начало_записи_с', 't_конец_ПС_СТ_с', 't_начало_нагрузки_с', 't_конец_ДОО_с',
    't_начало_восстановления_с', 'сдвиг_газ_RR_с', 'длительность_нагрузки_с',
    'минут_разгона', 'доля_т4_правильно_%', 'доля_т4_от_ПС_СТ_%',
    'базовая_нагрузка_Вт_расчёт', 'возраст'],
    'что это': [
    'Ноль шкалы газоанализатора. Все t в файле — в этой шкале.',
    'Номинальный конец предстарта и старта (30+30 с) в шкале RR.',
    'load_start из gas.make: оценка по RR (ПС+СТ+сдвиг), уточнённая по точке максимального ускорения VO2 в окне ±10 с.',
    't_начало_нагрузки_с + 180 с. Номинально, по протоколу; в данных эта граница не размечена.',
    'rec_start: пик ЧСС по газу, сверенный с надиром RR, спадом VE и меткой Фаза=RECOVERY.',
    'offset_s: сдвиг шкалы RR относительно шкалы газа, найден кросс-корреляцией кривых ЧСС.',
    't_начало_восстановления_с − t_начало_нагрузки_с. Разность меток, НЕ сумма кардиоинтервалов.',
    'длительность_нагрузки_с/60 − 3. Три минуты по ДОО вычитаются как номинальные.',
    'Доля нагрузки, на которую пришлась точка 4, от load_start. Правильный расчёт.',
    'То же, но от 60 с. Так считалось раньше — отсюда значения выше 100%.',
    'пик_нагрузки − 30·минут_разгона. Оценка, шум около 30 Вт (см. ответы, пункт 4).',
    'У лыжников из таблицы, у баскетболистов восстановлен как 220 − Макс.ЧСС.']})

with pd.ExcelWriter(OUT, engine='openpyxl') as w:
    T.to_excel(w, 'Длительность_и_нагрузка', index=False)
    NOTE.to_excel(w, 'Пояснения_к_столбцам', index=False)

print('записан', OUT)
print('строк:', len(T), '| с разметкой границ:', int(T['t_начало_нагрузки_с'].notna().sum()))
for sp in ('лыжник', 'баскетболист'):
    s = T[T['вид_спорта'] == sp]
    print(f"  {sp}: n={len(s)}, границы есть у {int(s['t_начало_нагрузки_с'].notna().sum())}, "
          f"доля т4 Ме {s['доля_т4_правильно_%'].median():.0f}%")
