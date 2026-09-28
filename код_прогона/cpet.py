# -*- coding: utf-8 -*-
"""Единая точка входа для экспериментов CPET/RR.

Примеры:
  python3 cpet.py --цель мощность     --признаки газ --спорт оба --выход ГазМощность
  python3 cpet.py --цель экономичность --признаки газ --спорт оба --выход ГазЭкономичность
  python3 cpet.py --список-целей

Всегда: 2 плеяды (ведущие + по доменам), Отчёт_краткий, Отчёт_полный,
Значимые_признаки.xlsx, свод по виду спорта. Плеяды рисуются классическим
draw() из analyze.py — трогать нельзя.
"""
import os, re, sys, json, argparse, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
C = '/Users/mac-os/Documents/Батя/outputs/_cache'
sys.path.insert(0, C); sys.path.insert(0, C+'/prog')

B = '/Users/mac-os/Documents/Батя'
Y = '/Users/mac-os/Yandex.Disk.localized/Отчёты_CPET_RR/ФИНАЛ_ВСЕ'
STAMP = '2026-08-25'
FULL = f'{B}/Полная_таблица_признаков_{STAMP}.xlsx'
SHEET = {'лыжники': 'Лыжники', 'баскет': 'Баскетболисты', 'контроль': 'Контроль'}
RUS = {'лыжники': 'Лыжники', 'баскет': 'Баскетболисты', 'контроль': 'Контроль'}

# ---------------------------------------------------------------- цели
def _pow_col(d):
    return next(c for c in d.columns if 'ик нагрузки' in str(c))


def _hr_peak(d):
    h = pd.to_numeric(d.get('ЧСС_пик'), errors='coerce')
    if h is None or h.isna().all():
        h = pd.to_numeric(d.get('газ_пик_ЧСС'), errors='coerce')
    return h


def _resid(y, x):
    """Остаток y после линейной регрессии на x (очистка от x)."""
    ok = y.notna() & x.notna()
    if ok.sum() < 8:
        return y * np.nan
    a, b = np.polyfit(x[ok], y[ok], 1)
    return y - (a * x + b)


def target_мощность(d):
    return pd.to_numeric(d[_pow_col(d)], errors='coerce'), 'Пик мощности (Вт)'


def target_экономичность(d):
    """Вт на литр O2, очищенные от самой мощности.

    Почему так: пиковая ЧСС между людьми почти не меняется (r с мощностью
    -0.05..+0.33), поэтому W/ЧСС на 95% повторяет мощность, а остаток W/ЧСС
    на 99% повторяет саму ЧСС. W/VO2 — классическая gross efficiency, она
    ортогональна и мощности (после очистки), и пиковой ЧСС.
    """
    W = pd.to_numeric(d[_pow_col(d)], errors='coerce')
    vo2 = pd.to_numeric(d.get('газ_т4_VO2'), errors='coerce')
    return _resid(W / vo2 * 1000.0, W), 'Экономичность, Вт/(л·мин⁻¹), очищ. от мощности'


def target_пульсовая_цена(d):
    """Вт на удар, очищенные от мощности. Внимание: почти зеркало пиковой ЧСС."""
    W = pd.to_numeric(d[_pow_col(d)], errors='coerce')
    return _resid(W / _hr_peak(d), W), 'Пульсовая цена, Вт/уд, очищ. от мощности'


def _delta(d):
    """Прирост нагрузки: 30 Вт/мин × минуты разгона.

    Протокол: ПС 30 с + СТ 30 с + 3 мин по ДОО + разгон +30 Вт/мин.
    Столбец `мин_нагрузки` отсчитывается уже после ПС+СТ, поэтому вычитаем три,
    а не четыре. Базовая нагрузка по ДОО индивидуальна и в проекте не измерена,
    поэтому цель — именно прирост, а не абсолютная мощность.
    """
    T = pd.to_numeric(d.get('мин_нагрузки'), errors='coerce')
    # разгон должен был начаться: при нагрузке короче трёх минут человек до него
    # не дошёл, прирост не определён (встречается только в контрольной группе)
    return 30.0 * (T.where(T > 3.0) - 3.0)


def target_прирост(d):
    return _delta(d), 'Прирост нагрузки, Вт (30 Вт/мин × минуты разгона)'


def target_экономичность_прироста(d):
    """Кислородная цена прироста ΔVO₂/ΔW, очищенная от самого прироста.

    Числитель и знаменатель — оба приросты от конца ДОО-фазы до точки 4,
    поэтому базовая нагрузка сокращается и ДОО не нужен. Это ΔVO₂/ΔWR
    из клинического CPET. После очистки показатель не связан ни с приростом,
    ни со своим знаменателем, ни с пиковой ЧСС.
    """
    t4 = pd.to_numeric(d.get('газ_т4_t'), errors='coerce')
    tk = pd.to_numeric(d.get('газ_коноо_t'), errors='coerce')
    V4 = pd.to_numeric(d.get('газ_т4_VO2'), errors='coerce')
    Vk = pd.to_numeric(d.get('газ_коноо_VO2'), errors='coerce')
    dW = 30.0 * (t4 - tk) / 60.0
    y = (V4 - Vk) / dW.where(dW > 15)          # при коротком разгоне отношение разлетается
    y = y.where(y.between(0, 40))
    return _resid(y, _delta(d)), 'Кислородная цена прироста, мл·мин⁻¹·Вт⁻¹, очищ. от прироста'


def target_хронотропная_цена(d):
    """Удары сердца на ватт прироста, очищенные от прироста.

    Числитель — кардиоинтервалы за разгон: полный счёт за нагрузку минус
    накопленные удары к концу 3-й минуты, то есть за вычетом ДОО-фазы.
    Идея эксперта: считать удары, а не пиковую ЧСС.
    """
    # КИ_за_разгон — точный счёт от конца ДОО-фазы (load_start+180) до rec_start
    # на шкале RR. Прежняя формула (КИ_за_нагрузку − Ннак_N3) вычитала лишнее:
    # Ннак_N3 считается от начала записи и включает предстарт со стартом, поэтому
    # отрезок 0–60 с вычитался дважды — недосчёт около 90 ударов. На ранги внутри
    # группы почти не влияло (Спирмен 0,9945), но абсолютные значения искажало,
    # а между группами по-разному, потому что делится на разный прирост.
    N = pd.to_numeric(d.get('КИ_за_разгон'), errors='coerce')
    Δ = _delta(d)
    y = (N / Δ.where(Δ > 30)).where(lambda v: v.between(0, 30))
    return _resid(y, Δ), 'Хронотропная цена прироста, уд/Вт, очищ. от прироста'


TARGETS = {
    'мощность': (target_мощность, 'Пик мощности на велоэргометре'),
    'прирост': (target_прирост, 'Прирост нагрузки: 30 Вт/мин × минуты разгона'),
    'экономичность_прироста': (target_экономичность_прироста,
                               'Кислородная цена прироста ΔVO₂/ΔW, очищенная от прироста'),
    'хронотропная_цена': (target_хронотропная_цена,
                          'Удары сердца на ватт прироста, очищенные от прироста'),
    'экономичность': (target_экономичность, 'Вт на литр O₂, очищенные от мощности'),
    'пульсовая_цена': (target_пульсовая_цена, 'Вт на удар, очищенные от мощности'),
}

# ------------------------------------------------------- наборы признаков
LEAK = ('таргет_', 'Пик нагрузки', 'пик нагрузки', 'Wmax', 'PWC', 'мин_нагрузки',
        'Ннак_N4', 'Ннак_N5', 'NewVar', 'Результативность',
        'КИ_за_нагрузку', 'КИ_всего_до_восст', 'КИ_сумма_мс', 'КИ_за_разгон',
        )

# Моменты, заданные порогом ЧСС: там ЧСС по построению равна порогу (разброс
# 1,3-1,5 уд/мин), а HRR = ЧСС_пик - порог, то есть повторяет пиковую ЧСС
# на r = 0,997. Оба признака бессодержательны при ЛЮБОМ пороге и при любом
# суффиксе имени (встречались варианты `ЧСС<100_нов_HRR`), поэтому ловим
# регулярным выражением, а не перечислением.
LEAK_RE = re.compile(r'восст_ЧСС<\d+.*_(ЧСС|HRR)$')

# Дополнительные исключения под конкретную цель: то, из чего цель построена.
EXTRA_LEAK = {
    'экономичность_прироста': ('газ_т4_', 'газ_коноо_'),
    'хронотропная_цена': ('КИ_за_разгон',),
    'экономичность': ('газ_т4_',),
}


def pick_features(d, kind, target_key=None):
    """Отбор колонок-признаков. kind: газ | rr | все."""
    extra = EXTRA_LEAK.get(target_key, ())
    cols = []
    for c in d.columns:
        s = str(c)
        if any(k in s for k in LEAK) or any(k in s for k in extra) or LEAK_RE.search(s):
            continue
        if s.startswith('газ_') and s.endswith('_t'):
            continue                      # абсолютное время = длительность теста
        if s in ('Unnamed: 0', 'источник_RR', 'кластер'):
            continue
        is_gas = s.startswith('газ_') or s in ('ЧСС_пик', 'МПК (мл/мин/кг )')
        if kind == 'газ' and not is_gas:
            continue
        if kind == 'rr' and is_gas:
            continue
        cols.append(c)
    return cols


# ---------------------------------------------------------------- прогон
# Исключения задаются ПО ЦЕЛИ, а не по человеку: убираем того, у кого недостоверна
# именно та величина, из которой построена данная цель.
#
# ФИО в код не зашиваем — репозиторий публичный, а это медицинские данные.
# Список лежит рядом с таблицей признаков в `исключения.json`:
#     {"мощность": {"Фамилия Имя": "почему исключён"}, ...}
# Пример структуры — `исключения.пример.json`. Файла нет — никто не исключается.
ИСКЛ_ФАЙЛ = os.path.join(B, 'исключения.json')


def _загрузить_исключения():
    try:
        with open(ИСКЛ_ФАЙЛ, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f'  ВНИМАНИЕ: {ИСКЛ_ФАЙЛ} не прочитан ({e}) — никто не исключается')
        return {}


ИСКЛЮЧЕНИЯ_ПО_ЦЕЛИ = _загрузить_исключения()


def run(target_key, feat_kind, sports, outname, split=True):
    from report2 import group_report, domain_summary
    from analyze import analyze
    from sport_report import sport_report

    fn, tdesc = TARGETS[target_key]
    xs = pd.ExcelFile(FULL)
    root = f'{Y}/{outname}'
    os.makedirs(root, exist_ok=True)
    note = (f' Целевая переменная: <b>{tdesc}</b>. Набор признаков: <b>{feat_kind}</b>. '
            f'Сравнение только внутри вида спорта.')
    summary = []

    for sp in sports:
        d = xs.parse(SHEET[sp])
        y, tname = fn(d)
        feats = pick_features(d, feat_kind, target_key)
        tab = d[[d.columns[0]] + feats].copy()
        tab.insert(1, tname, y.values)
        вон = ИСКЛЮЧЕНИЯ_ПО_ЦЕЛИ.get(target_key, {})
        имена = tab[tab.columns[0]].astype(str)
        убрано = [n for n in имена if n in вон]
        if убрано:
            tab = tab[~имена.isin(вон)]
            for n in убрано:
                print(f'    исключён {n}: {вон[n]}')
        tab = tab[tab[tname].notna()].reset_index(drop=True)
        med = tab[tname].median()
        tab.insert(2, 'кластер', np.where(tab[tname] >= med, 'сильные', 'слабые'))

        parts = [('Все', tab)]
        if split:
            parts += [('Сильные', tab[tab['кластер'] == 'сильные']),
                      ('Слабые', tab[tab['кластер'] == 'слабые'])]
        res = {}
        for sub, part in parts:
            outdir = f'{root}/{RUS[sp]}/{sub}'
            os.makedirs(outdir, exist_ok=True)
            src = f'{outdir}/_вход.xlsx'
            part.to_excel(src, index=False)
            group_report(src, outdir, f'{RUS[sp]} — {sub.lower()}', tname, tname,
                         sport_note=note)
            # Квартили значимых показателей рядом с отчётом — обязательно.
            # Эксперт выуживал их из общей описательной статистики полтора часа,
            # поэтому каждый отчёт теперь сопровождается своим файлом.
            try:
                from квартили_значимых import сделать as _кв
                _кв(outdir, тихо=True)
            except Exception as _e:
                print(f'    ВНИМАНИЕ: не собрались квартили значимых: {_e}')
            dd, yy, F, LEAD, DOM2, SHORT = analyze(src, tname)
            res[(RUS[sp], sub)] = {'n': len(dd), 'nsig': int((F['p'] < 0.05).sum()),
                                   'DS': domain_summary(F), 'LEAD': LEAD, 'DOM2': DOM2}
            summary.append((RUS[sp], sub, len(dd), int((F['p'] < 0.05).sum())))
            print(f'  {RUS[sp]}/{sub}: n={len(dd)} значимых={int((F["p"]<0.05).sum())}')
        if split:
            sport_report(res, RUS[sp], f'{root}/{RUS[sp]}', tname, note=note)
    print(f'\nГотово -> {root}')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--цель', dest='tgt', choices=list(TARGETS))
    p.add_argument('--признаки', dest='feat', default='газ', choices=['газ', 'rr', 'все'])
    p.add_argument('--спорт', dest='sport', default='оба',
                   choices=['оба', 'лыжники', 'баскет', 'контроль', 'все три'])
    p.add_argument('--выход', dest='out')
    p.add_argument('--без-кластеров', dest='nosplit', action='store_true')
    p.add_argument('--список-целей', dest='ls', action='store_true')
    a = p.parse_args()
    if a.ls:
        for k, (_, v) in TARGETS.items():
            print(f'  {k:18s} {v}')
        return
    sports = ({'оба': ['лыжники', 'баскет'],
               'все три': ['лыжники', 'баскет', 'контроль']}).get(a.sport, [a.sport])
    run(a.tgt, a.feat, sports, a.out or a.tgt, split=not a.nosplit)


if __name__ == '__main__':
    main()
