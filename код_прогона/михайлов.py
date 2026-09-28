# -*- coding: utf-8 -*-
"""Точки Михайлова: восстановленный код обеих версий.

Восстановлено 28.09.2026 из рабочего кэша (allmk.py, mikh2.py, mikh3.py, rec.py).
В таблице признаков сосуществуют ОБА семейства, посчитанные РАЗНЫМИ методами:

    М15 / М25 / М35   — версия 1, относительные пороги. ОШИБОЧНАЯ, см. ниже.
    Мих2 / Мих3 / Мих4 — версия 3, абсолютные пороги. Рабочая.
    Возв25 / Возв50 / Возв75 — возврат разброса на восстановлении (версия 3).

Разными они остались потому, что М-семейство не пересчитали после исправления
метода. На данных это видно: у лыжников М35 даёт 27 с при ЧСС 117, а Мих3 —
187 с при ЧСС 146; Спирмен между ними 0,04, то есть это не сдвиг одной величины,
а две разные величины.

Для новых работ использовать ТОЛЬКО Мих*/Возв*.
"""
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────
#  ВЕРСИЯ 3 (рабочая): разброс вокруг сглаженного тренда, абсолютный порог
# ────────────────────────────────────────────────────────────────────────────
WIN_SD = 30.0          # окно скользящего разброса, с
MIN_PTS_SD = 5         # минимум интервалов в окне
HOLD_LOAD = 25.0       # окно подтверждения на нагрузке, с
HOLD_REC = 25.0        # окно подтверждения на восстановлении, с
MIN_PTS_HOLD = 3       # минимум точек в окне подтверждения
TOL_DOWN = 1.3         # на нагрузке удержание: все значения < порог * 1.3
TOL_UP = 0.7           # на восстановлении: все значения > порог * 0.7
THRESHOLDS_MS = (2, 3, 4)          # абсолютные пороги разброса, мс
RETURN_FRACS = ((0.25, 'Возв25'), (0.50, 'Возв50'), (0.75, 'Возв75'))


def scatter_around_trend(ov, T, smooth_fn, win=WIN_SD, min_pts=MIN_PTS_SD):
    """Разброс интервалов вокруг сглаженного тренда — «толщина дорожки» тахограммы.

    ov — сырые RR-интервалы, мс; T — накопленное время, с.
    smooth_fn — gas.smooth_curve (Хампель + гаусс, sigma=5).
    Возвращает массив sd той же длины, что ov: стандартное отклонение остатка
    в скользящем ВРЕМЕННОМ окне win секунд, оканчивающемся на текущем ударе.
    """
    resid = ov - smooth_fn(ov, sigma=5)
    sd = np.full(len(T), np.nan)
    for i in range(len(T)):
        m = (T > T[i] - win) & (T <= T[i])
        if m.sum() >= min_pts:
            sd[i] = np.std(resid[m])
    return sd


def mikhailov_load(sd, T, load_start, rec_start, thr_ms):
    """Момент устойчивого исчезновения вариабельности на нагрузке.

    Первый удар, на котором разброс опустился ниже thr_ms И остался ниже
    thr_ms * 1.3 на протяжении следующих 25 с (не менее 3 точек).
    Условие удержания нужно, чтобы не поймать случайную яму.
    Возвращает абсолютное время в секундах или None.
    """
    load = (T >= load_start) & (T < rec_start)
    for i in np.where(load)[0]:
        if np.isfinite(sd[i]) and sd[i] < thr_ms:
            j = (T >= T[i]) & (T <= T[i] + HOLD_LOAD) & load
            v = sd[j][np.isfinite(sd[j])]
            if len(v) >= MIN_PTS_HOLD and np.all(v < thr_ms * TOL_DOWN):
                return float(T[i])
    return None


def mikhailov_recovery(sd, T, rec_start, thr_ms):
    """Момент устойчивого возврата вариабельности на восстановлении.

    Зеркально: первый удар выше thr_ms, остающийся выше thr_ms * 0.7 в течение 25 с.
    """
    rv = T >= rec_start
    for i in np.where(rv)[0]:
        if np.isfinite(sd[i]) and sd[i] > thr_ms:
            j = (T >= T[i]) & (T <= T[i] + HOLD_REC) & rv
            v = sd[j][np.isfinite(sd[j])]
            if len(v) >= MIN_PTS_HOLD and np.all(v > thr_ms * TOL_UP):
                return float(T[i])
    return None


def return_points(sd, T, load_start, rec_start):
    """Возв25/50/75 — возврат разброса к 25/50/75% от уровня ПОКОЯ.

    Уровень покоя — медиана разброса за 60 с до начала нагрузки.
    Условие удержания то же, что в mikhailov_recovery.
    """
    rest = np.nanmedian(sd[(T < load_start) & (T >= max(0.0, load_start - 60))])
    out = {'разброс_покой': float(rest) if np.isfinite(rest) else None}
    if not np.isfinite(rest):
        return out
    for frac, tag in RETURN_FRACS:
        t = mikhailov_recovery(sd, T, rec_start, frac * rest)
        if t is not None:
            out[tag + '_с'] = t - rec_start
            out[tag + '_абс'] = t
    return out


def features_v3(ov, T, hr, load_start, rec_start, smooth_fn):
    """Полный набор признаков рабочей версии: Мих2/3/4 и Возв25/50/75."""
    sd = scatter_around_trend(ov, T, smooth_fn)
    f = {
        'разброс_старт': float(np.nanmedian(sd[(T >= load_start) & (T < load_start + 30)])),
        'разброс_конец_нагр': float(np.nanmedian(sd[(T >= rec_start - 45) & (T < rec_start)])),
    }
    for thr in THRESHOLDS_MS:
        t = mikhailov_load(sd, T, load_start, rec_start, thr)
        if t is not None:
            f[f'Мих{thr}_нагр_с'] = t - load_start
            f[f'Мих{thr}_нагр_ЧСС'] = float(np.interp(t, T, hr))
            f[f'Мих{thr}_нагр_доля'] = (t - load_start) / max(rec_start - load_start, 1e-9)
        t2 = mikhailov_recovery(sd, T, rec_start, thr)
        if t2 is not None:
            f[f'Мих{thr}_восст_с'] = t2 - rec_start
            f[f'Мих{thr}_восст_ЧСС'] = float(np.interp(t2, T, hr))
    f.update(return_points(sd, T, load_start, rec_start))
    return f


# ────────────────────────────────────────────────────────────────────────────
#  ВЕРСИЯ 1 (историческая): та, которой посчитаны колонки М15/М25/М35
# ────────────────────────────────────────────────────────────────────────────
#  Приведена ради воспроизводимости уже посчитанных колонок. Не использовать.
#
#  Две ошибки, из-за которых точка срабатывала слишком рано:
#   1) база = МАКСИМУМ сглаженного |ΔRR| по первой трети нагрузки. Максимум —
#      не уровень покоя, а самый шумный удар, поэтому порог завышался, и любое
#      первое затишье его пробивало.
#   2) срабатывание по первому пересечению, без проверки удержания.
#  Плюс окно скользящего среднего задано в УДАРАХ (20), а не в секундах,
#  поэтому его физическая длительность менялась по ходу теста: на пульсе 180
#  двадцать ударов это 6–7 с, на пульсе 70 — около 17 с.

LEVELS_V1 = ((0.15, 'М15'), (0.25, 'М25'), (0.35, 'М35'))


def features_v1_historical(ov, T, hr, load_start, rec_start):
    """Воспроизводит колонки М15/М25/М35 ровно так, как они посчитаны."""
    rms = pd.Series(np.abs(np.diff(ov))).rolling(20, min_periods=5).mean().values
    tr = T[1:]
    rl = (tr >= load_start) & (tr < rec_start)
    rv = tr >= rec_start
    f = {}
    if rl.sum() <= 10:
        return f
    base = np.nanmax(rms[rl][:max(3, int(rl.sum() // 3))])
    for lvl, tag in LEVELS_V1:
        idx = np.where(rl & (rms < lvl * base))[0]
        if len(idx):
            tm = float(tr[idx[0]])
            f[tag + '_нагр_с'] = tm - load_start
            f[tag + '_нагр_ЧСС'] = float(np.interp(tm, T, hr))
            f[tag + '_нагр_доля'] = (tm - load_start) / max(rec_start - load_start, 1e-9)
        got = None
        for i in np.where(rv)[0]:
            if rms[i] > lvl * base:
                got = float(tr[i])
                break
        if got is not None:
            f[tag + '_восст_с'] = got - rec_start
            f[tag + '_восст_ЧСС'] = float(np.interp(got, T, hr))
    return f


# ────────────────────────────────────────────────────────────────────────────
#  ВЕРСИЯ 2 (промежуточная): относительные пороги, но от уровня ПОКОЯ
# ────────────────────────────────────────────────────────────────────────────
#  Прогнана только на 10 людях и в таблицу не вошла. Теги были М15/М25/М50
#  и В15/В25/В50 — при сборке таблицы их перепутали с колонками версии 1,
#  поэтому имя М35 в таблице и уровень 0,50 в коде версии 2 друг другу
#  не соответствуют. Оставлена для истории.

def rmssd_time_window(ov, T, win=30.0, min_pts=4):
    """RMSSD в скользящем ВРЕМЕННОМ окне (в отличие от версии 1)."""
    d = np.abs(np.diff(ov))
    td = T[1:]
    out = np.full(len(td), np.nan)
    for i in range(len(td)):
        m = (td > td[i] - win) & (td <= td[i])
        if m.sum() >= min_pts:
            out[i] = np.sqrt(np.mean(d[m] ** 2))
    return td, out
