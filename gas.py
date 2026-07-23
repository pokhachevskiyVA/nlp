# -*- coding: utf-8 -*-
"""GAS.ipynb — обработка газового анализа (CPET) + RR-интервалы.

Новое в этой версии:
  1) Запрос количества минут восстановления с клавиатуры.
  2) Вертикальная линия НАЧАЛА ВОССТАНОВЛЕНИЯ на каждом графике
     (отсчитывается N минут от конца временного ряда).
  3) Интерактив: клик по любой точке -> синхронная вертикальная линия и
     маркер в том же моменте времени на ВСЕХ графиках + копирование
     координат в буфер обмена. Клавиша Del / Backspace / Esc — очистить.
  4) (Черновик) Авто-поиск физиологических точек 1-4 из статьи Лелявиной:
     лактатный/аэробный порог, ацидотический (pH) порог, точка
     респираторной компенсации (RCP), аэробный лимит. Рисуются как
     подсказки, которые учёные проверяют глазами.

Обратная совместимость: gas.make() без аргументов работает в Colab как
раньше (берёт последний загруженный .rr и .xlsx, спрашивает минуты с
клавиатуры, скачивает html).
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# ------------------------------------------------------------------ #
#  Вспомогательные функции обработки сигнала
# ------------------------------------------------------------------ #
def replace_outliers_with_neighbors(data, threshold=1):
    """Заменяет выбросы на среднее значение соседей."""
    mean = np.mean(data)
    std_dev = np.std(data)
    outlier_mask = np.abs(data - mean) > threshold * std_dev
    cleaned_data = data.copy()
    for i in cleaned_data.index:
        if outlier_mask[i]:
            neighbors = []
            if i > 2:
                neighbors.append(cleaned_data[i - 1])
            if i < cleaned_data.index[-1]:
                neighbors.append(cleaned_data[i + 1])
            if neighbors:
                cleaned_data[i] = np.mean(neighbors)
    return cleaned_data


def gaussian_smoothing(data, sigma):
    """Гауссовское сглаживание с сохранением длины."""
    return gaussian_filter1d(data, sigma=sigma, mode='nearest')


def hampel_filter(series, win=7, k=3.0):
    """Фильтр Хампеля: заменяет выбросы на локальную медиану.

    Аккуратнее прежней чистки по глобальному 1σ (та убирала ~треть точек):
    точка считается выбросом, только если отклоняется от ЛОКАЛЬНОЙ медианы
    более чем на k·MAD в скользящем окне. Форма кривой сохраняется.
    """
    s = pd.Series(np.asarray(series, dtype=float)).interpolate(limit_direction='both')
    med = s.rolling(win, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(win, center=True, min_periods=1).median() * 1.4826
    bad = (s - med).abs() > k * (mad + 1e-9)
    s[bad] = med[bad]
    return s.values


def smooth_curve(series, sigma=5, win=7, k=3.0):
    """Правильное сглаживание для отображения и детекции: Хампель + гаусс."""
    return gaussian_smoothing(hampel_filter(series, win=win, k=k), sigma=sigma)


def _smooth_series(series, sigma=5):
    """Совместимость: сглаживание ряда для детекции точек (Хампель + гаусс)."""
    return smooth_curve(series, sigma=sigma)


def detect_recovery_start_curve(t, y, frac=0.2, hold_s=14.0):
    """Старт восстановления по СГЛАЖЕННОЙ кривой RR (той же, что на графике).

    RR падает во время нагрузки до минимума и разворачивается вверх на
    восстановлении. Ищем ПЕРВЫЙ устойчивый подъём после минимума: первый
    момент, с которого наклон держится положительным (не меньше доли frac от
    максимального наклона подъёма) на протяжении окна hold_s. Берём именно
    первый (а не самый крутой) подъём — это и есть начало восстановления.

    Возвращает время (в тех же единицах, что t) или None.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 12:
        return None
    T = t[-1]
    reg = np.where((t >= 0.30 * T) & (t <= 0.99 * T))[0]
    if len(reg) < 8:
        return None
    a, z = int(reg[0]), int(reg[-1])
    m = a + int(np.argmin(y[a:z + 1]))                # минимум RR
    sl = np.gradient(y, t)
    rise = sl[m:z + 1]
    rise = rise[rise > 0]
    if len(rise) == 0:
        return float(t[m])
    smax = float(np.max(rise))
    thr = frac * smax
    dt = float(np.median(np.diff(t))) or 1.0
    w = max(3, int(hold_s / dt))
    onset = m
    for k in range(m, max(m + 1, z - w)):
        seg = sl[k:k + w]
        if seg.mean() > thr and (seg > -0.02 * smax).all():
            onset = k
            break
    return float(t[onset])


# ------------------------------------------------------------------ #
#  Авто-расчёт начала восстановления по RR-интервалам (задача 1, авто)
# ------------------------------------------------------------------ #
def _clean_rr(r, win=11):
    """Чистка RR от артефактов (отходящие электроды, раскачивание и т.п.).

    1) физиологические границы [250, 2000] мс;
    2) удаление выбросов по локальному MAD относительно скользящей медианы;
    3) база — скользящая медиана (гасит плотные пачки артефактов).
    """
    s = pd.Series(np.asarray(r, dtype=float))
    s[(s < 250) | (s > 2000)] = np.nan
    s = s.interpolate(limit_direction='both')
    med = s.rolling(win, center=True, min_periods=1).median()
    resid = (s - med).abs()
    mad = resid.rolling(win, center=True, min_periods=1).median() * 1.4826 + 1e-9
    s[resid > 4 * mad] = np.nan
    s = s.interpolate(limit_direction='both')
    return s.rolling(win, center=True, min_periods=1).median().values


def detect_recovery_start_rr(rr_intervals, sigma=6, hold_s=15.0):
    """Начало восстановления по RR: момент, когда после нагрузки RR-интервалы
    НАЧИНАЮТ устойчиво расти («колено» кривой RR).

    Во время нагрузки ЧСС растёт -> RR падает до минимума на пике нагрузки;
    на восстановлении RR разворачивается вверх. Ищем именно НАЧАЛО этого
    подъёма: от минимума сглаженной (и очищенной от артефактов) кривой
    находим первый момент, с которого наклон становится устойчиво
    положительным и держится не менее hold_s секунд. Для острого «V» это
    сам минимум, для плоского «дна» — конец плато (где кривая уходит вверх).

    Ранее бралась точка, где RR уже поднялся на ~15% над дном, из-за чего
    метка систематически запаздывала — теперь берётся сам старт подъёма.

    Возвращает время в секундах (шкала RR совпадает с осью газоанализа)
    или None.
    """
    r = np.asarray(rr_intervals, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return None
    rc = _clean_rr(r)
    t = np.cumsum(rc) / 1000.0                      # секунды от начала теста
    rs = gaussian_smoothing(rc, sigma=sigma)
    T = t[-1]
    reg = np.where((t >= 0.30 * T) & (t <= 0.99 * T))[0]   # восстановление — поздно
    if len(reg) < 8:
        return None
    a, z = int(reg[0]), int(reg[-1])
    m = a + int(np.argmin(rs[a:z + 1]))             # вершина (минимум RR)
    slope = np.gradient(rs, t)
    rise = slope[m:z + 1]
    rise = rise[rise > 0]
    if len(rise) == 0:
        return float(t[m])
    thr = 0.25 * float(np.percentile(rise, 75))     # порог «уверенного» роста
    dt = float(np.median(np.diff(t))) or 0.5
    w = max(5, int(hold_s / dt))                    # окно устойчивости
    onset = m
    for k in range(m, max(m + 1, z - w)):
        seg = slope[k:k + w]
        if seg.min() > 0 and seg.mean() > thr:      # рост начался и держится
            onset = k
            break
    return float(t[onset])


def _dfa_alpha1(rr, nmin=4, nmax=16):
    """DFA α1 (короткая шкала) на отрезке RR."""
    rr = np.asarray(rr, dtype=float)
    x = np.cumsum(rr - rr.mean())
    N = len(x)
    ns = list(range(nmin, nmax + 1))
    F = []
    for n in ns:
        k = N // n
        if k < 2:
            F.append(np.nan)
            continue
        r = 0.0
        for j in range(k):
            s = x[j * n:(j + 1) * n]
            tt = np.arange(n)
            r += np.sum((s - np.polyval(np.polyfit(tt, s, 1), tt)) ** 2)
        F.append(np.sqrt(r / (k * n)))
    ns, F = np.array(ns), np.array(F)
    ok = np.isfinite(F) & (F > 0)
    return np.polyfit(np.log(ns[ok]), np.log(F[ok]), 1)[0] if ok.sum() >= 3 else np.nan


def rr_candidate_markers(t, rr_clean, rr_smoothed, rec_start,
                         win=110, step=10):
    """Кандидатные RR-маркеры порогов (для визуальной оценки учёными).

    Возвращает dict времён (сек) — часть может быть None:
      alpha1_075 — скользящий DFA α1 пересёк 0.75 (кандидат в аэробный порог VT1);
      alpha1_05  — DFA α1 пересёк 0.5 (кандидат в RCP/VT2);
      rmssd_floor — RMSSD упал ниже 15% исходного («исчезновение вариабельности»);
      trend_bp1/trend_bp2 — 1-й и 2-й переломы тренда сглаженной RR.
    ВНИМАНИЕ: по нашей проверке эти маркеры индивидуально совпадают с газовыми
    точками 1-3 плохо — это ориентиры, а не точный ответ.
    """
    t = np.asarray(t, float)
    rc = np.asarray(rr_clean, float)
    ys = np.asarray(rr_smoothed, float)
    res = {k: None for k in
           ('alpha1_075', 'alpha1_05', 'rmssd_floor', 'trend_bp1', 'trend_bp2')}
    load = t < rec_start
    if load.sum() < 20:
        return res
    tl = t[load]
    # --- DFA α1 скользящим окном ---
    try:
        ta, aa = [], []
        rcl = rc[load]
        for i in range(0, len(rcl) - win, step):
            aa.append(_dfa_alpha1(rcl[i:i + win]))
            ta.append(tl[i + win // 2])
        ta, aa = np.array(ta), np.array(aa)

        def cross(lvl):
            for i in range(1, len(aa)):
                if np.isfinite(aa[i - 1]) and np.isfinite(aa[i]) and aa[i - 1] >= lvl >= aa[i]:
                    f = (aa[i - 1] - lvl) / (aa[i - 1] - aa[i] + 1e-9)
                    return float(ta[i - 1] + f * (ta[i] - ta[i - 1]))
            return None
        res['alpha1_075'] = cross(0.75)
        res['alpha1_05'] = cross(0.5)
    except Exception:
        pass
    # --- RMSSD floor (исчезновение вариабельности) ---
    try:
        rms = pd.Series(np.abs(np.diff(rc))).rolling(20, min_periods=5).mean().values
        tr = t[1:]
        rl = tr < rec_start
        base = np.nanmax(rms[rl][:max(3, rl.sum() // 3)])
        for i in range(len(rms)):
            if rl[i] and rms[i] < 0.15 * base:
                res['rmssd_floor'] = float(tr[i])
                break
    except Exception:
        pass
    # --- переломы тренда сглаженной RR (2 излома, без разминки) ---
    try:
        yl = ys[load]
        lo = int(len(tl) * 0.12)
        idxs = [lo, len(tl) - 1]
        for _ in range(2):
            best = None
            for i in range(lo, len(tl) - 1):
                if i in idxs:
                    continue
                cand = sorted(idxs + [i])
                sse = 0.0
                for a, bb in zip(cand[:-1], cand[1:]):
                    xs, yy = tl[a:bb + 1], yl[a:bb + 1]
                    if len(xs) > 1:
                        sse += np.sum((yy - np.polyval(np.polyfit(xs, yy, 1), xs)) ** 2)
                if best is None or sse < best[0]:
                    best = (sse, i)
            idxs = sorted(idxs + [best[1]])
        bps = [float(tl[i]) for i in idxs[1:-1]]
        if len(bps) > 0:
            res['trend_bp1'] = bps[0]
        if len(bps) > 1:
            res['trend_bp2'] = bps[1]
    except Exception:
        pass
    return res


# краткая научная расшифровка маркеров (вставляется в RR-only HTML)
RR_MARKERS_HTML = """
<div style="font-family:sans-serif;max-width:1000px;margin:16px auto;padding:12px 18px;
     border:1px solid #ddd;border-radius:8px;background:#fafafa;color:#222">
  <h3 style="margin:4px 0">RR-маркеры порогов (кандидаты, смотреть «на глаз»)</h3>
  <ul style="line-height:1.5;margin:6px 0">
    <li><b style="color:#1f77b4">VT1?</b> — скользящий <b>DFA α1</b> пересекает
        <b>0.75</b>. α1 отражает «фрактальность» ритма; по литературе снижение до
        ~0.75 связывают с <b>аэробным (лактатным) порогом</b> — кандидат в точку&nbsp;1.</li>
    <li><b style="color:#d62728">VT2?</b> — <b>DFA α1</b> пересекает <b>0.5</b>;
        связывают с <b>точкой респираторной компенсации (RCP)</b> — кандидат в точку&nbsp;3.</li>
    <li><b style="color:#9467bd">ВСР↓</b> — <b>исчезновение вариабельности</b>
        (RMSSD падает до «пола»): краткосрочная вариабельность ритма гаснет по мере
        роста нагрузки (маркер по идее <i>Михайлова</i>).</li>
    <li><b style="color:#2ca02c">изл.1 / изл.2</b> — <b>переломы тренда</b> сглаженной
        RR (нагрузочной ритмограммы): перегибы скорости изменения RR
        (маркер по идее <i>Похачевского</i>).</li>
    <li><b>Чёрная линия</b> — начало восстановления (разворот RR вверх); слева от неё —
        нагрузка. Надир RR (минимум) ≈ аэробный лимит и практически совпадает с этой линией.</li>
  </ul>
  <p style="margin:6px 0;color:#a33"><b>Важно:</b> на нашей выборке (48 чел., метки из
     газа) эти маркеры <b>индивидуально</b> совпадают с газовыми точками 1–3 слабо
     (не лучше «среднего по группе»). Поэтому это <b>ориентиры для глаза</b>, а не
     точный автоматический ответ; надёжно по RR определяются только начало
     восстановления и аэробный лимит (надир). Подробности — в
     <i>RR_точки_исследование.html</i>.</p>
</div>
"""


def detect_recovery_start_vo2(times_sec, vo2, sigma=5, hold_s=15.0, min_gap_s=6.0):
    """Начало восстановления по VO2: момент, когда VO2 начинает устойчиво
    ПАДАТЬ (уходит с пика вниз). Совпадает с началом роста RR; спад VO2 резкий
    и однозначный — надёжный ориентир при наличии газовых данных.

    Сглаживание — то же (Хампель+гаусс), что и на графике: пик VO2 (метка
    100% МПК) и линия считаются по ОДНОЙ кривой. Линия ставится строго ПОСЛЕ
    пика (минимум min_gap_s), чтобы 100% МПК и точка 4 остались левее.

    Возвращает время в секундах или None.
    """
    tt = np.asarray(times_sec, dtype=float)
    v = np.asarray(vo2, dtype=float)
    ok = np.isfinite(tt) & np.isfinite(v)
    tt, v = tt[ok], v[ok]
    if len(tt) < 12:
        return None
    vs = smooth_curve(v, sigma=sigma)
    T = tt[-1]
    reg = np.where((tt >= 0.30 * T) & (tt <= 0.99 * T))[0]
    if len(reg) < 8:
        return None
    a, z = int(reg[0]), int(reg[-1])
    pk = a + int(np.argmax(vs[a:z + 1]))            # пик VO2 (апогей) = 100% МПК
    slope = np.gradient(vs, tt)
    fall = slope[pk:z + 1]
    fall = fall[fall < 0]
    dt = float(np.median(np.diff(tt))) or 1.0
    w = max(3, int(hold_s / dt))
    onset = pk
    if len(fall) > 0:
        thr = 0.25 * abs(float(np.percentile(fall, 25)))
        for k in range(pk, max(pk + 1, z - w)):
            seg = slope[k:k + w]
            if seg.max() < 0 and (-seg.mean()) > thr:   # устойчивый спад
                onset = k
                break
    # линия — строго после пика (чтобы 100% МПК и точка 4 были левее)
    return float(max(tt[onset], tt[pk] + min_gap_s))


# ------------------------------------------------------------------ #
#  Авто-поиск точек 1-4 (черновик, задача 4)
# ------------------------------------------------------------------ #
def _piecewise_breakpoint(t, y, lo_frac=0.1, hi_frac=0.9):
    """Ищет точку перелома двухсегментной кусочно-линейной модели.

    Возвращает индекс перелома (в переданных массивах) или None.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    if n < 8:
        return None
    lo = max(2, int(n * lo_frac))
    hi = min(n - 2, int(n * hi_frac))
    best_k, best_sse = None, np.inf
    for k in range(lo, hi):
        sse = 0.0
        for xs, ys in ((t[:k], y[:k]), (t[k:], y[k:])):
            if len(xs) < 2:
                sse = np.inf
                break
            p = np.polyfit(xs, ys, 1)
            sse += np.sum((ys - np.polyval(p, xs)) ** 2)
        if sse < best_sse:
            best_sse, best_k = sse, k
    return best_k


def _detect_points_v1(df, times, rec_start):
    """v1: простые независимые детекторы (fallback для v2).

    Точка 1 — минимум VE/VO2; точка 2 — перелом VCO2; точка 3 (RCP) —
    рост VE/VCO2; точка 4 — плато VO2. Возвращает dict {1..4: (t,val,подпись)}.
    """
    times = np.asarray(times, dtype=float)
    ex = times < rec_start
    if ex.sum() < 10:
        ex = np.ones_like(times, dtype=bool)
    te = times[ex]

    def col(name):
        return df[name].loc[2:].values[: len(times)]

    res = {1: None, 2: None, 3: None, 4: None}

    t3_idx = None
    try:
        vevco2 = _smooth_series(col('VE/VCO2'))[ex]
        start = int(len(vevco2) * 0.3)
        base_min = start + int(np.argmin(vevco2[start:]))
        d = np.diff(vevco2)
        for k in range(max(base_min, 3), len(vevco2) - 5):
            if d[k] > 0 and np.all(vevco2[k:k + 8] >= vevco2[k] - 1e-6):
                t3_idx = k
                break
        if t3_idx is not None:
            res[3] = (te[t3_idx], vevco2[t3_idx], 'RCP (респир. компенсация)')
    except Exception:
        pass

    t1_idx = None
    try:
        vevo2 = _smooth_series(col('VE/VO2'))[ex]
        upper = t3_idx if t3_idx is not None else int(len(vevo2) * 0.75)
        upper = max(upper, 6)
        t1_idx = int(np.argmin(vevo2[:upper]))
        res[1] = (te[t1_idx], vevo2[t1_idx], 'Лактатный/аэробный порог')
    except Exception:
        pass

    try:
        vco2 = _smooth_series(col('VCO2'))[ex]
        lo = (t1_idx + 2) if t1_idx is not None else int(len(vco2) * 0.3)
        hi = t3_idx if t3_idx is not None else int(len(vco2) * 0.85)
        if hi - lo >= 6:
            k = _piecewise_breakpoint(te[lo:hi], vco2[lo:hi])
            if k is not None:
                idx2 = lo + k
                res[2] = (te[idx2], vco2[idx2], 'Ацидотический (pH) порог')
    except Exception:
        pass

    try:
        vo2 = _smooth_series(col('VO2'))[ex]
        d = np.gradient(vo2, te)
        pos = d[d > 0]
        if len(pos):
            thr = 0.15 * np.median(pos)
            t4_idx = None
            for k in range(len(vo2) - 6, int(len(vo2) * 0.5), -1):
                if np.all(d[k:] < thr):
                    t4_idx = k
                else:
                    break
            if t4_idx is None:
                t4_idx = int(np.argmax(vo2))
            res[4] = (te[t4_idx], vo2[t4_idx], 'Аэробный лимит (плато VO2)')
    except Exception:
        pass

    return res


# ------------------------------------------------------------------ #
#  v2: детекция изломов по каждой кривой + кластеризация по времени
#  (реализация идеи А.В. Похачевского: один и тот же перелом виден на
#   нескольких кривых в один момент времени -> подтверждаем по консенсусу)
# ------------------------------------------------------------------ #
def _kink_strength(t, y, win):
    """Знаковая «сила излома» в каждой точке.

    Для каждого узла i строим линейную аппроксимацию слева и справа
    (окно win) и берём разность наклонов (правый − левый) на нормированной
    кривой. Положительное значение = вогнутость вверх (кривая загибается
    вверх, ускорение роста), отрицательное = загиб вниз (выход на плато).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    rng = np.nanmax(y) - np.nanmin(y)
    yr = (y - np.nanmin(y)) / (rng + 1e-9)
    ks = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - win), min(n, i + win + 1)
        if i - a >= 2 and b - i >= 2:
            sl = np.polyfit(t[a:i + 1], yr[a:i + 1], 1)[0]
            sr = np.polyfit(t[i:b], yr[i:b], 1)[0]
            ks[i] = sr - sl
    return ks


def _kink_peaks(t, y, win, sign, ntop=3, sep_frac=0.06):
    """Возвращает самые выраженные изломы кривой заданного знака.

    sign=+1 ищет загибы вверх, sign=-1 — вниз. Возвращает список
    (время, сила) — не более ntop, разнесённых по времени.
    """
    ks = _kink_strength(t, y, win) * sign
    idxs = [i for i in range(1, len(ks) - 1)
            if ks[i] > 0 and ks[i] >= ks[i - 1] and ks[i] >= ks[i + 1]]
    idxs.sort(key=lambda i: -ks[i])
    sep = (t[-1] - t[0]) * sep_frac
    chosen = []
    for i in idxs:
        if all(abs(t[i] - t[j]) > sep for j in chosen):
            chosen.append(i)
        if len(chosen) >= ntop:
            break
    return [(float(t[i]), float(ks[i])) for i in chosen]


def _cluster_times(cands, tol):
    """Кластеризация кандидатов (время, сила, кривая) по времени.

    Возвращает список кластеров с полями: t (взвеш. среднее время),
    curves (множество подтвердивших кривых), support (их число),
    strength (сумма сил).
    """
    cands = sorted(cands, key=lambda c: c[0])
    clusters = []
    for tm, st, cv in cands:
        placed = False
        for cl in clusters:
            if abs(tm - cl['t']) <= tol:
                cl['items'].append((tm, st, cv))
                ws = sum(x[1] for x in cl['items'])
                cl['t'] = sum(x[0] * x[1] for x in cl['items']) / (ws + 1e-9)
                placed = True
                break
        if not placed:
            clusters.append({'items': [(tm, st, cv)], 't': tm})
    for cl in clusters:
        cl['curves'] = set(x[2] for x in cl['items'])
        cl['support'] = len(cl['curves'])
        cl['strength'] = sum(x[1] for x in cl['items'])
    return clusters


def detect_points(df, times, rec_start, return_details=False, load_start=0.0):
    """v2: авто-поиск точек 1-4 через изломы кривых и их кластеризацию.

    Идея: перелом одной физиологической точки проявляется на нескольких
    кривых в один момент времени. Поэтому ищем изломы в каждой кривой
    отдельно, затем кластеризуем по времени и подтверждаем точку по числу
    согласных кривых. Всё считается в зоне нагрузки (t < начала
    восстановления). При неудаче отдельные точки берутся из v1.

    Возвращает dict {1..4: (t, val, подпись)}; при return_details=True —
    также ('clusters', ...) со списком консенсус-кластеров (для отладки
    и отображения «лишних» точек-кандидатов).
    """
    times = np.asarray(times, float)
    # зона нагрузки = [начало нагрузки … начало восстановления]
    ex = (times >= load_start) & (times < rec_start)
    if ex.sum() < 12:
        ex = times < rec_start
    if ex.sum() < 12:
        ex = np.ones_like(times, dtype=bool)
    te = times[ex]
    dur = te[-1] - te[0]
    win = max(6, int(len(te) * 0.06))          # окно наклона ~6% выборки
    tol = max(15.0, dur * 0.07)                 # допуск кластера ~7% времени

    def curve(name):
        return _smooth_series(df[name].loc[2:].values[: len(times)])[ex]

    res = {1: None, 2: None, 3: None, 4: None}
    v1 = _detect_points_v1(df, times, rec_start)

    # безопасно достаём сглаженные кривые
    C = {}
    for nm in ['VO2', 'VCO2', 'RER', 'VE', 'VE/VO2', 'VE/VCO2']:
        try:
            C[nm] = curve(nm)
        except Exception:
            C[nm] = None

    def val_at(name, t):
        y = C.get(name)
        if y is None:
            return float('nan')
        return float(y[int(np.argmin(np.abs(te - t)))])

    # Кол-во кривых, подтверждающих излом около момента t (для «уверенности»)
    def support_at(t, names, sign=+1):
        cnt = 0
        for nm in names:
            if C.get(nm) is None:
                continue
            try:
                pk = _kink_peaks(te, C[nm], win, sign, ntop=4)
                if any(abs(p[0] - t) <= tol for p in pk):
                    cnt += 1
            except Exception:
                pass
        return cnt

    def argmin_in(y, lo_t, hi_t):
        """Индекс минимума кривой y на интервале времени [lo_t, hi_t]."""
        m = (te >= lo_t) & (te <= hi_t)
        if m.sum() < 3:
            return None
        idx = np.where(m)[0]
        return int(idx[np.argmin(np.asarray(y)[idx])])

    def rise_onset(y, lo_t, hi_t):
        """Начало устойчивого терминального подъёма кривой (конец «корыта»).

        Берём минимум в окне; ищем ПОСЛЕДНИЙ момент, когда кривая ещё в
        полосе около минимума, но дальше уверенно растёт к концу. Это
        надёжнее argmin: у плоского «корыта» argmin хватает его начало, а
        нам нужен конец — точка, с которой начинается рост (VT2/RCP).
        """
        m = (te >= lo_t) & (te <= hi_t)
        if m.sum() < 4:
            return None
        idx = np.where(m)[0]
        yy = np.asarray(y)[idx]
        rng = float(np.nanmax(y) - np.nanmin(y)) + 1e-9
        band = 0.12 * rng
        vmin = float(np.nanmin(yy))
        yend = float(np.nanmean(yy[-3:]))
        cand = None
        for k in range(len(idx)):
            if yy[k] <= vmin + band and yend > yy[k] + 0.5 * band:
                cand = int(idx[k])          # последний «донный» индекс перед ростом
        if cand is None:
            cand = int(idx[int(np.argmin(yy))])
        return cand

    # --- Точка 3 (RCP): начало устойчивого роста вентиляторного        ---
    #     эквивалента VE/VCO2 (классический VT2).
    t3 = None
    try:
        i3 = rise_onset(C['VE/VCO2'], te[0] + 0.30 * dur, te[0] + 0.97 * dur)
        if i3 is not None:
            t3 = te[i3]
            s = support_at(t3, ['VE/VCO2', 'VE', 'RER'], +1)
            res[3] = (t3, val_at('VE/VCO2', t3),
                      'RCP (респир. компенсация); кривых: %d' % max(s, 1))
    except Exception:
        pass
    if res[3] is None and v1[3] is not None:
        res[3], t3 = v1[3], v1[3][0]

    # --- Точка 1 (лактатный/аэробный порог, VT1): минимум VE/VO2, ---
    #     когда вентиляция по O2 начинает расти (VE/VCO2 ещё стабилен).
    t1 = None
    try:
        upper = (t3 - 2 * tol) if t3 is not None else te[0] + 0.7 * dur
        upper = max(upper, te[0] + 0.25 * dur)
        i1 = argmin_in(C['VE/VO2'], te[0] + 0.08 * dur, upper)
        if i1 is not None:
            t1 = te[i1]
            s = support_at(t1, ['RER', 'VCO2', 'VE'], +1)   # изломы по статье
            res[1] = (t1, val_at('VE/VO2', t1),
                      'Лактатный/аэробный порог; кривых: %d' % s)
    except Exception:
        pass
    if res[1] is None and v1[1] is not None:
        res[1], t1 = v1[1], v1[1][0]

    # --- Точка 2 (ацидотический pH-порог): кластеризация изломов ВВЕРХ ---
    #     кривых VCO2 и VE строго между точкой 1 и RCP (самое сложное место,
    #     здесь идея кластеризации по кривым работает лучше всего).
    lo2 = (t1 + 0.5 * tol) if t1 is not None else te[0] + 0.3 * dur
    hi2 = (t3 - 0.3 * tol) if t3 is not None else te[0] + 0.85 * dur
    cands = []
    for nm in ['VCO2', 'VE', 'RER']:
        if C.get(nm) is None:
            continue
        try:
            for (tm, st) in _kink_peaks(te, C[nm], win, +1, ntop=4):
                if lo2 < tm < hi2:
                    cands.append((tm, st, nm))
        except Exception:
            pass
    clusters = _cluster_times(cands, tol)
    cons = [cl for cl in clusters if ({'VCO2', 'VE'} & cl['curves'])]
    if cons:
        # самый выраженный консенсус-перелом между порогами
        c2 = max(cons, key=lambda cl: (cl['support'], cl['strength']))
        res[2] = (c2['t'], val_at('VCO2', c2['t']),
                  'Ацидотический (pH) порог; кривых: %d' % c2['support'])
    elif v1[2] is not None and hi2 > v1[2][0] > lo2:
        res[2] = v1[2]

    # --- Точка 4 (аэробный лимит): НАЧАЛО плато VO2 — момент, когда VO2 ---
    #     перестаёт расти (апогей аэробного метаболизма). Это РАННИЙ край
    #     плато; начало восстановления (спад VO2) — поздний край, поэтому
    #     точка 4 всегда строго ВНУТРИ нагрузки и раньше восстановления.
    lo4 = max(te[0] + 0.55 * dur, (t3 if t3 is not None else te[0]))
    t4 = None
    try:
        vo2s = np.asarray(C['VO2'], dtype=float)
        sub = np.where(te > lo4)[0]
        if len(sub) >= 2:
            i0 = int(sub[0])
            i_pk = i0 + int(np.argmax(vo2s[i0:]))     # пик VO2 (апогей)
            v0, vpk = float(vo2s[i0]), float(vo2s[i_pk])
            # аэробный лимит = момент, когда VO2 прошёл 95% пути к пику
            # (подход к апогею). Для плато — начало плато, для крутого роста —
            # чуть раньше пика: всегда строго ВНУТРИ нагрузки, до восстановления.
            target = v0 + 0.95 * (vpk - v0)
            t4 = float(te[i_pk])
            for k in range(i0, i_pk + 1):
                if vo2s[k] >= target:
                    t4 = float(te[k])
                    break
    except Exception:
        t4 = None
    if t4 is not None:
        res[4] = (t4, val_at('VO2', t4), 'Аэробный лимит (начало плато VO2)')
    elif v1[4] is not None and (t3 is None or v1[4][0] > t3):
        res[4] = v1[4]

    if return_details:
        # Глобальные консенсус-кандидаты (изломы, подтверждённые >=2 кривыми)
        allc = []
        for nm in ['RER', 'VCO2', 'VE', 'VE/VO2', 'VE/VCO2']:
            if C.get(nm) is None:
                continue
            try:
                for (tm, stg) in _kink_peaks(te, C[nm], win, +1, ntop=4):
                    allc.append((tm, stg, nm))
            except Exception:
                pass
        gcl = [cl for cl in _cluster_times(allc, tol) if cl['support'] >= 2]
        res['clusters'] = clusters
        res['candidates'] = sorted(cl['t'] for cl in gcl)
    return res


# ------------------------------------------------------------------ #
#  Пользовательский JavaScript (задача 3)
# ------------------------------------------------------------------ #
CUSTOM_JS = r"""
<script>
document.addEventListener('DOMContentLoaded', function() {
    var gd = document.getElementsByClassName('js-plotly-plot')[0];
    if (!gd) { return; }

    // Правый клик не вызывает контекстное меню
    gd.oncontextmenu = function(event) { event.preventDefault(); };

    // Собираем список осей подграфиков (пропускаем верхние оси «% от МПК»,
    // у них задан overlaying — по ним крестик не рисуем)
    var xrefs = [];
    Object.keys(gd.layout).forEach(function(k) {
        var m = k.match(/^xaxis(\d*)$/);
        if (m && !gd.layout[k].overlaying) {
            xrefs.push({x: 'x' + m[1], y: 'y' + m[1]});
        }
    });
    xrefs.sort(function(a, b) {
        var na = parseInt(a.x.slice(1)) || 1, nb = parseInt(b.x.slice(1)) || 1;
        return na - nb;
    });

    // Реальные кривые = трейсы, чьё имя не начинается на «_» (служебные
    // трейсы для отрисовки верхних осей начинаются на «_»)
    var ORIG_TRACES = 0;
    gd.data.forEach(function(tr) {
        if (!(tr.name && tr.name.charAt(0) === '_')) { ORIG_TRACES++; }
    });
    var INIT_TRACES = gd.data.length;                    // все трейсы при загрузке
    var BASE_SHAPES = (gd.layout.shapes || []).slice();  // постоянные линии

    function removeExtraTraces() {
        if (gd.data.length > INIT_TRACES) {
            var idx = [];
            for (var i = INIT_TRACES; i < gd.data.length; i++) { idx.push(i); }
            return Plotly.deleteTraces(gd, idx);
        }
        return Promise.resolve();
    }

    function clearCrosshair() {
        removeExtraTraces().then(function() {
            Plotly.relayout(gd, {shapes: BASE_SHAPES});
        });
    }

    function drawCrosshair(xval) {
        removeExtraTraces().then(function() {
            // Вертикальные линии на всех подграфиках
            var shapes = BASE_SHAPES.slice();
            xrefs.forEach(function(r) {
                shapes.push({
                    type: 'line', xref: r.x, yref: r.y + ' domain',
                    x0: xval, x1: xval, y0: 0, y1: 1,
                    line: {color: 'red', width: 1.5, dash: 'dot'}
                });
            });
            // Маркер на каждой исходной кривой в ближайшем по времени узле
            var newTraces = [];
            for (var i = 0; i < ORIG_TRACES; i++) {
                var tr = gd.data[i];
                if (!tr.x || !tr.y) { continue; }
                var best = 0, bd = Infinity;
                for (var j = 0; j < tr.x.length; j++) {
                    var d = Math.abs(tr.x[j] - xval);
                    if (d < bd) { bd = d; best = j; }
                }
                var yv = tr.y[best];
                if (yv === null || yv === undefined || isNaN(yv)) { continue; }
                newTraces.push({
                    x: [tr.x[best]], y: [yv], mode: 'markers',
                    marker: {size: 11, color: 'red', symbol: 'circle-open',
                             line: {width: 2, color: 'red'}},
                    xaxis: tr.xaxis, yaxis: tr.yaxis,
                    hoverinfo: 'skip', showlegend: false
                });
            }
            Plotly.relayout(gd, {shapes: shapes}).then(function() {
                if (newTraces.length) { Plotly.addTraces(gd, newTraces); }
            });
        });
    }

    // Клик по точке: рисуем крестик и копируем координаты
    gd.on('plotly_click', function(data) {
        if (data.points && data.points.length > 0) {
            var p = data.points[0];
            drawCrosshair(p.x);
            if (p.text) {
                try { navigator.clipboard.writeText(p.text); } catch (e) {}
            }
        }
    });

    // Del / Backspace / Esc — очистить (начать с чистого листа)
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Delete' || e.key === 'Backspace' || e.key === 'Escape') {
            clearCrosshair();
        }
    });
});
</script>
"""


# ------------------------------------------------------------------ #
#  Поиск последнего загруженного файла (Colab)
# ------------------------------------------------------------------ #
def find_last_name(ext, directory='.'):
    files_ = [f for f in os.listdir(directory) if f.endswith(ext)]
    if not files_:
        raise FileNotFoundError(f"Файлы с расширением {ext} не найдены в {directory}")
    latest = max(files_, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    path = os.path.join(directory, latest)
    print(f"Последний файл: {path}")
    return path


# ------------------------------------------------------------------ #
#  RR-only режим: только по .rr файлу (без газового анализа)
# ------------------------------------------------------------------ #
def make_rr(rr_path=None, recovery_minutes=None, directory='.', out_dir='.',
            download=True, recovery_auto=True):
    """Строит HTML только по RR-файлу: RR-интервалы + сглаженная кривая и
    авто-метка начала восстановления. Газ и ось «% от МПК» не используются.

    Начало восстановления считается ПО RR: минимум сглаженной RR и первый
    устойчивый подъём после него (то же, что и в основном режиме).
    """
    if rr_path is None:
        rr_path = find_last_name('rr', directory)
    df1 = pd.read_csv(rr_path)
    df1.columns = ['Unnamed: 0']
    ovr = df1['Unnamed: 0'].dropna().reset_index(drop=True).astype(float)
    name = os.path.basename(rr_path)

    # шкала времени = накопленное время RR (сек); сглаживание — как на графике
    t = np.cumsum(hampel_filter(ovr.values)) / 1000.0
    rr_s = smooth_curve(ovr.values, sigma=5)

    # ---- начало восстановления по RR ----
    if recovery_minutes is None and recovery_auto:
        onset = detect_recovery_start_curve(t, rr_s)
        if onset is not None:
            recovery_minutes = round((t[-1] - onset) / 60.0, 1)
            print(f'RR-only: начало восстановления {onset:.0f} с '
                  f'-> {recovery_minutes} мин')
    if recovery_minutes is None:
        try:
            recovery_minutes = float(
                input('Введите минуты восстановления: ').strip().replace(',', '.'))
        except (ValueError, EOFError):
            recovery_minutes = 0
    rec_start = float(t[-1]) - recovery_minutes * 60.0

    # ---- маркеры-кандидаты по RR (для оценки учёными «на глаз») ----
    markers = rr_candidate_markers(t, hampel_filter(ovr.values), rr_s, rec_start)

    # ---- график: RR (сырой) + RR сглаженный ----
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.13)
    fig.add_trace(go.Scatter(
        x=t, y=ovr.values, mode='lines+markers', name='RR',
        marker=dict(size=4, color='green'),
        text=[f'Время: {tt:.0f} с, RR: {v:.0f} мс' for tt, v in zip(t, ovr.values)],
        hoverinfo='text'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=rr_s, mode='lines', name='RR_ga', line=dict(color='green'),
        text=[f'Время: {tt:.0f} с, RR сглаж.: {v:.0f} мс' for tt, v in zip(t, rr_s)],
        hoverinfo='text'), row=2, col=1)
    for i in (1, 2):
        fig.update_xaxes(title_text='Время (сек)', row=i, col=1)

    fig.update_layout(title=f'RR-интервалы {name}',
                      height=760, margin=dict(t=90), showlegend=False)

    # линия начала восстановления (чёрная сплошная) на обоих подграфиках
    for i in (1, 2):
        suf = '' if i == 1 else str(i)
        fig.add_shape(type='line', xref='x' + suf, yref='y' + suf + ' domain',
                      x0=rec_start, x1=rec_start, y0=0, y1=1,
                      line=dict(color='black', width=2))
    fig.add_annotation(x=rec_start, xref='x', yref='y domain', y=0.04,
                       text='← начало восстановления', showarrow=False,
                       font=dict(color='black', size=11),
                       xanchor='left', yanchor='bottom', xshift=4)
    # маркеры-кандидаты (для точек 1 и 3) — тонкие цветные линии с подписями
    #  key, подпись, цвет, y-высота подписи, сторона (для разноса близких меток)
    mk_style = [
        ('alpha1_075', 'VT1?', '#1f77b4', 0.02, 'right'),  # DFA α1=0.75 → аэр. порог
        ('trend_bp1',  'изл.1', '#2ca02c', 0.20, 'left'),  # 1-й перелом (Похачевский)
        ('rmssd_floor', 'ВСР↓', '#9467bd', 0.30, 'left'),  # исч. вариабельности (Михайлов)
        ('trend_bp2',  'изл.2', '#2ca02c', 0.20, 'left'),  # 2-й перелом тренда
        ('alpha1_05',  'VT2?', '#d62728', 0.13, 'left'),   # DFA α1=0.5 → RCP
    ]
    for key, lbl, color, ypos, side in mk_style:
        x = markers.get(key)
        if x is None:
            continue
        for i in (1, 2):
            suf = '' if i == 1 else str(i)
            fig.add_shape(type='line', xref='x' + suf, yref='y' + suf + ' domain',
                          x0=x, x1=x, y0=0, y1=1,
                          line=dict(color=color, width=1.2, dash='dot'))
        fig.add_annotation(x=x, xref='x2', yref='y2 domain', y=ypos,
                           text=lbl, showarrow=False,
                           font=dict(color=color, size=10),
                           xanchor=side, yanchor='bottom',
                           xshift=(-2 if side == 'right' else 2))
    for i, ttl in ((1, 'RR'), (2, 'RR сглаж.')):
        suf = '' if i == 1 else str(i)
        fig.add_annotation(text='<b>' + ttl + '</b>',
                           xref='x' + suf + ' domain', yref='y' + suf + ' domain',
                           x=0.5, y=1.05, showarrow=False,
                           xanchor='center', yanchor='bottom',
                           font=dict(size=13, color='#222'))

    html_path = os.path.join(out_dir, f"gas_RR_{name}.html")
    fig.write_html(html_path)
    # вставляем описание маркеров в тело страницы + пользовательский JS
    html = open(html_path, encoding='utf-8').read()
    html = html.replace('</body>', RR_MARKERS_HTML + '</body>')
    open(html_path, 'w', encoding='utf-8').write(html)
    with open(html_path, 'a', encoding='utf-8') as f:
        f.write(CUSTOM_JS)
    if download:
        try:
            from google.colab import files
            files.download(html_path)
        except Exception:
            pass
    return {'html_path': html_path, 'recovery_start': rec_start,
            'recovery_minutes': recovery_minutes, 'markers': markers}


# ------------------------------------------------------------------ #
#  Основная функция
# ------------------------------------------------------------------ #
def make(rr_path=None, gas_path=None, recovery_minutes=None,
         directory='.', out_dir='.', download=True, auto_detect=True,
         show_candidates=False, recovery_auto=True, align_by_recovery=True,
         prestart_s=30.0, start_s=30.0):
    """Строит HTML с графиками газового анализа.

    Параметры (все необязательные — по умолчанию поведение как в Colab):
      rr_path, gas_path   — пути к .rr и .xlsx; если None, берётся последний
                            загруженный файл в `directory`.
      recovery_minutes    — минуты восстановления; если None, спрашивается
                            с клавиатуры (задача 1).
      out_dir             — куда сохранить html.
      download            — скачать html (только в Colab).
      auto_detect         — рисовать авто-подсказки точек 1-4 (задача 4).
      show_candidates     — дополнительно рисовать серые линии-кандидаты
                            (изломы, подтверждённые >=2 кривыми) — «лишние»
                            точки в помощь учёному.
      recovery_auto       — если минуты не заданы явно, определять начало
                            восстановления автоматически по RR-интервалам
                            (по умолчанию True; без ввода с клавиатуры).

    Возвращает dict: {'html_path':..., 'recovery_start':..., 'points':...}.
    """
    # ---- Загрузка RR ----
    if rr_path is None:
        rr_path = find_last_name('rr', directory)
    df1 = pd.read_csv(rr_path)
    df1.columns = ['Unnamed: 0']

    df_res = pd.DataFrame()
    df_res['ОВР'] = df1['Unnamed: 0'].dropna().reset_index(drop=True)
    df_res['ВРЕМЯ'] = df_res['ОВР'].cumsum()
    df_res = df_res.reset_index()

    name = os.path.basename(rr_path)

    # ---- Загрузка газового анализа ----
    if gas_path is None:
        try:
            gas_path = find_last_name('xlsx', directory)
        except FileNotFoundError:
            print('Газовый файл (.xlsx) не найден — RR-only режим.')
            return make_rr(rr_path=rr_path, recovery_minutes=recovery_minutes,
                           out_dir=out_dir, download=download,
                           recovery_auto=recovery_auto)
    df = pd.read_excel(gas_path)

    df['RER'] = df['RQ']
    df['t'] = df['t'].apply(lambda x: str(x))
    df['VE'] = df['VE/VCO2'].iloc[2:] * df['VCO2'].iloc[2:]

    time_series = pd.Series(df_res['ОВР'])
    cumulative_time = time_series.cumsum()                 # мс, часы RR

    target_times = pd.to_timedelta(df['t'].loc[2:].values).total_seconds() * 1000  # мс, часы газа

    # --- Синхронизация RR и газа по точке ВОССТАНОВЛЕНИЯ ---
    # Газоанализ пишется непрерывно (предстарт/старт/ручное переключение на
    # кардио), а RR — строго по периодам, поэтому их шкалы сдвинуты. Совмещаем
    # надир RR (перегиб на восстановление) с пиком VO2 (главный горб): сдвигаем
    # часы RR на offset, чтобы эти моменты совпали. RR — приоритетная шкала.
    offset_ms = 0.0
    if align_by_recovery:
        try:
            gt = target_times / 1000.0
            vo2s = smooth_curve(df['VO2'].loc[2:].astype(float).values, sigma=5)
            gr = np.where((gt >= 0.30 * gt[-1]) & (gt <= 0.99 * gt[-1]))[0]
            t_vo2peak = float(gt[gr[int(np.argmax(vo2s[gr]))]])
            ct = cumulative_time.values / 1000.0
            rrs = smooth_curve(time_series.values, sigma=5)
            rr_ = np.where((ct >= 0.30 * ct[-1]) & (ct <= 0.99 * ct[-1]))[0]
            t_rrnadir = float(ct[rr_[int(np.argmin(rrs[rr_]))]])
            offset_ms = (t_vo2peak - t_rrnadir) * 1000.0
            print(f'Синхронизация RR↔газ по восстановлению: сдвиг RR на '
                  f'{offset_ms / 1000:.0f} с (пик VO2 {t_vo2peak:.0f} с, '
                  f'надир RR {t_rrnadir:.0f} с)')
        except Exception:
            offset_ms = 0.0

    closest_indices = []
    for target in target_times:
        closest_indices.append((cumulative_time - (target - offset_ms)).abs().idxmin())
    selected_elements = time_series.iloc[closest_indices]

    ser = selected_elements.reset_index(drop=True)
    empty_elements = pd.Series([None, None])
    new_ovr = pd.concat([empty_elements, ser], ignore_index=True)
    df['RR'] = new_ovr

    # ---- Кислородный пульс O2pulse = VO2 / ЧСС, где ЧСС = 60000 / RR(мс) ---
    #      т.е. O2pulse = VO2 * RR / 60000 (мл O2 за сердечное сокращение)
    df['O2pulse'] = (df['VO2'].iloc[2:].astype(float)
                     * df['RR'].iloc[2:].astype(float) / 60000.0)

    # ---- Сглаживание показателей (Хампель + гаусс, σ=4) ----
    list_periods = ['VO2', 'VCO2', 'RER', 'VE', 'VE/VO2', 'VE/VCO2', 'RR', 'O2pulse']
    for period in list_periods:
        arr = smooth_curve(df[period].iloc[2:].astype(float), sigma=5)
        arr_with_nan = np.insert(arr, 0, [np.nan, np.nan])
        df[f'{period}_ga'] = arr_with_nan

    # ---- Временной ряд по оси X (секунды) ----
    times_sec = pd.to_timedelta(df['t'].loc[2:]).dt.total_seconds().values

    # ================================================================ #
    #  ЗАДАЧА 1: минуты восстановления с клавиатуры
    # ================================================================ #
    # Метка прибора (Фаза=RECOVERY) — только для сверки/запаса.
    faza_min = None
    if 'Фаза' in df.columns:
        try:
            ph = df['Фаза'].loc[2:].astype(str)
            rec_mask = ph.str.upper().str.contains('RECOVERY')
            if rec_mask.any():
                faza_min = round((times_sec.max()
                                  - times_sec[rec_mask.values].min()) / 60.0, 1)
        except Exception:
            faza_min = None

    # ГЛАВНОЕ: длительность восстановления считаем МАТЕМАТИЧЕСКИ ПО RR
    # (в будущем газа может не быть). Восстановление = конец записи RR минус
    # момент начала устойчивого роста RR-интервалов. Отсчёт ведём ОТ КОНЦА
    # записи — так расчёт устойчив к рассинхрону/разной длине записей RR и газа
    # (обе останавливаются вместе в конце теста).
    # VO2-спад (и VE) — ТОЛЬКО для сверки/валидации и как запас, если RR не вышел.
    if recovery_minutes is None and recovery_auto:
        # RR-расчёт (для будущего, когда газа нет, и для сверки): длительность
        # восстановления = «конец записи RR − старт подъёма RR», по сглаженной
        # RR на её собственных часах, отсчёт от конца (устойчиво к рассинхрону).
        rr_min = None
        try:
            r = np.asarray(df_res['ОВР'].values, dtype=float)
            r = r[np.isfinite(r)]
            t_rr = np.cumsum(hampel_filter(r)) / 1000.0
            rr_on = detect_recovery_start_curve(t_rr, smooth_curve(r))
            rr_min = round((t_rr[-1] - rr_on) / 60.0, 1)
        except Exception:
            rr_min = None
        # ОСНОВНАЯ линия при наличии газа — по спаду VO2 (строго после пика):
        # гарантирует, что 100% МПК и точка 4 остаются в нагрузке, а VO2/VE
        # падают на восстановлении.
        vo2_onset = None
        try:
            vo2_onset = detect_recovery_start_vo2(times_sec, df['VO2'].loc[2:].values)
        except Exception:
            vo2_onset = None

        if vo2_onset is not None:
            recovery_minutes = round((times_sec.max() - vo2_onset) / 60.0, 1)
            r_txt = f'{rr_min} мин' if rr_min is not None else '—'
            f_txt = f', Фаза {faza_min} мин' if faza_min is not None else ''
            print(f'Авто: начало восстановления по спаду VO2 {vo2_onset:.0f}с '
                  f'-> {recovery_minutes} мин. Сверка по RR: {r_txt}{f_txt}')
        elif rr_min is not None:                 # газа нет — по RR
            recovery_minutes = rr_min
            print(f'Газа нет; начало восстановления по RR: {recovery_minutes} мин')

    # Запасной путь: метка прибора, затем ручной ввод (если авто не сработало)
    if recovery_minutes is None:
        if faza_min is not None:
            recovery_minutes = faza_min
            print(f'Авто по RR не удалось; беру метку прибора: {faza_min} мин')
        else:
            try:
                recovery_minutes = float(
                    input('Введите количество минут восстановления: ')
                    .strip().replace(',', '.'))
            except (ValueError, EOFError):
                recovery_minutes = 0

    # ================================================================ #
    #  ЗАДАЧА 2: время начала восстановления (отсчёт от конца ряда)
    # ================================================================ #
    t_end = float(np.nanmax(times_sec))
    rec_start = t_end - recovery_minutes * 60.0
    print(f'Конец теста: {t_end:.0f} с; '
          f'восстановление {recovery_minutes} мин -> '
          f'начало восстановления на {rec_start:.0f} с')

    # ---- НАЧАЛО НАГРУЗКИ (вторая опорная точка) ----
    # RR-запись строгая: предстарт -> (пауза) -> старт -> (пауза) -> нагрузка.
    # Начало нагрузки на шкале RR = предстарт + старт + паузы (паузы по ≤3-5 с,
    # задаются суммарно через pause_s). Переносим на шкалу газа найденным
    # сдвигом синхронизации (offset). Всё левее — предстарт/старт, в анализ
    # нагрузки и в проценты не входит.
    preload = float(prestart_s) + float(start_s)
    load_start = 0.0
    try:
        est = offset_ms / 1000.0 + preload         # оценка по RR
        load_start = float(min(max(est, 0.0), rec_start - 60.0))
        # УТОЧНЕНИЕ (учёт операторской паузы ±несколько секунд, автоматически):
        # подтягиваем к фактическому старту роста VO2 — точке максимального
        # ускорения VO2 (перелом вверх) в окне ±W вокруг оценки. Окно узкое
        # (масштаб операторской погрешности), чтобы не переопределять RR-оценку.
        W = 10.0
        vga = df['VO2_ga'].loc[2:].to_numpy(dtype=float)
        m = (times_sec >= load_start - W) & (times_sec <= load_start + W) & np.isfinite(vga)
        idx = np.where(m)[0]
        if len(idx) >= 5:
            best, bestk = int(idx[0]), -1e18
            for i in idx:
                lo, hi = max(0, i - 4), min(len(vga) - 1, i + 4)
                if i - lo >= 2 and hi - i >= 2 and np.isfinite(vga[lo]) and np.isfinite(vga[hi]):
                    sr = (vga[hi] - vga[i]) / (times_sec[hi] - times_sec[i] + 1e-9)
                    sl = (vga[i] - vga[lo]) / (times_sec[i] - times_sec[lo] + 1e-9)
                    if sr - sl > bestk:
                        bestk, best = sr - sl, i
            load_start = float(times_sec[best])
        print(f'Начало нагрузки: оценка по RR (предстарт {prestart_s:.0f}+старт '
              f'{start_s:.0f}) ≈ {est:.0f} с, уточнено по старту роста VO2 -> '
              f'{load_start:.0f} с. Окно нагрузки: {load_start:.0f}–{rec_start:.0f} с')
    except Exception:
        load_start = 0.0

    # ================================================================ #
    #  ЗАДАЧА 4 (черновик): авто-поиск точек 1-4
    # ================================================================ #
    points = {}
    candidates = []
    if auto_detect:
        try:
            points = detect_points(df, times_sec, rec_start,
                                   return_details=show_candidates,
                                   load_start=load_start)
            candidates = points.get('candidates', []) if show_candidates else []
            print('Авто-подсказки точек (проверьте глазами):')
            for num in (1, 2, 3, 4):
                p = points.get(num)
                if p:
                    print(f'  Точка {num}: t={p[0]:.0f} с, {p[2]} (знач.≈{p[1]:.2f})')
                else:
                    print(f'  Точка {num}: не найдена')
        except Exception as e:
            print(f'Авто-поиск точек не выполнен: {e}')
            points = {}

    # ================================================================ #
    #  Построение графиков
    # ================================================================ #
    # % от МПК для подсказки: VO2(t) / пик VO2 * 100 (по сглаженной кривой)
    _vo2ga = df['VO2_ga'].loc[2:].astype(float)
    _vmax = float(np.nanmax(_vo2ga.values)) if len(_vo2ga) else np.nan
    df['pctVO2max'] = _vo2ga / _vmax * 100.0 if _vmax and _vmax == _vmax else np.nan

    def plot_single(fig, row, col, df, period, color):
        tt = pd.to_timedelta(df['t'].loc[2:]).dt.total_seconds()
        fig.add_trace(go.Scatter(
            x=tt,
            y=df[period].loc[2:],
            mode='lines+markers',
            name=period,
            marker=dict(size=4, color=color),
            text=[f'Время: {time:.0f} с, {period}: {value:.2f}, '
                  f'RR: {rr:.0f} мс, %МПК: {pct:.0f}%'
                  for time, value, rr, pct in zip(
                      tt, df[period].loc[2:], df['RR'].loc[2:],
                      df['pctVO2max'].loc[2:])],
            hoverinfo='text'
        ), row=row, col=col)

    list_periods_all = ['VO2', 'VCO2', 'RER', 'VE', 'VE/VO2', 'VE/VCO2', 'RR', 'O2pulse']
    list_periods_all += [x + '_ga' for x in list_periods_all]
    colors = ['blue', 'orange', 'green', 'red', '#8c564b', 'purple', 'green', 'teal']
    colors += colors
    count_graphs = len(list_periods_all)
    n_rows = (count_graphs + 1) // 2

    # без стандартных заголовков сверху (там теперь ось «% от МПК») —
    # название кривой печатаем ВНУТРИ панели (слева вверху)
    fig = make_subplots(rows=n_rows, cols=2, vertical_spacing=0.075)
    for i, period in enumerate(list_periods_all):
        row = i // 2 + 1
        col = i % 2 + 1
        plot_single(fig, row, col, df, period, colors[i])
        fig.update_xaxes(title_text='Время (сек)', row=row, col=col)
        suf = '' if i == 0 else str(i + 1)
        # название панели — СВЕРХУ ПО ЦЕНТРУ, выше верхней оси «% от МПК»
        fig.add_annotation(text='<b>' + period + '</b>',
                           xref='x' + suf + ' domain', yref='y' + suf + ' domain',
                           x=0.5, y=1.24, showarrow=False,
                           xanchor='center', yanchor='bottom',
                           font=dict(size=13, color='#222'))

    fig.update_layout(
        title=f'Динамика показателей газового анализа {name}',
        height=300 + n_rows * 250,
        margin=dict(t=110),
        showlegend=False,
    )

    # ---- Вторая ось абсцисс СВЕРХУ: % от МПК (VO2 в % от максимума) ----
    # МПК = пик VO2 за тест. Метку каждого % ставим по РЕАЛЬНОМУ пересечению
    # восходящей кривой VO2 с уровнем pct%·МПК (с линейной интерполяцией по
    # времени). Уровни НИЖЕ VO2 покоя не рисуем: в покое VO2 — это личный
    # базовый уровень (~15-25% МПК), поэтому 0 с НЕ равно фиксированному %.
    try:
        vo2_ga = df['VO2_ga'].loc[2:].to_numpy(dtype=float)
        pk_i = int(np.nanargmax(vo2_ga))
        vmax = float(vo2_ga[pk_i])
        seg, ts = vo2_ga[:pk_i + 1], times_sec[:pk_i + 1]
        v_rest = float(np.nanmin(seg))                       # ~ уровень покоя
        tick_t, tick_txt = [], []
        for pct in (20, 40, 60, 80, 100):
            target = pct / 100.0 * vmax
            if pct == 100:
                tick_t.append(float(ts[pk_i])); tick_txt.append('100%'); continue
            if target <= v_rest:            # уровень ниже покоя — не показываем
                continue
            cross = None
            for k in range(1, len(seg)):    # первое восходящее пересечение
                if seg[k - 1] < target <= seg[k]:
                    f = (target - seg[k - 1]) / (seg[k] - seg[k - 1] + 1e-9)
                    cross = ts[k - 1] + f * (ts[k] - ts[k - 1])
                    break
            if cross is not None:
                tick_t.append(float(cross)); tick_txt.append(f'{pct}%')
        x_lo, x_hi = float(np.nanmin(times_sec)), float(np.nanmax(times_sec))
        if tick_t:
            top_layout = {}
            for i in range(1, count_graphs + 1):
                suf = '' if i == 1 else str(i)
                topkey = f'xaxis{count_graphs + i}'      # уникальные верхние оси
                # matches='x{i}' жёстко привязывает верхнюю ось к нижней:
                # они синхронны и при масштабировании/перетаскивании
                ax = dict(overlaying='x' + suf, side='top', anchor='y' + suf,
                          matches='x' + suf, tickmode='array',
                          tickvals=tick_t, ticktext=tick_txt, tickangle=0,
                          showgrid=False, ticks='outside',
                          tickfont=dict(size=9, color='#555'))
                top_layout[topkey] = ax
                # подпись оси — в ТУ ЖЕ СТРОКУ, что и проценты, справа (в зоне
                # восстановления, где эта ось всё равно не считается)
                fig.add_annotation(text='% от МПК',
                                   xref='x' + suf + ' domain',
                                   yref='y' + suf + ' domain',
                                   x=1.0, y=1.06, showarrow=False,
                                   xanchor='right', yanchor='bottom',
                                   font=dict(size=9, color='#555'))
                # служебный (невидимый) трейс, чтобы верхняя ось отрисовалась
                fig.add_trace(go.Scatter(
                    x=[x_lo, x_hi], y=[np.nan, np.nan],
                    xaxis='x' + str(count_graphs + i), yaxis='y' + suf,
                    mode='markers', name='_ax', opacity=0,
                    hoverinfo='skip', showlegend=False))
            fig.update_layout(**top_layout)
    except Exception as e:
        print(f'Верхняя ось %МПК не построена: {e}')

    # ---- Постоянные вертикальные линии на ВСЕХ подграфиках ----
    def add_vline_all(x, color, dash='dash', width=2):
        for i in range(1, count_graphs + 1):
            suf = '' if i == 1 else str(i)
            fig.add_shape(type='line', xref='x' + suf, yref='y' + suf + ' domain',
                          x0=x, x1=x, y0=0, y1=1,
                          line=dict(color=color, width=width, dash=dash))

    # Задача 2: линия начала восстановления (чёрная, сплошная)
    add_vline_all(rec_start, 'black', dash='solid', width=2)
    # подписи выносим ВНИЗ подграфика (у верхнего края теперь ось «% от МПК»)
    fig.add_annotation(x=rec_start, xref='x', yref='y domain', y=0.04,
                       text='← начало восстановления', showarrow=False,
                       font=dict(color='black', size=11),
                       xanchor='left', yanchor='bottom', xshift=4)

    # Линия НАЧАЛА НАГРУЗКИ (чёрная, сплошная): всё левее — предстарт/старт,
    # в анализ нагрузки и в проценты не входит.
    if load_start and load_start > times_sec.min() + 1:
        add_vline_all(load_start, 'black', dash='solid', width=2)
        fig.add_annotation(x=load_start, xref='x', yref='y domain', y=0.04,
                           text='начало нагрузки →', showarrow=False,
                           font=dict(color='black', size=11),
                           xanchor='right', yanchor='bottom', xshift=-4)

    # Доп. кандидаты (серые тонкие линии) — рисуем ПОД основными точками
    pt_times = {points[n][0] for n in (1, 2, 3, 4)
                if points.get(n)} if points else set()
    if auto_detect and show_candidates and candidates:
        for ct in candidates:
            if all(abs(ct - pt) > 8 for pt in pt_times):   # не дублируем точки 1-4
                add_vline_all(ct, 'rgba(150,150,150,0.6)', dash='dot', width=1)

    # Задача 4: авто-подсказки точек 1-4 (тонкие цветные линии + номера)
    if auto_detect and points:
        pt_colors = {1: '#1f77b4', 2: '#9467bd', 3: '#d62728', 4: '#8c564b'}
        for num in (1, 2, 3, 4):
            p = points.get(num)
            if not p:
                continue
            add_vline_all(p[0], pt_colors[num], dash='dashdot', width=1.5)
            # номер — чуть ПРАВЕЕ линии, чтобы не наезжал на неё
            fig.add_annotation(x=p[0], xref='x', yref='y domain', y=0.04,
                               text=str(num), showarrow=False,
                               font=dict(color=pt_colors[num], size=13),
                               xanchor='left', yanchor='bottom', xshift=3)

    # ---- Запись HTML ----
    html_path = os.path.join(out_dir, f"gas_{name}.html")
    fig.write_html(html_path)
    with open(html_path, 'a', encoding='utf-8') as f:
        f.write(CUSTOM_JS)

    # ---- Скачивание (только Colab) ----
    if download:
        try:
            from google.colab import files
            files.download(html_path)
        except Exception:
            pass  # локальный запуск — просто оставляем файл

    return {'html_path': html_path, 'recovery_start': rec_start,
            'recovery_minutes': recovery_minutes, 'points': points}
