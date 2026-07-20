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


def _smooth_series(series, sigma=5):
    """Очистка выбросов + сглаживание произвольного ряда (для детекции)."""
    s = pd.Series(np.asarray(series, dtype=float))
    s = s.reset_index(drop=True)
    s = s.interpolate(limit_direction='both')
    mean, std = s.mean(), s.std()
    if std > 0:
        mask = np.abs(s - mean) > 3 * std
        s[mask] = np.nan
        s = s.interpolate(limit_direction='both')
    return gaussian_smoothing(s.values, sigma=sigma)


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


def detect_points(df, times, rec_start, return_details=False):
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

    # --- Точка 4 (аэробный лимит): выход VO2 на плато. Комбинируем ---
    #     порог по наклону (v1) и поздний загиб вниз VO2.
    t4 = None
    try:
        pk = [p for p in _kink_peaks(te, C['VO2'], win, -1, ntop=3)
              if p[0] > te[0] + 0.6 * dur]
        if pk:
            t4 = max(pk, key=lambda p: p[0])[0]
            res[4] = (t4, val_at('VO2', t4), 'Аэробный лимит (плато VO2)')
    except Exception:
        pass
    if res[4] is None and v1[4] is not None:
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

    // Собираем список осей всех подграфиков
    var xrefs = [];
    Object.keys(gd.layout).forEach(function(k) {
        var m = k.match(/^xaxis(\d*)$/);
        if (m) { var s = m[1]; xrefs.push({x: 'x' + s, y: 'y' + s}); }
    });
    xrefs.sort(function(a, b) {
        var na = parseInt(a.x.slice(1)) || 1, nb = parseInt(b.x.slice(1)) || 1;
        return na - nb;
    });

    var ORIG_TRACES = gd.data.length;                    // исходные кривые
    var BASE_SHAPES = (gd.layout.shapes || []).slice();  // постоянные линии

    function removeExtraTraces() {
        if (gd.data.length > ORIG_TRACES) {
            var idx = [];
            for (var i = ORIG_TRACES; i < gd.data.length; i++) { idx.push(i); }
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
#  Основная функция
# ------------------------------------------------------------------ #
def make(rr_path=None, gas_path=None, recovery_minutes=None,
         directory='.', out_dir='.', download=True, auto_detect=True,
         show_candidates=False):
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
        gas_path = find_last_name('xlsx', directory)
    df = pd.read_excel(gas_path)

    df['RER'] = df['RQ']
    df['t'] = df['t'].apply(lambda x: str(x))
    df['VE'] = df['VE/VCO2'].iloc[2:] * df['VCO2'].iloc[2:]

    time_series = pd.Series(df_res['ОВР'])
    cumulative_time = time_series.cumsum()

    target_times = pd.to_timedelta(df['t'].loc[2:].values).total_seconds() * 1000
    closest_indices = []
    for target in target_times:
        closest_indices.append((cumulative_time - target).abs().idxmin())
    selected_elements = time_series.iloc[closest_indices]

    ser = selected_elements.reset_index(drop=True)
    empty_elements = pd.Series([None, None])
    new_ovr = pd.concat([empty_elements, ser], ignore_index=True)
    df['RR'] = new_ovr

    # ---- Сглаживание показателей ----
    list_periods = ['VO2', 'VCO2', 'RER', 'VE', 'VE/VCO2', 'RR']
    for period in list_periods:
        data = replace_outliers_with_neighbors(df[period].iloc[2:].astype(float))
        arr = gaussian_smoothing(data, sigma=5)
        arr_with_nan = np.insert(arr, 0, [np.nan, np.nan])
        df[f'{period}_ga'] = arr_with_nan

    # ---- Временной ряд по оси X (секунды) ----
    times_sec = pd.to_timedelta(df['t'].loc[2:]).dt.total_seconds().values

    # ================================================================ #
    #  ЗАДАЧА 1: минуты восстановления с клавиатуры
    # ================================================================ #
    # Подсказка: если в CPET-файле есть столбец «Фаза» с меткой RECOVERY,
    # оцениваем длительность восстановления и предлагаем как значение
    # по умолчанию (учёный может подтвердить Enter или ввести своё).
    suggested = None
    if 'Фаза' in df.columns:
        try:
            ph = df['Фаза'].loc[2:].astype(str)
            rec_mask = ph.str.upper().str.contains('RECOVERY')
            if rec_mask.any():
                rec_times = times_sec[rec_mask.values]
                suggested = round((times_sec.max() - rec_times.min()) / 60.0, 1)
        except Exception:
            suggested = None

    if recovery_minutes is None:
        prompt = 'Введите количество минут восстановления'
        if suggested is not None:
            prompt += f' (Enter = предложенное {suggested}): '
        else:
            prompt += ': '
        while True:
            try:
                raw = input(prompt).strip().replace(',', '.')
                if raw == '' and suggested is not None:
                    recovery_minutes = suggested
                else:
                    recovery_minutes = float(raw)
                break
            except (ValueError, EOFError):
                print('Не удалось прочитать число, попробуйте ещё раз.')
                recovery_minutes = suggested if suggested is not None else 0
                break

    # ================================================================ #
    #  ЗАДАЧА 2: время начала восстановления (отсчёт от конца ряда)
    # ================================================================ #
    t_end = float(np.nanmax(times_sec))
    rec_start = t_end - recovery_minutes * 60.0
    print(f'Конец теста: {t_end:.0f} с; '
          f'восстановление {recovery_minutes} мин -> '
          f'начало восстановления на {rec_start:.0f} с')

    # ================================================================ #
    #  ЗАДАЧА 4 (черновик): авто-поиск точек 1-4
    # ================================================================ #
    points = {}
    candidates = []
    if auto_detect:
        try:
            points = detect_points(df, times_sec, rec_start,
                                   return_details=show_candidates)
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
    def plot_single(fig, row, col, df, period, color):
        tt = pd.to_timedelta(df['t'].loc[2:]).dt.total_seconds()
        fig.add_trace(go.Scatter(
            x=tt,
            y=df[period].loc[2:],
            mode='lines+markers',
            name=period,
            marker=dict(size=4, color=color),
            text=[f'Время: {time:.0f} с, {period}: {value:.2f}, RR: {rr:.0f} мс'
                  for time, value, rr in zip(tt, df[period].loc[2:], df['RR'].loc[2:])],
            hoverinfo='text'
        ), row=row, col=col)

    list_periods_all = ['VO2', 'VCO2', 'RER', 'VE', 'VE/VCO2', 'RR']
    list_periods_all += [x + '_ga' for x in list_periods_all]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'green']
    colors += colors
    count_graphs = len(list_periods_all)

    fig = make_subplots(rows=(count_graphs + 1) // 2, cols=2,
                        subplot_titles=list_periods_all)
    for i, period in enumerate(list_periods_all):
        row = i // 2 + 1
        col = i % 2 + 1
        plot_single(fig, row, col, df, period, colors[i])
        fig.update_xaxes(title_text='Время (сек)', row=row, col=col)

    fig.update_layout(
        title=f'Динамика показателей газового анализа {name}',
        height=600 + (count_graphs // 2) * 150,
        showlegend=False,
    )

    # ---- Постоянные вертикальные линии на ВСЕХ подграфиках ----
    def add_vline_all(x, color, dash='dash', width=2):
        for i in range(1, count_graphs + 1):
            suf = '' if i == 1 else str(i)
            fig.add_shape(type='line', xref='x' + suf, yref='y' + suf + ' domain',
                          x0=x, x1=x, y0=0, y1=1,
                          line=dict(color=color, width=width, dash=dash))

    # Задача 2: линия начала восстановления (чёрная, сплошная)
    add_vline_all(rec_start, 'black', dash='solid', width=2)
    # подпись выносим ВПРАВО от линии (в пустую зону восстановления),
    # чтобы она не наезжала на номера точек 1-4 слева
    fig.add_annotation(x=rec_start, xref='x', yref='y domain', y=1.06,
                       text='← начало восстановления', showarrow=False,
                       font=dict(color='black', size=11),
                       xanchor='left', xshift=4)

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
            fig.add_annotation(x=p[0], xref='x', yref='y domain', y=1.06,
                               text=str(num), showarrow=False,
                               font=dict(color=pt_colors[num], size=13),
                               xanchor='center')

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
