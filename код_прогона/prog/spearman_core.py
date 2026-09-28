# -*- coding: utf-8 -*-
"""Ядро: чтение/очистка данных + расчёт корреляции Спирмена + выгрузка в xlsx."""
import re
import numpy as np
import pandas as pd
from scipy import stats
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

# ----- настройки -----
POINT_PREFIX = "точка"          # столбцы точек поля
ALPHA_05 = 0.05
ALPHA_01 = 0.01
MIN_N = 3                        # минимум пар для расчёта

# ===== универсальная конфигурация результата (баскетбол / лыжи / любой) =====
_RESULT_COLS = None             # заданные вручную столбцы-результаты
_ORIENT = "high"                # 'high' = больше лучше (попадания); 'low' = меньше лучше (время)


def configure_results(cols=None, orientation="high"):
    """Задать столбцы-результаты вручную и ориентацию ('high'/'low')."""
    global _RESULT_COLS, _ORIENT
    _RESULT_COLS = list(cols) if cols else None
    _ORIENT = orientation if orientation in ("high", "low") else "high"


def detect_results(df, preset="auto"):
    """Автоопределение столбцов-результатов по названию.
    preset: 'basket' (точка*), 'ski' (дистанции км), 'auto'."""
    if preset in ("basket", "auto"):
        pts = [c for c in df.columns if str(c).lower().startswith("точка")]
        if pts and preset != "ski":
            return pts
    if preset in ("ski", "auto"):
        ski = [c for c in df.columns
               if re.search(r"\bкм\b|\bkm\b|\d\s*км", str(c).lower())]
        if ski:
            return ski
    return []


def result_columns(df):
    """Итоговый список столбцов-результатов (ручные или авто)."""
    if _RESULT_COLS:
        return [c for c in _RESULT_COLS if c in df.columns]
    return detect_results(df, "auto")


def orientation():
    return _ORIENT


def effect_word(r):
    """Как показатель влияет на УСПЕХ (с учётом ориентации сырого результата).
    high: r>0 → улучшает; low (время): r>0 → ухудшает."""
    if _ORIENT == "low":
        return "улучшает" if r < 0 else "ухудшает"
    return "улучшает" if r > 0 else "ухудшает"

# регулярки для спец-значений
BP_RE  = re.compile(r"^\s*-?\d+(?:[.,]\d+)?\s*/\s*-?\d+(?:[.,]\d+)?\s*$")          # 122/72
VAL_PCT_RE = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*\(\s*(-?\d+(?:[.,]\d+)?)\s*%?\s*\)\s*$")  # 186 (96%)
NUM_RE = re.compile(r"^\s*-?\d+(?:[.,]\d+)?\s*$")


def _to_float(x):
    """строку с запятой/точкой -> float, иначе NaN."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("\xa0", "").replace(" ", "")
    if s == "":
        return np.nan
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _clean_header(h, idx):
    if h is None:
        return f"col_{idx}"
    return str(h).replace("\n", " ").strip()


def _dedup(names):
    """уникальные имена столбцов (повторы получают суффикс __2, __3...)."""
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}__{seen[n]}")
        else:
            seen[n] = 1
            out.append(n)
    return out


def load_and_clean(path, sheet=0):
    """Читает xlsx, чистит числа, разбивает АД и 'значение (%)' на два столбца.
    Возвращает (df_числовой, name_col_series)."""
    wb = load_workbook(path, data_only=True)
    ws = wb.worksheets[sheet] if isinstance(sheet, int) else wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    headers = [_clean_header(h, i) for i, h in enumerate(rows[0])]
    headers = _dedup(headers)
    data = rows[1:]
    raw = pd.DataFrame(data, columns=headers)

    # первый столбец — имена игроков (берём как метки строк)
    name_col = raw.iloc[:, 0].astype("string")
    raw = raw.iloc[:, 1:]
    # убираем полностью пустые строки (без имени и без данных)
    keep = ~(name_col.isna() & raw.isna().all(axis=1))
    raw, name_col = raw[keep].reset_index(drop=True), name_col[keep].reset_index(drop=True)

    new_cols = {}            # имя -> Series (в нужном порядке)
    for col in raw.columns:
        vals = raw[col]
        strv = vals.dropna().astype(str).str.strip()
        n = max(len(strv), 1)
        bp_frac  = strv.str.match(BP_RE).sum() / n if len(strv) else 0
        pct_frac = strv.str.match(VAL_PCT_RE).sum() / n if len(strv) else 0

        if bp_frac >= 0.5:                       # давление 122/72 -> два столбца
            sys_, dia_ = [], []
            for v in vals:
                if v is None or str(v).strip() == "":
                    sys_.append(np.nan); dia_.append(np.nan); continue
                m = BP_RE.match(str(v))
                if m:
                    a, b = str(v).replace(",", ".").split("/")
                    sys_.append(_to_float(a)); dia_.append(_to_float(b))
                else:
                    sys_.append(_to_float(v)); dia_.append(np.nan)
            new_cols[f"{col} (сист.)"]  = pd.Series(sys_)
            new_cols[f"{col} (диаст.)"] = pd.Series(dia_)
        elif pct_frac >= 0.5:                     # 186 (96%) -> значение + %
            val_, pct_ = [], []
            for v in vals:
                if v is None or str(v).strip() == "":
                    val_.append(np.nan); pct_.append(np.nan); continue
                m = VAL_PCT_RE.match(str(v))
                if m:
                    val_.append(_to_float(m.group(1))); pct_.append(_to_float(m.group(2)))
                else:
                    val_.append(_to_float(v)); pct_.append(np.nan)
            new_cols[f"{col} (знач.)"] = pd.Series(val_)
            new_cols[f"{col} (%)"]    = pd.Series(pct_)
        else:                                      # обычный числовой столбец
            new_cols[col] = vals.map(_to_float)

    num = pd.DataFrame(new_cols)
    num = _dedup_df(num)
    return num, name_col


def _dedup_df(df):
    df = df.copy()
    df.columns = _dedup(list(df.columns))
    return df


def compute_spearman(df, point_prefix=POINT_PREFIX, min_n=MIN_N):
    """Возвращает (rho, pval, nobs) — таблицы [показатели x точки].
    Попарное удаление пропусков для каждой пары."""
    point_cols = [c for c in df.columns if str(c).lower().startswith(point_prefix.lower())]
    # показатели = все остальные числовые столбцы, где есть хоть какие-то данные
    other_cols = [c for c in df.columns
                  if c not in point_cols and df[c].notna().sum() >= min_n
                  and df[c].nunique(dropna=True) > 1]

    rho = pd.DataFrame(index=other_cols, columns=point_cols, dtype=float)
    pval = pd.DataFrame(index=other_cols, columns=point_cols, dtype=float)
    nobs = pd.DataFrame(index=other_cols, columns=point_cols, dtype=float)

    for p in point_cols:
        pv = df[p]
        for o in other_cols:
            ov = df[o]
            mask = pv.notna() & ov.notna()
            n = int(mask.sum())
            nobs.at[o, p] = n
            if n < min_n or pv[mask].nunique() < 2 or ov[mask].nunique() < 2:
                rho.at[o, p] = np.nan
                pval.at[o, p] = np.nan
                continue
            r, pp = stats.spearmanr(pv[mask], ov[mask])
            rho.at[o, p] = r
            pval.at[o, p] = pp
    return rho, pval, nobs, point_cols, other_cols


# ----- стили xlsx -----
FILL_05 = PatternFill("solid", fgColor="FFF4CCCC")   # светло-красный  (p<0.05)
FILL_01 = PatternFill("solid", fgColor="FFE06666")   # ярко-красный    (p<0.01)
HDR_FILL = PatternFill("solid", fgColor="FF305496")
HDR_FONT = Font(color="FFFFFFFF", bold=True)
THIN = Side(style="thin", color="FFD9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _write_matrix(ws, mat, title, pval=None, color_by_p=False, fmt="0.000"):
    ws.cell(1, 1, title).font = Font(bold=True, size=12)
    # шапка
    ws.cell(2, 1, "Показатель").font = HDR_FONT
    ws.cell(2, 1).fill = HDR_FILL
    for j, col in enumerate(mat.columns, start=2):
        c = ws.cell(2, j, str(col)); c.font = HDR_FONT; c.fill = HDR_FILL
        c.alignment = Alignment(horizontal="center", wrap_text=True)
    for i, idx in enumerate(mat.index, start=3):
        rc = ws.cell(i, 1, str(idx)); rc.font = Font(bold=True)
        rc.alignment = Alignment(wrap_text=True, vertical="center")
        for j, col in enumerate(mat.columns, start=2):
            v = mat.iat[i - 3, j - 2]
            cell = ws.cell(i, j)
            cell.border = BORDER
            cell.alignment = Alignment(horizontal="center")
            if pd.isna(v):
                cell.value = None
            else:
                cell.value = float(v)
                cell.number_format = fmt
            if color_by_p and pval is not None:
                p = pval.iat[i - 3, j - 2]
                if not pd.isna(p):
                    if p < ALPHA_01:
                        cell.fill = FILL_01
                    elif p < ALPHA_05:
                        cell.fill = FILL_05
    ws.freeze_panes = "B3"
    ws.column_dimensions["A"].width = 34
    from openpyxl.utils import get_column_letter
    for j in range(2, len(mat.columns) + 2):
        ws.column_dimensions[get_column_letter(j)].width = 12


def export_xlsx(out_path, rho, pval, nobs, n_players):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active; ws.title = "Корреляции (rho)"
    _write_matrix(ws, rho, "Коэффициенты Спирмена (точки поля × остальные показатели). "
                           "Красным выделена статистическая значимость.",
                  pval=pval, color_by_p=True, fmt="0.000")
    # легенда
    base = len(rho.index) + 4
    ws.cell(base, 1, "Легенда:").font = Font(bold=True)
    c1 = ws.cell(base + 1, 1, "p < 0.05 (значимо)"); c1.fill = FILL_05
    c2 = ws.cell(base + 2, 1, "p < 0.01 (высоко значимо)"); c2.fill = FILL_01
    ws.cell(base + 3, 1, f"Игроков в выборке: {n_players}")

    ws2 = wb.create_sheet("p-value")
    _write_matrix(ws2, pval, "p-value (двусторонний) для коэффициентов Спирмена",
                  pval=pval, color_by_p=True, fmt="0.0000")

    ws3 = wb.create_sheet("N (число пар)")
    _write_matrix(ws3, nobs, "Число валидных пар наблюдений по каждой паре",
                  color_by_p=False, fmt="0")
    wb.save(out_path)
    return out_path
