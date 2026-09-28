# -*- coding: utf-8 -*-
"""ML-набор для баскетбола (аналог лыжников):
линейная регрессия (уравнение) + RandomForest + GradientBoosting + перцептрон (MLP).
Предсказываем попадания с каждой точки, PC1, сумму; плюс pooled-модель
(одно уравнение на кластер: номер точки и кластер как признаки).
Оценка честная: train R² и LOO-CV R² (для pooled — CV с группировкой по игроку)."""
import re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, LeaveOneOut, GroupKFold, KFold
from sklearn.metrics import r2_score
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

from spearman_core import load_and_clean, POINT_PREFIX
from glossary import describe
from pleiad_engine import (point_columns, build_targets, make_clusters,
                           candidate_indicators, rank_by_target, _short, equation_str,
                           _ols, HDR, HDRF, SIG, SIG5, THIN, BORD, _hrow, glossary_sheet)
from report_text import write_methodology


def _second_target(df):
    """Ориентированная (выше=лучше) интегральная цель для кластеризации."""
    tg, _ = build_targets(df)
    return list(tg.values())[1]

# --- гиперпараметры моделей (консервативные из-за малого n) ---
def make_models(n_features):
    return {
        "Линейная": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=3, min_samples_leaf=3, random_state=0),
        "ГрадБустинг": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2,
            min_samples_leaf=3, subsample=0.8, random_state=0),
        "Перцептрон(MLP)": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(16,), activation="relu",
                         alpha=0.01, max_iter=2000, random_state=0)),
    }

GB_PARAMS = dict(n_estimators=150, learning_rate=0.05, max_depth=2,
                 min_samples_leaf=3, subsample=0.8)
RF_PARAMS = dict(n_estimators=300, max_depth=3, min_samples_leaf=3)
MLP_ARCH = "входы → 16 нейронов (ReLU) → 1 выход"


def select_features(df, target, mask=None, k=8, method="spearman"):
    cols = candidate_indicators(df, target, mask)
    ranked = rank_by_target(df, target, cols, mask, method)
    return ranked.head(k)["показатель"].tolist(), ranked


def _cv_r2(model, X, y, groups=None, cv_mode="fast"):
    """Кросс-валидация. groups → GroupKFold по игроку (pooled).
    cv_mode: 'fast' = 5-блочная (быстро), 'loo' = leave-one-out (точнее, медленно)."""
    try:
        if groups is not None:
            cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
            pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        elif cv_mode == "loo":
            pred = cross_val_predict(model, X, y, cv=LeaveOneOut())
        else:
            cv = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=0)
            pred = cross_val_predict(model, X, y, cv=cv)
        return r2_score(y, pred)
    except Exception:
        return np.nan


def fit_target(df, target, mask=None, k=8, method="spearman", cv_mode="fast"):
    """Обучает все 4 модели на выбранных признаках. Возвращает словарь с
    train/CV R², уравнением линейной модели, важностями и спецификацией перцептрона."""
    feats, ranked = select_features(df, target, mask, k, method)
    idx = df.index if mask is None else df.index[mask]
    data = pd.concat([target.loc[idx].rename("__y__"), df.loc[idx, feats]], axis=1).dropna()
    out = {"features": feats, "ranked": ranked, "n": len(data), "models": {}}
    if len(data) < max(6, len(feats) + 2):
        out["error"] = f"мало данных (n={len(data)}) для {len(feats)} признаков"
        return out
    X = data[feats].values.astype(float)
    y = data["__y__"].values.astype(float)

    for name, model in make_models(len(feats)).items():
        model.fit(X, y)
        tr = r2_score(y, model.predict(X))
        cv = _cv_r2(model, X, y, cv_mode=cv_mode)
        rec = {"train_r2": tr, "cv_r2": cv}
        if name == "RandomForest":
            rec["importances"] = dict(zip(feats, model.feature_importances_))
        if name == "ГрадБустинг":
            rec["importances"] = dict(zip(feats, model.feature_importances_))
        if name == "Перцептрон(MLP)":
            mlp = model.named_steps["mlpregressor"]
            sc = model.named_steps["standardscaler"]
            n_par = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
            rec["mlp"] = dict(arch=MLP_ARCH, n_params=int(n_par),
                              W1=mlp.coefs_[0], b1=mlp.intercepts_[0],
                              W2=mlp.coefs_[1], b2=mlp.intercepts_[1],
                              scaler_mean=sc.mean_, scaler_scale=sc.scale_)
        out["models"][name] = rec

    # явное линейное уравнение (через numpy OLS для коэф., se, p)
    m = _ols(y, X)
    out["linear_eq"] = equation_str(target.name, feats, m)
    out["linear_ols"] = m
    return out


def _pooled_design(df, method="spearman", k=8):
    """Строит матрицу длинного формата (игрок×точка) для pooled-модели.
    Возвращает X, y, groups, feats, Xcols, pts, cluster_bin."""
    pts = point_columns(df)
    summ = df[pts].sum(axis=1, min_count=1)
    cl = make_clusters(_second_target(df))
    cluster_bin = (cl == "сильные").astype(int)

    long_rows = []
    for pi, pc in enumerate(pts):
        for i in df.index:
            long_rows.append((i, pi, pc, df.at[i, pc]))
    long = pd.DataFrame(long_rows, columns=["player", "pt_idx", "pt_name", "y"])

    phys = [c for c in df.columns if c not in set(pts)
            and df[c].notna().sum() >= 0.6 * len(df) and df[c].nunique(dropna=True) > 1]
    phys = [c for c in phys
            if abs(stats.spearmanr(df[c], summ, nan_policy="omit")[0] or 0) < 0.95]
    scores = []
    for c in phys:
        rs = []
        for pc in pts:
            m = df[c].notna() & df[pc].notna()
            if m.sum() >= 6 and df[c][m].nunique() > 1 and df[pc][m].nunique() > 1:
                rr = stats.spearmanr(df[c][m], df[pc][m])[0]
                if not np.isnan(rr):
                    rs.append(abs(rr))
        if rs:
            scores.append((c, float(np.mean(rs))))
    scores.sort(key=lambda t: t[1], reverse=True)
    feats = [c for c, _ in scores[:k]]

    long["cluster"] = long["player"].map(cluster_bin)
    for c in feats:
        long[c] = long["player"].map(df[c])
    long = long.dropna(subset=["y"] + feats + ["cluster"])
    dummies = pd.get_dummies(long["pt_idx"], prefix="точка", drop_first=True)
    Xdf = pd.concat([long[feats + ["cluster"]].reset_index(drop=True),
                     dummies.reset_index(drop=True)], axis=1)
    return (Xdf.values.astype(float), long["y"].values.astype(float),
            long["player"].values, feats, list(Xdf.columns), list(pts),
            int(long["player"].nunique()))


def pooled_model(df, targets_points=None, method="spearman", k=8):
    """Одно уравнение на кластер: длинный формат (игрок×точка).
    Признаки: физиология (top-k) + номер точки (one-hot) + кластер (сильный=1)."""
    X, y, groups, feats, Xcols, pts, n_players = _pooled_design(df, method, k)
    res = {"feats": feats, "Xcols": Xcols, "n_rows": len(y),
           "n_players": n_players, "models": {}}
    for name, model in make_models(X.shape[1]).items():
        model.fit(X, y)
        tr = r2_score(y, model.predict(X))
        cv = _cv_r2(model, X, y, groups=groups)
        res["models"][name] = {"train_r2": tr, "cv_r2": cv}
    m = _ols(y, X)
    res["linear_ols"] = m
    res["pt_names"] = list(pts)
    return res


# ---------- обучение одной модели (для сохранения и предсказания) ----------
_MODEL_TAG = {"Линейная": "lin", "RandomForest": "rf",
              "ГрадБустинг": "gb", "Перцептрон(MLP)": "mlp"}


def fit_one_model(df, target, mask, model_name, k=8, method="spearman"):
    """Обучает ОДНУ выбранную модель и возвращает её объект + признаки + качество."""
    feats, ranked = select_features(df, target, mask, k, method)
    idx = df.index if mask is None else df.index[mask]
    data = pd.concat([target.loc[idx].rename("__y__"), df.loc[idx, feats]], axis=1).dropna()
    if len(data) < len(feats) + 3:
        raise ValueError(f"Мало данных: n={len(data)} при {len(feats)} признаках.")
    X = data[feats].values.astype(float); y = data["__y__"].values.astype(float)
    model = make_models(len(feats))[model_name]
    model.fit(X, y)
    return dict(model=model, features=feats,
                train_r2=round(float(r2_score(y, model.predict(X))), 3),
                cv_r2=round(float(_cv_r2(model, X, y, cv_mode="fast")), 3), n=len(data))


def pooled_fit_one(df, model_name, k=8, method="spearman"):
    """Обучает ОДНУ pooled-модель, возвращает объект + дизайн для предсказания."""
    X, y, groups, feats, Xcols, pts, n_players = _pooled_design(df, method, k)
    model = make_models(X.shape[1])[model_name]
    model.fit(X, y)
    return dict(model=model, features=feats, Xcols=Xcols, pts=pts,
                train_r2=round(float(r2_score(y, model.predict(X))), 3),
                cv_r2=round(float(_cv_r2(model, X, y, groups=groups)), 3),
                n=len(y), n_players=n_players)


def make_model_id(target_name, scope, model_name):
    t = re.sub(r"[^0-9A-Za-zА-Яа-я]+", "_", str(target_name)).strip("_")[:24]
    s = {"все игроки": "все", "кластер: сильные": "сильные",
         "кластер: слабые": "слабые", "pooled": "pooled"}.get(scope, scope)
    return f"{t}__{s}__{_MODEL_TAG.get(model_name, model_name)}"


# ---------- xlsx-отчёт ----------
def _cmp_header(ws, r):
    _hrow(ws, r, ["Модель", "R² (train)", "R² (LOO-CV)", "Комментарий"])


def _fill_cmp(ws, r, models):
    _cmp_header(ws, r); r += 1
    order = ["Линейная", "RandomForest", "ГрадБустинг", "Перцептрон(MLP)"]
    best = max((v.get("cv_r2", np.nan) for v in models.values()
               if not np.isnan(v.get("cv_r2", np.nan))), default=np.nan)
    for name in order:
        if name not in models:
            continue
        v = models[name]
        ws.cell(r, 1, name)
        ws.cell(r, 2, None if np.isnan(v["train_r2"]) else round(v["train_r2"], 3))
        cvc = ws.cell(r, 3, None if np.isnan(v["cv_r2"]) else round(v["cv_r2"], 3))
        note = ""
        if not np.isnan(v["cv_r2"]) and v["cv_r2"] == best:
            cvc.fill = SIG5; note = "лучшая по CV"
        if not np.isnan(v["train_r2"]) and not np.isnan(v["cv_r2"]) and v["train_r2"] - v["cv_r2"] > 0.3:
            note = (note + "; " if note else "") + "переобучение (train≫CV)"
        ws.cell(r, 4, note)
        for j in range(1, 5):
            ws.cell(r, j).border = BORD
        r += 1
    return r + 1


def build_model_targets(df, choice, specific=""):
    """Готовит {название: Series} и флаг pooled по выбору пользователя (универсально)."""
    pts = point_columns(df)
    tg, _ = build_targets(df)
    if choice.startswith("все результаты"):
        return {pc: df[pc] for pc in pts}, False
    if choice.startswith("общие"):
        return dict(tg), True          # PC1+композит И pooled-модель
    if choice.startswith("всё сразу"):
        d = {pc: df[pc] for pc in pts}; d.update(tg)
        return d, True
    if choice.startswith("только pooled"):
        return {}, True
    if choice.startswith("конкретный"):
        c = specific.strip()
        if c in df.columns:
            return {c: df[c]}, False
        raise ValueError(f"Столбец «{c}» не найден. Проверьте название.")
    if choice in pts:
        return {choice: df[choice]}, False
    if choice in tg:
        return {choice: tg[choice]}, False
    return dict(tg), False


def models_xlsx(out_path, df, targets, method="spearman", k=8, include_pooled=False,
                cv_mode="fast"):
    wb = Workbook(); wb.remove(wb.active)
    summary = []          # (цель, охват, лучшая модель, CV R², уравнение)
    used = []             # признаки, реально использованные в моделях
    for tname, tser in targets.items():
        ws = wb.create_sheet(_short(tname, 28))
        r = 1
        ws.cell(r, 1, f"Модели для «{tname}»").font = Font(bold=True, size=12); r += 2
        _cat0, _desc0 = describe(tname)
        ws.cell(r, 1, f"Что это: {_desc0}").font = Font(italic=True); r += 2
        clusters = make_clusters(_second_target(df))
        for scope, mask in [("все игроки", None),
                            ("кластер: сильные", (clusters == "сильные").values),
                            ("кластер: слабые", (clusters == "слабые").values)]:
            res = fit_target(df, tser, mask, k, method, cv_mode=cv_mode)
            n = int(mask.sum()) if mask is not None else len(df)
            ws.cell(r, 1, f"▶ {scope} (n={n})").font = Font(bold=True, size=11); r += 1
            if "error" in res:
                ws.cell(r, 1, res["error"]); r += 2; continue
            used += list(res["features"])
            ws.cell(r, 1, "Уравнение (линейная):").font = Font(bold=True)
            ws.cell(r, 2, res["linear_eq"]); r += 1
            r = _fill_cmp(ws, r, res["models"])
            _bm = max(res["models"].items(),
                      key=lambda kv: (kv[1]["cv_r2"] if not np.isnan(kv[1]["cv_r2"]) else -9))
            summary.append((tname, scope, _bm[0], _bm[1]["cv_r2"], res["linear_eq"]))
            # важности лучшего дерева
            imp = res["models"].get("ГрадБустинг", {}).get("importances")
            if imp:
                ws.cell(r, 1, "Важность признаков (ГрадБустинг):").font = Font(italic=True); r += 1
                _hrow(ws, r, ["показатель", "важность", "категория", "что это (для врача)"]); r += 1
                for f, val in sorted(imp.items(), key=lambda t: t[1], reverse=True):
                    cat, desc = describe(f)
                    ws.cell(r, 1, _short(f, 40)); ws.cell(r, 2, round(float(val), 3))
                    ws.cell(r, 3, cat); ws.cell(r, 4, desc).alignment = Alignment(wrap_text=True)
                    for j in range(1, 5): ws.cell(r, j).border = BORD
                    r += 1
            # спецификация перцептрона
            mlp = res["models"].get("Перцептрон(MLP)", {}).get("mlp")
            if mlp:
                ws.cell(r, 1, f"Перцептрон: {mlp['arch']}, параметров: {mlp['n_params']}"
                              f" (при n={res['n']} → риск переобучения)").font = Font(italic=True)
                r += 1
            r += 1
        ws.column_dimensions["A"].width = 40
        ws.column_dimensions["B"].width = 14
        ws.column_dimensions["C"].width = 24
        ws.column_dimensions["D"].width = 60

    if include_pooled:
        ws = wb.create_sheet("POOLED (все точки)")
        r = 1
        ws.cell(r, 1, "Единая модель на кластер: строка = бросок игрока с точки.").font = Font(bold=True, size=12); r += 1
        ws.cell(r, 1, "Признаки: физиология + номер точки (one-hot) + кластер (сильный=1). "
                      "CV с группировкой по игроку.").font = Font(italic=True); r += 2
        res = pooled_model(df, method=method, k=k)
        used += list(res["feats"])
        ws.cell(r, 1, f"Строк: {res['n_rows']} | игроков: {res['n_players']} | "
                      f"физио-признаков: {len(res['feats'])}").font = Font(bold=True); r += 2
        r = _fill_cmp(ws, r, res["models"])
        _bm = max(res["models"].items(),
                  key=lambda kv: (kv[1]["cv_r2"] if not np.isnan(kv[1]["cv_r2"]) else -9))
        summary.append(("POOLED (все точки)", "все игроки", _bm[0], _bm[1]["cv_r2"], "см. лист POOLED"))
        # коэффициенты линейной pooled-модели
        ws.cell(r, 1, "Коэффициенты линейной pooled-модели:").font = Font(bold=True); r += 1
        _hrow(ws, r, ["Член", "Коэффициент"]); r += 1
        m = res["linear_ols"]; names = ["(константа)"] + res["Xcols"]
        for i, nm in enumerate(names):
            ws.cell(r, 1, _short(str(nm), 40)); ws.cell(r, 2, round(float(m["beta"][i]), 5))
            for j in (1, 2):
                ws.cell(r, j).border = BORD
            r += 1
        ws.column_dimensions["A"].width = 42
        ws.column_dimensions["B"].width = 15
    glossary_sheet(wb, df, used)
    write_methodology(wb, "ml")
    wb.save(out_path)
    return out_path, summary
