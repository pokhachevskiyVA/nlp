"""Газовые признаки без RR-файла того же теста.

Приём: синтетический RR-ряд строится из СОБСТВЕННОГО столбца ЧСС газового файла,
поэтому синхронизация RR<->газ становится тождественной (сдвиг 0), и вся штатная
логика gas.make (точки 1-4, %МПК, O2pulse, маркеры восстановления) работает на
временной шкале газа. Из результата берутся ТОЛЬКО газовые признаки.
"""
import sys, json, io, os, contextlib, warnings, tempfile
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/sessions/lucid-ecstatic-cannon/mnt/outputs')
import gas
C = '/sessions/lucid-ecstatic-cannon/mnt/outputs/_cache'

CURVES = ['VO2', 'VCO2', 'RER', 'VE', 'VE/VO2', 'VE/VCO2', 'RR', 'O2pulse']
PTS = ['т1', 'т2', 'т3', 'т4', 'восст_1мин', 'восст_2мин',
       'восст_замедление', 'восст_ЧСС<100', 'восст_max_RER']


def synth_rr(xl):
    """RR-ряд (мс) по столбцу ЧСС газового файла; шкала совпадает со шкалой газа."""
    df = pd.read_excel(xl)
    t = pd.to_timedelta(df['t'].loc[2:].apply(str)).dt.total_seconds().values.astype(float)
    hr = pd.to_numeric(df['ЧСС'].loc[2:], errors='coerce').values.astype(float)
    hr[(hr < 30) | (hr > 230)] = np.nan
    ok = np.isfinite(t) & np.isfinite(hr)
    t, hr = t[ok], hr[ok]
    if len(t) < 20:
        raise ValueError('мало точек ЧСС')
    rr, cur = [], float(t[0])
    while cur < t[-1]:
        h = float(np.interp(cur, t, hr))
        iv = 60000.0 / h
        rr.append(iv)
        cur += iv / 1000.0
    # предстарт: добить ряд от нуля, чтобы накопленное время = времени газа
    pre = []
    if t[0] > 0:
        h0 = float(hr[0]); iv0 = 60000.0 / h0
        pre = [iv0] * int(t[0] * 1000.0 / iv0)
    return np.array(pre + rr), df, t


def preload_s(df, t):
    """Секунда начала нагрузки: первый отсчёт фазы EXERCISE."""
    if 'Фаза' in df.columns:
        ph = df['Фаза'].loc[2:].astype(str).str.upper().values
        n = min(len(ph), len(t))
        idx = np.where(ph[:n] == 'EXERCISE')[0]
        if len(idx):
            return float(t[idx[0]])
    return 60.0


def extract(surname, xl, sport, istochnik):
    rrv, df, t = synth_rr(xl)
    pl = preload_s(df, t)
    fd, tmp = tempfile.mkstemp(suffix='.rr')
    with os.fdopen(fd, 'w') as fh:
        fh.write('ОВР\n' + '\n'.join(f'{v:.1f}' for v in rrv) + '\n')
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = gas.make(rr_path=tmp, gas_path=xl, report_only=True,
                         prestart_s=pl / 2.0, start_s=pl / 2.0)
    finally:
        os.unlink(tmp)
    f = {'surname': surname, 'sport': sport, 'источник': istochnik,
         'ЧСС_пик': r.get('hr_peak')}
    det = r.get('detail') or {}
    for pt in PTS:
        q = det.get(pt) or {}
        f[f'газ_{pt}_t'] = q.get('t')
        f[f'газ_{pt}_%МПК'] = q.get('pct')
        for c in CURVES:
            f[f'газ_{pt}_{c}'] = q.get(c)
        if 'ЧСС' in q:
            f[f'газ_{pt}_ЧСС'] = q.get('ЧСС')
        if 'HRR' in q:
            f[f'газ_{pt}_HRR'] = q.get('HRR')
    f['газ_пик_ЧСС'] = r.get('hr_peak')
    f['_load_start'] = r.get('load_start')
    f['_rec_start'] = r.get('rec_start')
    return f


if __name__ == '__main__':
    tasks = json.load(open(C + '/gasonly_list.json'))
    out = open(C + '/gasonly.jsonl', 'a')
    for sn, xl, sport, ist in tasks:
        try:
            out.write(json.dumps(extract(sn, xl, sport, ist),
                                 ensure_ascii=False, default=float) + '\n')
            print('ok', sn)
        except Exception as e:
            out.write(json.dumps({'surname': sn, 'error': str(e)[:120]},
                                 ensure_ascii=False) + '\n')
            print('ERR', sn, str(e)[:120])
    out.close()
