# -*- coding: utf-8 -*-
"""Точный счёт кардиоинтервалов ЗА РАЗГОН для всех 88 человек.

Ошибка, которую это исправляет: формула `КИ_за_нагрузку − Ннак_N3` вычитает лишнее.
`Ннак_N3` считается от начала записи (0 → 240 с) и включает предстарт со стартом,
а `КИ_за_нагрузку` начинается с `load_start`. В результате отрезок 0–60 с вычитается
дважды — систематический недосчёт около 90 ударов. Смещение почти постоянное
(Спирмен между вариантами 0,9945), поэтому на ранги внутри группы почти не влияет,
но абсолютные значения искажает, а между группами — по-разному, потому что делится
на разный прирост.

Правильно: удары от конца ДОО-фазы (`load_start + 180`) до `rec_start`, на шкале RR
(то есть минус сдвиг газ↔RR).
"""
import os, numpy as np, pandas as pd
B='/Users/mac-os/Documents/Батя'
os.chdir(B)

def ки(путь,t_от,t_до):
    rr=[]
    for s in open(путь,encoding='utf-8',errors='ignore'):
        s=s.strip()
        if not s or s.startswith(';'): continue
        try: v=float(s.replace(',','.'))
        except ValueError: continue
        if 250<=v<=2500: rr.append(v)
    if len(rr)<60: return np.nan
    t=np.cumsum(rr)/1000.0
    return float(((t>=t_от)&(t<t_до)).sum())

# --- спортсмены: границы из разметки, пути из словаря ---
D=pd.read_excel('длительность_и_нагрузка.xlsx',sheet_name=0)
S=pd.read_excel('Словарь_ФИО_и_путей.xlsx',sheet_name=0)
путь=dict(zip(S['ФИО_в_таблице_признаков'].astype(str),S['путь_RR'].astype(str)))
рез={}
for _,r in D.iterrows():
    имя=str(r['ФИО']); p=путь.get(имя)
    ls=pd.to_numeric(pd.Series([r.get('t_начало_нагрузки_с')]),errors='coerce').iloc[0]
    rs=pd.to_numeric(pd.Series([r.get('t_начало_восстановления_с')]),errors='coerce').iloc[0]
    off=pd.to_numeric(pd.Series([r.get('сдвиг_газ_RR_с')]),errors='coerce').iloc[0]
    if not p or p=='nan' or not os.path.exists(p): continue
    if not (np.isfinite(ls) and np.isfinite(rs)): continue
    if not np.isfinite(off): off=0.0
    if rs-ls<=180: рез[имя]=np.nan; continue
    рез[имя]=ки(p,ls+180-off,rs-off)
# --- контроль ---
k=pd.read_csv('outputs/_cache/контроль_признаки.csv')
import unicodedata
def фио(п):
    b=os.path.basename(str(п)).split('_')[0].strip().split()
    return unicodedata.normalize('NFC',' '.join(b[:2]))
for _,r in k.iterrows():
    ls,rs,off=r['load_start'],r['rec_start'],r['сдвиг_газ_RR_с']
    имя=фио(r['файл_rr'])
    рез[имя]=ки(r['файл_rr'],ls+180-off,rs-off) if (np.isfinite(ls) and rs-ls>180) else np.nan
print("посчитано:",len(рез),"| непусто:",sum(1 for v in рез.values() if np.isfinite(v)))

# --- вливаем в таблицу новым столбцом ---
xs=pd.ExcelFile('Полная_таблица_признаков_2026-08-25.xlsx')
листы={}
for л in xs.sheet_names:
    d=xs.parse(л)
    d['КИ_за_разгон']=[рез.get(str(i),np.nan) for i in d.iloc[:,0]]
    листы[л]=d
    n=d['КИ_за_разгон'].notna().sum()
    print("  %-14s строк %2d, КИ_за_разгон заполнено %2d, Ме=%s"%(л,len(d),n,
          ("%.0f"%d['КИ_за_разгон'].median()) if n else "—"))
with pd.ExcelWriter('Полная_таблица_признаков_2026-08-25.xlsx',engine='openpyxl') as w:
    for л,d in листы.items(): d.to_excel(w,sheet_name=л,index=False)
print("столбец КИ_за_разгон добавлен")
