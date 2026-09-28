# -*- coding: utf-8 -*-
"""Сравнение трёх групп: лыжники, баскетболисты, контроль."""
import sys, os
sys.path.insert(0,'/Users/mac-os/Documents/Батя/outputs/_cache')
sys.path.insert(0,'/Users/mac-os/Documents/Батя/outputs/_cache/prog')
import numpy as np, pandas as pd
from scipy import stats as st
import cpet
B='/Users/mac-os/Documents/Батя'
Y=f'{cpet.Y}/ГазоваяЭргометрия_Контроль_2026-08-25'
xs=pd.ExcelFile(cpet.FULL)

def набор(sheet):
    d=xs.parse(sheet)
    T=pd.to_numeric(d['мин_нагрузки'],errors='coerce')
    Δ=30*(T.where(T>3)-3)
    t4=pd.to_numeric(d['газ_т4_t'],errors='coerce'); tk=pd.to_numeric(d['газ_коноо_t'],errors='coerce')
    V4=pd.to_numeric(d['газ_т4_VO2'],errors='coerce'); Vk=pd.to_numeric(d['газ_коноо_VO2'],errors='coerce')
    dW=30*(t4-tk)/60
    о2=((V4-Vk)/dW.where(dW>15)).where(lambda v:v.between(0,40))
    N=pd.to_numeric(d['КИ_за_разгон'],errors='coerce')
    хрон=(N/Δ.where(Δ>30)).where(lambda v:v.between(0,30))
    return pd.DataFrame({
      'Длительность нагрузки, мин':T,
      'Прирост нагрузки, Вт':Δ,
      'Пиковая ЧСС, уд/мин':pd.to_numeric(d['ЧСС_пик'],errors='coerce'),
      'VO2 в точке 4, мл/мин':V4,
      'VO2 в конце ДОО-фазы, мл/мин':Vk,
      'Доля от пика в конце ДОО-фазы, %':pd.to_numeric(d['газ_коноо_%МПК'],errors='coerce'),
      'O2-пульс в точке 4, мл/уд':pd.to_numeric(d['газ_т4_O2pulse'],errors='coerce'),
      'VE/VO2 в точке 3':pd.to_numeric(d['газ_т3_VE/VO2'],errors='coerce'),
      'RER в точке 4':pd.to_numeric(d['газ_т4_RER'],errors='coerce'),
      'Кислородная цена прироста, мл/мин/Вт':о2,
      'Хронотропная цена прироста, уд/Вт':хрон,
      'Кардиоинтервалов за разгон':N,
      'Кардиоинтервалов за нагрузку':pd.to_numeric(d['КИ_за_нагрузку'],errors='coerce'),
      'Средняя ЧСС за нагрузку, уд/мин':pd.to_numeric(d['КИ_за_нагрузку'],errors='coerce')/T,
    }), d.iloc[:,0].values

Г={}; ФИО={}
for г,л in (('Лыжники','Лыжники'),('Баскетболисты','Баскетболисты'),('Контроль','Контроль')):
    Г[г],ФИО[г]=набор(л)
показатели=list(Г['Лыжники'].columns)

def кв(s):
    s=pd.Series(s).dropna()
    return (len(s),s.median(),s.quantile(.25),s.quantile(.75)) if len(s) else (0,np.nan,np.nan,np.nan)
def f_(v,z=1): return "—" if not np.isfinite(v) else ("%.*f"%(z,v)).replace(".",",")

строки=[]
for п in показатели:
    м0=pd.Series(Г['Лыжники'][п]).dropna().median()
    зн=0 if abs(м0)>=100 else (1 if abs(м0)>=10 else 2)
    р={'показатель':п}; ряды={}
    for г in Г:
        n,м,q1,q3=кв(Г[г][п]); ряды[г]=pd.Series(Г[г][п]).dropna()
        р[г]="%s [%s; %s]"%(f_(м,зн),f_(q1,зн),f_(q3,зн)); р[г+' n']=n
    сп=[v for v in ряды.values() if len(v)>3]
    р['p, три группы']=st.kruskal(*сп).pvalue if len(сп)==3 else np.nan
    for г in ('Лыжники','Баскетболисты'):
        a,b=ряды['Контроль'],ряды[г]
        р['p, контроль–'+г]=st.mannwhitneyu(a,b).pvalue if len(a)>3 and len(b)>3 else np.nan
    строки.append(р)
Т=pd.DataFrame(строки)[['показатель']+[c for г in ('Лыжники','Баскетболисты','Контроль') for c in (г,г+' n')]
                       +['p, три группы','p, контроль–Лыжники','p, контроль–Баскетболисты']]
os.makedirs(Y,exist_ok=True)
with pd.ExcelWriter(f'{Y}/Сравнение_трёх_групп.xlsx',engine='openpyxl') as w:
    Т.to_excel(w,sheet_name='Сравнение',index=False)
    for г in Г:
        d=Г[г].copy(); d.insert(0,'ФИО',ФИО[г]); d.to_excel(w,sheet_name=г[:31],index=False)
Т.to_pickle(f'{B}/outputs/_cache/сравнение3.pkl')
pd.set_option('display.width',250)
print(Т[['показатель','Лыжники','Баскетболисты','Контроль','p, три группы','p, контроль–Лыжники']].to_string(index=False))
