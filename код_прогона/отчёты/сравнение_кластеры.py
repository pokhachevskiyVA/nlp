# -*- coding: utf-8 -*-
"""Сравнение «сильный кластер / слабый кластер / контроль» отдельно по видам спорта.

Кластеры — по медиане прироста нагрузки (основная мера работоспособности).
Формат тот же, что в Сравнение_трёх_групп.xlsx: Ме [25Пц; 75Пц], Краскел–Уоллис
по трём группам и Манн–Уитни попарно.
"""
import sys, os
sys.path.insert(0,'/Users/mac-os/Documents/Батя/outputs/_cache')
sys.path.insert(0,'/Users/mac-os/Documents/Батя/outputs/_cache/prog')
import numpy as np, pandas as pd
from scipy import stats as st
import cpet
B='/Users/mac-os/Documents/Батя'
ПРИР=f'{cpet.Y}/ГазоваяЭргометрия_2026-08-24/Прирост'
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
    Nн=pd.to_numeric(d['КИ_за_нагрузку'],errors='coerce')
    t=pd.DataFrame({
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
      'Кардиоинтервалов за нагрузку':Nн,
      'Средняя ЧСС за нагрузку, уд/мин':Nн/T,
    })
    t.insert(0,'ФИО',d.iloc[:,0].values)
    return t

ВСЕ={с:набор(с) for с in ('Лыжники','Баскетболисты','Контроль')}
КЛ={}
for с in ('Лыжники','Баскетболисты'):
    v=pd.read_excel(f'{ПРИР}/{с}/Все/_вход.xlsx')
    КЛ[с]=dict(zip(v.iloc[:,0].astype(str),v['кластер']))

def f_(v,z=1): return "—" if not np.isfinite(v) else ("%.*f"%(z,v)).replace(".",",")
def кв(s):
    s=pd.Series(s).dropna()
    return (len(s),s.median(),s.quantile(.25),s.quantile(.75)) if len(s) else (0,np.nan,np.nan,np.nan)

def сравнить(группы, файл):
    """группы: список (подпись, датафрейм)."""
    показатели=[c for c in группы[0][1].columns if c!='ФИО']
    строки=[]
    for п in показатели:
        м0=pd.Series(группы[0][1][п]).dropna().median()
        зн=0 if abs(м0)>=100 else (1 if abs(м0)>=10 else 2)
        р={'показатель':п}; ряды={}
        for имя,d in группы:
            n,м,q1,q3=кв(d[п]); ряды[имя]=pd.Series(d[п]).dropna()
            р[имя]="%s [%s; %s]"%(f_(м,зн),f_(q1,зн),f_(q3,зн)); р[имя+' n']=n
        сп=[v for v in ряды.values() if len(v)>3]
        р['p, три группы']=st.kruskal(*сп).pvalue if len(сп)==3 else np.nan
        имена=[и for и,_ in группы]
        for i in range(3):
            for j in range(i+1,3):
                a,b=ряды[имена[i]],ряды[имена[j]]
                р['p, %s–%s'%(имена[i],имена[j])]=st.mannwhitneyu(a,b).pvalue if len(a)>3 and len(b)>3 else np.nan
        строки.append(р)
    Т=pd.DataFrame(строки)
    имена=[и for и,_ in группы]
    кол=['показатель']+[c for и in имена for c in (и,и+' n')]+['p, три группы']+\
        ['p, %s–%s'%(имена[i],имена[j]) for i in range(3) for j in range(i+1,3)]
    Т=Т[кол]
    os.makedirs(os.path.dirname(файл),exist_ok=True)
    with pd.ExcelWriter(файл,engine='openpyxl') as w:
        Т.to_excel(w,sheet_name='Сравнение',index=False)
        import re as _re
        for и,d in группы:
            лист=_re.sub(r'[:\\/?*\[\]]','',и)[:31]
            d.to_excel(w,sheet_name=лист,index=False)
    return Т

for спорт,кратко in (('Лыжники','Лыжники'),('Баскетболисты','Баскет')):
    d=ВСЕ[спорт]
    сил=d[[КЛ[спорт].get(str(f))=='сильные' for f in d['ФИО']]].reset_index(drop=True)
    сла=d[[КЛ[спорт].get(str(f))=='слабые'  for f in d['ФИО']]].reset_index(drop=True)
    группы=[(f'{спорт}: сильные',сил),(f'{спорт}: слабые',сла),('Контроль',ВСЕ['Контроль'])]
    папка=f'{cpet.Y}/ГазоваяЭргометрия_{кратко}_и_Контроль_2026-08-25'
    Т=сравнить(группы,f'{папка}/Сравнение_трёх_групп.xlsx')
    print("=====",спорт,"| сильные %d, слабые %d, контроль %d"%(len(сил),len(сла),len(ВСЕ['Контроль'])))
    pd.set_option('display.width',260)
    print(Т[[Т.columns[0],группы[0][0],группы[1][0],'Контроль','p, три группы']].to_string(index=False))
    print()
