# -*- coding: utf-8 -*-
"""Контрольная группа: сопоставление файлов, прогон gas.make, счёт КИ, сборка признаков."""
import sys, os, re, io, json, glob, contextlib, warnings
sys.path.insert(0,'/Users/mac-os/Documents/Батя/outputs/_cache')
sys.path.insert(0,'/Users/mac-os/Documents/Батя/outputs/_cache/prog')
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
B='/Users/mac-os/Documents/Батя'
os.chdir(B)
RRД='Баскетболисты/Баскетбол велоэргометр/Группа контроля'
ГАЗД='Баскетболисты/Баскетбол Газовый анализ/Группа контроля СРЕТ'

def норм(s):
    import unicodedata
    s=unicodedata.normalize('NFC',str(s)).strip().lower()
    for л,к in zip('cCeEoOaApPxXyYkKtTHMB','сСеЕоОаАрРхХуУкКтТНМВ'): s=s.replace(л,к)
    return re.sub(r'\s+',' ',s)

# --- пары по фамилии, с ручными исключениями по написанию ---
СИНОНИМЫ={'лопотецкий':'лопотетский','тошзода':'тошзода'}
пары=[]
газ_файлы={}
for p in glob.glob(os.path.join(ГАЗД,'*.xlsx')):
    m=re.match(r'(\d{8})\s+(.+?)\s*\(CPET',os.path.basename(p))
    if not m: continue
    слова=норм(m.group(2)).split()
    газ_файлы[p]=(m.group(1),слова)
for rp in sorted(glob.glob(os.path.join(RRД,'*.rr'))):
    фам=норм(os.path.basename(rp).split()[0])
    фам_alt=СИНОНИМЫ.get(фам,фам)
    найдено=None
    for gp,(дата,слова) in газ_файлы.items():
        if фам in слова or фам_alt in слова:
            найдено=gp; break
    пары.append(dict(файл_rr=rp,файл_газ=найдено,фамилия=фам))
есть=[p for p in пары if p['файл_газ']]
print("RR всего %d, с газом %d, без газа: %s"%(len(пары),len(есть),
      [p['фамилия'] for p in пары if not p['файл_газ']]))

# --- счёт кардиоинтервалов из .rr ---
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

import gas
СУФФ=['%МПК','VO2','VCO2','RER','VE','VE/VO2','VE/VCO2','RR','O2pulse','ЧСС','HRR']
КЛЮЧ={'т1':'т1','т2':'т2','т3':'т3','т4':'т4','коноо':'коноо','восст_1мин':'восст_1мин',
       'восст_2мин':'восст_2мин','восст_замедление':'восст_замедление',
       'восст_max_RER':'восст_max_RER','восст_ЧСС<100':'восст_ЧСС<100','восст_ЧСС<125':'восст_ЧСС<125'}
ПОЛЕ={'%МПК':'pct','VO2':'VO2','VCO2':'VCO2','RER':'RER','VE':'VE','VE/VO2':'VE/VO2',
      'VE/VCO2':'VE/VCO2','RR':'RR','O2pulse':'O2pulse','ЧСС':'ЧСС','HRR':'HRR'}
строки=[]
for i,p in enumerate(есть,1):
    буф=io.StringIO()
    try:
        with contextlib.redirect_stdout(буф):
            r=gas.make(rr_path=p['файл_rr'],gas_path=p['файл_газ'],report_only=True,
                       prestart_s=30,start_s=30,hr_thresholds=(100,125),doo_s=180.0)
    except Exception as e:
        print("  %2d/%d %-14s ОШИБКА: %s"%(i,len(есть),p['фамилия'],str(e)[:80])); continue
    ls,rs,off=r.get('load_start'),r.get('rec_start'),r.get('offset_s') or 0.0
    d=r.get('detail',{})
    зап={'ФИО':None,'мин_нагрузки':(rs-ls)/60.0 if ls is not None and rs is not None else np.nan,
         'ЧСС_пик':r.get('hr_peak'),'load_start':ls,'rec_start':rs,'сдвиг_газ_RR_с':off}
    for имя,к in КЛЮЧ.items():
        точка=d.get(к) or {}
        зап['газ_%s_t'%имя]=точка.get('t')
        for суф in СУФФ:
            if суф in ПОЛЕ: зап['газ_%s_%s'%(имя,суф)]=точка.get(ПОЛЕ[суф])
    # КИ на шкале RR: газовые метки минус сдвиг
    if ls is not None and rs is not None:
        зап['КИ_за_нагрузку']=ки(p['файл_rr'],ls-off,rs-off)
        зап['КИ_всего_до_восст']=ки(p['файл_rr'],0,rs-off)
        for n in (1,2,3,4,5):
            зап['Ннак_N%d'%n]=ки(p['файл_rr'],0,60.0+60.0*n)
    зап['файл_rr']=p['файл_rr']; зап['файл_газ']=p['файл_газ']; зап['фамилия']=p['фамилия']
    строки.append(зап)
    print("  %2d/%d %-14s load=%s rec=%s мин_нагр=%.2f КИ=%s"%(
        i,len(есть),p['фамилия'],ls,rs,зап['мин_нагрузки'],зап.get('КИ_за_нагрузку')))
d=pd.DataFrame(строки)
d.to_csv(f'{B}/outputs/_cache/контроль_признаки.csv',index=False)
print("\nготово, строк:",len(d))
