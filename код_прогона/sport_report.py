# -*- coding: utf-8 -*-
import os,sys,html,warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
sys.path.insert(0,'/Users/mac-os/Documents/Батя/outputs/_cache')
from report2 import CSS,GLOSS,esc,star,cls,ftable
from analyze import DOM_RU

def sport_report(res, sport, outdir, target_name, note=''):
    """res: {(sport,nm):{n,nsig,DS,LEAD,DOM2}} для nm в Все/Сильные/Слабые"""
    NMS=[k[1] for k in res if k[0]==sport]
    order=[x for x in ['Все','Сильные','Слабые'] if x in NMS]
    H=[f'<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><title>{esc(sport)} — свод</title>{CSS}</head><body>']
    H.append(f'<h1>{esc(sport)} — сводный отчёт</h1>')
    H.append(f'<p class="lead">Целевая переменная: <b>{esc(target_name)}</b>. Сравниваются подгруппы внутри одного вида спорта: вся группа, сильные и слабые (деление по медиане цели).{note}</p>')
    H.append('<h2>1. Объём и число значимых связей</h2>')
    h='<table><tr><th class="l">Подгруппа</th><th>Человек</th><th>Значимых признаков (p&lt;0.05)</th></tr>'
    for nm in order:
        v=res[(sport,nm)]
        h+=f'<tr><td class="l">{esc(nm)}</td><td>{v["n"]}</td><td>{v["nsig"]}</td></tr>'
    H.append(h+'</table>')
    H.append('<h2>2. Ведущие признаки в каждой подгруппе</h2>')
    for nm in order:
        v=res[(sport,nm)]
        H.append(f'<h3>{esc(nm)} (n={v["n"]})</h3>')
        H.append(ftable(v['LEAD'].head(8)))
    H.append('<h2>3. Динамика систем признаков: сильные против слабых</h2>')
    H.append('<p>Для каждой системы показан максимум |r| среди её <b>значимых</b> признаков. '
             'Прочерк означает, что в этой подгруппе у системы нет ни одной значимой связи. '
             'Сравнение ведётся только внутри одного вида спорта.</p>')
    doms=set()
    for nm in order: doms|=set(res[(sport,nm)]['DS']['домен'])
    rows=[]
    for dm in doms:
        row={'домен':dm,'домен_ru':DOM_RU.get(dm,dm)}
        for nm in order:
            DS=res[(sport,nm)]['DS']; g=DS[DS['домен']==dm]
            row[nm]=float(g['макс |r|'].iloc[0]) if len(g) and g['значимых'].iloc[0]>0 else np.nan
            row[nm+'_k']=int(g['значимых'].iloc[0]) if len(g) else 0
        rows.append(row)
    T=pd.DataFrame(rows)
    T['_s']=T[[nm for nm in order]].max(axis=1)
    T=T.sort_values('_s',ascending=False)
    h='<table><tr><th class="l">Система признаков</th>'+''.join(f'<th>{esc(nm)}<br><span class="small">макс |r| / значимых</span></th>' for nm in order)+'<th class="l">Динамика</th></tr>'
    dyn_txt=[]
    for _,x in T.iterrows():
        cells=''
        for nm in order:
            v=x[nm]
            cells+= f'<td class="{cls(v) if v==v else "b"}">{f"{v:.2f}" if v==v else "—"}<br><span class="small">{int(x[nm+"_k"])}</span></td>'
        # словесная динамика сильные vs слабые
        s=x.get('Сильные',np.nan); w=x.get('Слабые',np.nan)
        if s==s and w==w:
            d=s-w
            if abs(d)<0.10: t='примерно одинаково'
            elif d>0: t=f'<b>сильнее у сильных</b> (+{d:.2f})'
            else: t=f'<b>сильнее у слабых</b> ({d:.2f})'
        elif s==s and w!=w: t='<b>только у сильных</b>'
        elif w==w and s!=s: t='<b>только у слабых</b>'
        else: t='не значимо ни там, ни там'
        dyn_txt.append((x['домен_ru'],t,s,w))
        h+=f'<tr><td class="l">{esc(x["домен_ru"])}</td>{cells}<td class="l small">{t}</td></tr>'
    H.append(h+'</table>')
    # словесное описание динамики
    H.append('<h3>Что происходит с системами при переходе от слабых к сильным</h3><ul>')
    for dom,t,s,w in dyn_txt:
        if 'только у сильных' in t:
            H.append(f'<li><b>{esc(dom)}</b> — значимые связи появляются <b>только у сильных</b> (макс |r| = {s:.2f}). У слабых система не работает: различия внутри неё не связаны с результатом.</li>')
        elif 'только у слабых' in t:
            H.append(f'<li><b>{esc(dom)}</b> — значима <b>только у слабых</b> (макс |r| = {w:.2f}). У сильных перестаёт различать: вероятно, там показатель выходит на общий высокий уровень и разброс исчезает.</li>')
        elif 'сильнее у сильных' in t:
            H.append(f'<li><b>{esc(dom)}</b> — вклад растёт от слабых к сильным: {w:.2f} → {s:.2f}. С ростом мастерства эта система становится более определяющей.</li>')
        elif 'сильнее у слабых' in t:
            H.append(f'<li><b>{esc(dom)}</b> — вклад падает: {w:.2f} у слабых против {s:.2f} у сильных. Система важна на начальном уровне, но у подготовленных перестаёт быть узким местом.</li>')
        elif 'одинаково' in t:
            H.append(f'<li><b>{esc(dom)}</b> — вклад стабилен ({w:.2f} и {s:.2f}), от уровня подготовки не зависит.</li>')
        else:
            H.append(f'<li><b>{esc(dom)}</b> — значимых связей нет ни в одной подгруппе.</li>')
    H.append('</ul>')
    H.append(GLOSS)
    H.append('<p class="small">Подробности по каждой подгруппе — в файле <code>Отчёт.html</code> внутри соответствующей папки, там же обе плеяды и таблица значимых признаков.</p>')
    H.append('</body></html>')
    os.makedirs(outdir,exist_ok=True)
    open(f'{outdir}/00_Отчёт_{sport}.html','w',encoding='utf-8').write('\n'.join(H))
