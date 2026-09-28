# -*- coding: utf-8 -*-
import os,sys,html,warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
sys.path.insert(0,'/Users/mac-os/Documents/Батя/outputs/_cache')
from analyze import analyze, draw, DOM_RU

CSS="""<style>
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 22px;color:#1a1a1a;line-height:1.6}
h1{font-size:23px;color:#14315e;margin-bottom:2px} h2{font-size:19px;color:#14315e;margin-top:30px;border-bottom:2px solid #e3e8f0;padding-bottom:5px}
h3{font-size:16px;color:#1a3e6e;margin:18px 0 5px}
.lead{color:#555;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0}
th,td{border:1px solid #d5dbe3;padding:6px 9px;text-align:center} th{background:#14315e;color:#fff} td.l,th.l{text-align:left}
td.g{background:#e2efda;font-weight:700} td.y{background:#fff2cc;font-weight:700} td.b{background:#f8d7da;font-weight:700}
tr.sec td{background:#eef1f6;font-weight:700;color:#14315e;text-align:left}
.box{background:#f3f6fb;border-left:4px solid #14315e;padding:10px 14px;margin:12px 0}
.warn{background:#fff8e6;border:1px solid #f0d98a;border-radius:6px;padding:10px 14px;margin:12px 0}
.ok{background:#e2efda;border:1px solid #b5d3a0;border-radius:6px;padding:10px 14px;margin:12px 0}
code{background:#f0f3f7;padding:1px 5px;border-radius:4px;font-size:12.5px;color:#14315e;font-weight:600}
.small{color:#666;font-size:12.5px}
.gloss{background:#fafbfd;border:1px solid #e0e5ec;border-radius:6px;padding:10px 16px;margin:12px 0;font-size:12.5px;color:#444}
.gloss b{color:#14315e} .gloss div{margin:3px 0}
img{max-width:100%;border:1px solid #e0e5ec;border-radius:6px;margin:8px 0}
</style>"""
GLOSS="""<div class="gloss"><b>Как читать</b>
<div><b>r</b> — коэффициент корреляции Спирмена с целевой переменной. Знак показывает направление: «+» — чем больше признак, тем выше результат; «−» — наоборот.</div>
<div><b>p</b> — вероятность случайно получить такую связь. Значимыми считаем признаки при p&lt;0.05: * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001.</div>
<div><b>Плеяда 1 «Ведущие»</b> — самые сильные значимые связи, независимо от того, к какой системе организма относится признак.</div>
<div><b>Плеяда 2 «По доменам»</b> — сильнейший <i>значимый</i> признак от каждой системы. Нужна, чтобы кардиопоказатели (их в таблице больше тысячи) не вытеснили стабилометрию, анатомию и психологию, которых десятки.</div>
<div><b>Газовые точки.</b> Порог 1–4 определяется одновременно по нескольким кривым (VO₂, VCO₂, RER, VE, O₂-пульс). В плеяде такая точка показана <b>один раз</b> — по сильнейшей из своих кривых, чтобы не дублировать одну и ту же находку.</div>
<div><b>Признак-двойник</b> — показатель, у которого взаимная |r| с уже отобранным ≥ 0.90; из такой пары остаётся один представитель.</div>
<div><b>Домены (системы признаков):</b> Кардио нагрузки и восстановления (RR-интервалы по минутам, pNN, перегибы, точка Михайлова) · Газообмен: пороги 1–4, восстановление, сводные (МПК, VO₂) · Пульсовая цена (Вт на удар) · ВСР покоя · Стабилометрия · Анатомия и состав тела · Психология.</div>
</div>"""

def esc(x): return html.escape(str(x))
def star(p): return '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ''))
def cls(r):
    a=abs(r); return 'g' if a>=0.6 else ('y' if a>=0.4 else '')

def ftable(df,maxrows=None):
    if not len(df): return '<p class="small">— значимых признаков не найдено —</p>'
    d=df.head(maxrows) if maxrows else df
    h='<table><tr><th class="l">Признак</th><th class="l">Система</th><th>r</th><th>p</th><th>n</th></tr>'
    for _,x in d.iterrows():
        h+=(f'<tr><td class="l"><code>{esc(x["признак"])}</code></td><td class="l">{esc(x["домен_ru"])}</td>'
            f'<td class="{cls(x["r"])}">{x["r"]:+.3f}{star(x["p"])}</td><td>{x["p"]:.1e}</td><td>{int(x["n"])}</td></tr>')
    return h+'</table>'

def domain_summary(F,alpha=0.05):
    S=F[F['p']<alpha]
    rows=[]
    for dm,g in F.groupby('домен'):
        sg=g[g['p']<alpha]
        rows.append({'домен':dm,'домен_ru':DOM_RU.get(dm,dm),'всего':len(g),'значимых':len(sg),
                     'доля':len(sg)/len(g) if len(g) else 0,
                     'макс |r|':sg['|r|'].max() if len(sg) else np.nan,
                     'топ':sg.iloc[0]['признак'] if len(sg) else '—',
                     'знак':('+' if len(sg) and sg.iloc[0]['r']>0 else ('−' if len(sg) else ''))})
    return pd.DataFrame(rows).sort_values('макс |r|',ascending=False)

def group_report(src,outdir,group,target_col,target_name,sport_note=''):
    os.makedirs(outdir,exist_ok=True)
    d,y,F,LEAD,DOM2,SHORT=analyze(src,target_col)
    n=len(d)
    tshort='Результат' if 'езультатив' in target_col else ('Пик Вт' if 'ощност' in target_col else target_col[:10])
    draw(LEAD,d,y,f'{outdir}/Плеяда_1_ведущие.png',f'Плеяда — {group}\nведущие значимые признаки | {target_name}',target_name,by_domain=False,target_short=tshort)
    draw(DOM2,d,y,f'{outdir}/Плеяда_2_по_доменам.png',f'Плеяда — {group}\nпо сильнейшему значимому признаку от каждой системы | {target_name}',target_name,by_domain=True,target_short=tshort)
    DS=domain_summary(F)
    with pd.ExcelWriter(f'{outdir}/Значимые_признаки.xlsx',engine='openpyxl') as w:
        F[F['p']<0.05][['признак','домен_ru','r','p','n']].to_excel(w,index=False,sheet_name='Все значимые')
        SHORT[['признак','домен_ru','r','p','n']].to_excel(w,index=False,sheet_name='Без повторов маркеров')
        LEAD[['признак','домен_ru','r','p','n']].to_excel(w,index=False,sheet_name='Плеяда 1')
        DOM2[['признак','домен_ru','r','p','n']].to_excel(w,index=False,sheet_name='Плеяда 2')
        DS.to_excel(w,index=False,sheet_name='Сводка по системам')
    nsig=int((F['p']<0.05).sum())
    H=[f'<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><title>{esc(group)}</title>{CSS}</head><body>']
    H.append(f'<h1>{esc(group)}</h1>')
    H.append(f'<p class="lead">Целевая переменная: <b>{esc(target_name)}</b>. В группе <b>{n}</b> человек. '
             f'Проверено признаков: {len(F)}, из них <b>статистически значимых {nsig}</b> (p&lt;0.05).{sport_note}</p>')
    # ЧАСТЬ 1
    H.append('<h2>Часть 1. Ведущие признаки</h2>')
    H.append('<img src="Плеяда_1_ведущие.png" alt="плеяда ведущих">')
    H.append('<h3>Что показывает первая плеяда</h3>')
    H.append(ftable(LEAD))
    if len(LEAD):
        top=LEAD.iloc[0]
        dom_lead=LEAD['домен_ru'].value_counts()
        txt=(f'Сильнейшая связь — <b>{esc(top["признак"])}</b> ({esc(top["домен_ru"])}), r = {top["r"]:+.2f}{star(top["p"])}. '
             f'В верхушке преобладает система <b>{esc(dom_lead.index[0])}</b> ({dom_lead.iloc[0]} из {len(LEAD)} признаков).')
        others=[k for k in dom_lead.index[1:4]]
        if others: txt+=' Также представлены: '+', '.join(esc(o) for o in others)+'.'
        H.append(f'<div class="ok">{txt}</div>')
        pos=(LEAD['r']>0).sum(); neg=len(LEAD)-pos
        H.append(f'<p class="small">Из {len(LEAD)} ведущих признаков {pos} связаны с целью положительно, {neg} — отрицательно.</p>')
    # ЧАСТЬ 2
    H.append('<h2>Часть 2. Динамика по системам признаков</h2>')
    H.append('<p>Первая плеяда показывает только самое сильное — и там почти всегда доминируют кардио и газообмен, просто потому что таких признаков в таблице на порядок больше. '
             'Вторая плеяда устроена иначе: от каждой системы берётся <b>сильнейший значимый</b> признак. Так видно вклад тех систем, которые в общем списке теряются.</p>')
    H.append('<img src="Плеяда_2_по_доменам.png" alt="плеяда по доменам">')
    H.append('<h3>Представленность систем</h3>')
    h='<table><tr><th class="l">Система признаков</th><th>Проверено</th><th>Значимых</th><th>Доля значимых</th><th>Макс |r|</th><th class="l">Сильнейший признак</th></tr>'
    for _,x in DS.iterrows():
        mr=x['макс |r|']
        h+=(f'<tr><td class="l">{esc(x["домен_ru"])}</td><td>{int(x["всего"])}</td><td>{int(x["значимых"])}</td>'
            f'<td>{x["доля"]*100:.0f}%</td>'
            f'<td class="{cls(mr) if mr==mr else "b"}">{f"{mr:.2f}" if mr==mr else "—"}</td>'
            f'<td class="l"><code>{esc(x["топ"])}</code> {esc(x["знак"])}</td></tr>')
    H.append(h+'</table>')
    H.append('<h3>Разбор по системам</h3>')
    for _,x in DS.iterrows():
        if x['значимых']==0:
            H.append(f'<p><b>{esc(x["домен_ru"])}</b> — <span class="small">значимых связей нет ни у одного из {int(x["всего"])} признаков.</span></p>'); continue
        sub=F[(F['домен']==x['домен'])&(F['p']<0.05)].head(5)
        strength=('сильная' if x['макс |r|']>=0.6 else ('умеренная' if x['макс |r|']>=0.4 else 'слабая'))
        H.append(f'<p><b>{esc(x["домен_ru"])}</b> — {int(x["значимых"])} значимых из {int(x["всего"])}, максимум |r| = {x["макс |r|"]:.2f} ({strength} связь).</p>')
        H.append(ftable(sub))
    H.append(GLOSS)
    H.append('<div class="box">Это <b>краткая</b> версия: от каждого маркера оставлен один представитель — '
             'газовая точка показана один раз (по сильнейшей кривой), точка Михайлова один раз (по лучшему порогу), '
             'pNN один раз (наибольший порог). Полный список всех вариантов — в файле <code>Отчёт_полный.html</code>.</div>')
    H.append(f'<p class="small">Рядом: <code>Отчёт_полный.html</code>, <code>Плеяда_1_ведущие.png</code>, <code>Плеяда_2_по_доменам.png</code>, <code>Значимые_признаки.xlsx</code> (листы: все значимые, обе плеяды, сводка по системам).</p>')
    H.append('</body></html>')
    open(f'{outdir}/Отчёт_краткий.html','w',encoding='utf-8').write('\n'.join(H))
    _full_report(outdir,group,target_name,n,F,SHORT,DS,sport_note)
    return dict(n=n,nsig=nsig,F=F,LEAD=LEAD,DOM2=DOM2,DS=DS)


def _full_report(outdir,group,target_name,n,F,SHORT,DS,sport_note=''):
    """Полный отчёт: все значимые признаки без схлопывания маркеров."""
    S=F[F['p']<0.05].copy()
    keep=set(SHORT['признак'])
    H=[f'<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><title>{esc(group)} — полный</title>{CSS}</head><body>']
    H.append(f'<h1>{esc(group)} — полный отчёт</h1>')
    H.append(f'<p class="lead">Целевая переменная: <b>{esc(target_name)}</b>. В группе <b>{n}</b> человек. '
             f'Показаны <b>все {len(S)} значимых признаков</b> (p&lt;0.05), включая все варианты одного маркера: '
             f'газовую точку по каждой кривой, точку Михайлова при каждом пороге, pNN при каждом пороге.{sport_note}</p>')
    H.append('<div class="box">Этот отчёт — для проверки полноты: здесь ничего не схлопнуто. '
             'Если порог 1 оказался значимым по VO₂, VCO₂ и RER — все три строки будут видны. '
             'Краткая версия с одним представителем на маркер — в файле <code>Отчёт_краткий.html</code>.</div>')
    # маркерные семейства
    H.append('<h2>1. Семейства маркеров: сколько вариантов оказалось значимым</h2>')
    G=S[S['группа_маркера'].notna()].copy()
    if len(G):
        G['гр']=G['группа_маркера'].astype(str)
        rows=[]
        for gk,g in G.groupby('гр'):
            best=g.sort_values('|r|',ascending=False).iloc[0]
            rows.append((gk.replace("('газточка', '",'').replace("('михайлов', '",'Михайлов ').replace("('pnn', '",'pNN ').replace("')",''),
                         len(g), best['признак'], best['r'], best['p'],
                         ', '.join(sorted(set(g['признак']))[:8])))
        rows.sort(key=lambda x:-abs(x[3]))
        h='<table><tr><th class="l">Маркер</th><th>Значимых вариантов</th><th class="l">Сильнейший</th><th>r</th><th class="l">Все значимые варианты</th></tr>'
        for nm,cnt,b,r,p,allv in rows:
            h+=(f'<tr><td class="l"><b>{esc(nm)}</b></td><td>{cnt}</td><td class="l"><code>{esc(b)}</code></td>'
                f'<td class="{cls(r)}">{r:+.3f}{star(p)}</td><td class="l small">{esc(allv)}</td></tr>')
        H.append(h+'</table>')
        H.append('<p class="small">В краткой версии от каждого такого семейства остаётся один представитель — сильнейший '
                 '(для pNN сначала выбирается наибольший порог, затем сильнейший из них).</p>')
    else:
        H.append('<p class="small">— повторяющихся маркеров среди значимых нет —</p>')
    # все значимые по системам
    H.append('<h2>2. Все значимые признаки по системам</h2>')
    for _,x in DS.iterrows():
        if x['значимых']==0: continue
        sub=S[S['домен']==x['домен']].copy()
        H.append(f'<h3>{esc(x["домен_ru"])} — {int(x["значимых"])} значимых из {int(x["всего"])}</h3>')
        sub=sub.assign(**{'в краткой версии':['да' if a in keep else '' for a in sub['признак']]})
        hh='<table><tr><th class="l">Признак</th><th>r</th><th>p</th><th>n</th><th>В краткой версии</th></tr>'
        for _,z in sub.iterrows():
            hh+=(f'<tr><td class="l"><code>{esc(z["признак"])}</code></td><td class="{cls(z["r"])}">{z["r"]:+.3f}{star(z["p"])}</td>'
                 f'<td>{z["p"]:.1e}</td><td>{int(z["n"])}</td><td>{z["в краткой версии"]}</td></tr>')
        H.append(hh+'</table>')
    H.append(GLOSS)
    H.append('<p class="small">Рядом: <code>Отчёт_краткий.html</code> (по одному представителю на маркер, с плеядами), '
             '<code>Значимые_признаки.xlsx</code>.</p>')
    H.append('</body></html>')
    open(f'{outdir}/Отчёт_полный.html','w',encoding='utf-8').write('\n'.join(H))
