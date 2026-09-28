# -*- coding: utf-8 -*-
"""Анализ группы: значимые признаки, классические плеяды (стиль программы Даниила)."""
import os,sys,math,re,warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy import stats
from itertools import combinations
sys.path.insert(0,'/Users/mac-os/Documents/Батя/outputs/_cache')
from domains import domain

DOM_RU={'RR_нагрузка':'Кардио: нагрузка','RR_восстановление':'Кардио: восстановление',
 'газ_точки':'Газообмен: пороги','газ_восстановление':'Газообмен: восстановление',
 'газ_сводные':'Газообмен: сводные','пульсовая_цена':'Пульсовая цена',
 'ВСР_покой':'ВСР покоя','стабилометрия':'Стабилометрия','анатомия':'Анатомия и состав тела',
 'психология':'Психология','прочее':'Прочее'}

def marker_group(c):
    """Группа взаимозаменяемых записей одного маркера.
    Газовая точка т1 по VO2/VCO2/RER -> одна группа. Точка Михайлова при разных порогах -> одна.
    pNN одного периода -> одна (предпочтение наибольшему порогу)."""
    s=str(c)
    m=re.match(r'^(газ_(?:т[1-4]|восст_[^_]+(?:_[^_]+)?))_',s)
    if m: return ('газточка',m.group(1))
    m=re.match(r'^Мих\d+_(нагр|восст)_(\w+)$',s)
    if m: return ('михайлов',m.group(1)+'_'+m.group(2))
    m=re.match(r'^(ПС|СТ|\d+[НВ])_pNN(\d+)$',s)
    if m: return ('pnn',m.group(1))
    return None

def pnn_rank(c):
    m=re.match(r'^(?:ПС|СТ|\d+[НВ])_pNN(\d+)$',str(c))
    return int(m.group(1)) if m else -1

def analyze(src, target_col, alpha=0.05, redun=0.90):
    d=pd.read_excel(src)
    y=pd.to_numeric(d[target_col],errors='coerce')
    rows=[]
    for c in d.columns:
        s=str(c)
        if s==target_col: continue
        dm=domain(c)
        if dm in ('служебное','результат'): continue
        v=pd.to_numeric(d[c],errors='coerce'); m=v.notna()&y.notna()
        if m.sum()<max(8,int(0.5*len(d))) or v[m].nunique()<3: continue
        r,p=stats.spearmanr(v[m],y[m])
        if not np.isfinite(r): continue
        rows.append({'признак':s,'домен':dm,'домен_ru':DOM_RU.get(dm,dm),'r':float(r),'p':float(p),
                     'n':int(m.sum()),'группа_маркера':marker_group(c),'pnn':pnn_rank(c)})
    F=pd.DataFrame(rows)
    if not len(F): return d,y,F,F,F,F
    F['|r|']=F['r'].abs()
    F=F.sort_values('|r|',ascending=False).reset_index(drop=True)
    S=F[F['p']<alpha].copy()                      # ПОЛНЫЙ набор значимых
    # ----- КРАТКИЙ набор: один представитель на маркер -----
    def pick(g):
        g=g.copy()
        if len(g) and g['группа_маркера'].iloc[0] and g['группа_маркера'].iloc[0][0]=='pnn':
            # предпочтение наибольшему порогу pNN
            mx=g['pnn'].max()
            g=g[g['pnn']==mx]
        return g.sort_values('|r|',ascending=False).head(1)
    parts=[]
    grp=S[S['группа_маркера'].notna()]
    if len(grp):
        parts.append(grp.groupby(grp['группа_маркера'].astype(str),sort=False,group_keys=False).apply(pick))
    parts.append(S[S['группа_маркера'].isna()])
    SHORT=pd.concat(parts).sort_values('|r|',ascending=False) if parts else S
    def dedup(df,topn):
        if not len(df): return df
        cols=list(df['признак'])[:300]
        X=d[cols].apply(pd.to_numeric,errors='coerce'); R=X.rank()
        M=R.corr(min_periods=6).abs().fillna(0).values
        pos={c:i for i,c in enumerate(cols)}
        kept=[]
        for c in cols:
            if any(M[pos[c],pos[k]]>=redun for k in kept): continue
            kept.append(c)
            if len(kept)>=topn: break
        return df[df['признак'].isin(kept)].copy()
    K=auto_top_k(len(d))
    LEAD=dedup(SHORT,K)
    D2=[]
    for dm,g in SHORT.groupby('домен',sort=False): D2.append(dedup(g,1).head(1))
    DOM2=pd.concat(D2).sort_values('|r|',ascending=False) if D2 else SHORT.head(0)
    return d,y,F,LEAD,DOM2,SHORT

def _wrap(name, width=15, max_lines=3):
    s=re.sub(r"\s+"," ",str(name).replace("\n"," ")).strip()
    tokens=re.split(r"[ ,]+",s); lines,cur=[],""
    for w in tokens:
        if len(cur)+len(w)+1<=width: cur=(cur+" "+w).strip()
        else:
            if cur: lines.append(cur)
            cur=w
    if cur: lines.append(cur)
    if len(lines)>max_lines: lines=lines[:max_lines]; lines[-1]+="…"
    return "\n".join(lines)

def _short(name,n=22):
    s=re.sub(r"\s+"," ",str(name)).strip()
    return s if len(s)<=n else s[:n-1]+"…"

def auto_top_k(n):
    """Как в программе: число узлов по объёму выборки."""
    if n is None: return 8
    if n<22: return 6
    if n<40: return 8
    return 9

DCOL={'RR_нагрузка':'#c0392b','RR_восстановление':'#e67e22','газ_точки':'#16a085',
 'газ_восстановление':'#1abc9c','газ_сводные':'#27ae60','пульсовая_цена':'#8e44ad',
 'ВСР_покой':'#d35400','стабилометрия':'#2980b9','анатомия':'#7f8c8d','психология':'#c2185b','прочее':'#95a5a6'}

def draw(df, d, y, out_png, title, target_name, by_domain=False, alpha=0.05, target_short=None):
    """Точный порт плеяды из программы: все значимые рёбра, мелкие узлы, подписи снаружи."""
    if not len(df): return False
    feats=list(df['признак']); k=len(feats)
    tname=target_short or _short(target_name,10)
    nodes=[tname]+feats
    data=pd.concat([y.rename(tname), d[feats].apply(pd.to_numeric,errors='coerce')],axis=1)
    pos={tname:(0.0,0.0)}
    for i,c in enumerate(feats):
        ang=2*np.pi*i/max(k,1)+np.pi/2
        pos[c]=(np.cos(ang),np.sin(ang))
    def corr(a,b):
        m=a.notna()&b.notna()
        if m.sum()<4 or a[m].nunique()<2 or b[m].nunique()<2: return np.nan,np.nan
        r,p=stats.spearmanr(a[m],b[m]); return r,p
    edges=[]
    for a,b in combinations(nodes,2):
        r,p=corr(data[a],data[b])
        if np.isfinite(r) and p<alpha: edges.append((a,b,r,p))
    fig,ax=plt.subplots(figsize=(13,13))
    def _lblpos(a,b):
        if a==tname or b==tname:
            per=b if a==tname else a
            return pos[per][0]*0.62,pos[per][1]*0.62
        mx,my=(pos[a][0]+pos[b][0])/2,(pos[a][1]+pos[b][1])/2
        dd=(mx**2+my**2)**0.5
        if dd<0.5:
            if dd<0.08:
                dx,dy=pos[b][0]-pos[a][0],pos[b][1]-pos[a][1]
                nrm=(dx**2+dy**2)**0.5 or 1
                return -dy/nrm*0.5, dx/nrm*0.5
            f=0.55/dd; return mx*f,my*f
        return mx,my
    for a,b,r,p in edges:
        x=[pos[a][0],pos[b][0]]; yy=[pos[a][1],pos[b][1]]
        ax.plot(x,yy,color=("#c0392b" if r>0 else "#2471a3"),
                lw=0.7+6.5*abs(r),alpha=0.65,zorder=1,solid_capstyle="round")
        mx,my=_lblpos(a,b)
        ax.text(mx,my,f"{r:+.2f}",ha="center",va="center",fontsize=8,fontweight="bold",zorder=4,
                color=("#7b241c" if r>0 else "#1a5276"),
                bbox=dict(boxstyle="round,pad=0.12",fc="white",ec="none",alpha=0.85))
    dm_of=dict(zip(df['признак'],df['домен']))
    for node,(x,yy) in pos.items():
        is_t = node==tname
        col = "#f1c40f" if is_t else (DCOL.get(dm_of.get(node),'#aed6f1') if by_domain else "#aed6f1")
        ax.scatter([x],[yy],s=(2800 if is_t else 750),c=col,edgecolors="#333",linewidths=1.3,zorder=2)
    ax.text(0,0,tname,ha="center",va="center",fontsize=11,fontweight="bold",zorder=6,
            bbox=dict(boxstyle="circle,pad=0.3",fc="#f1c40f",ec="#333",lw=1.3))
    for node in feats:
        x,yy=pos[node]; lx,ly=x*1.34,yy*1.34
        ha="left" if x>0.08 else ("right" if x<-0.08 else "center")
        va="bottom" if yy>0.08 else ("top" if yy<-0.08 else "center")
        lbl=_wrap(node)
        if by_domain: lbl+="\n"+_wrap(DOM_RU.get(dm_of.get(node),''),20,1)
        ax.text(lx,ly,lbl,ha=ha,va=va,fontsize=8.5,zorder=3,linespacing=0.95)
    ax.set_xlim(-1.95,1.95); ax.set_ylim(-1.8,1.8); ax.axis("off")
    ax.set_title(title,fontsize=11,fontweight="bold")
    if by_domain:
        seen={}
        for c in feats: seen[DOM_RU.get(dm_of.get(c),'')]=DCOL.get(dm_of.get(c),'#aed6f1')
        hs=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=v,markeredgecolor='#333',
                       markeredgewidth=1,markersize=9,label=kk) for kk,v in seen.items() if kk]
        ax.legend(handles=hs,loc='lower center',ncol=4,frameon=False,fontsize=8,bbox_to_anchor=(0.5,-0.03))
    fig.savefig(out_png,dpi=110,bbox_inches='tight'); plt.close(fig)
    return True
