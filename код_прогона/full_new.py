import sys,json,warnings,io,contextlib,numpy as np,pandas as pd; warnings.filterwarnings('ignore')
sys.path.insert(0,'/sessions/lucid-ecstatic-cannon/mnt/outputs'); import gas,os
os.chdir('/sessions/lucid-ecstatic-cannon/mnt/Батя')
C='/sessions/lucid-ecstatic-cannon/mnt/outputs/_cache'
start=int(sys.argv[1]); cnt=int(sys.argv[2])
pairs=json.load(open(C+'/pairs_new.json'))
out=open(C+'/new_feats.jsonl','a')
def load_rr(rp):
    v=[]
    for ln in open(rp,encoding='utf-8',errors='ignore'):
        try: v.append(float(ln.strip()))
        except: pass
    return np.array(v)
for sn,rp,xl,grp in pairs[start:start+cnt]:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r=gas.make(rr_path=rp,gas_path=xl,report_only=True,prestart_s=30,start_s=30)
        ov=load_rr(rp); T=np.cumsum(ov)/1000.0
        rr_s=gas.smooth_curve(ov,sigma=5); hr=gas.smooth_curve(60000.0/ov,sigma=5)
        df=pd.read_excel(xl); df['t']=df['t'].apply(str)
        tg=pd.to_timedelta(df['t'].loc[2:]).dt.total_seconds().values.astype(float)
        off,_=gas.hr_sync_offset(tg,df['ЧСС'].loc[2:].values,ov)
        if off is None: off=0
        rec=r['rec_start']-off; ls=(r.get('load_start') or 60)-off
        f={'surname':sn,'sport':grp,'источник':'RR+газ','ЧСС_пик':r.get('hr_peak'),
           'мин_нагрузки':(rec-ls)/60.0}
        # газовые точки
        curves=['VO2','VCO2','RER','VE','VE/VO2','VE/VCO2','RR','O2pulse']
        for pt in ['т1','т2','т3','т4','восст_1мин','восст_2мин','восст_замедление','восст_ЧСС<100','восст_max_RER']:
            q=(r.get('detail') or {}).get(pt) or {}
            f[f'газ_{pt}_t']=q.get('t'); f[f'газ_{pt}_%МПК']=q.get('pct')
            for c in curves: f[f'газ_{pt}_{c}']=q.get(c)
            if 'ЧСС' in q: f[f'газ_{pt}_ЧСС']=q.get('ЧСС')
            if 'HRR' in q: f[f'газ_{pt}_HRR']=q.get('HRR')
        f['газ_пик_ЧСС']=r.get('hr_peak')
        # периоды
        segs={'ПС':(0,30),'СТ':(30,60)}
        for i in range(9):
            a=60+i*60
            if a>=rec: break
            segs[f'{i+1}Н']=(a,min(a+60,rec))
        for i in range(6):
            a=rec+i*60
            if a>=T[-1]: break
            segs[f'{i+1}В']=(a,min(a+60,T[-1]))
        for p,(a,b) in segs.items():
            m=(T>=a)&(T<b); s=ov[m]
            if len(s)<2: continue
            x=np.arange(len(s)); sl,ic=np.polyfit(x,s,1); dd=np.abs(np.diff(s))
            f[f'{p}_наклон']=float(sl); f[f'{p}_отрезок']=float(ic)
            f[f'{p}_RMSSD']=float(dd.mean()); f[f'{p}_SDNN']=float(np.std(s)); f[f'{p}_ЧСС']=float(60000/np.median(s))
            for thr in range(2,52,2): f[f'{p}_pNN{thr}']=float(100*np.mean(dd>=thr)) if len(dd) else None
        # перегибы
        m=(T>=ls)&(T<=rec); tl=T[m]-ls; yl=rr_s[m]
        if len(tl)>=12:
            k=gas._piecewise_breakpoint(tl,yl,0.10,0.90)
            if k and k>=3:
                a_,b_=np.polyfit(tl[:k],yl[:k],1)
                f['Н_a']=float(a_); f['Н_b']=float(b_); f['Н_перегиб_с']=float(tl[k]); f['Н_перегиб_ЧСС']=float(60000/yl[k])
        m2=(T>=rec)&(T<=rec+300); tr=T[m2]-rec; yr=rr_s[m2]
        if len(tr)>=12:
            k=gas._piecewise_breakpoint(tr,yr,0.08,0.90)
            if k and k>=3:
                a_,b_=np.polyfit(tr[:k],yr[:k],1)
                f['В_a']=float(a_); f['В_b']=float(b_); f['В_перегиб_с']=float(tr[k]); f['В_перегиб_ЧСС']=float(60000/yr[k])
        # Михайлов
        resid=ov-rr_s; sd=np.full(len(T),np.nan)
        for i in range(len(T)):
            mm=(T>T[i]-30)&(T<=T[i])
            if mm.sum()>=5: sd[i]=np.std(resid[mm])
        load=(T>=ls)&(T<rec)
        for thr in (2,3,4):
            got=None
            for i in np.where(load)[0]:
                if np.isfinite(sd[i]) and sd[i]<thr:
                    j=(T>=T[i])&(T<=T[i]+25)&load; v=sd[j][np.isfinite(sd[j])]
                    if len(v)>=3 and np.all(v<thr*1.3): got=float(T[i]); break
            if got is not None:
                f[f'Мих{thr}_нагр_с']=got-ls; f[f'Мих{thr}_нагр_ЧСС']=float(np.interp(got,T,hr))
        # окна минут
        for nm,(s0,s1) in {'м1':(0,60),'м2':(60,120),'м3':(120,180),'м12':(0,120),'м23':(60,180),'м13':(0,180)}.items():
            mm=(T>=ls+s0)&(T<=ls+s1)
            if mm.sum()>=6:
                a_,b_=np.polyfit(T[mm]-ls-s0,rr_s[mm],1)
                f[f'{nm}_a']=float(a_); f[f'{nm}_b']=float(b_)
                f[f'{nm}_ЧССкон']=float(60000/rr_s[mm][-1]); f[f'{nm}_ЧССср']=float(np.mean(hr[mm]))
                dd=np.abs(np.diff(ov[mm]))
                if len(dd): f[f'{nm}_RMSSD']=float(dd.mean())
        # суммы/удары
        def blk(t0,tag,limit=None):
            for mn in (1,2,3):
                a=t0+(mn-1)*60; b=a+60
                if limit is not None and a>=limit: break
                sel=(T>=a)&(T<b)
                if sel.sum()>=5:
                    f[f'{tag}{mn}_N']=int(sel.sum()); f[f'{tag}{mn}_Sum']=float(ov[sel].sum()); f[f'{tag}{mn}_срRR']=float(ov[sel].mean())
                idx=np.where(T>=a)[0]
                for nb in (30,50,100):
                    if len(idx)>=nb and (limit is None or T[idx[nb-1]]<=limit+5): f[f'{tag}{mn}_Σ{nb}']=float(ov[idx[0]:idx[0]+nb].sum())
            idx=np.where(T>=t0)[0]
            for nb in (50,100,150,200):
                if len(idx)>=nb and (limit is None or T[idx[nb-1]]<=limit+5): f[f'{tag}нак_Σ{nb}']=float(ov[idx[0]:idx[0]+nb].sum())
        blk(ls,'Н',limit=rec); blk(rec,'В')
        selL=(T>=ls)&(T<rec); selR=(T>=rec)
        if selL.sum()>5: f['Н_всего_N']=int(selL.sum()); f['Н_всего_Sum']=float(ov[selL].sum())
        if selR.sum()>5: f['В_всего_N']=int(selR.sum()); f['В_всего_Sum']=float(ov[selR].sum())
        for mn in range(1,10):
            a=ls+(mn-1)*60; b=a+60
            if b<=rec:
                s2=(T>=a)&(T<b)
                if s2.sum()>=5: f[f'НN{mn}']=int(s2.sum())
        for mn in range(1,7):
            a=rec+(mn-1)*60; b=a+60
            if b<=T[-1]:
                s2=(T>=a)&(T<b)
                if s2.sum()>=5: f[f'ВN{mn}']=int(s2.sum())
        for mn in (1,2,3,4,5):
            b=ls+mn*60
            if b<=rec: f[f'Ннак_N{mn}']=int(((T>=ls)&(T<b)).sum())
        for i in (1,2,3):
            tt=rec+i*60
            if T[0]<=tt<=T[-1]: f[f'ЧСС_{i}мин_восст']=float(np.interp(tt,T,hr))
        mm=(T>=rec-60)&(T<=rec)
        if mm.sum()>3: f['ЧСС_посл_мин']=float(60000/np.median(rr_s[mm]))
        out.write(json.dumps(f,ensure_ascii=False,default=float)+"\n")
    except Exception as e:
        out.write(json.dumps({'surname':sn,'error':str(e)[:60]},ensure_ascii=False)+"\n")
out.close(); print(f"{start}+{cnt} done")
