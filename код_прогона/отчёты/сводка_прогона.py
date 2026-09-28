# -*- coding: utf-8 -*-
"""Расшифровка признаков + свод по прогону ГазоваяЭргометрия_2026-08-24."""
import sys, os, io, re
sys.path.insert(0,'/Users/mac-os/Documents/Батя/outputs/_cache')
sys.path.insert(0,'/Users/mac-os/Documents/Батя/outputs/_cache/prog')
import pandas as pd, numpy as np
from glossary import describe
import cpet

Y = f'{cpet.Y}/ГазоваяЭргометрия_2026-08-24'
ЦЕЛИ = [('Прирост','прирост'),('ЭкономичностьПрироста','экономичность_прироста'),
        ('ХронотропнаяЦена','хронотропная_цена'),('Мощность','мощность')]
СП = ['Лыжники','Баскетболисты']

CSS = """<style>
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1060px;margin:24px auto;padding:0 22px;color:#1a1a1a;line-height:1.62}
h1{font-size:24px;color:#14315e;margin-bottom:2px}
h2{font-size:19px;color:#14315e;margin-top:30px;border-bottom:2px solid #e3e8f0;padding-bottom:5px}
h3{font-size:16px;color:#1a3e6e;margin:18px 0 6px}
.lead{color:#555;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:12px 0}
th,td{border:1px solid #d5dbe3;padding:6px 9px;text-align:center}
th{background:#14315e;color:#fff;font-weight:600}
td.l,th.l{text-align:left}
tr.sec td{background:#eef1f6;font-weight:700;color:#14315e;text-align:left}
tr:nth-child(even) td{background:#fafbfd}
td.g{background:#e2efda!important;font-weight:700} td.y{background:#fff2cc!important;font-weight:700} td.b{background:#f8d7da!important;font-weight:700}
.box{background:#f3f6fb;border-left:4px solid #14315e;padding:11px 15px;margin:14px 0}
.warn{background:#fff8e6;border:1px solid #f0d98a;border-radius:6px;padding:11px 15px;margin:14px 0}
.ok{background:#e2efda;border:1px solid #b5d3a0;border-radius:6px;padding:11px 15px;margin:14px 0}
.err{background:#fdecec;border:1px solid #e6a9a9;border-radius:6px;padding:11px 15px;margin:14px 0}
code{background:#f0f3f7;padding:1px 5px;border-radius:4px;font-size:12.5px;color:#14315e;font-weight:600}
.small{color:#666;font-size:12.5px} a{color:#14315e}
.fml{background:#fbfcfe;border:1px solid #dde3ec;border-radius:6px;padding:12px 16px;margin:12px 0;font-size:15px;text-align:center;font-family:Georgia,serif}
.wrap{overflow-x:auto} img{max-width:100%;border:1px solid #e0e5ec;border-radius:6px}
</style>"""

def esc(t): return str(t).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def f_(v,z=2): return ("%.*f"%(z,v)).replace(".",",")
def s_(v,z=2): return ("%+.*f"%(z,v)).replace(".",",")

# ---------- какие признаки реально участвовали ----------
xs = pd.ExcelFile(cpet.FULL)
исп = {}
for пап, ключ in ЦЕЛИ:
    for sp, sheet in zip(СП, ['Лыжники','Баскетболисты']):
        d = xs.parse(sheet)
        for c in cpet.pick_features(d,'газ',ключ):
            исп.setdefault(str(c), set()).add(пап)
все = sorted(исп)

# ---------- 1. Расшифровка ----------
h=[f"<!doctype html><meta charset='utf-8'><title>Расшифровка признаков</title>{CSS}"]
A=h.append
A("<h1>Расшифровка названий признаков</h1>")
A("<p class='lead'>Прогон «Газовая эргометрия» от 24 августа 2026. "
  f"Всего в корреляциях участвовало {len(все)} газовых показателей.</p>")
A("<h2>Как устроено имя</h2>")
A("<div class='fml'>газ_<b>момент теста</b>_<b>показатель</b></div>")
A("<p>Например <code>газ_т4_O2pulse</code> — кислородный пульс в точке 4, "
  "<code>газ_восст_1мин_VO2</code> — потребление кислорода на первой минуте восстановления.</p>")
A("<h2>Моменты теста</h2>")
A("<div class='box'>Протокол: предстарт 30 с + старт 30 с + <b>3 минуты постоянной нагрузки по ДОО</b> "
  "+ <b>разгон +30 Вт/мин</b> до отказа + восстановление. Времена <code>газ_*_t</code> "
  "отсчитываются от первой записи газоанализатора.</div>")
A("<table><tr><th class='l'>Обозначение</th><th class='l'>Момент</th></tr>")
from глоссарий_газ import МОМЕНТ
for ключ, оп in МОМЕНТ:
    A(f"<tr><td class='l'><code>{esc(ключ)}</code></td><td class='l'>{esc(оп)}</td></tr>")
A("</table>")
A("<p class='small'>Точки 1 и 2 у большинства достигаются ещё <b>внутри трёхминутной нагрузки по ДОО</b> "
  "(точка 1 у 100% лыжников и 90% баскетболистов, точка 2 у 79% и 68%), точки 3 и 4 — уже на разгоне.</p>")
A("<h2>Показатели</h2>")
A("<table><tr><th class='l'>Суффикс</th><th class='l'>Единицы</th><th class='l'>Что это</th></tr>")
ЕД={'VO2':'мл/мин','VCO2':'мл/мин','RER':'безразмерный','VE':'мл/мин','VE/VO2':'безразмерный',
    'VE/VCO2':'безразмерный','RR':'мс','O2pulse':'мл/уд','ЧСС':'уд/мин','HRR':'уд/мин','%МПК':'%'}
from глоссарий_газ import ПОКАЗАТЕЛЬ
for к,(кат,оп) in ПОКАЗАТЕЛЬ.items():
    if к=='t': continue
    A(f"<tr><td class='l'><code>{esc(к)}</code></td><td class='l'>{ЕД.get(к,'')}</td><td class='l'>{esc(оп)}</td></tr>")
A("</table>")
A("<div class='warn'><b>Обратите внимание на <code>RR</code>:</b> в газовых выгрузках это "
  "<b>интервал RR в миллисекундах</b> (60 000, делённое на ЧСС), а не частота дыхания. "
  "Проверено тождеством VO₂ / O₂-пульс = 60 000 / RR — оно выполняется точно.</div>")
A("<h2>Полный список признаков этого прогона</h2>")
A("<div class='wrap'><table><tr><th class='l'>Признак</th><th class='l'>Система</th>"
  "<th class='l'>Что означает</th><th class='l'>В каких целях</th></tr>")
пред=None
for c in все:
    кат, оп = describe(c)
    if кат != пред:
        A(f"<tr class='sec'><td colspan='4'>{esc(кат)}</td></tr>"); пред=кат
    A(f"<tr><td class='l'><code>{esc(c)}</code></td><td class='l'>{esc(кат)}</td>"
      f"<td class='l'>{esc(оп)}</td><td class='l'>{', '.join(sorted(исп[c]))}</td></tr>")
A("</table></div>")
A("<h2>Что в корреляции не участвовало и почему</h2>")
A("<table><tr><th class='l'>Что исключено</th><th class='l'>Причина</th></tr>"
  "<tr><td class='l'><code>газ_*_t</code> — абсолютные времена моментов</td>"
  "<td class='l'>это длительность теста под другим именем, то есть утечка цели</td></tr>"
  "<tr><td class='l'><code>газ_восст_ЧСС&lt;125_ЧСС</code>, <code>газ_восст_ЧСС&lt;100_ЧСС</code></td>"
  "<td class='l'>по построению равны порогу: разброс 1,3–1,5 уд/мин на всю группу</td></tr>"
  "<tr><td class='l'><code>газ_восст_ЧСС&lt;125_HRR</code>, <code>газ_восст_ЧСС&lt;100_HRR</code></td>"
  "<td class='l'>HRR = пиковая ЧСС минус порог, поэтому повторяет пиковую ЧСС на r = 0,997</td></tr>"
  "<tr><td class='l'><code>мин_нагрузки</code>, <code>Wmax</code>, <code>PWC</code>, <code>пик нагрузки</code>, "
  "<code>таргет_*</code>, <code>Ннак_N4/N5</code>, <code>КИ_*</code></td>"
  "<td class='l'>прямые производные длительности теста</td></tr>"
  "<tr><td class='l'>для цели «экономичность прироста» ещё <code>газ_т4_*</code> и <code>газ_коноо_*</code></td>"
  "<td class='l'>из них построена сама цель</td></tr>"
  "<tr><td class='l'>для цели «хронотропная цена» ещё <code>Ннак_N3</code></td>"
  "<td class='l'>то же</td></tr></table>")
io.open(f'{Y}/Расшифровка_признаков.html','w',encoding='utf-8').write("\n".join(h))
print("расшифровка готова, признаков:",len(все))
