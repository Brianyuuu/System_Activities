import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'inputs'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 if not rows:return
 tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 tmp.replace(p)
