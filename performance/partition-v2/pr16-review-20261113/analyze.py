# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import json,math,statistics,itertools,functools
O=Path(__file__).resolve().parent;plan=json.loads((O/'preregistered-plan.json').read_text())
def betacf(a,b,x):
 qab=a+b;qap=a+1;qam=a-1;c=1.;d=1-qab*x/qap;d=1/max(abs(d),1e-300)*(1 if d>=0 else -1);h=d
 for m in range(1,10001):
  aa=m*(b-m)*x/((qam+2*m)*(a+2*m));d=1+aa*d;d=d if abs(d)>1e-300 else 1e-300;c=1+aa/c;c=c if abs(c)>1e-300 else 1e-300;d=1/d;h*=d*c
  aa=-(a+m)*(qab+m)*x/((a+2*m)*(qap+2*m));d=1+aa*d;d=d if abs(d)>1e-300 else 1e-300;c=1+aa/c;c=c if abs(c)>1e-300 else 1e-300;d=1/d;delta=d*c;h*=delta
  if abs(delta-1)<3e-14:return h
 raise RuntimeError('Incomplete beta did not converge')
def ibeta(a,b,x):
 if x<=0:return 0.
 if x>=1:return 1.
 bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
 return bt*betacf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-bt*betacf(b,a,1-x)/b
def sf(t,df):
 q=.5*ibeta(df/2,.5,df/(df+t*t))
 return q if t>=0 else 1-q
@functools.lru_cache(None)
def critical(df):
 lo,hi=0.,100.
 for _ in range(80):
  mid=(lo+hi)/2
  if sf(mid,df)>.025:lo=mid
  else:hi=mid
 return (lo+hi)/2
assert abs(critical(15)-2.131449545559323)<1e-10
assert abs(critical(7)-2.364624251592784)<1e-10
assert abs(sf(0,15)-.5)<1e-15

def holm(pvalues,family=None):
 n=family or len(pvalues);ranked=sorted(enumerate(pvalues),key=lambda t:t[1]);out=[None]*len(pvalues);current=0.
 for j,(idx,p) in enumerate(ranked):current=max(current,(n-j)*p);out[idx]=min(1.,current)
 return out

def summarize(ds):
 n=len(ds);mu=statistics.mean(ds);se=statistics.stdev(ds)/math.sqrt(n) if n>1 else None;c=critical(n-1) if n>1 else None;ci=[math.exp(mu-c*se),math.exp(mu+c*se)] if c else None;t=mu/se if se and se>0 else (float('inf') if mu>0 else -float('inf'));p=sf(t,n-1) if math.isfinite(t) else (0. if t>0 else 1.)
 return {'quartets':n,'ratio':math.exp(mu),'ratio_95ci':ci,'log_effects':ds,'student_t_one_sided_p':p}

def randomization(z,order):
 n=len(z);obs=statistics.mean((1 if q=='ABBA' else -1)*v for q,v in zip(order,z));total=sum(z);count=0;draws=0;positive=order.count('ABBA')
 for chosen in itertools.combinations(range(n),positive):
  stat=(2*sum(z[j] for j in chosen)-total)/n;draws+=1;count+=stat>=obs-1e-14
 return count/draws,draws

out={}
for campaign in ['primary','controls']:
 orders=plan[campaign]['order'];blocks=[]
 for i,order in enumerate(orders):
  four=[]
  for pos,letter in enumerate(order):
   role='parent' if letter=='A' else 'candidate';label=f'{campaign}-q{i:02d}-p{pos}-{role}';p=O/'runs'/f'{label}.metadata.json'
   if not p.exists():break
   try:m=json.loads(p.read_text())
   except json.JSONDecodeError:break
   assert m['accepted'],label
   rows={r['id']:r for r in map(json.loads,(O/'runs'/f'{label}.jsonl').read_text().splitlines())};four.append((role,rows))
  if len(four)==4:blocks.append(four)
 if not blocks:continue
 records=[]
 for key in blocks[0][0][1]:
  ds=[];gpu=[];zs=[];aa_parent=[];aa_candidate=[];times=[]
  for block in blocks:
   a=[r[key]['wall_median_ms'] for role,r in block if role=='parent'];b=[r[key]['wall_median_ms'] for role,r in block if role=='candidate'];ga=[r[key]['event_median_ms'] for role,r in block if role=='parent'];gb=[r[key]['event_median_ms'] for role,r in block if role=='candidate'];all_log=[math.log(r[key]['wall_median_ms']) for _,r in block]
   ds.append((sum(map(math.log,b))-sum(map(math.log,a)))/2);gpu.append((sum(map(math.log,gb))-sum(map(math.log,ga)))/2);zs.append((all_log[1]+all_log[2]-all_log[0]-all_log[3])/2);aa_parent.append(a[1]/a[0]);aa_candidate.append(b[1]/b[0]);times.append({'parent_ms':a,'candidate_ms':b})
  r={'id':key,**summarize(ds),'gpu_ratio':math.exp(statistics.mean(gpu)),'quartet_process_medians':times,'same_parent_later_earlier_ratios':aa_parent,'same_candidate_later_earlier_ratios':aa_candidate}
  if len(blocks)==len(orders):r['randomization_one_sided_p'],r['randomization_assignments']=randomization(zs,orders)
  records.append(r)
 if len(blocks)==len(orders):
  for field,pfield in [('student_t_holm','student_t_one_sided_p'),('randomization_holm','randomization_one_sided_p')]:
   for family,suffix in [(len(records),str(len(records))),(42,'42_sensitivity')]:
    values=holm([r[pfield] for r in records],family)
    for r,p in zip(records,values):r[field+'_'+suffix]=p
 out[campaign]={'complete_quartets':len(blocks),'planned_quartets':len(orders),'records':records}
 print(campaign,'complete quartets',len(blocks),'/',len(orders))
 if len(blocks)==len(orders):
  for r in records:print(r['id'],'ratio',round(r['ratio'],5),'CI',[round(x,5) for x in r['ratio_95ci']],'Holm t',round(r['student_t_holm_'+str(len(records))],6),'Holm random',round(r['randomization_holm_'+str(len(records))],6))
(O/'summary-live.json').write_text(json.dumps(out,indent=2)+'\n')
