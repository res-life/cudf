# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import csv,io,json,statistics,collections
O=Path(__file__).resolve().parent
out={}
for f in sorted(O.glob('ncu2-f*.csv')):
    s=f.read_text()
    assert '"ID",' in s, f.name
    rows=list(csv.DictReader(io.StringIO(s[s.index('"ID",'):])) )
    groups=collections.defaultdict(list)
    sequence=[];geometry={}
    for r in rows:
        if r['Metric Name']!='gpu__time_duration.sum':continue
        name=r['Kernel Name'];sequence.append(name)
        groups[name].append(float(r['Metric Value'].replace(',','')))
        geometry[name]={k:v for k,v in r.items() if k in ['Block Size','Grid Size','Registers Per Thread','Static Shared Memory Per Block','Dynamic Shared Memory Per Block']}
    out[f.stem]={'launches':len(sequence),'sequence':sequence,'kernels':{k:{'count':len(v),'median_ns':statistics.median(v),'mean_ns':statistics.mean(v),'min_ns':min(v),'max_ns':max(v),**geometry[k]} for k,v in groups.items()}}
assert len(out)==8
comparisons={}
for i in [0,1]:
    records={role:[v for k,v in out.items() if k.startswith(f'ncu2-f{i}-') and k.endswith('-'+role)] for role in ['parent','candidate']}
    seqs=[x['sequence'] for vs in records.values() for x in vs]
    assert len(seqs)==4
    assert all(s==seqs[0] for s in seqs),'Kernel sequence changed'
    per_kernel={}
    for name in records['parent'][0]['kernels']:
        d={role:[v['kernels'][name]['median_ns'] for v in records[role]] for role in records}
        d['candidate_over_parent']=statistics.mean(d['candidate'])/statistics.mean(d['parent'])
        per_kernel[name]=d
    comparisons[str(i)]={'identical_launch_sequence':True,'launches':len(seqs[0]),'kernels':per_kernel}
(O/'profile-summary.json').write_text(json.dumps({'profiles':out,'comparisons':comparisons},indent=2)+'\n')
print('All four parent/candidate profiles have identical launch sequences for both targets.')
for f,d in comparisons.items():
    for name,v in d['kernels'].items():
        print('fixture',f,name.split('(')[0][:115], 'ratio',round(v['candidate_over_parent'],4))
