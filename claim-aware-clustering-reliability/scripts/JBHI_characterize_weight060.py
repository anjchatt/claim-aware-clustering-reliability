#!/usr/bin/env python3
"""Characterize selected w=.600 versus w=1.000 R2 reconstructed partitions.

Uses the same preprocessing functions as the 06C reconstruction. The analysis
evaluates two weights (40 seeded fits per weight), selects the lowest-inertia
fit, and can compare the result with saved 06C summary files.

This is not a recreation of the original 06B primary partitions.
"""
from __future__ import annotations
import argparse, importlib.util, json, hashlib, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

UTIL=['number_outpatient','number_emergency','number_inpatient']

def p_hash(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def opts():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_mutually_exclusive_group(required=True)
    s.add_argument('--raw-csv',type=Path,help='Public UCI diabetic_data.csv')
    s.add_argument('--dev-matrix',type=Path,help='Already encoded EXACT R2 dev matrix in 06C format')
    p.add_argument('--heldout-matrix',type=Path)
    p.add_argument('--dev-meta',type=Path)
    p.add_argument('--heldout-meta',type=Path)
    p.add_argument('--split-manifest',type=Path)
    p.add_argument('--validator',type=Path,required=True,help='Path to jbhi_06C_validation_v2.py')
    p.add_argument('--run-manifest',type=Path,help='Manifest corresponding to submitted 480 fit run')
    p.add_argument('--reference-fit-csv',type=Path,help='Saved fit_level_seed_runs(2).csv from n_init=20 run')
    p.add_argument('--reference-cross-csv',type=Path,help='Saved cross_weight_partition_agreement(2).csv')
    p.add_argument('--out',type=Path,default=Path('JBHI_w060_characterization'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--seeds-per-weight',type=int,default=40)
    p.add_argument('--n-init',type=int,default=20)
    p.add_argument('--age-divisor',type=float,default=100.0)
    p.add_argument('--weight-transform',choices=['direct','sqrt'],default='direct')
    p.add_argument('--scaler',choices=['robust','standard','none'],default='robust')
    return p.parse_args()

def get_module(path):
    spec=importlib.util.spec_from_file_location('jbhi_06C_existing',str(path))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def align_assignment(y_a,y_b):
    aa=np.asarray(y_a,int);bb=np.asarray(y_b,int)
    confusion=np.zeros((2,2),dtype=int)
    np.add.at(confusion,(aa,bb),1)
    rows,cols=linear_sum_assignment(-confusion)
    mapping={int(col):int(row) for row,col in zip(rows,cols)}
    return np.array([mapping[int(v)] for v in bb]),confusion,mapping

def block_map(columns):
    ix={
      'Age':[i for i,c in enumerate(columns) if c=='clinical__age' or c=='clinical__age_midpoint' or c.startswith('clinical__age_')],
      'Diagnosis family 1':[i for i,c in enumerate(columns) if 'diag_family_1_' in c or 'diag1_family_' in c or 'diag_family_1__' in c],
      'Diagnosis family 2':[i for i,c in enumerate(columns) if 'diag_family_2_' in c or 'diag2_family_' in c or 'diag_family_2__' in c],
      'Diagnosis family 3':[i for i,c in enumerate(columns) if 'diag_family_3_' in c or 'diag3_family_' in c or 'diag_family_3__' in c],
      'Utilization':[i for i,c in enumerate(columns) if c.startswith('util__')],
    }
    assigned=set(sum(ix.values(),[]));extra=[i for i in range(len(columns)) if i not in assigned]
    if extra: ix['Other encoded clinical coordinates']=extra
    return {k:v for k,v in ix.items() if v}

def centroid_dispersion(x,y,columns,weight):
    n=len(y);global_mu=x.mean(axis=0)
    groups=block_map(columns)
    between=np.zeros(x.shape[1],dtype=float)
    for k in np.unique(y):
        v=x[y==k]; between+=len(v)*((v.mean(axis=0)-global_mu)**2)
    tot=float(between.sum())
    return [{'weight':weight,'feature_block':block,'between_dispersion':float(between[ids].sum()),
             'share_pct':100*float(between[ids].sum())/tot if tot else float('nan')}
            for block,ids in groups.items()]

def raw_clinical_table(mod, raw_csv, dev_ids):
    df=pd.read_csv(raw_csv,dtype={'patient_nbr':str,'encounter_id':str},low_memory=False)
    df['_encnum']=pd.to_numeric(df.encounter_id,errors='raise')
    df=df.sort_values(['_encnum','patient_nbr'],kind='stable').drop_duplicates('patient_nbr',keep='first')
    if df.patient_nbr.duplicated().any(): raise AssertionError('Duplicate patients')
    df=df.set_index('patient_nbr').loc[dev_ids.astype(str)].reset_index()
    for j in (1,2,3): df[f'diagnosis_family_{j}']=df[f'diag_{j}'].map(mod.icd9_family)
    df['age_midpoint']=df.age.map(mod.age_midpoint)
    return df

def main():
    a=opts(); a.out.mkdir(parents=True,exist_ok=True)
    if not a.validator.exists(): raise FileNotFoundError(a.validator)
    mod=get_module(a.validator)
    if a.dev_matrix:
        if not a.heldout_matrix: raise ValueError('Matrix mode needs --heldout-matrix')
        dev, held, dm, hm, source, notes=mod.load_matrix(a)
    else:
        dev, held, dm, hm, source, notes=mod.load_raw(a)
    if not dm.empty and 'patient_nbr' in dm and dm.patient_nbr.duplicated().any():
        raise ValueError('Duplicate patient_nbr in dev metadata')
    if a.run_manifest:
        old=json.loads(a.run_manifest.read_text())
        assert old.get('seed')==a.seed, 'Run seed does not match supplied manifest'
        assert old.get('weight_transform')==a.weight_transform, 'Weight transform differs from manifest'
        assert old.get('audit_fit_n_init')==a.n_init, 'n_init differs from manifest'
        assert old.get('n_independent_seeds_per_weight')==a.seeds_per_weight, 'Seed count differs from manifest'
        assert a.scaler in old.get('scalers',[]), 'Scaler differs from manifest'
        print('Manifest parameter checks: PASS. Input hashes are separately reported.')
    ix=np.array([i for i,c in enumerate(dev.columns) if c.startswith('util__')],dtype=int)
    if len(ix)!=3: raise ValueError('Expected 3 utilization coordinates in R2 matrix')
    selected={};labels_out=[];disp=[]
    with threadpool_limits(limits=1):
        for w in (.6,1.0):
            x,hx=mod.transforms(dev,held,a.scaler,w,ix,a.weight_transform)
            best=None;infos=[]
            for j in range(a.seeds_per_weight):
                seed=a.seed+100003*j
                m=KMeans(n_clusters=2,n_init=a.n_init,random_state=seed,algorithm='lloyd')
                y=m.fit_predict(x)
                inf=float(m.inertia_)
                infos.append((seed,inf,sorted(np.bincount(y,minlength=2).tolist())))
                if best is None or inf < best[0]: best=(inf,seed,y,m)
            inertia,seed,yl,m=best
            yh=m.predict(hx) if len(hx) else np.empty(0,dtype=int)
            selected[w]=dict(yl=yl,yh=yh,inertia=inertia,seed=seed)
            disp.extend(centroid_dispersion(x,yl,dev.columns,w))
            print(f'w={w:g}: selected seed={seed}, inertia={inertia:.6f}, sizes={sorted(np.bincount(yl).tolist())}')
            for cohort, labs, meta in [('development',yl,dm),('heldout',yh,hm)]:
                if not len(labs): continue
                if not meta.empty and 'patient_nbr' in meta:
                    ids=meta.patient_nbr.astype(str).to_numpy()
                else: ids=np.array([f'ROW_{i}' for i in range(len(labs))])
                labels_out.extend({'cohort':cohort,'patient_nbr':pid,'weight':w,'cluster':int(lab)}
                                  for pid,lab in zip(ids,labs))
    y6=selected[.6]['yl']; y1=selected[1.0]['yl']
    a1,cm,mapping=align_assignment(y6,y1)
    dev_ari=adjusted_rand_score(y6,y1)
    changed=100*float((a1!=y6).mean())
    print('DEV cross-weight ARI=',round(dev_ari,6),'changed%=',round(changed,4))
    pd.DataFrame(cm,index=['w060_label0','w060_label1'],columns=['w100_label0','w100_label1']).to_csv(a.out/'raw_label_transition.csv')
    pd.crosstab(pd.Series(y6,name='w060_cluster'),pd.Series(a1,name='w100_aligned_to_w060')).to_csv(a.out/'aligned_transition.csv')
    pd.DataFrame(labels_out).to_csv(a.out/'patient_assignments_w060_w100.csv',index=False)
    pd.DataFrame(disp).to_csv(a.out/'feature_block_concentration.csv',index=False)
    (a.out/'feature_block_concentration.json').write_text(json.dumps(disp,indent=2))
    if not a.dev_matrix and not dm.empty and 'patient_nbr' in dm:
        meta=raw_clinical_table(mod,a.raw_csv,dm.patient_nbr)
        contrasts=[]
        for w,y in [(0.6,y6),(1.0,y1)]:
            counts=pd.Series(y).value_counts()
            for col in ['diagnosis_family_1','diagnosis_family_2','diagnosis_family_3','age']:
                vals=meta[col].fillna('Missing').astype(str)
                ct=pd.crosstab(vals,pd.Series(y,name='cluster'))
                if set(ct.columns)!={0,1}: continue
                p0=ct[0]/counts[0];p1=ct[1]/counts[1]
                for cat in ct.index:
                    contrasts.append({'weight':w,'feature':col,'category':cat,
                                      'cluster0_n':int(ct.loc[cat,0]),'cluster1_n':int(ct.loc[cat,1]),
                                      'cluster0_pct':100*float(p0[cat]),'cluster1_pct':100*float(p1[cat]),
                                      'difference_cluster1_minus_0_pp':100*float(p1[cat]-p0[cat])})
        cd=pd.DataFrame(contrasts)
        cd['abs_diff']=cd.difference_cluster1_minus_0_pp.abs()
        cd.sort_values(['weight','abs_diff'],ascending=[True,False]).drop(columns='abs_diff').to_csv(a.out/'diagnosis_age_cluster_contrasts.csv',index=False)
        util=meta[UTIL].apply(pd.to_numeric,errors='coerce')
        summaries=[]
        for w,y in [(0.6,y6),(1.0,y1)]:
            for k in (0,1):
                sub=meta.loc[y==k]; u=util.loc[y==k]
                summaries.append({'weight':w,'cluster':k,'n':len(sub),
                                  'age_midpoint_mean':float(pd.to_numeric(sub.age_midpoint).mean()),
                                  **{f'{c}_mean':float(u[c].mean()) for c in UTIL}})
        pd.DataFrame(summaries).to_csv(a.out/'cluster_age_utilization_summary.csv',index=False)
    held_summary={}
    if len(selected[.6]['yh']):
        h6=selected[.6]['yh']; h1=selected[1.0]['yh']
        ha,_,_=align_assignment(h6,h1)
        held_summary={'ari':float(adjusted_rand_score(h6,h1)),'changed_percent':100*float((ha!=h6).mean())}
        print('HELD cross-weight ARI=',round(held_summary['ari'],6),'changed%=',round(held_summary['changed_percent'],4))
    checks={}
    if a.reference_cross_csv:
        ref=pd.read_csv(a.reference_cross_csv)
        row=ref[(ref.scaler==a.scaler)&(np.isclose(ref.weight_1,.6))&(np.isclose(ref.weight_2,1.0))]
        if len(row)!=1:raise ValueError('Expected exact w0.6-w1 comparison in reference CSV')
        exp=float(row.iloc[0].development_ari);pct=float(row.iloc[0].development_changed_pct_aligned)
        checks['cross_weight_matches_saved']=(abs(exp-dev_ari)<1e-6 and abs(pct-changed)<1e-4)
        if not checks['cross_weight_matches_saved']: print('WARNING: COHORT/PIPELINE MISMATCH; values do not reproduce saved cross-weight CSV!')
    if a.reference_fit_csv:
        fit=pd.read_csv(a.reference_fit_csv)
        for w in [.6,1.0]:
            arow=fit[(fit.scaler==a.scaler)&(np.isclose(fit.weight,w))&(fit.n_init==a.n_init)]
            if arow.empty: raise ValueError('Reference fit CSV missing requested config')
            best=arow.loc[arow.inertia.idxmin()]
            checks[f'fit_inertia_w{w:g}_matches_saved']=abs(float(best.inertia)-selected[w]['inertia'])<1e-5
    record={'source':source,'n_dev':len(dev),'n_heldout':len(held),'weights':[.6,1.0],
            'scaler':a.scaler,'n_init':a.n_init,'seeds_per_weight':a.seeds_per_weight,
            'selected_seed_by_weight':{str(w):selected[w]['seed'] for w in selected},
            'selected_cluster_sizes':{str(w):np.bincount(selected[w]['yl'],minlength=2).tolist() for w in selected},
            'development_crossweight_ari':float(dev_ari),'development_changed_pct':changed,
            'heldout':held_summary,'saved_results_comparison':checks,
            'source_sha256':{str(p):p_hash(p) for p in [a.validator,a.raw_csv or a.dev_matrix] if p},
            'WARNING':'Not a verified copy of original 06B cohorts. Do not use as published characterization unless saved 06C selected fits and cohorts match.'}
    (a.out/'characterization_manifest.json').write_text(json.dumps(record,indent=2))
    print('Results:',a.out.resolve())

if __name__=='__main__':main()
