#!/usr/bin/env python3
"""JBHI 06D: optional complementary clustering-stability comparators.

Optional research comparison; not part of the reported JBHI results.
These are additional diagnostics, not substitutes for
clinical adjudication, independent cohorts, or exact historical reproduction.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from jbhi_06C_validation_v2 import load_raw, load_matrix, transforms, sha256


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    inp=p.add_mutually_exclusive_group(required=True)
    inp.add_argument('--raw-csv',type=Path)
    inp.add_argument('--dev-matrix',type=Path)
    p.add_argument('--split-manifest',type=Path)
    p.add_argument('--dev-meta',type=Path)
    p.add_argument('--out',type=Path,default=Path('jbhi_06D_results'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--bootstrap-reps',type=int,default=30)
    p.add_argument('--prediction-reps',type=int,default=30)
    p.add_argument('--n-init',type=int,default=20)
    p.add_argument('--test-fraction',type=float,default=.20)
    p.add_argument('--age-divisor',type=float,default=100.0)
    p.add_argument('--quick',action='store_true')
    a=p.parse_args()
    if a.quick:
        a.bootstrap_reps=4
        a.prediction_reps=4
        a.n_init=2
    if not 0<a.test_fraction<1: p.error('--test-fraction must be 0..1')
    return a


def jaccard(a,b):
    a=np.asarray(a,dtype=bool);b=np.asarray(b,dtype=bool)
    return float((a&b).sum()/(a|b).sum()) if (a|b).sum() else np.nan


def min_prediction_strength(test_groups, reference_groups):
    """For each cluster fitted on test patients, check pair co-membership under
    independent training-fitted model; use combinatorial counts, not n^2 matrix.
    """
    scores=[]
    for k in np.unique(test_groups):
        members=reference_groups[test_groups==k]
        n=len(members)
        if n<2:
            return np.nan
        total=n*(n-1)//2
        _,counts=np.unique(members,return_counts=True)
        agreeing=int(np.sum(counts*(counts-1)//2))
        scores.append(float(agreeing/total))
    return float(min(scores))


def fit(x,k,seed,n_init):
    model=KMeans(n_clusters=k,random_state=seed,n_init=n_init,algorithm='lloyd')
    model.fit(x)
    return model


def main():
    a=cli();a.out.mkdir(parents=True,exist_ok=True)
    if a.raw_csv:
        o=SimpleNamespace(raw_csv=a.raw_csv,split_manifest=a.split_manifest,seed=a.seed,age_divisor=a.age_divisor)
        dev,_,_,_,mode,_=load_raw(o)
        inp=a.raw_csv
    else:
        o=SimpleNamespace(dev_matrix=a.dev_matrix,heldout_matrix=None,dev_meta=a.dev_meta,heldout_meta=None)
        dev,_,_,_,mode,_=load_matrix(o)
        inp=a.dev_matrix
    u=np.array([i for i,c in enumerate(dev.columns) if c.startswith('util__')],dtype=int)
    c=np.array([i for i,c in enumerate(dev.columns) if c.startswith('clinical__')],dtype=int)
    if len(u)!=3 or not len(c): raise ValueError('Requires R2 clinical__ and three util__ columns')
    r2,_=transforms(dev,dev.iloc[:0], 'robust',1.0,u,'direct')
    clinical=r2[:,c]
    candidates={'R1_clinical_K2':clinical,'R2_clinical_util_K2':r2}
    raw_rows=[]; ps_rows=[]; summary=[]
    with threadpool_limits(limits=1):
        for name,x in candidates.items():
            base=fit(x,2,a.seed,a.n_init)
            base_y=base.predict(x)
            for r in range(a.bootstrap_reps):
                seed=a.seed+97*r
                idx=np.random.default_rng(seed).choice(len(x),len(x),replace=True)
                f=fit(x[idx],2,seed,a.n_init)
                pred=f.predict(x)
                for k in np.unique(base_y):
                    values=[jaccard(base_y==k,pred==n) for n in np.unique(pred)]
                    raw_rows.append({'candidate':name,'bootstrap_repetition':r,'baseline_cluster':int(k),
                                     'max_refit_assignment_jaccard':max(values),
                                     'bootstrap_unique_patients':int(len(set(idx))),
                                     'interpretation':'Jaccard of full-cohort predicted assignments after bootstrap refit'})
            for r in range(a.prediction_reps):
                seed=a.seed+1291*r
                ix=np.arange(len(x))
                i,j=train_test_split(ix,test_size=a.test_fraction,random_state=seed)
                f_tr=fit(x[i],2,seed,a.n_init)
                f_te=fit(x[j],2,seed+1,a.n_init)
                score1=min_prediction_strength(f_te.predict(x[j]),f_tr.predict(x[j]))
                score2=min_prediction_strength(f_tr.predict(x[i]),f_te.predict(x[i]))
                ps_rows.append({'candidate':name,'repetition':r,'seed':seed,
                                'test_cluster_pair_agreement':score1,
                                'train_cluster_pair_agreement':score2,
                                'symmetric_mean_prediction_strength':float(np.nanmean([score1,score2]))})
            vals=np.array([z['max_refit_assignment_jaccard'] for z in raw_rows if z['candidate']==name],float)
            ps=np.array([z['symmetric_mean_prediction_strength'] for z in ps_rows if z['candidate']==name],float)
            summary.append({'candidate':name,'n_patients':len(x),'n_encoded_features':x.shape[1],
                            'bootstrap_refit_jaccard_mean':np.nanmean(vals),
                            'bootstrap_refit_jaccard_min':np.nanmin(vals),
                            'prediction_strength_mean':np.nanmean(ps),
                            'prediction_strength_min':np.nanmin(ps),
                            'bootstrap_repetitions':a.bootstrap_reps,
                            'prediction_repetitions':a.prediction_reps})
            print(name, summary[-1],flush=True)
    pd.DataFrame(raw_rows).to_csv(a.out/'bootstrap_refit_jaccard.csv',index=False)
    pd.DataFrame(ps_rows).to_csv(a.out/'prediction_strength.csv',index=False)
    pd.DataFrame(summary).to_csv(a.out/'complementary_stability_summary.csv',index=False)
    details={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
             'source_sha256':sha256(inp),'method':'06D complementary diagnostics, not original 06B reproduction',
             'input_mode':mode,'quick':a.quick,'n_dev':len(dev),'n_init':a.n_init,
             'random_seed':a.seed,'bootstrap_reps':a.bootstrap_reps,
             'prediction_reps':a.prediction_reps,
             'data_split_for_ps':a.test_fraction,
             'bootstrap_jaccard_definition':'baseline full-cohort partition vs full-cohort predictions from bootstrap refit',
             'prediction_strength_definition':'min per-test-cluster pair co-membership under independently fitted model, averaged symmetric directions',
             'limitations':['Bootstrap Jaccard is predictive on the same source cohort; not external validation',
                            'Prediction strength is unsupervised and cannot establish clinical phenotype validity',
                            'If starting from original matrices, do not re-robust-scale already transformed utilization columns',
                            'Raw mode diagnosis and age mappings must be reconciled to historical notebooks'],
             'scikit_learn':sklearn.__version__,'scipy':scipy.__version__}
    (a.out/'06D_manifest.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print('Saved optional comparator evidence to',a.out.resolve())

if __name__=='__main__':main()
