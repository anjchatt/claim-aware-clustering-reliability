#!/usr/bin/env python3
"""JBHI Section V: provenance-aware reporting and visualisation add-on.

This script DOES NOT refit K-means, manufacture patient labels, validate phenotypes,
or establish that the 06C reconstruction reproduces the primary cohort.

Inputs:
  - Primary candidate summary CSV (values reported in manuscript Table II)
  - 06C multistart vs single-start summary and weight-pair ARI CSV exports
  - Optional distance/negative-control CSV export
  - Optional TWO patient-level CSV files for a genuine transition matrix

Outputs:
  - Six-candidate review-evidence JSON/CSV (5 primary + 1 distinct reconstruction)
  - Figures: audit evidence matrix, restart seed stability, cross-weight ARI,
    optional feature-distance shares and optional patient transition matrix
  - audit_manifest.json recording SHA-256 hashes and exact CLI/settings

All review thresholds are exploratory and post hoc; no clinical acceptance is implied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

SCHEMA = "JBHI-claim-audit/0.2.0"
PRIMARY_COLUMNS = ["candidate_id", "cohort_id", "silhouette", "seed_ari", "subsample_ari", "refit_masking_ari_20pct", "utilization_only_ari", "reported_source"]
THRESHOLDS = {"seed_ari_min": .80, "subsample_ari_min": .80, "refit_masking_ari_min": .70,
              "utilization_equivalence_ari_min": .95, "cross_specification_ari_min": .80}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary", type=Path, required=True, help="Five-candidate primary summary CSV")
    p.add_argument("--restart", type=Path, required=True, help="06C 12-row n_init=1 vs 20 comparison CSV")
    p.add_argument("--cross-weight", type=Path, required=True, help="06C cross_weight_partition_agreement CSV")
    p.add_argument("--multistart", type=Path, help="06C weight_transition_summary CSV")
    p.add_argument("--distance", type=Path, help="06C distance_and_negative_control CSV")
    p.add_argument("--labels-a", type=Path, help="Optional CSV: patient_nbr,label for w=0.600")
    p.add_argument("--labels-b", type=Path, help="Optional CSV: patient_nbr,label for w=1.000 (SAME cohort)")
    p.add_argument("--label-id", default="patient_nbr", help="Patient ID header in label files")
    p.add_argument("--label-column", default="label", help="Assignment header in label files")
    p.add_argument("--label-cohort", default="development", help="Cohort label for optional transition matrix")
    p.add_argument("--intended-use", default="Exploratory healthcare patient segmentation; no clinical deployment endorsement")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def table(path, required):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    x = pd.read_csv(path)
    missing = sorted(set(required) - set(x.columns))
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}; available: {list(x.columns)}")
    return x


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str)+"\n",encoding="utf-8")


def diagnostic(value, metric, status="observed", note="", limit=None, direction="below"):
    if status != "observed":
        return {"status": status, "value": None, "metric": metric, "flagged": False, "note": note}
    v=float(value)
    if not np.isfinite(v):
        raise ValueError(f"Non-finite measured value for {metric}")
    bad=limit is not None and ((v<limit) if direction == "below" else (v>=limit))
    return {"status": status, "value": round(v,9), "metric":metric, "flagged": bool(bad),
            "threshold":limit, "note":note}


def review_flags(ds):
    return [key for key,item in ds.items() if isinstance(item,dict) and item.get("flagged",False)]


def primary_records(df, path, intended):
    if len(df)!=5 or df["candidate_id"].duplicated().any():
        raise ValueError("Primary table must contain exactly five unique candidate IDs. Do not silently replace primary evidence.")
    records=[]
    for _,row in df.iterrows():
        name=str(row.candidate_id)
        if name not in {"R1 K=2","R1 K=5","R1 K=8","R2 K=2","R3 K=2"}:
            raise ValueError(f"Unexpected candidate {name!r}")
        d={
          "internal_quality": diagnostic(row.silhouette,"mean_silhouette",note="Descriptive, no acceptance cutoff"),
          "seed_stability": diagnostic(row.seed_ari,"mean_pairwise_seed_ARI",limit=THRESHOLDS["seed_ari_min"]),
          "subsample_stability": diagnostic(row.subsample_ari,"mean_pairwise_80pct_subsample_ARI",limit=THRESHOLDS["subsample_ari_min"]),
          "refit_masking": diagnostic(row.refit_masking_ari_20pct,"mean_ARI_20pct_diagnosis_masking_with_refit",limit=THRESHOLDS["refit_masking_ari_min"]),
          "feature_domain": diagnostic(row.utilization_only_ari,"full_vs_utilization_only_ARI",limit=THRESHOLDS["utilization_equivalence_ari_min"],direction="above") if pd.notna(row.utilization_only_ari) else diagnostic(None,"full_vs_utilization_only_ARI",status="not_applicable",note="R1 contains no utilization block"),
          "cross_specification":diagnostic(None,"cross_weight_ARI",status="not_evaluated",note="Separate reconstruction cannot be assigned to primary cohort")
        }
        flags=review_flags(d)
        records.append({
          "schema":SCHEMA,"candidate_id":name,"evidence_stream":"PRIMARY_MANUSCRIPT_TABLE_II",
          "intended_claim":intended,"cohort_id":str(row.cohort_id),
          "source_evidence":[{"file":path.name,"sha256":digest(path),"source_classification":str(row.reported_source)}],
          "diagnostics":d,"review_flags":flags,
          "review_disposition":"PROPOSED_FURTHER_REVIEW" if flags else "INSUFFICIENT_EVIDENCE_FOR_CLINICAL_ENDORSEMENT",
          "clinical_validation_status":"not_established",
          "caveat":"Aggregate values copied from manuscript Table II, not recalculated from raw model predictions. Flags post hoc and illustrative."
        })
    return records


def choose_pair(df, scaler, a=.6, b=1.):
    sub=df.loc[(df.scaler==scaler) &
               (((np.isclose(df.weight_1,a)) & (np.isclose(df.weight_2,b))) |
                ((np.isclose(df.weight_1,b)) & (np.isclose(df.weight_2,a))))]
    if len(sub)!=1:
        raise ValueError(f"Expected precisely one cross-weight result for {scaler}, {a} vs {b}: {len(sub)}")
    return sub.iloc[0]


def reconstruction_record(cross, restart, p_cross, p_restart, intended):
    row=choose_pair(cross,"robust")
    r0=restart.loc[(restart.scaler=="robust")&np.isclose(restart.weight,0.6)]
    r1=restart.loc[(restart.scaler=="robust")&np.isclose(restart.weight,1.0)]
    if len(r0)!=1 or len(r1)!=1: raise ValueError("Missing reconstruction endpoints at robust weights 0.6 and 1.0")
    r0, r1=r0.iloc[0], r1.iloc[0]
    d={
      "internal_quality":diagnostic(None,"silhouette",status="not_evaluated",note="Use separate endpoint sampled-silhouette values; do not mix primary and reconstruction"),
      "seed_stability_w060":diagnostic(r0.seed_ari_all_pairs_ninit20,"mean_pairwise_seed_ARI_at_weight_0.6",limit=THRESHOLDS["seed_ari_min"]),
      "seed_stability_w100":diagnostic(r1.seed_ari_all_pairs_ninit20,"mean_pairwise_seed_ARI_at_weight_1.0",limit=THRESHOLDS["seed_ari_min"]),
      "subsample_stability":diagnostic(None,"subsample_ARI",status="not_evaluated"),
      "refit_masking":diagnostic(None,"masking_ARI",status="not_evaluated"),
      "feature_domain_w060":diagnostic(r0.util_agreement_best_fit_ninit20,"full_vs_util_only_ARI_weight_0.6",limit=THRESHOLDS["utilization_equivalence_ari_min"],direction="above"),
      "feature_domain_w100":diagnostic(r1.util_agreement_best_fit_ninit20,"full_vs_util_only_ARI_weight_1.0",limit=THRESHOLDS["utilization_equivalence_ari_min"],direction="above"),
      "cross_specification":diagnostic(row.development_ari,"ARI_weight_0.6_vs_1.0_robust_development",limit=THRESHOLDS["cross_specification_ari_min"]),
      "aligned_changes_development_pct":diagnostic(row.development_changed_pct_aligned,"percent_changed_label_aligned_development"),
      "cross_specification_heldout":diagnostic(row.heldout_ari_fixed_development_models,"ARI_weight_0.6_vs_1.0_fixed_models_heldout",limit=THRESHOLDS["cross_specification_ari_min"]),
      "aligned_changes_heldout_pct":diagnostic(row.heldout_changed_pct_aligned,"percent_changed_label_aligned_heldout")
    }
    return {"schema":SCHEMA,"candidate_id":"R2 reconstruction: robust w=0.600 vs w=1.000",
      "evidence_stream":"SEPARATE_RAW_SOURCE_RECONSTRUCTION_NOT_PRIMARY_REPLICATION",
      "intended_claim":intended,"cohort_id":"RECONSTRUCTION_SEPARATE_SPLIT",
      "source_evidence":[{"file":p_cross.name,"sha256":digest(p_cross)},
                         {"file":p_restart.name,"sha256":digest(p_restart)}],
      "diagnostics":d,"review_flags":review_flags(d),"review_disposition":"PROPOSED_SPECIFICATION_REVIEW",
      "clinical_validation_status":"not_established",
      "caveat":"40 seeded fits per weight; n_init=20 versus 1 per matching output table. Cross-weight partition selected by within-weight minimum inertia. Held-out models not refitted. Post hoc review flags."}


def primary_matrix(records,out):
    names=[r["candidate_id"] for r in records[:5]]
    colspec=[("internal_quality","Silhouette\n(descriptive)"),("seed_stability","Seed ARI"),
             ("subsample_stability","Subsample ARI"),("refit_masking","20% refit\nmasking ARI"),
             ("feature_domain","Utilization-only\nARI"),("cross_specification","Cross-spec\nARI")]
    cells=np.zeros((len(names),len(colspec)),int)
    labels=[]
    for i,r in enumerate(records[:5]):
        rr=[]
        for j,(key,_) in enumerate(colspec):
            d=r["diagnostics"][key]
            st=d["status"]
            if st=="observed":
                if key=="internal_quality": cells[i,j]=3
                else: cells[i,j]=2 if d["flagged"] else 1
                rr.append(f'{d["value"]:.3f}')
            elif st=="not_applicable": cells[i,j]=4;rr.append("N/A")
            else: cells[i,j]=0;rr.append("NE")
        labels.append(rr)
    fig,ax=plt.subplots(figsize=(11.7,4.5))
    cmap=ListedColormap(["#e7ebef","#d8e9df","#f3d6d5","#dce4f3","#f2f2f2"])
    ax.imshow(cells,cmap=cmap,vmin=-.5,vmax=4.5,aspect="auto")
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            ax.text(j,i,labels[i][j],ha="center",va="center",fontsize=10.3,fontweight="medium")
    ax.set_xticks(range(len(colspec)),[c[1] for c in colspec],fontsize=9.3)
    ax.set_yticks(range(len(names)),names,fontsize=10)
    ax.tick_params(axis="both",length=0)
    ax.set_xticks(np.arange(-.5,cells.shape[1],1), minor=True)
    ax.set_yticks(np.arange(-.5,cells.shape[0],1), minor=True)
    ax.grid(which="minor",color="white",linewidth=3)
    ax.tick_params(which="minor",bottom=False,left=False)
    fig.suptitle("Primary clustering candidates: five diagnostic dimensions",fontsize=13.2,fontweight="bold",y=.995)
    fig.text(.5,.025,"Green = no illustrative flag; pink = illustrative review flag; blue = descriptive only; NE = not evaluated; N/A = not applicable.",ha="center",fontsize=8.5)
    fig.text(.5,.002,"Source: manuscript Table II, n=10,000. Reconstruction is a separate evidence record. No clinical approval implied.",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.055,1,.94]);fig.savefig(out/"Fig_Audit_PrimaryCandidates.png",dpi=300,bbox_inches="tight");fig.savefig(out/"Fig_Audit_PrimaryCandidates.svg",bbox_inches="tight");plt.close(fig)


def restart_plot(data,out):
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),sharey=True)
    for ax,scaler in zip(axes,["robust","standard"]):
        sub=data.loc[data.scaler==scaler].sort_values("weight")
        if len(sub)<2: raise ValueError(f"Expected multiple {scaler} weights in restart data")
        ax.plot(sub.weight,sub.seed_ari_all_pairs_ninit1,color="#B75B44",marker="o",linewidth=2,label="1 restart")
        ax.plot(sub.weight,sub.seed_ari_all_pairs_ninit20,color="#24648B",marker="s",linewidth=2,label="20 restarts")
        ax.axhline(.8,ls="--",color="#959da6",lw=.8)
        ax.set_title(scaler.title()+" scaling",fontsize=11.5)
        ax.set_xlabel("Utilization-coordinate weight")
        ax.set_xticks(sub.weight,[f"{x:.3f}" for x in sub.weight],rotation=50)
        ax.grid(axis="y",alpha=.2)
        ax.set_ylim(-.03,1.05)
    axes[0].set_ylabel("Mean pairwise seed ARI")
    axes[1].legend(loc="lower right",frameon=False)
    fig.suptitle("Reconstruction: initialization agreement under matched weights",fontsize=13,fontweight="bold")
    fig.text(.5,.005,"Source: separate raw-source R2 reconstruction; 40 independently seeded fits/configuration. Dashed line: exploratory review criterion.",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(out/"Fig_Reconstruction_RestartStability.png",dpi=300,bbox_inches="tight");fig.savefig(out/"Fig_Reconstruction_RestartStability.svg",bbox_inches="tight");plt.close(fig)


def cross_weight_plot(data,out):
    weights=sorted(set(data.weight_1).union(set(data.weight_2)))
    if len(weights)>12:
        raise ValueError("Cross-weight heatmap expects compact 6-weight comparison; supply matched grid")
    fig,axes=plt.subplots(1,2,figsize=(12.4,5.1),sharex=True,sharey=True)
    ims=[]
    for ax,scale in zip(axes,["robust","standard"]):
        q=data.loc[data.scaler==scale]
        grid=np.full((len(weights),len(weights)),np.nan)
        for i in range(len(weights)):grid[i,i]=1.
        for _,r in q.iterrows():
            a=weights.index(float(r.weight_1));b=weights.index(float(r.weight_2))
            grid[a,b]=grid[b,a]=float(r.development_ari)
        if np.isnan(grid).any(): raise ValueError(f"Incomplete cross-weight pairs for {scale}")
        img=ax.imshow(grid,vmin=-.08,vmax=1,cmap="RdYlBu",aspect="equal")
        ims.append(img)
        ax.set_xticks(range(len(weights)),[f"{w:g}" for w in weights],rotation=60,fontsize=8)
        ax.set_yticks(range(len(weights)),[f"{w:g}" for w in weights],fontsize=8)
        ax.set_title(f"{scale.title()} scaling")
        for i in range(len(weights)):
            for j in range(len(weights)):
                ax.text(j,i,f"{grid[i,j]:.3f}",ha="center",va="center",fontsize=7.4,color="white" if grid[i,j]>.70 or grid[i,j]<.15 else "black")
        ax.set_xlabel("Utilization-coordinate weight")
    axes[0].set_ylabel("Utilization-coordinate weight")
    fig.subplots_adjust(left=.07,right=.87,bottom=.23,top=.84,wspace=.15)
    color_axis=fig.add_axes([.915,.255,.02,.50])
    fig.colorbar(ims[-1],cax=color_axis,label="Cross-weight development ARI")
    fig.suptitle("Reconstruction: patient grouping changes between weights",fontsize=13,fontweight="bold",y=.985)
    fig.text(.5,.035,"Each cell compares minimum-inertia fits at two weights in reconstructed development patients; not primary-cohort replication.",ha="center",fontsize=8)
    fig.savefig(out/"Fig_Reconstruction_CrossWeightARI.png",dpi=300,bbox_inches="tight");fig.savefig(out/"Fig_Reconstruction_CrossWeightARI.svg",bbox_inches="tight");plt.close(fig)


def distance_plot(path,out):
    d=table(path,["scaler","clinical_share","valid_sampled_pairs","experiment"])
    d=d.loc[d.clinical_share.notna()].copy()
    for k in ("robust","standard"):
        if len(d.loc[d.scaler==k])!=1: raise ValueError(f"Need one distance summary for {k}")
    d=d.set_index("scaler").loc[["robust","standard"]]
    clinical=100*d.clinical_share.to_numpy(float)
    util=100-clinical
    fig,ax=plt.subplots(figsize=(8.7,3.7))
    y=np.arange(2)
    ax.barh(y,clinical,color="#447CA0",label="Clinical block")
    ax.barh(y,util,left=clinical,color="#C18B58",label="Utilization block")
    for i in range(2):
        ax.text(clinical[i]/2,i,f"{clinical[i]:.1f}%",ha="center",va="center",color="white",fontsize=11,fontweight="bold")
        ax.text(clinical[i]+util[i]/2,i,f"{util[i]:.1f}%",ha="center",va="center",color="white",fontsize=11,fontweight="bold")
    ax.set_yticks(y,["Robust scaling","Standard scaling"])
    ax.invert_yaxis()
    ax.set_xlim(0,100)
    ax.set_xlabel("Share of mean sampled-pair squared Euclidean distance (%)")
    ax.legend(ncol=2,loc="lower center",bbox_to_anchor=(.5,1.0),frameon=False)
    fig.suptitle("Reconstruction: feature blocks contribute to distance",fontsize=13,fontweight="bold",y=1.12)
    fig.text(.5,.012,"Source: reconstruction distance audit; 19,999 valid sampled pairs. Distance share does not identify feature importance.",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(out/"Fig_Reconstruction_DistanceShares.png",dpi=300,bbox_inches="tight");fig.savefig(out/"Fig_Reconstruction_DistanceShares.svg",bbox_inches="tight");plt.close(fig)


def transition_plot(path_a,path_b,id_col,label_col,cohort,out):
    a=table(path_a,[id_col,label_col])[[id_col,label_col]].copy()
    b=table(path_b,[id_col,label_col])[[id_col,label_col]].copy()
    if a[id_col].duplicated().any() or b[id_col].duplicated().any():
        raise ValueError("Duplicate patient IDs in label files")
    a[id_col]=a[id_col].astype(str)
    b[id_col]=b[id_col].astype(str)
    if set(a[id_col])!=set(b[id_col]):
        raise ValueError("Patient IDs differ across supplied label files; require EXACT same patients")
    merged=a.merge(b,on=id_col,how="inner",validate="one_to_one",suffixes=("_a","_b"))
    y0=merged[f"{label_col}_a"].astype(str).to_numpy()
    y1=merged[f"{label_col}_b"].astype(str).to_numpy()
    ua,ub=sorted(set(y0)),sorted(set(y1))
    if len(ua)!=2 or len(ub)!=2:raise ValueError("Transition plot supports 2 clusters in both specifications")
    count=np.array([[int(np.sum((y0==x)&(y1==z))) for z in ub] for x in ua])
    ri,ci=linear_sum_assignment(-count)
    mapping={ub[j]:ua[i] for i,j in zip(ri,ci)}
    aligned=np.array([mapping[z] for z in y1])
    diff=float(np.mean(y0!=aligned))*100
    ari=float(adjusted_rand_score(y0,y1))
    # Display columns in alignment order to make diagonal correspond to matched memberships.
    j_order=[ub.index(next(z for z,m in mapping.items() if m==x)) for x in ua]
    aligned_count=count[:,j_order]
    if aligned_count.sum()!=len(merged):raise AssertionError("Transition counts do not sum to cohort")
    fig,ax=plt.subplots(figsize=(6.6,4.7))
    ax.imshow(aligned_count,cmap="Blues")
    ax.set_xticks([0,1],[f"Matched B group {i}" for i in range(2)])
    ax.set_yticks([0,1],[f"A group {x}" for x in ua])
    ax.set_xlabel("Weight 1.000 (optimally aligned labels)")
    ax.set_ylabel("Weight 0.600")
    for i in range(2):
        for j in range(2): ax.text(j,i,f"{aligned_count[i,j]:,}",ha="center",va="center",fontsize=13,color="white" if aligned_count[i,j]>aligned_count.max()*.52 else "black")
    ax.set_title(f"R2 reconstruction, {cohort}: matched patient transitions\nARI={ari:.4f}; changed={diff:.2f}% (n={len(merged):,})",fontsize=11.5)
    fig.tight_layout();fig.savefig(out/f"Fig_PatientTransitions_{cohort}.png",dpi=300,bbox_inches="tight");fig.savefig(out/f"Fig_PatientTransitions_{cohort}.svg",bbox_inches="tight");plt.close(fig)
    pd.DataFrame(aligned_count,index=[f"A_{x}" for x in ua],columns=[f"Matched_B_{i}" for i in range(2)]).to_csv(out/f"patient_transition_counts_{cohort}.csv")
    return {"cohort":cohort,"n":len(merged),"ARI":ari,"changed_pct":diff,"count_matrix":aligned_count.tolist(),"source":[path_a.name,path_b.name]}


def main():
    a=arguments()
    a.out.mkdir(parents=True,exist_ok=True)
    loaded={}
    for name,path,required in [
        ("primary",a.primary,PRIMARY_COLUMNS),
        ("restart",a.restart,["scaler","weight","seed_ari_all_pairs_ninit1","seed_ari_all_pairs_ninit20","util_agreement_best_fit_ninit20"]),
        ("cross_weight",a.cross_weight,["scaler","weight_1","weight_2","development_ari","development_changed_pct_aligned","heldout_ari_fixed_development_models","heldout_changed_pct_aligned"]),
    ]:
        loaded[name]=table(path,required)
    if a.multistart: loaded["multistart"]=table(a.multistart,["scaler","weight","seed_ari_all_pairs","n_independent_fits"])
    if a.distance:loaded["distance"]=table(a.distance,["scaler","clinical_share","valid_sampled_pairs","experiment"])
    restart=loaded["restart"]
    if set(restart.scaler)!={"robust","standard"} or len(restart)!=12:
        raise ValueError("Expected exactly 12 restart-comparison rows: 2 scalers x 6 shared weights")
    if loaded.get("multistart") is not None:
        m=loaded["multistart"]
        joined=restart.merge(m[["scaler","weight","seed_ari_all_pairs","n_independent_fits"]],on=["scaler","weight"],validate="one_to_one")
        if len(joined)!=12 or (joined.n_independent_fits!=40).any():
            raise ValueError("Multistart summary must contain the 12 matching 40-fit configs")
        if not np.allclose(joined.seed_ari_all_pairs,joined.seed_ari_all_pairs_ninit20,atol=1e-8):
            raise ValueError("Restart and multistart summaries do NOT agree; examine file provenance")
    records=primary_records(loaded["primary"],a.primary,a.intended_use)
    records.append(reconstruction_record(loaded["cross_weight"],restart,a.cross_weight,a.restart,a.intended_use))
    save_json({"schema":SCHEMA,"generated_from":"reported aggregate tables; no patient-level inference","threshold_policy":{"calibration":"none","selection":"post_exploratory","thresholds":THRESHOLDS},"records":records},a.out/"claim_aware_evidence_records.json")
    flat=[]
    for rec in records:
        flat.append({"candidate_id":rec["candidate_id"],"evidence_stream":rec["evidence_stream"],"cohort_id":rec["cohort_id"],"review_flags":";".join(rec["review_flags"]),"review_disposition":rec["review_disposition"],
              **{f"{k}_value":v["value"] for k,v in rec["diagnostics"].items()},
              **{f"{k}_status":v["status"] for k,v in rec["diagnostics"].items()}})
    pd.DataFrame(flat).to_csv(a.out/"claim_aware_evidence_records.csv",index=False)
    primary_matrix(records,a.out)
    restart_plot(restart,a.out)
    cross_weight_plot(loaded["cross_weight"],a.out)
    if a.distance:distance_plot(a.distance,a.out)
    if bool(a.labels_a)!=bool(a.labels_b):
        raise ValueError("Provide BOTH --labels-a and --labels-b; transition matrix cannot be generated from percentages")
    trans=None
    if a.labels_a and a.labels_b:
        trans=transition_plot(a.labels_a,a.labels_b,a.label_id,a.label_column,a.label_cohort,a.out)
        save_json(trans,a.out/f"patient_transition_summary_{a.label_cohort}.json")
    else:
        print("SKIPPED patient-level transition matrix: both raw assignment files are required.")
    paths=[p for p in [a.primary,a.restart,a.cross_weight,a.multistart,a.distance,a.labels_a,a.labels_b] if p]
    save_json({"created_utc":datetime.now(timezone.utc).isoformat(),"script":Path(__file__).name,"script_sha256":digest(Path(__file__)),
      "schema":SCHEMA,"input_sha256":{str(p.resolve()):digest(p) for p in paths},
      "platform":platform.platform(),"python":sys.version.split()[0],"numpy":np.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__,
      "scipy_note":"Optimal permutation solved using scipy.optimize.linear_sum_assignment in optional patient transition plot",
      "notes":["This report is produced from saved manuscript tables and reconstruction summaries, not K-means refitting.",
               "Primary and reconstruction are intentionally separate evidence records.",
               "The uploaded reconstruction run's original Python and scikit-learn versions CANNOT be inferred from this environment.",
               "Do not interpret post hoc flags as clinical safety thresholds.",
               "Please verify primary values, simulation settings and reconstruction run_manifest against original executed scripts."],
      "optional_patient_transition_computed":trans is not None},a.out/"sectionV_audit_manifest.json")
    (a.out/"figure_captions.txt").write_text(
      "Fig. A. Claim-aware primary-candidate evidence matrix (development n=10,000). "
      "Silhouette is descriptive; other flags use uncalibrated retrospective review criteria. "
      "NE indicates not evaluated and N/A not applicable. Source: reported manuscript Table II.\n\n"
      "Fig. B. Matched single-restart versus 20-restart mean pairwise seed ARI in the separate raw-source R2 reconstruction "
      "(40 independent seeded fits per weight, six shared utilization weights, robust and standard scaling). "
      "Source: JBHI_06C_ninit1_vs_ninit20_comparison.csv.\n\n"
      "Fig. C. Development patient-partition agreement between minimum-inertia K-means fits at each pair of utilization weights. "
      "Robust and standard scaling shown separately. Source: cross_weight_partition_agreement.csv; "
      "this is NOT primary cohort replication or independent held-out fitting.\n\n"
      "Fig. D (optional). Shares of sampled-pair squared Euclidean distance attributable to clinical versus utilization blocks "
      "in the reconstructed R2 representation; not feature-importance estimates. Source: distance_and_negative_control.csv.\n\n"
      "Fig. E (only if patient-level files provided). Label-aligned patient transition counts for two weights on an IDENTICAL cohort. "
      "The same patient identifiers must be present in both CSVs.\n",encoding="utf-8")
    print(f"Created {len(records)} separated evidence records and graphics at {a.out}")
    for f in sorted(a.out.iterdir()):print("  -",f.name)

if __name__=="__main__":
    main()
