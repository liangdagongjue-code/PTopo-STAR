from pathlib import Path
import numpy as np
import pandas as pd
import json

out=Path(__file__).resolve().parents[1]
root=out.parent; data=out/"formal_results/aggregate"
checks=[]
def check(name,condition,detail=None):
    checks.append(dict(check=name,pass_=bool(condition),detail=detail))
    if not condition: print("FAILED",name,detail,flush=True)

splits=pd.read_csv(out/"derived_data/support_test_splits.csv").fillna("")
one=next((out/"formal_results/pretraining").glob("**/target_embeddings_*.npz"))
with np.load(one,allow_pickle=False) as a: ids=a["target_ids"].copy()
groups=np.array([np.argmax(np.array(x.split("_"),dtype=float)) for x in ids])
split_lookup={}
for row in splits.itertuples():
    support=set(row.support_ids.split("|")) if row.support_ids else set()
    test=set(row.test_ids.split("|"))
    family=set(ids[groups==row.held_out_family])
    check("Disjoint and exhaustive split "+str((row.held_out_family,row.shots,row.repeat)),
          not support&test and support|test==family and len(support)==row.shots and len(test)==row.n_test)
    target=np.flatnonzero(groups==row.held_out_family)
    rng=np.random.default_rng(91731+100003*row.held_out_family+1009*row.shots+row.repeat)
    regenerated=set(ids[rng.choice(target,size=row.shots,replace=False)]) if row.shots else set()
    check("Deterministic support "+str((row.held_out_family,row.shots,row.repeat)),regenerated==support)
    split_lookup[(row.held_out_family,row.shots,row.repeat)]=test
check("488 unique splits",len(splits)==488)
print("All support and test splits checked.",flush=True)

s=pd.read_csv(data/"scalar_predictions.csv.gz")
c=pd.read_csv(data/"curve_metrics.csv.gz")
keys=["task","held_out_family","shots","seed","repeat","sample_id"]
check("Scalar prediction keys unique",not s.duplicated(keys+["method"]).any())
check("Curve metric keys unique",not c.duplicated(keys+["method"]).any())
for frame,label in [(s,"scalar"),(c,"curve")]:
    grouped=frame.groupby(["task","held_out_family","shots","seed","repeat","method"])
    matched=True
    for key,g in grouped:
        if set(g.sample_id)!=split_lookup[(key[1],key[2],key[4])]:
            matched=False;break
    check("All "+label+" methods use the declared test samples",matched,len(grouped))
    check("All "+label+" prediction records have the correct family",
          all(int(f)==int(np.argmax(np.array(sid.split("_"),dtype=float)))
              for f,sid in frame[["held_out_family","sample_id"]].drop_duplicates().itertuples(index=False,name=None)))
check("Scalar truths constant across methods and repetitions",
      s.groupby(["task","sample_id"]).y_true.nunique().max()==1)

wide=s.pivot(index=keys,columns="method",values="y_pred")
star=s[s.method=="ptopo_star"].set_index(keys)
reconstructed=wide.explicit_ridge+star.gate*(wide.ptopo_fusion-wide.explicit_ridge)
check("Scalar gated predictions satisfy the stated residual equation",
      np.allclose(reconstructed,wide.ptopo_star,rtol=1e-10,atol=1e-12),
      float(np.max(abs(reconstructed-wide.ptopo_star))))
check("Scalar ungated residual equals ordinary PTopo fusion",
      np.allclose(wide.ptopo_residual_ungated,wide.ptopo_fusion,rtol=1e-12,atol=1e-12))

selected=pd.read_csv(out/"derived_data/gate_selections.csv").rename(columns={"fold":"held_out_family"})
joined=star.reset_index().merge(selected,on=["task","held_out_family","seed"],suffixes=("","_selection"),validate="many_to_one")
g=joined.scale*np.exp(-joined.lambda_distance*np.maximum(joined.distance_norm-1,0)**2
                     -joined.lambda_uncertainty*np.maximum(joined.uncertainty_norm-1,0)**2)
check("Saved gates match source-selected coefficients",np.allclose(g,joined.gate,rtol=1e-12,atol=1e-12))

energy=s[(s.task=="energy_dissipation")&s.shots.isin([1,2,4])]
p=energy.pivot(index=keys,columns="method",values="squared_error")
per_design=p[["ptopo_star","explicit_ridge"]].groupby(["held_out_family","sample_id"]).mean()
sd=float(energy.normalizer_sd.iloc[0])
family=np.sqrt(per_design.groupby("held_out_family").mean())/sd
family["paired_difference"]=family.ptopo_star-family.explicit_ridge
check("Independent primary paired effect",np.isclose(family.paired_difference.mean(),-.290358211043852,rtol=1e-12,atol=1e-12))
family["effect_without_this_family"]=[family.paired_difference.drop(i).mean() for i in family.index]
names=energy.groupby("held_out_family").held_out_name.first()
family.insert(0,"family",names)
family.to_csv(out/"derived_data/family_omission_sensitivity.csv")

summary=pd.read_csv(data/"summary_metrics.csv")
pooled=summary[summary.shots.isin([1,2,4])].groupby(["method","task"]).macro_nrmse.mean().unstack()
pooled.to_csv(out/"derived_data/pooled_outcomes_independent_review.csv")
gates={}
for frame in [s,c]:
    for task,z in frame[(frame.method=="ptopo_star")&frame.shots.isin([1,2,4])].groupby("task"):
        gates[task]={"mean":float(z.gate.mean()),"median":float(z.gate.median()),
                     "fraction_below_0_9":float((z.gate<.9).mean()),"records":len(z)}
counts=pd.Series(groups).value_counts().sort_index()
checks.append(dict(check="All paired family effects favour PTopo-STAR",pass_=bool((family.paired_difference<0).all()),detail=family.paired_difference.tolist()))
report={"checks":checks,"passed":sum(x["pass_"] for x in checks),"total":len(checks),
        "family_counts":counts.to_dict(),"family_omission_range":[float(family.effect_without_this_family.min()),float(family.effect_without_this_family.max())],
        "gates":gates,"study_status":"Reanalysis of existing formal predictions; no new training."}
(out/"verification/independent_data_review.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="checks"},ensure_ascii=False,indent=2),flush=True)
print(family.to_string(),flush=True)
