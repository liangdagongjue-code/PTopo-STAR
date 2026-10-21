from pathlib import Path
import json,itertools,zipfile,sys,importlib.util,ast
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
formal=root/"formal_results"
out=root
v=root/"verification"
checks=[]
def check(name,ok,detail=None):
    checks.append({"check":name,"pass":bool(ok),"detail":detail})
    if not ok:print("CHECK FAILED",name,detail,flush=True)

models=list(formal.glob("pretraining/**/model_*.pt"))
embeddings=list(formal.glob("pretraining/**/target_embeddings_*.npz"))
archives=sorted(formal.glob("downstream_shards/*_curve_predictions_repeat0.npz"))
check("160 original formal models recovered",len(models)==160)
check("200 learned and random embeddings recovered",len(embeddings)==200)
check("40 original formal curve archives recovered",len(archives)==40)
check("Checkpoint zip CRC",all(zipfile.ZipFile(p).testzip() is None for p in models))
truths={}
archive_rows=0
for p in archives:
    with np.load(p,allow_pickle=False) as a:
        check("Curve archive schema "+p.name,{"metadata","predictions","truth","strain"}<=set(a.files))
        check("Finite curve archive "+p.name,all(np.isfinite(a[k]).all() for k in ["predictions","truth","strain"]))
        check("Curve array shape "+p.name,a["predictions"].shape==a["truth"].shape==a["strain"].shape and a["truth"].shape[1]==120)
        archive_rows+=len(a["metadata"])
        for row,stress,strain in zip(a["metadata"],a["truth"],a["strain"]):
            sid=str(row["sample_id"])
            if sid in truths:
                check_equal=np.array_equal(stress,truths[sid][0]) and np.array_equal(strain,truths[sid][1])
                if not check_equal:raise ValueError("Inconsistent experimental trace "+sid)
            else:truths[sid]=(stress.copy(),strain.copy())
check("274 unique physical traces recovered",len(truths)==274)
endpoint=pd.read_csv(out/"derived_data/endpoint_verification.csv")
recomputed_energy={}
for _,r in endpoint.iterrows():
    stress,strain=truths[str(r["sample_id"])]
    stress=np.asarray(stress,dtype=np.float32);strain=np.asarray(strain,dtype=np.float32)
    energy=float(np.trapezoid(stress[:60]-stress[60:][::-1],dx=float(np.max(strain))/60))
    peak=float(stress.max())
    recomputed_energy[str(r["sample_id"])]=energy
print("Recovered model and curve files checked.",flush=True)

aggregate=formal/"aggregate"
scalar=pd.read_csv(aggregate/"scalar_predictions.csv.gz")
curve=pd.read_csv(aggregate/"curve_metrics.csv.gz")
runs=pd.read_csv(aggregate/"run_metrics.csv")
stored_energy=scalar.loc[scalar.task=="energy_dissipation"].groupby("sample_id").y_true.first()
endpoint_checks=[np.isclose(energy,float(stored_energy.loc[sid]),rtol=1e-6,atol=1e-7) for sid,energy in recomputed_energy.items()]
check("All 274 energy endpoints reproduced from recovered traces",len(endpoint_checks)==274 and all(endpoint_checks))
check("Formal scalar prediction row count",len(scalar)==1715340,len(scalar))
check("Formal curve-metric row count",len(curve)==857670,len(curve))
keys=["task","method","shots","held_out_family","seed","repeat"]
for frame,value,mode in [(scalar,"squared_error","scalar"),(curve,"curve_rmse","curve")]:
    temp=frame.assign(_q=frame[value]**2 if mode=="curve" else frame[value])
    rebuilt=temp.groupby(keys).agg(mse=("_q","mean"),sd=("normalizer_sd","first")).reset_index()
    rebuilt["nrmse_rebuilt"]=np.sqrt(rebuilt["mse"])/rebuilt["sd"]
    joined=rebuilt.merge(runs[keys+["nrmse_global_sd"]],on=keys,validate="one_to_one")
    diff=float(np.max(np.abs(joined["nrmse_rebuilt"]-joined["nrmse_global_sd"])))
    check("All "+mode+" run NRMSE values rebuilt",len(joined)==len(rebuilt) and diff<1e-12,{"rows":len(joined),"max_abs_difference":diff})
for name in ["family_metrics.csv","summary_metrics.csv","summary_family_bootstrap.csv","inference.csv","negative_transfer.csv"]:
    a=pd.read_csv(aggregate/name);b=pd.read_csv(out/"derived_data"/name)
    numeric=[c for c in a.select_dtypes(include="number").columns if c in b]
    aligned=len(a)==len(b) and list(a.columns)==list(b.columns)
    equal=aligned and all(np.allclose(a[c],b[c],equal_nan=True,rtol=1e-12,atol=1e-12) for c in numeric)
    textcols=[c for c in a.columns if c not in numeric]
    equal=equal and all(a[c].fillna("").astype(str).equals(b[c].fillna("").astype(str)) for c in textcols)
    check("Remote aggregate matches handoff "+name,equal,{"rows":len(a),"numeric_columns":len(numeric)})

# Load only inspected pure aggregation functions; training and model imports are unnecessary.
source=ast.parse((out/"supplementary_software/aggregate_results.py").read_text(encoding="utf-8"))
names={"paired_sample_table","family_effects","exact_sign_flip_pvalue","holm_adjust","negative_transfer_summary"}
namespace={"np":np,"pd":pd,"itertools":itertools}
exec(compile(ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),"<aggregation functions>","exec"),namespace)
saved=pd.read_csv(out/"derived_data/inference.csv")
results=[];pvalues=[]
specs=[("ptopo_star","explicit_ridge"),("ptopo_fusion","masked_only_fusion"),("ptopo_star","ptopo_residual_ungated")]
for idx,(method,reference) in enumerate(specs):
    collapsed=namespace["paired_sample_table"](scalar,"energy_dissipation",[1,2,4],method,reference)
    effects=namespace["family_effects"](collapsed)
    point=float(effects.mean());pvalue=namespace["exact_sign_flip_pvalue"](effects)
    families=sorted(collapsed.held_out_family.unique())
    arrays={f:collapsed.loc[collapsed.held_out_family==f,["method_mse","reference_mse"]].to_numpy() for f in families}
    sd=float(collapsed.normalizer.iloc[0])
    rng=np.random.default_rng(20260901+idx);boot=[]
    for _ in range(10000):
        sampled_families=rng.choice(families,size=len(families),replace=True)
        diffs=[]
        for f in sampled_families:
            x=arrays[f];sample=x[rng.integers(0,len(x),size=len(x))]
            means=sample.mean(axis=0)
            diffs.append((np.sqrt(means[0])-np.sqrt(means[1]))/sd)
        boot.append(float(np.mean(diffs)))
    lo,hi=np.quantile(boot,[.025,.975])
    row=saved[(saved["method"]==method)&(saved["reference"]==reference)&(saved["shots"]=="1|2|4")].iloc[0]
    observed=[point,float(lo),float(hi),pvalue]
    expected=[float(row[c]) for c in ["effect_nrmse_method_minus_reference","ci95_low","ci95_high","family_exact_sign_flip_p"]]
    check("Primary paired inference reproduced "+method+" vs "+reference,np.allclose(observed,expected,rtol=1e-12,atol=1e-12),observed)
    results.append({"method":method,"reference":reference,"effect":point,"ci95":[float(lo),float(hi)],"exact_p":pvalue})
    pvalues.append(pvalue)
    print("Reproduced paired inference",method,"vs",reference,flush=True)
adjusted=namespace["holm_adjust"](pvalues)
for i,r in enumerate(results):r["holm_p"]=adjusted[i]
rebuilt=namespace["negative_transfer_summary"](runs)
saved_nt=pd.read_csv(out/"derived_data/negative_transfer.csv")
ntkeys=["task","shots","method"]
j=rebuilt.merge(saved_nt,on=ntkeys,suffixes=("_new","_saved"),validate="one_to_one")
check("All 120 negative-transfer summaries reproduced",len(j)==120 and np.allclose(j.negative_transfer_rate_new,j.negative_transfer_rate_saved,rtol=1e-12,atol=1e-12),len(j))

# Export auditable plotted values, then regenerate the formal Figure 4.
figure_data=out/"derived_data/figure4"
figure_data.mkdir(exist_ok=True)
selected=scalar[(scalar.task=="energy_dissipation")&(scalar.shots==2)]
a=selected[selected.method=="ptopo_star"];b=selected[selected.method=="explicit_ridge"]
pairkeys=["held_out_family","sample_id","seed","shots","repeat"]
paired=a.merge(b[pairkeys+["absolute_error"]],on=pairkeys,suffixes=("_star","_explicit"),validate="one_to_one")
paired["delta_error"]=(paired.absolute_error_star-paired.absolute_error_explicit)/paired.normalizer_sd
points=paired.groupby(["held_out_family","sample_id"],as_index=False).agg(gate=("gate","mean"),distance=("distance_norm","mean"),uncertainty=("uncertainty_norm","mean"),delta_error=("delta_error","mean"))
points.to_csv(figure_data/"gate_points.csv",index=False)
spec=importlib.util.spec_from_file_location("publication_assets",out/"supplementary_software/make_publication_assets.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.configure_style()
module.figure4_gate_and_curves(aggregate,formal/"downstream_shards",out/"figures")
metadata,predicted,truth,strain=module.load_curve_archive(formal/"downstream_shards")
seed=int(np.median(np.unique(metadata["seed"])))
curve_rows=[]
for family in [0,4,5,7]:
    mask=(metadata["seed"]==seed)&(metadata["shots"]==2)&(metadata["held_out_family"]==family)
    candidates=sorted(set(metadata[mask&(metadata["method"]=="explicit_ridge")]["sample_id"])&set(metadata[mask&(metadata["method"]=="ptopo_star")]["sample_id"]))
    sid=candidates[0]
    for idx in np.flatnonzero(mask&(metadata["sample_id"]==sid)&np.isin(metadata["method"],["explicit_ridge","ptopo_star"])):
        for n in range(120):
            curve_rows.append({"family":family,"sample_id":str(sid),"seed":seed,"shots":2,"method":str(metadata[idx]["method"]),"point":n+1,"strain":float(strain[idx,n]),"measured_stress_mpa":float(truth[idx,n]),"predicted_stress_mpa":float(predicted[idx,n])})
pd.DataFrame(curve_rows).to_csv(figure_data/"selected_stress_traces.csv",index=False)
check("Figure 4 regenerated from formal results",(out/"figures/figure4_gate_curves.pdf").exists(),{"gate_points":len(points),"plotted_curve_rows":len(curve_rows)})
summary={"checks":checks,"passed":sum(c["pass"] for c in checks),"total":len(checks),"paired_inference":results,"formal_curve_archive_rows":archive_rows,
         "method":"Recovered outputs were re-aggregated; no model training or new physical experiment was performed."}
(v/"formal_recovery_validation.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"passed":summary["passed"],"total":summary["total"],"failures":[c for c in checks if not c["pass"]]},ensure_ascii=False),flush=True)
