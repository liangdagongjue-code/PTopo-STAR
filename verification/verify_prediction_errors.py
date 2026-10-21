from pathlib import Path
import json
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
formal=root/"formal_results"
p=root/"verification/formal_recovery_validation.json"
report=json.loads(p.read_text(encoding="utf-8"))
def check(name,ok,detail=None):
    report["checks"].append({"check":name,"pass":bool(ok),"detail":detail})

valid=True
for path in formal.glob("pretraining/**/target_embeddings_*.npz"):
    with np.load(path,allow_pickle=False) as a:
        valid &= a["embeddings"].shape==(274,64) and len(set(a["target_ids"]))==274 and np.isfinite(a["embeddings"]).all()
check("All 200 embedding archives contain finite 274 by 64 features",valid)

scalar=pd.read_csv(formal/"aggregate/scalar_predictions.csv.gz",
                   usecols=["y_true","y_pred","squared_error","absolute_error"])
error=scalar.y_pred-scalar.y_true
check("Scalar squared errors match recorded predictions",
      np.allclose(error**2,scalar.squared_error,rtol=1e-10,atol=1e-12),
      float(np.max(np.abs(error**2-scalar.squared_error))))
check("Scalar absolute errors match recorded predictions",
      np.allclose(np.abs(error),scalar.absolute_error,rtol=1e-10,atol=1e-12),
      float(np.max(np.abs(np.abs(error)-scalar.absolute_error))))

rows=[]
for path in formal.glob("downstream_shards/*_curve_predictions_repeat0.npz"):
    with np.load(path,allow_pickle=False) as a:
        e=a["predictions"].astype(float)-a["truth"].astype(float)
        frame=pd.DataFrame.from_records(a["metadata"])
        frame["recomputed_rmse"]=np.sqrt(np.mean(e**2,axis=1))
        frame["recomputed_mae"]=np.mean(np.abs(e),axis=1)
        rows.append(frame)
rebuilt=pd.concat(rows,ignore_index=True)
saved=pd.read_csv(formal/"aggregate/curve_metrics.csv.gz")
keys=["seed","held_out_family","shots","repeat","method","sample_id"]
j=rebuilt.merge(saved[keys+["curve_rmse","curve_mae"]],on=keys,validate="one_to_one")
check("Every archived stress prediction matches a metric record",len(j)==len(rebuilt),len(j))
for metric in ["rmse","mae"]:
    # Prediction archives store float32 traces; metrics were computed before this cast.
    delta=float(np.max(np.abs(j["recomputed_"+metric]-j["curve_"+metric])))
    check("Curve "+metric+" reproduced from float32 archived predictions",
          np.allclose(j["recomputed_"+metric],j["curve_"+metric],rtol=2e-5,atol=2e-7),delta)

report["passed"]=sum(x["pass"] for x in report["checks"])
report["total"]=len(report["checks"])
p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"passed":report["passed"],"total":report["total"],
                  "failures":[c for c in report["checks"] if not c["pass"]]},ensure_ascii=False))
