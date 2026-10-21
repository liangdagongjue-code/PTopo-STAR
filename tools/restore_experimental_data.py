"""Join original experimental data into a separate local verification copy."""
from pathlib import Path
import argparse, gzip, hashlib, json, shutil, urllib.request, zipfile
import numpy as np
import pandas as pd

SOURCE_URL='https://raw.githubusercontent.com/maxineAPS/energy-dissipation-metamaterial-dataset/ce432eda188c2eb0b470cf0203442d89c129eaa6/dataset.zip'
SOURCE_SHA='c983f31ce9c32c106ba78b068c6e7a83369a1c87bf89baaeafcfbb0f8f3386b2'

def get_source(work,source_zip=None):
    work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    archive=Path(source_zip).resolve() if source_zip else work/'upstream/dataset.zip'
    if not archive.exists():
        if source_zip: raise FileNotFoundError(archive)
        archive.parent.mkdir(parents=True,exist_ok=True)
        partial=archive.with_suffix('.download')
        request=urllib.request.Request(SOURCE_URL,headers={'User-Agent':'PTopo-STAR-reproducibility/1.0'})
        with urllib.request.urlopen(request,timeout=60) as src,partial.open('wb') as dst:
            shutil.copyfileobj(src,dst)
        if hashlib.sha256(partial.read_bytes()).hexdigest()!=SOURCE_SHA:
            raise ValueError('Source archive checksum mismatch; original dataset version was not obtained.')
        partial.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError('Source archive checksum mismatch.')
    samples={}
    data_dir=work/'datasets/processed/tpms274_raw/dataset'
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.startswith('dataset/') or not name.endswith('.json'):continue
            rel=Path(name).relative_to('dataset')
            target=(data_dir/rel).resolve()
            if not target.is_relative_to(data_dir.resolve()): raise ValueError('Unsafe archive path.')
            raw=z.read(name);data=json.loads(raw);sid=Path(name).stem
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            if sid in samples: raise ValueError('Duplicate source identifier.')
            samples[sid]={
                'strain':np.asarray(data['strain'],dtype=np.float32),
                'stress':np.asarray(data['stress'],dtype=np.float32),
                'energy':float(np.float32(data['energy_dissipation'])),
                'peak':float(np.max(np.asarray(data['stress'],dtype=np.float32)))}
    if len(samples)!=274: raise ValueError(f'Expected 274 records, got {len(samples)}')
    return samples

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--source-zip',type=Path)
    args=parser.parse_args()
    package=Path(__file__).resolve().parents[1];work=args.work_dir.resolve()
    if work==package or work.is_relative_to(package):
        parser.error('Use a work directory outside the public package.')
    samples=get_source(work,args.source_zip)
    restored=work/'restored'
    if restored.exists():
        parser.error('The restored folder already exists. Choose a new work directory to preserve prior work.')
    shutil.copytree(package,restored)
    for p in (restored/'formal_results/downstream_shards').glob('*curve_predictions*.npz'):
        with np.load(p,allow_pickle=False) as a:
            metadata=a['metadata'].copy();predictions=a['predictions'].copy()
        ids=[str(row['sample_id']) for row in metadata]
        np.savez_compressed(p,metadata=metadata,predictions=predictions,
                            truth=np.stack([samples[s]['stress'] for s in ids]),
                            strain=np.stack([samples[s]['strain'] for s in ids]))
    for name in ['scalar_predictions.csv.gz','curve_metrics.csv.gz']:
        source=package/'formal_results/aggregate'/name
        target=restored/'formal_results/aggregate'/name
        first=True
        with gzip.open(target,'wt',encoding='utf-8',newline='') as dst:
            for frame in pd.read_csv(source,chunksize=150000,float_precision='round_trip'):
                energy=frame.sample_id.map({k:v['energy'] for k,v in samples.items()})
                peak=frame.sample_id.map({k:v['peak'] for k,v in samples.items()})
                if energy.isna().any() or peak.isna().any():raise ValueError('Unknown experimental identifier.')
                if name.startswith('scalar'):
                    truth=np.where(frame.task=='energy_dissipation',energy,peak)
                    if not set(frame.task)<={'energy_dissipation','peak_stress'}:raise ValueError('Unknown scalar endpoint.')
                    frame.insert(frame.columns.get_loc('y_pred'),'y_true',truth)
                    frame.insert(frame.columns.get_loc('y_pred')+1,'error',frame.y_pred-frame.y_true)
                else:
                    frame.insert(frame.columns.get_loc('predicted_curve_energy'),'official_energy',energy)
                    frame.insert(frame.columns.get_loc('predicted_curve_peak'),'true_peak',peak)
                frame.to_csv(dst,index=False,header=first);first=False
    p=restored/'derived_data/figure4/selected_stress_traces.csv'
    frame=pd.read_csv(p,float_precision='round_trip')
    if not frame.point.between(1,120).all() or not (frame.point==frame.point.astype(int)).all():
        raise ValueError('Selected curve points must use the archived 1-based index (1 to 120).')
    strain=[float(samples[s]['strain'][int(i)-1]) for s,i in zip(frame.sample_id,frame.point)]
    stress=[float(samples[s]['stress'][int(i)-1]) for s,i in zip(frame.sample_id,frame.point)]
    loc=frame.columns.get_loc('predicted_stress_mpa')
    frame.insert(loc,'strain',strain);frame.insert(loc+1,'measured_stress_mpa',stress)
    frame.to_csv(p,index=False)
    (restored/'LOCAL_RESTORATION.txt').write_text(
        'This local verification copy includes separately obtained third-party observations.\n'
        'The public package licences do not grant rights to those observations.\n'
        'Do not upload this restored directory as the author-owned public deposit.\n'
        'Source archive SHA-256: '+SOURCE_SHA+'\n',encoding='utf-8')
    print(json.dumps({'restored_directory':str(restored),'source_records':len(samples),'source_sha256':SOURCE_SHA}),flush=True)

if __name__=='__main__':main()
