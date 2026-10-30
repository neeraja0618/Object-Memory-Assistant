"""Install a source-specific combination; never change failed individual reports.

Requires the user's actual V2 and V3 result ZIPs, validates checksums, recomputes
the combined development screen from saved predictions, and performs real CPU
inference before changing the current app. Close the app before installation.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from combined_detector import CombinedDetector,fuse_predictions,overlap

PATCH=Path(__file__).resolve().parent
ROOT=PATCH if (PATCH/'config.json').is_file() else PATCH.parent


def read_result(path,version):
    with zipfile.ZipFile(path) as archive:
        report=json.loads(archive.read('training_report.json'))
        if report.get('version')!=version or report.get('error'):
            raise ValueError('Select the completed Demo_V'+str(version)+'_Results.zip.')
        if archive.getinfo('best.pt').file_size>150_000_000:
            raise ValueError('Unexpected model size.')
        weights=archive.read('best.pt')
        if hashlib.sha256(weights).hexdigest()!=report.get('weights_sha256'):
            raise ValueError('Model checksum does not match its report.')
        extra={name:json.loads(archive.read(name)) for name in
               (['selected_predictions.json'] if version==2 else
                ['selected_predictions.json','warm_start_predictions.json','training_predictions_NOT_TEST.json'])}
        extra['dataset_manifest.json']=json.loads(archive.read('dataset_manifest.json'))
    return report,weights,extra


def evaluate(items):
    names=('bottle','moisturizer','book')
    stats={n:dict(tp=0,fp=0,fn=0,ground_truth=0) for n in names}
    for row in items:
        used=set()
        for t in row['truth']:
            stats[t['name']]['ground_truth']+=1
        for p in row['predictions']:
            matches=[(overlap(p['box'],t['box']),i) for i,t in enumerate(row['truth'])
                     if i not in used and t['name']==p['name']]
            value,index=max(matches,default=(0,-1))
            if value>=.5:
                used.add(index);stats[p['name']]['tp']+=1
            else:
                stats[p['name']]['fp']+=1
        for i,t in enumerate(row['truth']):
            if i not in used:
                stats[t['name']]['fn']+=1
    for row in stats.values():
        tp,fp,fn=(row[k] for k in ('tp','fp','fn'))
        row['precision']=tp/(tp+fp) if tp+fp else None
        row['recall']=tp/(tp+fn) if tp+fn else None
    return stats


def screen(v2,v3,extra2,extra3):
    if v3['warm_start_weights_sha256']!=v2['weights_sha256']:
        raise ValueError('V3 was not fine-tuned from this V2 checkpoint. Select the matching ZIPs.')
    validation=[]
    old={r['image']:r for r in extra3['warm_start_predictions.json']}
    for row in extra3['selected_predictions.json']:
        validation.append(dict(row,predictions=fuse_predictions(old[row['image']]['predictions'],row['predictions'])))
    stats=evaluate(validation)
    # V2 validation previously included the demo video; V3 now trains on it.
    # Pair exact same image coordinates, and call this TRAINING/demo fit.
    old_demo={Path(r['image']).name:r for r in extra2['selected_predictions.json'] if r['group']=='development_demo'}
    hashes2={Path(r['image']).name:r['sha256'] for r in extra2['dataset_manifest.json']['images']}
    hashes3={Path(r['image']).name:r['sha256'] for r in extra3['dataset_manifest.json']['images']}
    demo=[]
    for row in extra3['training_predictions_NOT_TEST.json']:
        if row['group']=='development_demo':
            name=Path(row['image']).name
            if name not in old_demo or hashes2.get(name)!=hashes3.get(name):
                raise ValueError('Missing paired saved demo prediction. No results were invented.')
            demo.append(dict(row,predictions=fuse_predictions(old_demo[name]['predictions'],row['predictions'])))
    fit=evaluate(demo)
    reasons=[]
    for name in ('bottle','moisturizer'):
        if (stats[name]['precision'] or 0)<.8 or (stats[name]['recall'] or 0)<.75:
            reasons.append(name+' combined development precision/recall below screen')
    if (stats['bottle']['recall'] or 0)+.05 < (v3['warm_start']['bottle']['recall'] or 0):
        reasons.append('Combined bottle recall below V2 baseline')
    for name in ('bottle','moisturizer','book'):
        if fit[name]['ground_truth']<3 or (fit[name]['precision'] or 0)<.8 or (fit[name]['recall'] or 0)<.8:
            reasons.append(name+' combined training/demo fit below screen')
    result=dict(mode='V2 bottle + V3 moisturizer/book; contradictory overlapping boxes omitted',
                confidence=.3,iou=.5,development=stats,training_demo_NOT_TEST=fit,
                passed=not reasons,reasons=reasons,
                limitations=['Saved-prediction checks only; real camera/video latency and identity matching are not verified.',
                             'V3 trained on dem.mp4 and V2 saw desk photos. Neither is independent testing.',
                             'Broader book detection remains weak. Phone/cup/remote are outside this demo model.',
                             'Two models run per processed frame; CPU processing can be slower.'])
    if reasons:
        raise ValueError('Combined screen failed: '+'; '.join(reasons))
    return result


def restore():
    records=sorted((ROOT/'data'/'combined_backups').glob('*/state.json'))
    if not records:
        raise RuntimeError('No combined-demo backup exists.')
    backup=records[-1].parent;state=json.loads(records[-1].read_text())
    for name,existed in state.items():
        target=ROOT/name
        if existed:
            shutil.copy2(backup/name,target)
        elif target.is_file():
            target.unlink()
    print('Previous configuration/code restored. Object registrations and memories were preserved.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--v2',type=Path);parser.add_argument('--v3',type=Path)
    parser.add_argument('--restore',action='store_true')
    args=parser.parse_args()
    if not (ROOT/'config.json').is_file():
        raise RuntimeError('Extract combined_demo_update INSIDE the existing project directory.')
    if args.restore:
        restore();return
    for name in ('app.py','recognition_app.py','identity_view.py','preview_overlay.py','base_app.py'):
        if not (ROOT/name).is_file():
            raise RuntimeError('This patch needs the existing Full Scene Evidence app. Missing: '+name)
    if not args.v2 or not args.v3:
        import tkinter as tk
        from tkinter import filedialog
        root=tk.Tk();root.withdraw()
        try:
            if not args.v2:
                selected=filedialog.askopenfilename(title='FIRST: select Demo_V2_Results.zip',filetypes=[('ZIP files','*.zip')])
                if not selected:return
                args.v2=Path(selected)
            if not args.v3:
                selected=filedialog.askopenfilename(title='SECOND: select Demo_V3_Results.zip',filetypes=[('ZIP files','*.zip')])
                if not selected:return
                args.v3=Path(selected)
        finally:
            root.destroy()
    r2,w2,e2=read_result(args.v2,2);r3,w3,e3=read_result(args.v3,3)
    report=screen(r2,r3,e2,e3)
    print('Combined saved-prediction screen passed. Checking actual model inference on this laptop...',flush=True)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory);(path/'v2.pt').write_bytes(w2);(path/'v3.pt').write_bytes(w3)
        detector=CombinedDetector(path/'v2.pt',path/'v3.pt')
        import numpy as np,time
        started=time.monotonic()
        detector.predict(np.zeros((640,640,3),dtype=np.uint8),device='cpu',imgsz=640,conf=.3,verbose=False)
        report['cpu_startup_check_seconds']=time.monotonic()-started
    names=['config.json','base_app.py','combined_detector.py']
    backup=ROOT/'data'/'combined_backups'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup.mkdir(parents=True)
    state={name:(ROOT/name).is_file() for name in names}
    for name,exists in state.items():
        if exists:shutil.copy2(ROOT/name,backup/name)
    (backup/'state.json').write_text(json.dumps(state))
    try:
        target=ROOT/'models'/'combined_demo';target.mkdir(parents=True,exist_ok=True)
        (target/'bottle_v2.pt').write_bytes(w2);(target/'objects_v3.pt').write_bytes(w3)
        config=json.loads((ROOT/'config.json').read_text())
        config.update(model='models/combined_demo/objects_v3.pt',remember_classes=['bottle','moisturizer','book'],
                      detection_confidence=.3,personal_enabled=True,reference_search=False,image_size=640)
        config['combined_detector']=dict(enabled=True,bottle_model='models/combined_demo/bottle_v2.pt',
                    other_model='models/combined_demo/objects_v3.pt',
                    sha256=dict(bottle_model=r2['weights_sha256'],other_model=r3['weights_sha256']))
        if config.get('memory_confidence',.65)<.3:config['memory_confidence']=.65
        if PATCH!=ROOT:
            shutil.copy2(PATCH/'base_app.py',ROOT/'base_app.py')
            shutil.copy2(PATCH/'combined_detector.py',ROOT/'combined_detector.py')
        (ROOT/'config.json').write_text(json.dumps(config,indent=2))
        (target/'combined_report.json').write_text(json.dumps(report,indent=2))
    except Exception:
        for name,existed in state.items():
            if existed:shutil.copy2(backup/name,ROOT/name)
            elif (ROOT/name).is_file():(ROOT/name).unlink()
        raise
    print('INSTALLED. Run run_windows.bat as usual. No Colab or further training needed.')
    print('Active categories: bottle, moisturizer, book. Registrations/history preserved.')
    print('Test fresh webcam/video, personal-name search and last-seen evidence now.')
    print('This is a development/demo candidate, not independently validated accuracy.')
    print('Backup:',backup)


if __name__=='__main__':
    main()
