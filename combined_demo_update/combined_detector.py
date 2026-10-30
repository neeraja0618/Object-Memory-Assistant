"""V2 bottles + V3 moisturizer/book. One result for the existing app worker.

Always infer both on the SAME frame. Never recycle a prediction from an older
frame. Overlapping contradictory identities are omitted, not guessed.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np

NAMES={0:'bottle',1:'moisturizer',2:'book'}


def overlap(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(1e-9,area-inter)


def fuse_predictions(bottle,other,confidence=.3):
    candidates=[dict(p) for p in bottle if p['name']=='bottle' and p['confidence']>=confidence]
    candidates += [dict(p) for p in other if p['name'] in ('moisturizer','book') and p['confidence']>=confidence]
    conflicts=set()
    for i,p in enumerate(candidates):
        for j in range(i):
            q=candidates[j]
            if p['name']!=q['name'] and overlap(p['box'],q['box'])>=.75:
                conflicts.update((i,j))
    # Do not allow duplicate boxes to become duplicate remembered objects.
    accepted=[]
    for i,p in sorted(enumerate(candidates),key=lambda pair:pair[1]['confidence'],reverse=True):
        if i in conflicts:
            continue
        if any(q['name']==p['name'] and overlap(p['box'],q['box'])>=.65 for q in accepted):
            continue
        accepted.append(p)
    return accepted


class BoxCollection(list):
    def cpu(self):
        return self


def unpack(result):
    rows=[]
    if result.boxes is not None:
        for box in result.boxes.cpu():
            rows.append(dict(name=result.names[int(box.cls.item())],
                             confidence=float(box.conf.item()),box=box.xyxy[0].tolist()))
    return rows


class CombinedDetector:
    names=NAMES

    def __init__(self,bottle_path,other_path):
        from ultralytics import YOLO
        self.bottle=YOLO(str(bottle_path))
        self.other=YOLO(str(other_path))
        for model in (self.bottle,self.other):
            if model.names!=NAMES:
                raise ValueError('Combined demo expects bottle, moisturizer, book in both checkpoints.')

    def predict(self,source,**kwargs):
        options=dict(kwargs)
        confidence=float(options.get('conf',.3))
        # Match the evaluation's class-aware NMS. No category is allowed to
        # suppress another category before the source-specific filter.
        options['agnostic_nms']=False
        options.pop('classes',None)
        first=self.bottle.predict(source,**options)[0]
        second=self.other.predict(source,**options)[0]
        rows=fuse_predictions(unpack(first),unpack(second),confidence)
        boxes=BoxCollection()
        ids={name:cid for cid,name in NAMES.items()}
        for row in rows:
            boxes.append(SimpleNamespace(cls=np.array([ids[row['name']]]),
                         conf=np.array([row['confidence']]),xyxy=np.array([row['box']])))
        return [SimpleNamespace(names=NAMES,boxes=boxes)]


def load_detector(config,root):
    root=Path(root)
    spec=config.get('combined_detector',{})
    if not spec.get('enabled'):
        from ultralytics import YOLO
        path=Path(config['model'])
        return YOLO(str(path if path.is_absolute() else root/path))
    paths=[]
    for key in ('bottle_model','other_model'):
        path=Path(spec[key]);path=path if path.is_absolute() else root/path
        if not path.is_file():
            raise FileNotFoundError('Combined model missing: '+str(path))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256'][key]:
            raise ValueError('Combined model checksum mismatch: '+path.name)
        paths.append(path)
    return CombinedDetector(*paths)
