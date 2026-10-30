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
    if config.get('automatic_detector', {}).get('enabled'):
        return AutomaticDetector(config, root)
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


def spatial_match(a, b):
    """Overlap or strong containment catches differently sized duplicate crops."""
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    smaller=min(max(0,a[2]-a[0])*max(0,a[3]-a[1]),max(0,b[2]-b[0])*max(0,b[3]-b[1]))
    return overlap(a,b)>=.5 or (smaller>0 and intersection/smaller>=.8)


def automatic_fusion(general, custom, confidence=.3, conflict_confidence=.55):
    """General categories remain visible; disputed categories cannot enter memory.

    Scores from separate models are not treated as calibrated probabilities.
    Custom predictions fill uncovered regions rather than winning by score.
    """
    accepted=[dict(p,category_source='general') for p in general if p['confidence']>=confidence]
    for original in custom:
        if original['confidence']<confidence:continue
        p=dict(original,category_source='custom_demo')
        hits=[q for q in accepted if spatial_match(p['box'],q['box'])]
        same=[q for q in hits if q['name']==p['name']]
        different=[q for q in hits if q['name']!=p['name']]
        if different:
            if p['confidence']>=conflict_confidence:
                for q in different:
                    q['detector_conflict']=True
                    q['conflict_label']=p['name']
            continue
        if same:
            for q in same:q['custom_corroborated']=True
            continue
        accepted.append(p)
    return accepted


class AutomaticDetector:
    def __init__(self,config,root):
        from ultralytics import YOLO
        import warnings
        root=Path(root)
        path=Path(config.get('general_model','yolo11s.pt'))
        self.general=YOLO(str(path if path.is_absolute() else root/path))
        self.names=dict(self.general.names)
        if 'moisturizer' not in self.names.values():
            self.names[max(self.names,default=-1)+1]='moisturizer'
        self.allowed=set(config.get('general_classes',self.general.names.values()))
        spec=config.get('automatic_detector',{})
        self.interval=max(1,int(spec.get('custom_every_n_frames',2)))
        self.conflict_confidence=float(spec.get('conflict_confidence',.55))
        self.calls=0
        self.custom=None
        self.capability_warning=''
        try:
            custom_config=dict(config)
            custom_config['automatic_detector']={'enabled':False}
            custom_config['combined_detector']=dict(config['combined_detector'],enabled=True)
            self.custom=load_detector(custom_config,root)
        except Exception as exc:
            self.capability_warning='Custom recognition unavailable; using general categories only. '+str(exc)
            warnings.warn(self.capability_warning,RuntimeWarning)

    def predict(self,source,**kwargs):
        options=dict(kwargs)
        confidence=float(options.get('conf',.3))
        options['agnostic_nms']=False
        options.pop('classes',None)
        general=[p for p in unpack(self.general.predict(source,**options)[0]) if p['name'] in self.allowed]
        custom=[]
        # General runs every time. Specialists run on this SAME frame every
        # second analysis; no cached boxes are placed onto a newer frame.
        if self.custom is not None and self.calls%self.interval==0:
            custom=unpack(self.custom.predict(source,**options)[0])
        self.calls+=1
        rows=automatic_fusion(general,custom,confidence,self.conflict_confidence)
        ids={name:cid for cid,name in self.names.items()}
        boxes=BoxCollection()
        for row in rows:
            if row['name'] not in ids:continue
            boxes.append(SimpleNamespace(cls=np.array([ids[row['name']]]),conf=np.array([row['confidence']]),
                xyxy=np.array([row['box']]),category_source=row['category_source'],
                detector_conflict=row.get('detector_conflict',False),conflict_label=row.get('conflict_label','')))
        return [SimpleNamespace(names=self.names,boxes=boxes)]
