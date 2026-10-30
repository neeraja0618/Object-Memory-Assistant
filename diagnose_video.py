"""Run the real local models on a clip; report evidence without altering saved memories."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import tempfile
import time

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',type=Path)
    parser.add_argument('--max-seconds',type=float,default=30)
    args=parser.parse_args()
    if args.video is None:
        import tkinter as tk
        from tkinter import filedialog
        window=tk.Tk();window.withdraw()
        chosen=filedialog.askopenfilename(title='Select a video for recognition diagnosis',filetypes=[('Video','*.mp4 *.mov *.avi *.mkv'),('All files','*.*')])
        window.destroy()
        if not chosen:return 0
        args.video=Path(chosen)
    folder=ROOT/'data'/'diagnostics'/datetime.now().strftime('%Y%m%d_%H%M%S')
    folder.mkdir(parents=True,exist_ok=True)
    report={'video':str(args.video),'settings':{},'frames':[],'errors':[],
            'note':'Actual model outputs on this clip; not an accuracy score without independent ground truth.'}
    cap=None
    try:
        import cv2
        from ultralytics import YOLO
        from personal import Registry
        from research import ResearchVision
        from research_engine import ResearchEngine
        from scene_monitor import SceneMonitor
        config=json.loads((ROOT/'config.json').read_text())
        report['settings']=config
        registry=Registry(ROOT/'data')
        report['registrations']=[{'name':o['name'],'views':len(o['references']),'readable_views':sum((registry.directory/r).is_file() for r in o['references'])} for o in registry.active()]
        print('Registrations:',report['registrations'],flush=True)
        cap=cv2.VideoCapture(str(args.video))
        if not cap.isOpened():raise RuntimeError('Cannot read selected video.')
        weights=Path(config['model'])
        print('Loading detector and registered appearance models...',flush=True)
        detector=YOLO(str(weights if weights.is_absolute() else ROOT/weights))
        vision=ResearchVision(registry,config) if config['personal_enabled'] and any(o['references'] for o in registry.active()) else None
        if vision:report['reference_analysis']=vision.report
        monitor=SceneMonitor() if config.get('camera_motion_warning',True) else None
        fps=cap.get(cv2.CAP_PROP_FPS) or 30
        stride=max(1,round(fps/config.get('video_sample_fps',2)))
        with tempfile.TemporaryDirectory() as temporary:
            engine=ResearchEngine(temporary,'diagnostic',config,registry)
            try:
                index=0
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    seconds=index/fps
                    index+=1
                    if seconds>args.max_seconds:break
                    if (index-1)%stride:continue
                    began=time.monotonic()
                    result=detector.predict(frame,conf=config['detection_confidence'],imgsz=config['image_size'],device='cpu',agnostic_nms=True,verbose=False)[0]
                    h,w=frame.shape[:2]
                    detections=[]
                    if result.boxes is not None:
                        for box in result.boxes.cpu():
                            name=result.names[int(box.cls.item())]
                            if name not in config['remember_classes']:continue
                            a,b,c,d=box.xyxy[0].tolist()
                            detections.append(dict(name=name,confidence=float(box.conf.item()),box=[a/w,b/h,c/w,d/h]))
                    moved=monitor.update(frame,[d['box'] for d in detections]) if monitor else False
                    if vision:detections=vision.analyse(frame,detections)
                    if moved:engine.warn_camera(f'{seconds:.2f}s')
                    engine.update(detections,seconds,f'{seconds:.2f}s')
                    report['frames'].append(dict(seconds=seconds,processing_seconds=time.monotonic()-began,camera_warning=moved,
                         detections=detections,tracks=[dict(id=t['id'],state=t['state'],raw=t['raw_name'],personal_id=t.get('personal_id',''),
                         suggestion=t.get('suggestion',{}),review_reason=t.get('review_reason','')) for t in engine.tracks.values() if t['last_detection']==seconds]))
                    print(f'{seconds:5.1f}s: '+', '.join(f"{d['name']} {d['confidence']:.0%} -> {d.get('personal_match',{}).get('name') or 'unmatched'}" for d in detections),flush=True)
                report['confirmed_memories']=list(engine.records.values())
                report['observation_count']=len(engine.history(recent_only=False))
            finally:engine.close()
    except Exception as exc:
        report['errors'].append(str(exc))
        print('Diagnostic error:',exc,flush=True)
    finally:
        if cap is not None:cap.release()
        target=folder/'recognition_report.json'
        target.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('Report saved:',target,flush=True)
        print('Your registrations and saved memories were not changed.')
    return 1 if report['errors'] else 0


if __name__=='__main__':raise SystemExit(main())
