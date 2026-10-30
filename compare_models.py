"""Export predictions and timings for identical sampled video frames. No fake accuracy metrics."""
import argparse
import csv
from pathlib import Path
import time
import cv2
from ultralytics import YOLO


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('video',type=Path)
    parser.add_argument('--models',nargs='+',default=['yolo11n.pt','yolo11s.pt'])
    parser.add_argument('--conf',type=float,default=.3)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--sample-every',type=float,default=1.0)
    parser.add_argument('--output',type=Path,default=Path('evaluation'))
    args=parser.parse_args()
    if not args.video.is_file() or args.sample_every<=0 or not 0<args.conf<=1:
        parser.error('Choose an existing video, positive sample interval and confidence in (0,1].')
    args.output.mkdir(parents=True,exist_ok=True)
    for model_number,weights in enumerate(args.models):
        model=YOLO(weights)
        cap=cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            raise RuntimeError('Cannot decode video.')
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not 1<=fps<=240:
            fps=30
        stride=max(1,round(fps*args.sample_every))
        name=f'{model_number+1}_{Path(weights).stem}'
        folder=args.output/name
        folder.mkdir(exist_ok=True)
        rows=[]
        timings=[]
        index=0
        try:
            while True:
                ok,frame=cap.read()
                if not ok:
                    break
                if index%stride==0:
                    began=time.perf_counter()
                    result=model.predict(frame,conf=args.conf,imgsz=args.imgsz,device='cpu',verbose=False)[0]
                    elapsed=time.perf_counter()-began
                    timings.append(elapsed)
                    cv2.imwrite(str(folder/f'frame_{index:06d}.jpg'),result.plot())
                    if result.boxes is not None:
                        for box in result.boxes.cpu():
                            rows.append([index,index/fps,result.names[int(box.cls.item())],float(box.conf.item()),
                                         *box.xyxy[0].tolist(),elapsed*1000])
                index+=1
        finally:
            cap.release()
        with (folder/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f)
            writer.writerow(['frame','video_seconds','prediction','confidence','x1','y1','x2','y2','inference_ms'])
            writer.writerows(rows)
        with (folder/'timings.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f)
            writer.writerow(['sample_number','inference_ms'])
            writer.writerows((i,t*1000) for i,t in enumerate(timings))
        print(f'{weights}: {len(timings)} sampled frames, {len(rows)} predictions, '
              f'mean inference {sum(timings)/max(1,len(timings))*1000:.0f} ms')
    print('Inspect the annotated frames against real object labels. Confidence and prediction count are not accuracy.')


if __name__=='__main__':
    main()
