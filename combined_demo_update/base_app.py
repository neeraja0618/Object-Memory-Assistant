import csv
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import cv2
from PIL import Image, ImageTk
from engine import Engine

ROOT = Path(__file__).resolve().parent


def video_clock(seconds):
    n = int(seconds)
    return f'Video {n//3600:02}:{n%3600//60:02}:{n%60:02}'


class App:
    def __init__(self, root):
        self.root = root
        root.title('Object Memory Assistant V2')
        root.geometry(f'{min(1320, root.winfo_screenwidth()-60)}x{min(880, root.winfo_screenheight()-90)}')
        root.minsize(760, 520)
        self.config = json.loads((ROOT/'config.json').read_text())
        self.engine = self.make_engine('idle-'+uuid.uuid4().hex, self.config.copy())
        self.queue = queue.Queue()
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.running = False
        self.token = 0
        self.raw_frames = {}
        self.evidence = {}
        self.best_evidence = {}
        self.dirty_evidence = set()
        self.dirty_best = set()
        self.search_results = []
        self.result_choice = tk.StringVar()
        self.best_picture = tk.BooleanVar(value=False)
        self.last_frame = None
        self.last_stamp = ''
        self.last_seconds = 0
        self.last_flush = time.monotonic()
        self.last_table = 0
        self.model = tk.StringVar(value=self.config['model'])
        self.det_conf = tk.StringVar(value=str(self.config['detection_confidence']))
        self.mem_conf = tk.StringVar(value=str(self.config['memory_confidence']))
        self.size = tk.StringVar(value=str(self.config['image_size']))
        self.camera = tk.StringVar(value=str(self.config['camera_index']))
        self.target = tk.StringVar(value='cell phone')
        self.correct_name = tk.StringVar(value='cell phone')
        self.status = tk.StringVar(value='Ready. First model start downloads weights. Settings apply to the next session.')
        self.answer = tk.StringVar(value='Choose an object name, then Find. Uncertain sightings need confirmation.')
        self.build()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(50, self.poll)

    def build(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Treeview', rowheight=26)
        self.root.minsize(760, 520)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        top = ttk.Frame(self.root, padding=(10, 8))
        top.grid(row=0, column=0, sticky='ew')
        ttk.Label(top, text='Object Memory Assistant', font=('Segoe UI', 16, 'bold')).pack(anchor='w')
        ttk.Label(top,text='1. Register photos   2. Start camera / video   3. Review uncertain matches   4. Find last sighting',wraplength=720).pack(anchor='w',pady=(3,0))
        controls = ttk.Frame(top)
        controls.pack(fill='x', pady=(6, 0))
        self.start_buttons = []
        for label, command in [('Start camera', self.start_webcam), ('Open video', self.start_video)]:
            button = ttk.Button(controls, text=label, command=command)
            button.pack(side='left', padx=(0, 6))
            self.start_buttons.append(button)
        self.pause_button=ttk.Button(controls,text='Pause video',command=self.pause)
        self.pause_button.pack(side='left',padx=(0,6))
        ttk.Button(controls, text='Stop', command=self.stop).pack(side='left', padx=(0,6))
        navigation=ttk.Frame(top)
        navigation.pack(fill='x',pady=(5,0))
        ttk.Button(navigation, text='My objects', command=self.manage_objects).pack(side='left', padx=(0,6))
        ttk.Button(navigation,text='History',command=self.timeline).pack(side='left',padx=(0,6))
        ttk.Button(navigation, text='Settings', command=self.advanced).pack(side='left')
        ttk.Label(self.root, textvariable=self.status, wraplength=720, padding=(10, 4)).grid(row=1, column=0, sticky='ew')
        tabs = ttk.Notebook(self.root)
        self.main_tabs=tabs
        tabs.grid(row=2, column=0, sticky='nsew', padx=10, pady=(0,10))
        live = ttk.Frame(tabs)
        tracks = ttk.Frame(tabs, padding=8)
        tabs.add(live, text='Camera & search')
        tabs.add(tracks, text='Seen objects')
        self.tracks_tab=tracks
        live.rowconfigure(0, weight=1)
        live.columnconfigure(0, weight=3)
        live.columnconfigure(1, weight=2)
        left = ttk.Frame(live)
        left.grid(row=0, column=0, sticky='nsew')
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.preview = ttk.Label(left, anchor='center', text='Start the camera or open a video', width=1)
        self.preview.grid(row=0, column=0, sticky='nsew')
        ttk.Label(left, text='Live preview. Recognition runs separately; see results in Seen objects.', wraplength=380).grid(row=1, column=0, sticky='ew', pady=5)
        right = ttk.Frame(live, padding=10)
        right.grid(row=0, column=1, sticky='nsew')
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)
        ttk.Label(right, text='Where did I last see it?', font=('Segoe UI', 12, 'bold')).grid(row=0,column=0,sticky='w')
        search = ttk.Frame(right)
        search.grid(row=1,column=0,sticky='ew',pady=7)
        search.columnconfigure(0,weight=1)
        ttk.Combobox(search,textvariable=self.target,values=self.config['remember_classes'],width=16).grid(row=0,column=0,sticky='ew')
        ttk.Button(search,text='Find',command=self.find).grid(row=0,column=1,padx=(5,0))
        self.result_picker=ttk.Combobox(right,textvariable=self.result_choice,state='readonly',width=20)
        self.result_picker.grid(row=2,column=0,sticky='ew')
        self.result_picker.bind('<<ComboboxSelected>>',lambda event:self.show_result())
        answer=ttk.Label(right,textvariable=self.answer,wraplength=300,justify='left')
        answer.grid(row=3,column=0,sticky='ew',pady=8)
        right.bind('<Configure>',lambda event:answer.configure(wraplength=max(180,event.width-24)))
        self.picture=ttk.Label(right,anchor='center',text='Last confirmed evidence',width=1)
        self.picture.grid(row=4,column=0,sticky='nsew')
        tracks.rowconfigure(1,weight=1)
        tracks.columnconfigure(0,weight=1)
        ttk.Label(tracks,text='Pause the video, select a detected item, then Confirm identity. Possible names are not confirmed memories.',wraplength=700).grid(row=0,column=0,sticky='ew',pady=(0,8))
        cols=('id','raw','saved','state','zone','last')
        self.table=ttk.Treeview(tracks,columns=cols,show='headings',height=5)
        for col,title,width in zip(cols,['ID','Prediction','Confirmed name','State','Location','Last sighting'],[45,150,150,150,150,190]):
            self.table.heading(col,text=title)
            self.table.column(col,width=width,minwidth=40)
        self.table.grid(row=1,column=0,sticky='nsew')
        scroll=ttk.Scrollbar(tracks,orient='vertical',command=self.table.yview)
        scroll.grid(row=1,column=1,sticky='ns')
        horizontal=ttk.Scrollbar(tracks,orient='horizontal',command=self.table.xview)
        horizontal.grid(row=2,column=0,sticky='ew')
        self.table.configure(yscrollcommand=scroll.set,xscrollcommand=horizontal.set)
        review=ttk.Frame(tracks)
        review.grid(row=3,column=0,sticky='ew',pady=(8,0))
        ttk.Button(review,text='Confirm identity',command=self.review_track).pack(side='left',padx=(0,5))
        ttk.Combobox(review,textvariable=self.correct_name,values=self.config['remember_classes'],state='readonly',width=14).pack(side='left')
        ttk.Button(review,text='Correct name',command=self.correct).pack(side='left',padx=5)
        ttk.Button(review,text='Reject',command=self.reject).pack(side='left')
        self.table.bind('<Double-1>',lambda event:self.review_track())

    def advanced(self):
        window=tk.Toplevel(self.root)
        window.title('Settings')
        window.geometry('650x540')
        window.transient(self.root)
        canvas=tk.Canvas(window,highlightthickness=0)
        scrollbar=ttk.Scrollbar(window,orient='vertical',command=canvas.yview)
        scrollbar.pack(side='right',fill='y')
        canvas.pack(side='left',fill='both',expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        content=ttk.Frame(canvas,padding=12)
        content_id=canvas.create_window((0,0),window=content,anchor='nw')
        content.bind('<Configure>',lambda event:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda event:canvas.itemconfigure(content_id,width=event.width))
        ttk.Label(content,text='Camera & locations',font=('Segoe UI',12,'bold')).pack(anchor='w')
        row=ttk.Frame(content);row.pack(fill='x',pady=8)
        ttk.Label(row,text='Camera number:').pack(side='left')
        ttk.Entry(row,textvariable=self.camera,width=6).pack(side='left',padx=8)
        ttk.Label(content,text='Registered-object matching is enabled. Changes apply on the next start.',wraplength=560).pack(anchor='w',pady=4)
        locations=ttk.LabelFrame(content,text='Set locations (stop after a clear frame first)',padding=8)
        locations.pack(fill='x',pady=8)
        ttk.Button(locations,text='Add location',command=lambda:self.mark('zone')).pack(side='left',padx=4)
        ttk.Button(locations,text='Reset locations',command=self.clear_areas).pack(side='left',padx=4)
        ttk.Label(content,text='Example: Left desk / Right desk. Search reports the last confirmed location, not a guaranteed current location.',wraplength=560).pack(anchor='w',pady=4)
        extra=ttk.LabelFrame(content,text='Advanced — optional',padding=8)
        def toggle():
            if extra.winfo_manager():
                extra.pack_forget();button.configure(text='Show advanced settings')
            else:
                extra.pack(fill='x',pady=8,before=done);button.configure(text='Hide advanced settings')
        button=ttk.Button(content,text='Show advanced settings',command=toggle);button.pack(anchor='w',pady=8)
        for i,(label,var) in enumerate([('Detector',self.model),('Detection threshold',self.det_conf),('Save threshold',self.mem_conf),('Input size',self.size)]):
            ttk.Label(extra,text=label).grid(row=i,column=0,sticky='w',pady=3)
            if i==0:ttk.Combobox(extra,textvariable=var,values=['yolo11n.pt','yolo11s.pt','yolo11m.pt'],width=25).grid(row=i,column=1,sticky='w')
            else:ttk.Entry(extra,textvariable=var,width=27).grid(row=i,column=1,sticky='w')
        ttk.Checkbutton(extra,text='Experimental unfamiliar-object search (slow)',variable=self.guided_on).grid(row=4,column=0,columnspan=2,sticky='w',pady=5)
        ttk.Label(extra,text='Approval policy').grid(row=5,column=0,sticky='w')
        ttk.Combobox(extra,textvariable=self.policy,values=['selective','always'],state='readonly',width=25).grid(row=5,column=1,sticky='w')
        tools=ttk.Frame(extra);tools.grid(row=6,column=0,columnspan=2,sticky='ew',pady=8)
        for i,(label,command) in enumerate([('Repair past observations',self.history_window),('Reference guidance',self.guidance),('Limit monitored area',lambda:self.mark('roi')),('Export evaluation data',self.export_observations),('Load trained detector',self.weights)]):
            ttk.Button(tools,text=label,command=command).grid(row=i//2,column=i%2,sticky='ew',padx=3,pady=3)
        done=ttk.Button(content,text='Done',command=window.destroy);done.pack(anchor='e',pady=8)

    def make_engine(self, session, config):
        return Engine(ROOT/'data', session, config)

    def build_personal_pipeline(self, config):
        return None

    def process_frame(self, pipeline, frame, detections, seconds):
        return detections, False

    def on_frame_observed(self,frame,detections,seconds,stamp):
        pass

    def on_gallery_report(self,report):
        pass

    def weights(self):
        path = filedialog.askopenfilename(filetypes=[('YOLO weights', '*.pt')])
        if path:
            self.model.set(path)

    def settings(self):
        config = self.config.copy()
        config.update(model=self.model.get().strip(), detection_confidence=float(self.det_conf.get()),
                      memory_confidence=float(self.mem_conf.get()), image_size=int(self.size.get()),
                      camera_index=int(self.camera.get()))
        if not 0 < config['detection_confidence'] <= config['memory_confidence'] <= 1:
            raise ValueError('Use 0 < detection confidence <= memory confidence <= 1.')
        if config['image_size'] < 160 or config['image_size'] % 32:
            raise ValueError('Input size must be a multiple of 32, at least 160 (e.g. 640 or 960).')
        if config['camera_index'] < 0 or not config['model']:
            raise ValueError('Choose model weights and a nonnegative camera index.')
        return config

    def start_webcam(self):
        try:
            self.start(int(self.camera.get()), False)
        except ValueError as exc:
            messagebox.showerror('Settings', str(exc))

    def start_video(self):
        path = filedialog.askopenfilename(filetypes=[('Videos', '*.mp4 *.avi *.mov *.mkv *.webm'), ('All files', '*.*')])
        if path:
            self.start(path, True)

    def start(self, source, is_video):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo('Worker busy', 'Wait for the current source to finish stopping.')
            return
        try:
            config = self.settings()
        except ValueError as exc:
            messagebox.showerror('Settings', str(exc))
            return
        self.persist()
        self.engine.close()
        self.config = config
        (ROOT/'config.json').write_text(json.dumps(config, indent=2))
        session = datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        self.engine = self.make_engine(session, config.copy())
        self.raw_frames, self.evidence, self.best_evidence = {}, {}, {}
        self.dirty_evidence, self.dirty_best = set(), set()
        self.last_frame = None
        self.last_stamp = ''
        self.last_seconds = 0
        self.token += 1
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.is_video = is_video
        self.pause_button.configure(text='Pause video')
        self.running = True
        for button in self.start_buttons:
            button.configure(state='disabled')
        self.refresh_table()
        self.live_preview = None
        self.capability_warning = ''
        self.status.set('Starting source. Camera preview runs separately while recognition models load.')
        self.worker_thread = threading.Thread(target=self.worker,
            args=(source, is_video, self.token, self.stop_event, self.pause_event, config.copy()), daemon=True)
        self.worker_thread.start()

    def worker(self, source, is_video, token, stop, pause, config):
        if not is_video:
            return self.webcam_worker(source, token, stop, config)
        cap = None
        try:
            from combined_detector import load_detector
            model = load_detector(config, ROOT)
            if stop.is_set():
                return
            pipeline = self.build_personal_pipeline(config)
            unsupported = set(config['remember_classes'])-set(model.names.values())
            if unsupported:
                raise ValueError('Configured categories missing from model: '+', '.join(sorted(unsupported))+
                                 '. Edit remember_classes in config.json for your custom model.')
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if not is_video and os.name == 'nt' else cv2.VideoCapture(source)
            if not cap.isOpened() and not is_video:
                cap.release()
                cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError('Cannot open source. Check camera index/permissions or use a standard MP4.')
            if not is_video:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not 1 <= fps <= 240:
                fps = 30
            stride = max(1, round(fps/config.get('video_sample_fps',8))) if is_video else 1
            index = 0
            start = time.monotonic()
            base = None
            while not stop.is_set():
                if pause.is_set():
                    paused_at = time.monotonic()
                    while pause.is_set() and not stop.wait(.05):
                        pass
                    if base is not None:
                        base += time.monotonic()-paused_at
                    if stop.is_set():
                        break
                ok, frame = cap.read()
                if not ok:
                    if is_video:
                        self.queue.put((token, 'info', 'Video finished. Search and timeline remain available.'))
                        break
                    raise RuntimeError('Camera stopped returning frames.')
                index += 1
                if is_video and (index-1) % stride:
                    continue
                seconds = (index-1)/fps if is_video else time.monotonic()-start
                stamp = video_clock(seconds) if is_video else datetime.now().astimezone().isoformat(timespec='seconds')
                if is_video:
                    if base is None:
                        base = time.monotonic()
                    if stop.wait(max(0, base+seconds-time.monotonic())):
                        break
                began = time.monotonic()
                result = model.predict(frame, conf=config['detection_confidence'], imgsz=config['image_size'],
                                       device='cpu', agnostic_nms=True, verbose=False)[0]
                h, w = frame.shape[:2]
                detections = []
                if result.boxes is not None:
                    for b in result.boxes.cpu():
                        name = result.names[int(b.cls.item())]
                        if name in config['remember_classes']:
                            x1,y1,x2,y2 = b.xyxy[0].tolist()
                            detections.append(dict(name=name, confidence=float(b.conf.item()), box=[x1/w,y1/h,x2/w,y2/h]))
                detections, camera_moved = self.process_frame(pipeline,frame,detections,seconds)
                if camera_moved:
                    self.queue.put((token,'camera_moved',stamp))
                self.queue.put((token, 'frame', (frame, detections, seconds, stamp, time.monotonic()-began)))
                while self.queue.qsize() > 3 and not stop.wait(0.05):
                    pass
                if not is_video:
                    stop.wait(max(0, 0.12-(time.monotonic()-began)))
        except Exception as exc:
            self.queue.put((token, 'error', str(exc)))
        finally:
            if cap is not None:
                cap.release()
            self.queue.put((token, 'done', None))

    def webcam_worker(self, source, token, stop, config):
        from live_capture import LatestFrame
        slot = LatestFrame()
        camera_done = threading.Event()
        camera_error = []
        began_session = time.monotonic()
        def capture():
            cap = None
            try:
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(source)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    raise RuntimeError('Cannot open camera. Check camera number and Windows camera permissions.')
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                while not stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError('Camera stopped returning frames.')
                    seconds = time.monotonic() - began_session
                    stamp = datetime.now().astimezone().isoformat(timespec='seconds')
                    slot.publish((frame, seconds, stamp))
                    self.live_preview = (token, frame)
                    stop.wait(.015)
            except Exception as exc:
                camera_error.append(str(exc))
            finally:
                if cap is not None:
                    cap.release()
                camera_done.set()
        camera = threading.Thread(target=capture, daemon=True)
        camera.start()
        try:
            from combined_detector import load_detector
            model = load_detector(config, ROOT)
            if stop.is_set():
                return
            unsupported = set(config['remember_classes']) - set(model.names.values())
            if unsupported:
                raise ValueError('Configured categories missing from model: '+', '.join(sorted(unsupported)))
            try:
                pipeline = self.build_personal_pipeline(config)
            except Exception as exc:
                pipeline = None
                self.queue.put((token, 'capability_error', 'Personal matching unavailable; basic detection only. '+str(exc)))
            sequence = 0
            while not stop.is_set():
                if camera_done.is_set():
                    if camera_error:
                        raise RuntimeError(camera_error[0])
                    break
                sequence_new, sample = slot.after(sequence)
                if sample is None:
                    stop.wait(.025)
                    continue
                sequence = sequence_new
                frame, seconds, stamp = sample
                began = time.monotonic()
                result = model.predict(frame, conf=config['detection_confidence'], imgsz=config['image_size'],
                                       device='cpu', agnostic_nms=True, verbose=False)[0]
                h,w = frame.shape[:2]
                detections=[]
                if result.boxes is not None:
                    for box in result.boxes.cpu():
                        name=result.names[int(box.cls.item())]
                        if name in config['remember_classes']:
                            a,b,c,d=box.xyxy[0].tolist()
                            detections.append(dict(name=name,confidence=float(box.conf.item()),box=[a/w,b/h,c/w,d/h]))
                detections,moved=self.process_frame(pipeline,frame,detections,seconds)
                if stop.is_set():
                    break
                if moved:
                    self.queue.put((token,'camera_moved',stamp))
                self.queue.put((token,'frame',(frame,detections,seconds,stamp,time.monotonic()-began)))
                while self.queue.qsize()>3 and not stop.wait(.05):
                    pass
                stop.wait(.05)
        except Exception as exc:
            self.queue.put((token,'error',str(exc)))
        finally:
            stop.set()
            camera.join(timeout=1)
            if camera.is_alive():
                # Keep the source busy until its owner has released it.
                self.queue.put((token,'info','Waiting for the camera driver to release the source.'))
                camera.join()
            self.queue.put((token,'done',None))

    def annotate(self, frame, seconds):
        out = frame.copy()
        h,w = out.shape[:2]
        for tid,t in self.engine.tracks.items():
            if abs(seconds-t['last_detection']) > 0.001:
                continue
            record = self.engine.records.get(tid)
            state = t['state']
            color = (60,210,70) if state in ('Confirmed','Reference-confirmed') else ((230,160,40) if state == 'User-labelled' else (0,160,255))
            x1,y1,x2,y2 = [int(v*s) for v,s in zip(t['box'], (w,h,w,h))]
            label = record['name'] if record and state in ('Confirmed','User-labelled','Reference-confirmed') else t['raw_name']
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
            cv2.putText(out, f"#{tid} {label} {t['raw_confidence']:.2f} [{state}]", (x1,max(18,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        for z in self.config.get('zones', []):
            a,b,c,d = [int(v*s) for v,s in zip(z['box'], (w,h,w,h))]
            cv2.rectangle(out, (a,b), (c,d), (200,200,80), 1)
            cv2.putText(out, z['name'], (a+3,b+17), cv2.FONT_HERSHEY_SIMPLEX, .5, (200,200,80), 1)
        roi = self.config.get('roi')
        if roi:
            a,b,c,d = [int(v*s) for v,s in zip(roi, (w,h,w,h))]
            cv2.rectangle(out, (a,b), (c,d), (255,210,80), 2)
        return out

    def show_image(self, widget, frame, size):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail(size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        widget.configure(image=photo, text='')
        widget.image = photo

    def persist(self):
        folder = ROOT/'data'/'snapshots'/self.engine.session
        if self.evidence:
            folder.mkdir(parents=True, exist_ok=True)
        for tid in list(self.dirty_evidence):
            frame = self.evidence[tid]
            if tid not in self.engine.records:
                continue
            path = folder/f'track_{tid}.jpg'
            temp = path.with_name(path.stem+'.tmp.jpg')
            if not cv2.imwrite(str(temp), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                raise RuntimeError('Unable to save evidence image.')
            temp.replace(path)
            self.engine.records[tid]['snapshot'] = str(path.relative_to(ROOT))
            self.dirty_evidence.discard(tid)
        for tid in list(self.dirty_best):
            path = folder/f'track_{tid}_best.jpg'
            temp = path.with_name(path.stem+'.tmp.jpg')
            if not cv2.imwrite(str(temp), self.best_evidence[tid], [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError('Unable to save best evidence image.')
            temp.replace(path)
            self.engine.records[tid]['best_snapshot'] = str(path.relative_to(ROOT))
            self.dirty_best.discard(tid)
        self.engine.flush()
        # Keep long videos from retaining an image array for every retired track.
        for tid, track in self.engine.tracks.items():
            if self.last_seconds-track['last_detection'] > self.config['missing_after_seconds']:
                self.raw_frames.pop(tid, None)
                self.evidence.pop(tid, None)
                self.best_evidence.pop(tid, None)
        self.last_flush = time.monotonic()

    def refresh_table(self):
        selected = self.table.selection()
        self.table.delete(*self.table.get_children())
        for tid,t in self.engine.tracks.items():
            r = self.engine.records.get(tid)
            if r:
                state = r['status'] if self.running else 'Historical / stopped'
            else:
                state = t['state'] if self.last_seconds-t['last_detection'] <= self.config['missing_after_seconds'] else 'Expired tentative'
            raw = f"{t['raw_name']} {t.get('raw_confidence',0):.0%}"
            self.table.insert('', 'end', iid=str(tid), values=(tid, raw, r['name'] if r else 'Not saved', state,
                    r['location'] if r else '', r['last_seen'] if r else ''))
        for item in selected:
            if self.table.exists(item):
                self.table.selection_set(item)

    def poll(self):
        try:
            for _ in range(8):
                try:
                    token,kind,payload = self.queue.get_nowait()
                except queue.Empty:
                    break
                if token != self.token:
                    continue
                if kind == 'gallery_report':
                    self.on_gallery_report(payload)
                elif kind == 'camera_moved':
                    if hasattr(self.engine,'warn_camera'):
                        self.engine.warn_camera(payload)
                    self.config['roi'],self.config['zones']=None,[]
                    (ROOT/'config.json').write_text(json.dumps(self.config,indent=2))
                    self.status.set('Camera view shifted. Memory updates suspended; Stop, clear/redefine areas and restart.')
                elif kind == 'frame':
                    frame,detections,seconds,stamp,elapsed = payload
                    self.last_frame,self.last_seconds,self.last_stamp = frame,seconds,stamp
                    records = self.engine.update(detections,seconds,stamp)
                    annotated = self.annotate(frame,seconds)
                    for tid,t in self.engine.tracks.items():
                        if abs(seconds-t['last_detection']) < 0.001:
                            self.raw_frames[tid] = frame
                    for record in records:
                        tid = record['id']
                        self.evidence[tid] = annotated
                        self.dirty_evidence.add(tid)
                        h,w = frame.shape[:2]
                        a,b,c,d = [max(0, int(v*s)) for v,s in zip(record['box'], (w,h,w,h))]
                        crop = frame[b:min(h,d),a:min(w,c)]
                        if crop.size:
                            # Sharpness is an evidence-selection heuristic, not detection accuracy.
                            quality = float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
                            if quality > record.get('best_quality', -1):
                                record['best_quality'], record['best_seen'] = quality, stamp
                                record['best_box'] = list(record['box'])
                                self.best_evidence[tid] = annotated
                                self.dirty_best.add(tid)
                    self.on_frame_observed(frame,detections,seconds,stamp)
                    if self.is_video or not self.running:
                        self.show_image(self.preview,annotated,(max(1,self.preview.winfo_width()-4),max(1,self.preview.winfo_height()-4)))
                    prefix = 'Paused | ' if self.pause_event.is_set() else ''
                    self.status.set(f'{prefix}{stamp} | {len(records)} confirmed/user-labelled observations this frame | '
                                    f'{len(self.engine.records)} saved tracks | Recognition {elapsed*1000:.0f} ms per analysed frame')
                    if getattr(self,'capability_warning',''):
                        self.status.set(self.status.get()+' | '+self.capability_warning)
                    if getattr(self.engine,'camera_warning',False):
                        self.status.set('CAMERA VIEW SHIFTED: memory updates suspended. Stop, clear/redefine areas and restart.')
                    if time.monotonic()-self.last_flush > 2:
                        self.persist()
                    if time.monotonic()-self.last_table > .5:
                        self.refresh_table()
                        self.last_table = time.monotonic()
                elif kind == 'capability_error':
                    self.capability_warning=payload
                    self.status.set(payload)
                elif kind == 'info':
                    self.status.set(payload)
                elif kind == 'error':
                    self.status.set('Error: '+payload)
                    messagebox.showerror('Error',payload)
                elif kind == 'done':
                    self.running = False
                    self.persist()
                    self.refresh_table()
                    for button in self.start_buttons:
                        button.configure(state='normal')
        except Exception as exc:
            self.stop_event.set()
            self.running = False
            self.status.set('Error: '+str(exc))
        live = getattr(self, 'live_preview', None)
        if live and live[0] == self.token and self.running:
            self.show_image(self.preview,live[1],(max(1,self.preview.winfo_width()-4),max(1,self.preview.winfo_height()-4)))
        self.root.after(50,self.poll)

    def find(self):
        self.persist()
        rows = self.engine.search(self.target.get())
        if not rows:
            self.search_results = []
            self.result_picker.configure(values=[])
            self.result_choice.set('')
            self.answer.set('No confirmed memory matches. An orange box is only a tentative prediction. '
                            'You can correct an uncertain track manually, or improve detection.')
            self.picture.configure(image='',text='No confirmed evidence')
            self.picture.image = None
            return
        self.search_results = rows
        choices = [f"{r['nickname'] or r['name']} | #{r['id']} | {r['last_seen']} | {r['session']}" for r in rows]
        self.result_picker.configure(values=choices)
        self.result_picker.current(0)
        self.show_result()

    def show_result(self):
        self.persist()
        index = self.result_picker.current()
        if not self.search_results or index < 0:
            return
        r = self.search_results[index].copy()
        if r['current_session'] and r['id'] in self.engine.records:
            r.update(self.engine.records[r['id']])
        active = r['current_session'] and self.running and not self.pause_event.is_set()
        state = r['status'] if active else 'Historical / monitoring paused or stopped; current location unknown'
        confidence = ('User-assigned label; model accuracy not verified' if r['user_labelled'] else
                      f"Mean supporting confidence: {r['confidence']:.0%}; label agreement: {r['agreement']:.0%}")
        self.answer.set(f"{r['nickname'] or r['name']} — {r['name']} / track #{r['id']}\n{state}\n"
                        f"Last confirmed observation: {r['last_seen']}\n{r['location']}\n{confidence}\n"
                        f"{len(self.search_results)} matching stored tracks; use the selector to view another.")
        use_best = self.best_picture.get() and r.get('best_snapshot')
        source = r['best_snapshot'] if use_best else r['snapshot']
        if use_best:
            self.answer.set(self.answer.get()+f"\nClearest evidence captured: {r['best_seen']} (may be earlier than last seen).")
        path = ROOT/source
        frame = cv2.imread(str(path)) if source and path.exists() else None
        if frame is not None:
            self.show_image(self.picture,frame,(max(1,self.picture.winfo_width()-4),max(1,self.picture.winfo_height()-4)))
        else:
            self.picture.configure(image='',text='Evidence image unavailable')
            self.picture.image = None

    def correct(self):
        selected = self.table.selection()
        if not selected:
            messagebox.showinfo('Choose a track','Select a row in the current-session table first.')
            return
        tid = int(selected[0])
        if self.last_seconds-self.engine.tracks[tid]['last_detection'] > self.config['missing_after_seconds']:
            messagebox.showinfo('Track expired','Correct an active or recently observed track, not a retired track.')
            return
        nickname = simpledialog.askstring('Optional nickname','Name such as “my black phone” (leave blank if unnecessary):',parent=self.root)
        if nickname is None:
            return
        record = self.engine.correct(tid,self.correct_name.get(),nickname,self.last_stamp)
        if tid in self.raw_frames:
            self.evidence[tid] = self.annotate(self.raw_frames[tid],self.engine.tracks[tid]['last_detection'])
            self.dirty_evidence.add(tid)
            self.best_evidence[tid] = self.evidence[tid]
            self.engine.records[tid]['best_seen'] = self.engine.tracks[tid]['last_stamp']
            self.engine.records[tid]['best_box'] = list(self.engine.tracks[tid]['box'])
            self.engine.records[tid]['best_quality'] = -1
            self.dirty_best.add(tid)
        self.persist()
        self.refresh_table()
        self.target.set(nickname or record['name'])
        self.find()

    def reject(self):
        selected = self.table.selection()
        if not selected:
            messagebox.showinfo('Choose a track','Select the incorrect detection in the table first.')
            return
        self.engine.reject(int(selected[0]), self.last_stamp)
        self.refresh_table()
        self.find()
        self.status.set('Track rejected and excluded from search. Corrections can restore it; the detector was not retrained.')

    def pause(self):
        if not self.running or not getattr(self, 'is_video', False):
            messagebox.showinfo('Video only','Pause is available during recorded-video playback. Use Stop for the webcam.')
            return
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_button.configure(text='Pause video')
            self.status.set('Video resumed.')
        else:
            self.pause_event.set()
            self.pause_button.configure(text='Resume video')
            self.persist()
            self.status.set('Pausing video after current inference. Search or correct an object while paused.')

    def mark(self, mode):
        if self.running or (self.worker_thread and self.worker_thread.is_alive()):
            messagebox.showinfo('Stop first','Stop monitoring, wait for the worker, then mark areas on the last frame.')
            return
        if self.last_frame is None:
            messagebox.showinfo('Need a frame','Start the camera/video, then Stop after a clear frame appears.')
            return
        dialog = tk.Toplevel(self.root)
        dialog.title('Drag a rectangle: '+('desk area' if mode=='roi' else 'named zone'))
        img = Image.fromarray(cv2.cvtColor(self.last_frame,cv2.COLOR_BGR2RGB))
        img.thumbnail((900,650))
        w,h = img.size
        photo = ImageTk.PhotoImage(img)
        canvas = tk.Canvas(dialog,width=w,height=h)
        canvas.pack()
        canvas.create_image(0,0,anchor='nw',image=photo)
        canvas.image = photo
        point = [None,None]
        def down(event):
            point[0] = (max(0,min(w,event.x)),max(0,min(h,event.y)))
            if point[1]:
                canvas.delete(point[1])
            point[1] = canvas.create_rectangle(event.x,event.y,event.x,event.y,outline='yellow',width=3)
        def move(event):
            if point[0]:
                canvas.coords(point[1],*point[0],max(0,min(w,event.x)),max(0,min(h,event.y)))
        def up(event):
            if point[0] is None:
                return
            x,y = point[0]
            a,b = max(0,min(w,event.x)),max(0,min(h,event.y))
            if abs(a-x)<15 or abs(b-y)<15:
                return
            box = [min(x,a)/w,min(y,b)/h,max(x,a)/w,max(y,b)/h]
            if mode=='roi':
                self.config['roi'] = box
            else:
                name = simpledialog.askstring('Zone name','For example: left shelf or laptop area',parent=dialog)
                if not name or not name.strip():
                    return
                self.config['zones'].append(dict(name=name.strip(),box=box))
            (ROOT/'config.json').write_text(json.dumps(self.config,indent=2))
            self.status.set('Area saved. Applies on next source start; clear it if the camera moves.')
            dialog.destroy()
        canvas.bind('<ButtonPress-1>',down)
        canvas.bind('<B1-Motion>',move)
        canvas.bind('<ButtonRelease-1>',up)
        ttk.Label(dialog,text='Drag on the image. Overlapping named zones use the first match.').pack()
        dialog.grab_set()

    def clear_areas(self):
        if self.running:
            messagebox.showinfo('Stop first','Stop before changing areas.')
            return
        self.config['roi'],self.config['zones'] = None,[]
        (ROOT/'config.json').write_text(json.dumps(self.config,indent=2))
        self.status.set('Areas cleared. Applies on next source start.')

    def timeline(self):
        self.persist()
        window = tk.Toplevel(self.root)
        window.title('History — newest events first')
        window.geometry('1150x480')
        cols = ('session','time','id','category','event','location')
        tree = ttk.Treeview(window,columns=cols,show='headings')
        for col,width in zip(cols,[200,180,50,100,380,210]):
            tree.heading(col,text=col.title())
            tree.column(col,width=width)
        scrollbar = ttk.Scrollbar(window,orient='vertical',command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        actions=ttk.Frame(window,padding=6);actions.pack(side='bottom',fill='x')
        ttk.Button(actions,text='Export history',command=self.export).pack(side='left')
        horizontal=ttk.Scrollbar(window,orient='horizontal',command=tree.xview)
        horizontal.pack(side='bottom',fill='x')
        tree.configure(xscrollcommand=horizontal.set)
        scrollbar.pack(side='right',fill='y')
        tree.pack(fill='both',expand=True)
        for row in self.engine.timeline():
            tree.insert('','end',values=row)

    def export(self):
        self.persist()
        path = filedialog.asksaveasfilename(defaultextension='.csv',initialfile='events_v2.csv',filetypes=[('CSV','*.csv')])
        if path:
            with open(path,'w',newline='',encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['session','time','track','category','event','image_location'])
                writer.writerows(reversed(self.engine.timeline()))
            self.status.set('Event log exported.')

    def stop(self):
        self.stop_event.set()
        self.running = False
        self.persist()
        self.refresh_table()
        self.status.set('Stopping source. A running recognition call or model download must finish before another source can start.')

    def close(self):
        self.stop_event.set()
        try:
            self.persist()
            self.engine.close()
        finally:
            self.root.destroy()


if __name__=='__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
