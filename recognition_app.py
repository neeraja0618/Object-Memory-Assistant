import csv
import json
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk,messagebox,filedialog
import cv2
import numpy as np
from PIL import Image
from personal_app import App as PersonalApp
from base_app import ROOT
from personal import Registry,Embedder
from research import ResearchVision,gallery_report,describe_report
from research_engine import ResearchEngine
from scene_monitor import SceneMonitor


class App(PersonalApp):
    def __init__(self,root):
        self.reference_report=None
        self.analysis_thread=None
        super().__init__(root)
        root.title('Object Memory Assistant — Hybrid Recognition Edition')
        self.policy=tk.StringVar(value=self.config['review_policy'])

    def find(self):
        super().find()
        if self.search_results:
            return
        from demo_search import possible_sightings
        # Tentative evidence is separate from confirmed memories; rejected candidates stay excluded.
        excluded={tid:set(t.get('denied_objects',set())) for tid,t in self.engine.tracks.items()}
        rejected={tid for tid,t in self.engine.tracks.items() if t.get('rejected')}
        active={item['id'] for item in self.registry.active()}
        evidence=[row for row in self.engine.history(recent_only=False) if not row.get('candidate_id') or row['candidate_id'] in active]
        rows=possible_sightings(evidence,self.target.get(),excluded,rejected)
        if not rows:
            detected=any(not t.get('rejected') for t in self.engine.tracks.values())
            self.answer.set('No confirmed sighting for this name.' +
                (' Some objects were detected but not identified. Open Seen objects to inspect them.' if detected else ' No matching evidence in this session.'))
            return
        row=rows[0]
        self.answer.set(f"POSSIBLE SIGHTING — NOT CONFIRMED\n{row['candidate_name'] or row['detector_name']}\n"
                        f"Observed: {row['stamp']}\nThe app needs identity confirmation. Open Seen objects, select the matching row and Confirm identity.\n"
                        f"{row.get('reason') or 'Reference identity has not passed confirmation checks.'}\n"
                        'This is tentative evidence, not a saved confirmed location.')
        path=ROOT/row['evidence'] if row.get('evidence') else None
        frame=cv2.imread(str(path)) if path and path.is_file() else None
        if frame is not None:
            h,w=frame.shape[:2]
            a,b,c,d=[max(0,int(v*z)) for v,z in zip(row['box'],(w,h,w,h))]
            crop=frame[b:min(h,d),a:min(w,c)]
            if crop.size:
                self.show_image(self.picture,crop,(max(1,self.picture.winfo_width()-4),max(1,self.picture.winfo_height()-4)))
        else:
            self.picture.configure(image='',text='Tentative observation exists, but its saved image is unavailable.')
            self.picture.image=None

    def make_engine(self,session,config):
        return ResearchEngine(ROOT/'data',session,config,self.registry)

    def settings(self):
        if self.analysis_thread and self.analysis_thread.is_alive():
            raise ValueError('Wait for reference analysis to finish before starting a source.')
        config=super().settings()
        config['review_policy']=self.policy.get()
        return config

    def build_personal_pipeline(self,config):
        registry=Registry(ROOT/'data')
        active=any(item['references'] for item in registry.active())
        if config['personal_enabled'] and active:
            self.queue.put((self.token,'info','Loading registered-object models and references. First use may download weights.'))
        vision=ResearchVision(registry,config) if config['personal_enabled'] and active else None
        if vision:self.queue.put((self.token,'gallery_report',vision.report))
        return dict(vision=vision,scene=SceneMonitor() if config.get('camera_motion_warning',True) else None)

    def on_gallery_report(self,report):
        self.reference_report=report

    def guidance(self):
        if self.running:
            if self.reference_report:self.report_window(self.reference_report)
            else:messagebox.showinfo('Loading','Reference analysis will be available after personal models finish loading.')
            return
        if not self.registration_allowed():return
        if self.analysis_thread and self.analysis_thread.is_alive():return
        active=[item for item in self.registry.active() if item['references']]
        if not active:
            messagebox.showinfo('Register first','Register objects with cropped reference photos first.');return
        token=self.token
        self.status.set('Analysing reference similarities; first use may download DINOv2. Results guide data collection, not accuracy claims.')
        def work():
            try:
                embedder=Embedder(self.config['embedding_model'],ROOT/'data'/'feature_cache')
                gallery=[]
                for item in active:
                    paths=[self.registry.directory/path for path in item['references'] if (self.registry.directory/path).is_file()]
                    if paths:gallery.append(dict(id=item['id'],name=item['name'],features=np.stack([embedder.reference(p) for p in paths]),colors=np.stack([ResearchVision.reference_color(p) for p in paths])))
                self.queue.put((token,'gallery_report',gallery_report(gallery,self.config)))
                self.queue.put((token,'info','Reference analysis finished. Click Registration guidance to view it.'))
            except Exception as exc:self.queue.put((token,'error',str(exc)))
        self.analysis_thread=threading.Thread(target=work,daemon=True);self.analysis_thread.start()

    def report_window(self,report):
        window=tk.Toplevel(self.root);window.title('Confusion-aware registration guidance');window.geometry('850x580')
        text=tk.Text(window,wrap='word',padx=10,pady=10);text.pack(fill='both',expand=True)
        text.insert('1.0',describe_report(report));text.configure(state='disabled')

    def refresh_table(self):
        super().refresh_table()
        for tid,t in self.engine.tracks.items():
            if self.table.exists(str(tid)) and t.get('review_needed'):
                values=list(self.table.item(str(tid),'values'));values[3]='Review requested'
                self.table.item(str(tid),values=values)

    def review_queue(self):
        window=tk.Toplevel(self.root);window.title('Selective review queue');window.geometry('850x370')
        tree=ttk.Treeview(window,columns=('track','candidate','reason'),show='headings')
        for col,width in [('track',60),('candidate',230),('reason',500)]:tree.heading(col,text=col.title());tree.column(col,width=width)
        tree.pack(fill='both',expand=True)
        for tid,t in self.engine.tracks.items():
            if t.get('review_needed') and self.last_seconds-t['last_detection']<=self.config['missing_after_seconds']:
                tree.insert('','end',iid=str(tid),values=(tid,t.get('suggestion',{}).get('name','Unknown'),t.get('review_reason','')))
        def review():
            selected=tree.selection()
            if selected:self.table.selection_set(selected[0]);self.review_track()
        ttk.Button(window,text='Review selected request',command=review).pack(pady=6)
        ttk.Label(window,text=f'{self.engine.review_requests} requests surfaced this session; cooldown/budget prevent repeated requests. Pause/Stop before confirming.').pack(pady=4)

    def save_pending_history(self,frame=None,seconds=None):
        pending=self.engine.pending_history
        groups={}
        for oid,row in list(pending.items()):
            if frame is not None and row['seconds']==seconds:
                groups.setdefault(row['seconds'],dict(frame=frame,ids=[]))['ids'].append(oid)
            else:
                tid=row['track'];t=self.engine.tracks.get(tid)
                if t and t['last_detection']==row['seconds'] and tid in self.raw_frames:
                    groups.setdefault(row['seconds'],dict(frame=self.raw_frames[tid],ids=[]))['ids'].append(oid)
        folder=ROOT/'data'/'history'/self.engine.session
        if groups:folder.mkdir(parents=True,exist_ok=True)
        for group in groups.values():
            path=folder/f"observation_{group['ids'][0]}.jpg"
            if not cv2.imwrite(str(path),group['frame'],[cv2.IMWRITE_JPEG_QUALITY,82]):raise RuntimeError('Unable to save history evidence.')
            self.engine.attach_history_image(group['ids'],str(path.relative_to(ROOT)))

    def on_frame_observed(self,frame,detections,seconds,stamp):
        self.save_pending_history(frame,seconds)
        import time
        now=time.monotonic()
        if now-getattr(self,'last_history_prune',0)>10:
            self.prune_history_images()
            self.last_history_prune=now

    def persist(self):
        super().persist()
        if hasattr(self.engine,'pending_history'):self.save_pending_history()

    def prune_history_images(self):
        rows=self.engine.db.execute('SELECT id,session,seconds,payload FROM observations ORDER BY id DESC').fetchall()
        keep=set();seen=set()
        for oid,session,seconds,payload in rows:
            path=json.loads(payload).get('evidence','')
            if not path:continue
            eligible=(session!=self.engine.session or seconds>=self.last_seconds-self.config['history_image_retention_seconds'])
            if eligible and len(keep)<self.config['max_history_images']:keep.add(path)
            seen.add(path)
        for path in seen-keep:
            file=ROOT/path
            if file.is_file():file.unlink()

    def history_window(self):
        self.persist()
        window=tk.Toplevel(self.root);window.title('Recent evidence and scoped memory repair');window.geometry('1100x700')
        ttk.Label(window,text='Select only observations you want reviewed. Repair is limited to the recent session window; raw originals and revisions are audited.',
                  wraplength=1050,padding=8).pack(fill='x')
        tree=ttk.Treeview(window,columns=('time','track','state','identity','candidate','score'),show='headings',selectmode='extended',height=10)
        for col,width in [('time',200),('track',55),('state',160),('identity',200),('candidate',200),('score',80)]:
            tree.heading(col,text=col.title());tree.column(col,width=width)
        tree.pack(fill='x',padx=8)
        picture=ttk.Label(window);picture.pack(fill='both',expand=True,pady=5)
        state={'rows':{}}
        def refresh():
            tree.delete(*tree.get_children());state['rows']={}
            for row in self.engine.history():
                oid=str(row['observation_id']);state['rows'][oid]=row
                tree.insert('','end',iid=oid,values=(row['stamp'],row['track'],row['state'],row['name'],row['candidate_name'],f"{row['score']:.3f}"))
        def show(event=None):
            selection=tree.selection()
            if not selection:return
            r=state['rows'][selection[-1]];path=ROOT/r['evidence'] if r['evidence'] else None
            frame=cv2.imread(str(path)) if path and path.exists() else None
            if frame is None:picture.configure(image='',text='History evidence expired/unavailable; repair is disabled for this observation.');picture.image=None;return
            h,w=frame.shape[:2];a,b,c,d=[max(0,int(v*s)) for v,s in zip(r['box'],(w,h,w,h))]
            crop=frame[b:min(h,d),a:min(w,c)]
            if crop.size:self.show_image(picture,crop,(700,250))
        tree.bind('<<TreeviewSelect>>',show)
        controls=ttk.Frame(window,padding=8);controls.pack(fill='x')
        name=tk.StringVar(value=self.registry.active()[0]['name'] if self.registry.active() else '')
        ttk.Label(controls,text='Target identity:').pack(side='left')
        ttk.Combobox(controls,textvariable=name,values=[o['name'] for o in self.registry.active()],state='readonly',width=25).pack(side='left',padx=5)
        def preview(mode):
            selected=tree.selection()
            if not selected or not name.get():messagebox.showinfo('Choose evidence','Select history rows and a registered identity.',parent=window);return
            target=next(o['id'] for o in self.registry.active() if o['name']==name.get())
            try:plan=self.engine.preview_repair([int(s) for s in selected],target,mode)
            except Exception as exc:messagebox.showerror('Repair',str(exc),parent=window);return
            dialog=tk.Toplevel(window);dialog.title('Review the proposed historical changes');dialog.geometry('850x500')
            text=tk.Text(dialog,wrap='word');text.pack(fill='both',expand=True)
            for p in plan:
                text.insert('end',f"Observation {p['observation_id']} / {p['before']['stamp']} / track {p['before']['track']}\n"
                    f"{p['before']['name']} ({p['before']['state']}) → {p['after']['name']} ({p['after']['state']})\n{p['reason']}\n\n")
            text.configure(state='disabled')
            if mode=='manual':ttk.Label(dialog,text='Apply only if YOU have inspected each selected observation and confirmed that identity.',wraplength=820).pack(pady=4)
            def apply():
                try:
                    count=self.engine.apply_repair(plan)
                    for tid in {p['after']['track'] for p in plan}:
                        self.evidence.pop(tid,None);self.best_evidence.pop(tid,None)
                        self.dirty_evidence.discard(tid);self.dirty_best.discard(tid)
                    self.persist();self.refresh_table();refresh();self.target.set(name.get());self.find()
                    self.status.set(f'{count} scoped history revisions applied; original observations preserved in audit.');dialog.destroy()
                except Exception as exc:messagebox.showerror('Repair',str(exc),parent=dialog)
            ttk.Button(dialog,text='Apply these reviewed changes',command=apply).pack(pady=8)
        for label,mode in [('Evidence-based repair','evidence'),('Mark disputed','dispute'),('I confirm selected identities','manual')]:
            ttk.Button(controls,text=label,command=lambda m=mode:preview(m)).pack(side='left',padx=4)
        refresh()

    def export_observations(self):
        self.persist()
        path=filedialog.asksaveasfilename(defaultextension='.json',initialfile='observations_and_repairs.json',filetypes=[('JSON','*.json')])
        if path:
            data=dict(session=self.engine.session,review_requests=self.engine.review_requests,observations=self.engine.history(False),repairs=[dict(observation=oid,reason=reason,before=json.loads(b),after=json.loads(a))
                    for oid,reason,b,a in self.engine.repair_audit()])
            Path(path).write_text(json.dumps(data,indent=2),encoding='utf-8');self.status.set('Observations and repair audit exported.')

    def show_result(self):
        super().show_result()
        i=self.result_picker.current()
        if i<0 or not self.search_results:return
        r=dict(self.search_results[i])
        if not r.get('personal_id'):return
        if r['current_session'] and r['id'] in self.engine.records:r.update(self.engine.records[r['id']])
        item=self.registry.get(r['personal_id']);name=item['name'] if item else r['name']
        status=r['status'] if r['current_session'] and self.running and not self.pause_event.is_set() else 'Historical; current physical location unknown'
        self.answer.set(f"{name} / track #{r['id']}\n{status}\nLast identified: {r['last_seen']}\nLast detected region: {r.get('last_detected',r['last_seen'])}"
                        f"\n{r['location']}\nIdentity basis: {r.get('identity_basis','Reference checks')}\nLatest similarity: {r.get('match_score',0):.3f} (not a probability)")
        if self.best_picture.get() and r.get('best_snapshot'):self.answer.set(self.answer.get()+f"\nClearest evidence: {r['best_seen']}")
        if self.engine.camera_warning:self.answer.set(self.answer.get()+'\nCamera shifted: old image zones may be unreliable.')


if __name__=='__main__':
    root=tk.Tk();App(root);root.mainloop()
