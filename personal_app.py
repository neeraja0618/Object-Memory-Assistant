"""V3 GUI: personal object registry, explicit identity review and reliable evidence."""
from pathlib import Path
import json
import threading
import tkinter as tk
from tkinter import ttk,messagebox,filedialog,simpledialog
import cv2
from PIL import Image,ImageTk,ImageOps
from base_app import App as BaseApp,ROOT
from personal import Registry,PersonalVision
from personal_engine import PersonalEngine
from scene_monitor import SceneMonitor


class App(BaseApp):
    def __init__(self,root):
        self.registry=Registry(ROOT/'data')
        super().__init__(root)
        root.title('Object Memory Assistant V3 — Personal Objects')
        self.personal_on=tk.BooleanVar(value=True)
        self.guided_on=tk.BooleanVar(value=self.config['reference_search'])
        self.table.heading('raw',text='Detector prediction')
        self.table.heading('saved',text='Object name / candidate')
        self.add_personal_choices()
        self.registration_capture=None

    def make_engine(self,session,config):
        return PersonalEngine(ROOT/'data',session,config,self.registry)

    def settings(self):
        if getattr(self,'registration_capture',None) is not None:
            raise ValueError('Close the registration-camera window before starting monitoring.')
        config=super().settings()
        config.update(personal_enabled=self.personal_on.get(),reference_search=self.guided_on.get())
        return config

    def build_personal_pipeline(self,config):
        # Worker owns model loading/inference; no model loading on the GUI thread.
        registry=Registry(ROOT/'data')
        active=any(item['references'] for item in registry.active())
        vision=PersonalVision(registry,config) if config['personal_enabled'] and active else None
        return dict(vision=vision,scene=SceneMonitor() if config.get('camera_motion_warning',True) else None)

    def process_frame(self,pipeline,frame,detections,seconds):
        if pipeline is None:
            return detections,False
        moved=pipeline['scene'].update(frame,[d['box'] for d in detections]) if pipeline['scene'] else False
        if moved:
            return [],True
        if pipeline['vision']:
            detections=pipeline['vision'].analyse(frame,detections)
        return detections,False

    def refresh_table(self):
        super().refresh_table()
        for tid,track in self.engine.tracks.items():
            if not self.table.exists(str(tid)):
                continue
            match=track.get('suggestion',{})
            if match.get('name') and not track.get('personal_id') and not track.get('rejected'):
                values=list(self.table.item(str(tid),'values'))
                values[2]=f"Possible: {match['name']}"
                values[3]='Needs confirmation'
                self.table.item(str(tid),values=values)

    def annotate(self,frame,seconds):
        image=super().annotate(frame,seconds)
        h,w=image.shape[:2]
        for track in self.engine.tracks.values():
            if abs(seconds-track['last_detection'])>.001 or track.get('personal_id') or track.get('rejected'):continue
            match=track.get('suggestion',{})
            if match.get('name'):
                a,b,_,d=track['box']
                cv2.putText(image,f"Review: {match['name']} [{match['score']:.2f}]",(int(a*w),min(h-8,int(d*h)+18)),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,(0,160,255),2)
        return image

    def add_personal_choices(self):
        names=[item['name'] for item in self.registry.active()]
        self.config['search_names']=names
        # Find the editable search combobox associated with self.target.
        def walk(widget):
            for child in widget.winfo_children():
                if isinstance(child,ttk.Combobox) and str(child.cget('textvariable'))==str(self.target):
                    child.configure(values=self.config['remember_classes']+names)
                walk(child)
        walk(self.root)

    def registration_allowed(self):
        if self.running or (self.worker_thread and self.worker_thread.is_alive()):
            messagebox.showinfo('Stop first','Stop the source and wait for it to finish before editing registrations. '
                                'You can review a track while a video is paused.')
            return False
        return True

    def crop_dialog(self,image,title,callback):
        image=ImageOps.exif_transpose(image).convert('RGB')
        original=image.copy()
        preview=image.copy()
        preview.thumbnail((850,550),Image.Resampling.LANCZOS)
        width,height=preview.size
        window=tk.Toplevel(self.root)
        window.title(title)
        photo=ImageTk.PhotoImage(preview)
        canvas=tk.Canvas(window,width=width,height=height)
        canvas.pack()
        canvas.create_image(0,0,image=photo,anchor='nw')
        canvas.image=photo
        start=[None]
        rectangle=[None]
        selected=[None]
        def down(event):
            start[0]=(max(0,min(width,event.x)),max(0,min(height,event.y)))
            if rectangle[0]: canvas.delete(rectangle[0])
            rectangle[0]=canvas.create_rectangle(*start[0],*start[0],outline='yellow',width=3)
        def move(event):
            if start[0]:
                canvas.coords(rectangle[0],*start[0],max(0,min(width,event.x)),max(0,min(height,event.y)))
        def up(event):
            if start[0] is None: return
            x,y=start[0]
            a,b=max(0,min(width,event.x)),max(0,min(height,event.y))
            selected[0]=[min(x,a),min(y,b),max(x,a),max(y,b)]
        def save():
            if selected[0] is None:
                messagebox.showinfo('Select object','Drag a tight box around the object.',parent=window)
                return
            x1,y1,x2,y2=selected[0]
            if x2-x1<20 or y2-y1<20:
                messagebox.showinfo('Larger crop','Select a larger view of the object.',parent=window)
                return
            box=(round(x1*original.width/width),round(y1*original.height/height),
                 round(x2*original.width/width),round(y2*original.height/height))
            try:
                callback(original.crop(box))
            except Exception as exc:
                messagebox.showerror('Registration',str(exc),parent=window)
                return
            window.destroy()
        canvas.bind('<ButtonPress-1>',down)
        canvas.bind('<B1-Motion>',move)
        canvas.bind('<ButtonRelease-1>',up)
        ttk.Label(window,text='Select only the object. Add distinct angles, not repeated copies of the same photo.').pack(pady=4)
        ttk.Button(window,text='Save selected object view',command=save).pack(pady=6)
        window.grab_set()

    def manage_objects(self):
        if not self.registration_allowed(): return
        window=tk.Toplevel(self.root)
        window.title('My objects — names & reference photos')
        window.geometry('850x510')
        ttk.Label(window,text='Register one physical item per name. Add at least three distinct views; five to ten are preferable.',
                  wraplength=820,padding=10).pack(fill='x')
        tree=ttk.Treeview(window,columns=('name','views'),show='headings',height=10)
        tree.heading('name',text='Personal name')
        tree.heading('views',text='Reference views')
        tree.pack(fill='both',expand=True,padx=10)
        def refresh():
            tree.delete(*tree.get_children())
            for item in self.registry.active():
                tree.insert('','end',iid=item['id'],values=(item['name'],len(item['references'])))
            self.add_personal_choices()
        def selected():
            selection=tree.selection()
            if not selection:
                messagebox.showinfo('Choose object','Select a registered object first.',parent=window)
                return None
            return selection[0]
        def new():
            name=simpledialog.askstring('Personal name','Examples: my blue bottle, scientific calculator, room keys',parent=window)
            if name is None: return
            try:
                item=self.registry.create(name)
                refresh()
                tree.selection_set(item['id'])
            except ValueError as exc:
                messagebox.showerror('Name',str(exc),parent=window)
        def save_ref(oid,image):
            added=self.registry.add_reference(oid,image)
            refresh()
            tree.selection_set(oid)
            self.status.set('Reference view saved. Personal models refresh on next source start.' if added else 'Identical reference already exists.')
        def upload():
            oid=selected()
            if not oid:return
            path=filedialog.askopenfilename(parent=window,filetypes=[('Images','*.png *.jpg *.jpeg *.webp *.bmp')])
            if not path:return
            try:
                with Image.open(path) as photo:
                    image=ImageOps.exif_transpose(photo).convert('RGB').copy()
                self.crop_dialog(image,'Crop reference photo',lambda crop:save_ref(oid,crop))
            except Exception as exc:
                messagebox.showerror('Photo',str(exc),parent=window)
        def last_frame():
            oid=selected()
            if not oid:return
            if self.last_frame is None:
                messagebox.showinfo('Need camera frame','Start camera, show object, Stop, then use this button.',parent=window)
                return
            image=Image.fromarray(cv2.cvtColor(self.last_frame,cv2.COLOR_BGR2RGB))
            self.crop_dialog(image,'Crop last camera/video frame',lambda crop:save_ref(oid,crop))
        def camera():
            oid=selected()
            if not oid:return
            self.capture_reference(oid,lambda:refresh())
        def rename():
            oid=selected()
            if not oid:return
            name=simpledialog.askstring('Rename','New unique name:',initialvalue=self.registry.get(oid)['name'],parent=window)
            if name is None:return
            try:
                self.registry.rename(oid,name)
                for r in self.engine.records.values():
                    if r.get('personal_id')==oid:r['name']=r['nickname']=name.strip()
                self.engine.flush()
                refresh()
            except ValueError as exc:messagebox.showerror('Name',str(exc),parent=window)
        def archive():
            oid=selected()
            if oid:
                self.registry.archive(oid)
                refresh()
                self.status.set('Registration archived. References and history remain on disk; it is excluded from active matching/search.')
        def manage_views():
            oid=selected()
            if not oid:return
            item=self.registry.get(oid)
            popup=tk.Toplevel(window);popup.title('Reference views — '+item['name'])
            popup.geometry('650x470')
            listing=tk.Listbox(popup,height=6);listing.pack(fill='x',padx=10,pady=5)
            preview=ttk.Label(popup);preview.pack(fill='both',expand=True)
            def update_list():
                listing.delete(0,'end')
                for i,path in enumerate(item['references']):listing.insert('end',f'View {i+1}: {Path(path).name[:16]}')
                refresh()
            def show(event=None):
                index=listing.curselection()
                if not index:return
                path=self.registry.directory/item['references'][index[0]]
                frame=cv2.imread(str(path))
                if frame is not None:self.show_image(preview,frame,(580,260))
            def remove():
                index=listing.curselection()
                if index:
                    self.registry.remove_reference(oid,item['references'][index[0]])
                    preview.configure(image='',text='View excluded from matching');preview.image=None
                    update_list()
            listing.bind('<<ListboxSelect>>',show)
            ttk.Button(popup,text='Exclude selected reference view',command=remove).pack(pady=5)
            ttk.Label(popup,text='Excluding a view removes it from the active gallery. Original files remain on disk.').pack(pady=5)
            update_list()
        controls=ttk.Frame(window,padding=8)
        controls.pack(fill='x', before=tree)
        for i,(label,command) in enumerate([('New object',new),('Add photo',upload),('Take photo',camera),('View/edit photos',manage_views)]):
            ttk.Button(controls,text=label,command=command).grid(row=0,column=i,sticky='ew',padx=3,pady=3)
        more=ttk.Menubutton(controls,text='More')
        menu=tk.Menu(more,tearoff=False)
        menu.add_command(label='Rename object',command=rename)
        menu.add_command(label='Deactivate object',command=archive)
        menu.add_command(label='Add photo from stopped frame',command=last_frame)
        menu.add_command(label='Check reference guidance',command=self.guidance)
        more.configure(menu=menu);more.grid(row=0,column=4,padx=3,pady=3)
        ttk.Label(window,text='Select an object, then add at least 3 tightly cropped photos from different angles. Photos stay on this computer.',wraplength=800,padding=8).pack(fill='x',before=tree)
        refresh()

    def capture_reference(self,oid,refresh):
        try:index=int(self.camera.get())
        except ValueError:
            messagebox.showerror('Camera','Use a numeric camera index.');return
        cap=cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release();messagebox.showerror('Camera','Cannot open registration camera. Close other camera apps and check permissions.');return
        if self.registration_capture is not None:
            cap.release()
            messagebox.showinfo('Camera already open','Close the existing registration-camera window first.')
            return
        self.registration_capture=cap
        window=tk.Toplevel(self.root)
        window.title('Show registered object, then freeze and crop')
        label=ttk.Label(window)
        label.pack()
        state={'frame':None,'closed':False,'frozen':False}
        def close():
            state['closed']=True
            cap.release()
            self.registration_capture=None
            window.destroy()
        def tick():
            if state['closed']:return
            if not state['frozen']:
                ok,frame=cap.read()
                if ok:
                    state['frame']=frame
                    self.show_image(label,frame,(750,480))
            window.after(80,tick)
        def capture():
            if state['frame'] is None:return
            state['frozen']=True
            image=Image.fromarray(cv2.cvtColor(state['frame'],cv2.COLOR_BGR2RGB))
            def save(crop):
                self.registry.add_reference(oid,crop)
                refresh()
                state['frozen']=False
                self.status.set('Camera reference saved. Rotate the object and capture another distinct view.')
            self.crop_dialog(image,'Crop captured object',save)
        ttk.Button(window,text='Freeze + select object',command=capture).pack(pady=5)
        ttk.Button(window,text='Resume preview',command=lambda:state.update(frozen=False)).pack(pady=3)
        window.protocol('WM_DELETE_WINDOW',close)
        tick()

    def review_track(self):
        selected=self.table.selection()
        if not selected:
            messagebox.showinfo('Select track','Select a track in the bottom table first.');return
        tid=int(selected[0]);track=self.engine.tracks[tid]
        if self.engine.camera_warning:
            messagebox.showinfo('Camera shifted','Restart the source and redefine zones before reviewing identities.');return
        if self.last_seconds-track['last_detection']>self.config['missing_after_seconds'] or tid not in self.raw_frames:
            messagebox.showinfo('Expired track','Review a currently/recently observed track.');return
        objects=self.registry.active()
        if not objects:
            messagebox.showinfo('Register first','Stop monitoring and register your personal object with reference photos.');return
        # Freeze the particular observation being reviewed. No later frame is silently added as a reference.
        captured=self.raw_frames[tid].copy()
        h,w=captured.shape[:2]
        a,b,c,d=[max(0,int(v*s)) for v,s in zip(track['box'],(w,h,w,h))]
        crop=captured[b:min(h,d),a:min(w,c)]
        if not crop.size:return
        stamp=track['last_stamp']
        reviewed_seconds=track['last_detection']
        match=dict(track.get('suggestion',{}))
        dialog=tk.Toplevel(self.root)
        dialog.title(f'Review personal identity — track #{tid}')
        dialog.geometry(f'790x{min(660,self.root.winfo_screenheight()-110)}')
        photos=ttk.Frame(dialog);photos.pack(fill='x',pady=8)
        observed=ttk.Frame(photos);observed.pack(side='left',expand=True)
        reference_frame=ttk.Frame(photos);reference_frame.pack(side='left',expand=True)
        ttk.Label(observed,text='Observed crop').pack()
        ttk.Label(reference_frame,text='Registered reference').pack()
        picture=ttk.Label(observed);picture.pack()
        reference_picture=ttk.Label(reference_frame);reference_picture.pack()
        self.show_image(picture,crop,(340,210))
        reason=match.get('reason','No reference match available; you may assign identity manually.')
        ttk.Label(dialog,text=f"Observed: {stamp}\nDetector: {track['raw_name']}\n"
                  f"Proposed identity: {match.get('name') or 'Unknown'}\n"
                  f"Combined similarity: {match.get('score',0):.3f} (not a probability)\n"
                  f"Required: {match.get('threshold',self.config['similarity_threshold']):.3f}; separation from next object: {match.get('margin',0):.3f}\n"
                  f"Saved views: {len(self.registry.get(match['object_id'])['references']) if match.get('object_id') and self.registry.get(match['object_id']) else 0}\n{reason}",wraplength=590).pack(pady=8)
        name=tk.StringVar(value=match.get('name') or objects[0]['name'])
        choice=ttk.Combobox(dialog,textvariable=name,values=[o['name'] for o in objects],state='readonly',width=40)
        choice.pack(pady=5)
        def show_reference(event=None):
            item=next(o for o in objects if o['name']==name.get())
            if item['references']:
                path=self.registry.directory/item['references'][0]
                ref=cv2.imread(str(path))
                if ref is not None:self.show_image(reference_picture,ref,(340,210));return
            reference_picture.configure(image='',text='No reference yet');reference_picture.image=None
        choice.bind('<<ComboboxSelected>>',show_reference)
        show_reference()
        add_example=tk.BooleanVar(value=False)
        ttk.Checkbutton(dialog,text='Also save THIS reviewed crop as an approved reference view',variable=add_example).pack(pady=5)
        def current_oid():return next(o['id'] for o in objects if o['name']==name.get())
        def approve():
            # If the scene advanced, require a paused/stopped view or review the new observation.
            latest=self.engine.tracks[tid]
            if latest['last_detection']!=reviewed_seconds:
                messagebox.showinfo('Observation changed','Pause the video or Stop the webcam, then reopen Review to confirm the shown observation.',parent=dialog)
                return
            try:
                oid=current_oid()
                record=self.engine.approve(tid,oid,stamp)
                self.evidence[tid]=self.annotate(captured,latest['last_detection'])
                self.dirty_evidence.add(tid)
                self.best_evidence[tid]=self.evidence[tid]
                record.update(best_seen=stamp,best_quality=-1,best_box=list(latest['box']))
                self.dirty_best.add(tid)
                if add_example.get():
                    self.registry.add_reference(oid,Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)))
                self.persist();self.refresh_table();self.target.set(record['name']);self.find()
                self.status.set('Identity confirmed for this track. New reference examples take effect on the next source start.')
                dialog.destroy()
            except Exception as exc:messagebox.showerror('Review',str(exc),parent=dialog)
        def deny():
            self.engine.deny(tid,current_oid(),stamp)
            self.refresh_table();dialog.destroy()
        buttons=ttk.Frame(dialog);buttons.pack(pady=8)
        ttk.Button(buttons,text='Yes, this is the selected object',command=approve).pack(side='left',padx=5)
        ttk.Button(buttons,text='Not this object',command=deny).pack(side='left',padx=5)
        ttk.Button(buttons,text='Leave unknown',command=dialog.destroy).pack(side='left',padx=5)

    def show_result(self):
        super().show_result()
        index=self.result_picker.current()
        if index<0 or not self.search_results:return
        r=self.search_results[index]
        if r.get('personal_id'):
            current=self.engine.records.get(r['id']) if r['current_session'] else None
            if current:r=dict(r,**current)
            item=self.registry.get(r['personal_id'])
            if item:
                text=self.answer.get().split('\n')
                text[0]=f"{item['name']} — personal object / track #{r['id']}"
                self.answer.set('\n'.join(text))
            self.answer.set(self.answer.get()+f"\nLast detected region: {r.get('last_detected',r['last_seen'])}"
                            f"\nLast confidently identified: {r['last_seen']}"
                            f"\nIdentity source: {r.get('identity_basis','User approval')}"
                            f"\nLatest appearance similarity: {r.get('match_score',0):.3f} (not a probability)")
        if self.engine.camera_warning:
            self.answer.set(self.answer.get()+'\nCAMERA VIEW SHIFTED: prior image zones may no longer describe current locations.')

    def correct(self):
        selected=self.table.selection()
        if selected and self.engine.tracks[int(selected[0])].get('personal_id'):
            self.review_track()
        else:
            super().correct()

    def closeup(self):
        self.persist()
        index=self.result_picker.current()
        if index<0 or not self.search_results:
            messagebox.showinfo('Search first','Find an object and select a stored sighting first.');return
        r=dict(self.search_results[index])
        if r['current_session'] and r['id'] in self.engine.records:r.update(self.engine.records[r['id']])
        use_best=self.best_picture.get() and r.get('best_snapshot')
        source=r.get('best_snapshot') if use_best else r.get('snapshot')
        box=r.get('best_box') if use_best else r.get('box')
        if not source or not box:return
        frame=cv2.imread(str(ROOT/source))
        if frame is None:return
        h,w=frame.shape[:2]
        a,b,c,d=[max(0,int(v*s)) for v,s in zip(box,(w,h,w,h))]
        crop=frame[b:min(h,d),a:min(w,c)]
        if not crop.size:return
        window=tk.Toplevel(self.root);window.title('Evidence close-up — '+r['name'])
        label=ttk.Label(window);label.pack(padx=10,pady=10)
        self.show_image(label,crop,(650,450))
        stamp=r['best_seen'] if use_best else r['last_seen']
        ttk.Label(window,text=f'{r["name"]} | {stamp}\nThis is saved evidence, not a current-location guarantee.',padding=10).pack()

    def close(self):
        if self.registration_capture is not None:
            self.registration_capture.release()
            self.registration_capture=None
        super().close()


if __name__=='__main__':
    root=tk.Tk()
    App(root)
    root.mainloop()
