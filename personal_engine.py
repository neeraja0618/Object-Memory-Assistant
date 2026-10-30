from collections import Counter,deque
from engine import Engine


class PersonalEngine(Engine):
    def __init__(self,directory,session,settings,registry):
        super().__init__(directory,session,settings)
        self.registry=registry
        self.camera_warning=False
    def warn_camera(self,stamp):
        if self.camera_warning:
            return
        self.camera_warning=True
        for r in self.records.values():
            r['status']='Camera shifted; historical evidence'
        self.db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',
                        (self.session,stamp,0,'scene','Camera-view shift warning; memory suspended','Previous zones unreliable'))
        self.flush()
    def association_score(self,track,detection,iou,distance,ratio):
        incoming=detection.get('personal_match',{})
        previous=track.get('suggestion',{})
        trusted=track.get('personal_id') or (previous.get('object_id') if previous.get('strong') else '')
        proposed=incoming.get('object_id') if incoming.get('strong') else ''
        if trusted and proposed and trusted!=proposed:
            # Prefer any consistent identity elsewhere. A lone conflicting
            # observation at the same position must still invalidate trust.
            return .01 if ratio>.2 and iou>.70 else None
        if trusted and proposed==trusted and ratio>.2 and distance<.35:
            return .6+iou+max(0,.35-distance)
        return super().association_score(track,detection,iou,distance,ratio)

    def stable_label(self,track):
        if track.get('model_disputed'):
            return None
        match=track.get('observation',{}).get('personal_match',{})
        if match.get('object_id') and not track.get('personal_id') and not track.get('override'):
            # A general detector category can be remembered without assigning
            # the competing personal identity. Custom/proposal categories still
            # use the original conservative guard.
            observation=track.get('observation',{})
            if not (self.settings.get('category_memory_enabled',False)
                    and observation.get('category_source')=='general'):
                return None
        return super().stable_label(track)

    def category_conflict(self,detection,match):
        if not self.settings.get('strict_personal_names',False):return False
        if detection.get('category_source')!='general':return False
        if detection.get('confidence',0)<self.settings.get('category_conflict_confidence',.7):return False
        category=detection['name']
        if category not in self.settings.get('identity_conflict_categories',[]):return False
        item=self.registry.get(match.get('object_id')) or {}
        # Hints are explicit settings/registration metadata, not guessed from
        # arbitrary personal names. A conflict preserves uncertainty; it does
        # not prove that the category detector is correct.
        hints=self.settings.get('personal_category_hints',{})
        expected=item.get('expected_category') or hints.get(match.get('name','').strip().casefold())
        return bool(expected and expected!=category)

    def update(self,detections,seconds,stamp):
        if self.camera_warning:
            return []
        counts=Counter(d.get('personal_match',{}).get('object_id') for d in detections if d.get('personal_match',{}).get('strong'))
        # Prepare identity guards before generic memory updates.
        for d in detections:
            match=d.get('personal_match',{})
            if self.category_conflict(d,match):
                match=dict(match,reference_strong=bool(match.get('strong')),strong=False,
                           category_conflict=True,display_allowed=False,
                           reason='Detector category and registered-object category disagree; explicit review required')
                d['personal_match']=match
            if match.get('object_id') and counts[match['object_id']]>1:
                match=dict(match,strong=False,reference_strong=False,reason='Same registered identity proposed in multiple regions')
                d['personal_match']=match
            d['identity_verified']=False
        for track in self.tracks.values():
            if seconds-track['last_detection']>self.settings['evidence_gap_seconds']:
                track.pop('suggestion_votes',None)
                track.pop('category_ready',None)
                track.pop('model_dispute_votes',None)
                track.pop('model_disputed',None)
                if 'identity_hits' in track:track['identity_hits'].clear()
            oid=track.get('personal_id')
            if not oid:
                continue
            item=self.registry.get(oid)
            if item and item.get('active',True):
                track['override']=item['name']
            else:
                track['override_guard']=True
        # Guard is set per associated detection inside the base engine hook below.
        result=super().update(detections,seconds,stamp)
        for tid,track in self.tracks.items():
            if abs(seconds-track['last_detection'])>.001:
                continue
            observation=track.get('observation',{})
            match=observation.get('personal_match',{})
            if match.get('object_id') in track.get('denied_objects',set()):
                match=dict(match,object_id='',name='',strong=False,reason='Candidate rejected for this track')
            if self.settings.get('strict_personal_names',False):
                votes=track.setdefault('suggestion_votes',deque(maxlen=max(2,int(self.settings.get('suggestion_observations',3)))))
                safe=bool(match.get('object_id') and match.get('strong') and not match.get('confused') and not match.get('category_conflict') and not track.get('model_disputed'))
                votes.append(match.get('object_id') if safe else '')
                ready=len(votes)==votes.maxlen and safe and all(v==match['object_id'] for v in votes)
                match=dict(match,display_allowed=bool(ready))
                if not ready and safe:match['reason']='Waiting for repeated strong reference matches; identity remains unconfirmed'
            track['suggestion']=match
            n=max(2,int(self.settings.get('suggestion_observations',3)))
            recent=list(track['votes'])[-n:]
            track['category_ready']=not track.get('model_disputed') and len(recent)==n and all(name==track['raw_name'] and confidence>=self.settings.get('possible_category_confidence',.5) for name,confidence in recent)
            track['quality_warning']=observation.get('quality_warning','')
            record=self.records.get(tid)
            if record:
                if not track.get('personal_id') and observation.get('category_source')=='general':
                    record['category_only']=True
                    record['identity_basis']='Repeated general-category observations; personal identity unverified'
                record['last_detected']=stamp
                record['last_detected_box']=track['box']
                if track.get('personal_id'):
                    record['category_only']=False
                    record['personal_id']=track['personal_id']
                    record['match_score']=match.get('score',0)
                    record['identity_basis']='User approval + visual reference checks'
                    if track['personal_id'] in track.get('denied_objects',set()):
                        record['status']='Identity disputed by user'
        return result
    def identity_check(self,track,detection):
        disputes=track.setdefault('model_dispute_votes',deque(maxlen=3))
        disputes.append(bool(detection.get('detector_conflict')))
        track['model_disputed']=any(disputes)
        if track['model_disputed']:
            detection['detector_conflict']=True
            detection['identity_verified']=False
            match=detection.get('personal_match',{})
            if match:
                detection['personal_match']=dict(match,strong=False,reference_strong=False,display_allowed=False,
                    reason='Models disagree on category; identity and memory updates withheld')
            if 'identity_hits' in track:track['identity_hits'].clear()
            return
        oid=track.get('personal_id')
        if not oid:
            return
        match=detection.get('personal_match',{})
        item=self.registry.get(oid)
        # An explicit user correction may resolve a detector-category conflict,
        # but future frames still need strong visual-reference evidence.
        visual=match.get('strong') or (track.get('identity_manual') and match.get('reference_strong'))
        good=bool(item and item.get('active',True) and match.get('object_id')==oid and visual
                  and oid not in track.get('denied_objects',set()))
        if good and track.get('identity_manual') and match.get('category_conflict'):
            detection['personal_match']=dict(match,strong=True,category_conflict=False,
                reason='Explicit user correction resolves category conflict; strong reference evidence still required')
        history=track.setdefault('identity_hits',deque(maxlen=self.settings['personal_confirmations']))
        history.append(good)
        detection['identity_verified']=len(history)>=self.settings['personal_confirmations'] and all(history)
    def approve(self,tid,oid,stamp,manual=True):
        if self.camera_warning:
            raise ValueError('Camera view shifted. Restart and redefine zones before confirming identity.')
        item=self.registry.get(oid)
        if not item or not item.get('active',True):
            raise ValueError('Choose an active registered object.')
        if tid not in self.tracks:
            raise ValueError('Choose a current track.')
        track=self.tracks[tid]
        record=super().correct(tid,item['name'],item['name'],stamp)
        track.update(personal_id=oid,override_guard=True,
                     identity_hits=deque([] if manual else [True]*self.settings['personal_confirmations'],
                                         maxlen=self.settings['personal_confirmations']))
        track.setdefault('denied_objects',set()).discard(oid)
        basis='Explicit user approval' if manual else 'Automatic selective reference assignment (experimental)'
        track['identity_manual']=manual
        record.update(personal_id=oid,category_only=False,last_detected=track['last_stamp'],identity_basis=basis,user_labelled=manual,
                      match_score=track.get('suggestion',{}).get('score',0))
        self.log(record,'User confirmed registered object identity' if manual else 'Selective policy assigned personal identity',stamp)
        self.flush()
        return record
    def deny(self,tid,oid,stamp):
        track=self.tracks[tid]
        track.setdefault('denied_objects',set()).add(oid)
        track['suggestion']=dict(object_id='',name='',score=0,strong=False,reason='User rejected identity candidate')
        if track.get('personal_id')==oid:
            track['identity_hits']=deque(maxlen=self.settings['personal_confirmations'])
            if tid in self.records:
                self.records[tid]['status']='Identity disputed by user'
        record=self.records.get(tid) or dict(id=tid,name=track['raw_name'],location='')
        self.log(record,'User rejected proposed personal identity',stamp)
        self.flush()
    def search(self,query):
        rows=super().search(query)
        active={item['id']:item for item in self.registry.active()}
        results=[]
        # Also search renamed personal items by persistent registration ID.
        for r in super().search(''):
            oid=r.get('personal_id')
            if not oid:
                continue
            if oid not in active:
                continue
            r['name']=r['nickname']=active[oid]['name']
            if query.strip().lower() in r['name'].lower():
                results.append(r)
        results.extend(r for r in rows if not r.get('personal_id'))
        return results
