"""Selective review, tentative observations and explicitly scoped history revision."""
from collections import deque
import json
from pathlib import Path
import shutil
import uuid
from engine import zone_for
from personal_engine import PersonalEngine


ACCEPTED={'User-confirmed','Reference-confirmed'}


class ResearchEngine(PersonalEngine):
    def __init__(self,directory,session,settings,registry):
        super().__init__(directory,session,settings,registry)
        self.db.execute('CREATE TABLE IF NOT EXISTS observations(id INTEGER PRIMARY KEY AUTOINCREMENT,session TEXT,track INTEGER,seconds REAL,payload TEXT,original TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS repairs(id INTEGER PRIMARY KEY AUTOINCREMENT,session TEXT,observation INTEGER,before TEXT,after TEXT,reason TEXT)')
        self.db.execute('CREATE INDEX IF NOT EXISTS observations_session_time ON observations(session,seconds)')
        self.pending_history={}
        self.last_history={}
        self.last_requests={}
        self.review_requests=0
        self.max_seconds=0

    def request_review(self,tid,match,reason,seconds,stamp):
        oid=match.get('object_id')
        if not oid:return
        track=self.tracks[tid]
        track['review_reason']=reason
        track['review_needed']=True
        key=(oid,reason)
        if seconds-self.last_requests.get(key,-1e9)<self.settings['review_cooldown_seconds']:
            return
        if self.review_requests>=self.settings['max_review_requests']:
            track['review_reason']='Review budget reached; identity remains unresolved'
            return
        self.review_requests+=1
        self.last_requests[key]=seconds
        record=self.records.get(tid) or dict(id=tid,name=match['name'],location=zone_for(track['box'],self.settings.get('zones',[])))
        self.log(record,'Review requested: '+reason,stamp)

    def update(self,detections,seconds,stamp):
        if self.camera_warning:return []
        self.max_seconds=max(self.max_seconds,seconds)
        for t in self.tracks.values():
            if seconds-t['last_detection']>self.settings['evidence_gap_seconds']:
                t.pop('auto_votes',None)
        super().update(detections,seconds,stamp)
        for tid,track in self.tracks.items():
            if abs(seconds-track['last_detection'])>.001:continue
            match=track.get('suggestion',{})
            oid=match.get('object_id')
            track['review_needed']=False
            if track.get('rejected'):continue
            if not track.get('personal_id'):
                votes=track.setdefault('auto_votes',deque(maxlen=self.settings['auto_identity_observations']))
                safe=bool(oid and match.get('strong') and not match.get('confused') and oid not in track.get('denied_objects',set()))
                votes.append(oid if safe else '')
                consistent=len(votes)==votes.maxlen and all(v==oid and v for v in votes)
                occupied=any(t['id']!=tid and t.get('personal_id')==oid
                             and seconds-t['last_detection']<=self.settings['missing_after_seconds'] for t in self.tracks.values()) if oid else False
                if self.settings['review_policy']=='selective' and consistent and not occupied:
                    self.approve(tid,oid,stamp,manual=False)
                elif oid and (self.settings['review_policy']=='always' or len(votes)==votes.maxlen):
                    reason='Registered objects look similar' if match.get('confused') else ('Possible identity switch / duplicate' if occupied else
                           ('Always-review policy' if self.settings['review_policy']=='always' else 'Insufficient identity evidence'))
                    self.request_review(tid,match,reason,seconds,stamp)
            else:
                if oid and (oid!=track['personal_id'] or not match.get('strong')):
                    track['conflict_count']=track.get('conflict_count',0)+1
                    if track['conflict_count']>=2:self.request_review(tid,match,'New evidence conflicts with trusted identity',seconds,stamp)
                else:track['conflict_count']=0
            record=self.records.get(tid)
            if record and track.get('personal_id'):
                manual=track.get('identity_manual',True)
                record['user_labelled']=manual
                if record['seconds']==seconds and record['status']=='User-labelled' and not manual:
                    record['status']=track['state']='Reference-confirmed'
                record['identity_basis']='Explicit user approval + appearance checks' if manual else 'Selective reference policy (experimental)'
            self.add_observation(tid)
        return [r for r in self.records.values() if r['seconds']==seconds and r['status'] in ('User-labelled','Reference-confirmed','Confirmed') and not r.get('rejected')]

    def approve(self,tid,oid,stamp,manual=True):
        record=super().approve(tid,oid,stamp,manual=manual)
        self.tracks[tid]['review_needed']=False
        if not manual:
            record['status']=self.tracks[tid]['state']='Reference-confirmed'
        self.add_observation(tid,force=True)
        return record

    def add_observation(self,tid,force=False):
        track=self.tracks[tid]
        r=self.records.get(tid)
        match=track.get('suggestion',{})
        seconds=track['last_detection']
        accepted=bool(r and r.get('personal_id') and r['seconds']==seconds and r['status'] in ('User-labelled','Reference-confirmed'))
        state=('User-confirmed' if r['user_labelled'] else 'Reference-confirmed') if accepted else 'Tentative'
        oid=r['personal_id'] if accepted else ''
        key=(state,oid,match.get('object_id',''))
        last=self.last_history.get(tid)
        if not force and last and key==last[1] and seconds-last[0]<self.settings['history_sample_seconds']:return
        payload=dict(track=tid,seconds=seconds,stamp=track['last_stamp'],state=state,object_id=oid,
            name=r['name'] if accepted else 'Unresolved',candidate_id=match.get('object_id',''),candidate_name=match.get('name',''),
            score=match.get('score',0),margin=match.get('margin',0),strong=bool(match.get('strong')),
            threshold=match.get('threshold',self.settings['similarity_threshold']),box=list(track['box']),
            detector_name=track['raw_name'],detector_confidence=track.get('raw_confidence',0),evidence='',
            reason=track.get('review_reason',''),basis=r.get('identity_basis','') if accepted else '')
        text=json.dumps(payload)
        cursor=self.db.execute('INSERT INTO observations(session,track,seconds,payload,original) VALUES(?,?,?,?,?)',
                               (self.session,tid,seconds,text,text))
        self.pending_history[cursor.lastrowid]=payload
        self.last_history[tid]=(seconds,key)
        return cursor.lastrowid

    def history(self,recent_only=True):
        self.flush()
        rows=self.db.execute('SELECT id,session,payload FROM observations WHERE session=? ORDER BY seconds DESC,id DESC',(self.session,)).fetchall()
        result=[]
        for oid,session,payload in rows:
            row=json.loads(payload)
            if recent_only and row['seconds']<self.max_seconds-self.settings['history_window_seconds']:continue
            row.update(observation_id=oid,session=session)
            result.append(row)
        return result

    def attach_history_image(self,ids,relative_path):
        for oid in ids:
            row=self.db.execute('SELECT payload,original FROM observations WHERE id=? AND session=?',(oid,self.session)).fetchone()
            if not row:continue
            payload,original=map(json.loads,row)
            payload['evidence']=original['evidence']=relative_path
            self.db.execute('UPDATE observations SET payload=?,original=? WHERE id=?',
                            (json.dumps(payload),json.dumps(original),oid))
            self.pending_history.pop(oid,None)
        self.db.commit()

    def preview_repair(self,observation_ids,target_id,mode='evidence'):
        if mode not in ('evidence','manual','dispute'):raise ValueError('Invalid repair mode.')
        item=self.registry.get(target_id)
        if not item or not item.get('active',True):raise ValueError('Choose an active registered object.')
        available={row['observation_id']:row for row in self.history()}
        plan=[]
        for oid in observation_ids:
            if oid not in available:raise ValueError('Repair is limited to this session and the configured recent window.')
            old=available[oid]
            if not old['evidence'] or not (self.directory.parent/old['evidence']).exists():
                raise ValueError('Evidence image is unavailable. Do not revise unseen observations.')
            new=dict(old)
            if mode=='manual' or (mode=='evidence' and old['candidate_id']==target_id and old['strong']):
                new.update(object_id=target_id,name=item['name'],state='User-confirmed' if mode=='manual' else 'Reference-confirmed',
                           basis='User reviewed this historical observation' if mode=='manual' else 'Repaired from stored supporting reference evidence')
                reason='Explicit per-observation confirmation' if mode=='manual' else 'Stored identity evidence supports target'
            elif mode=='evidence' and old['object_id']==target_id and old['state'] in ACCEPTED:
                reason='Keep existing supported identity'
            else:
                new.update(object_id='',name='Unresolved',state='Disputed',basis='Historical identity invalidated; not automatically reassigned')
                reason='Insufficient identity evidence; preserve uncertainty'
            new['repair_reason']=reason
            plan.append(dict(observation_id=oid,before=old,after=new,reason=reason))
        return plan

    def apply_repair(self,plan):
        available={row['observation_id'] for row in self.history()}
        if any(change['observation_id'] not in available for change in plan):
            raise ValueError('Repair window advanced after preview. Reopen the history and select recent evidence.')
        affected=set()
        with self.db:
            for change in plan:
                oid=change['observation_id']
                current=self.db.execute('SELECT payload FROM observations WHERE id=? AND session=?',(oid,self.session)).fetchone()
                if not current:raise ValueError('Observation changed or no longer belongs to this session.')
                now=json.loads(current[0]);before=dict(change['before'])
                before.pop('observation_id',None);before.pop('session',None)
                if now!=before:raise ValueError('Observation changed after preview. Create a new preview.')
                after=dict(change['after']);after.pop('observation_id',None);after.pop('session',None)
                self.db.execute('UPDATE observations SET payload=? WHERE id=?',(json.dumps(after),oid))
                self.db.execute('INSERT INTO repairs(session,observation,before,after,reason) VALUES(?,?,?,?,?)',
                                (self.session,oid,json.dumps(before),json.dumps(after),change['reason']))
                affected.add(after['track'])
            for tid in affected:self.rebuild_track(tid)
        self.flush()
        return len(plan)

    def rebuild_track(self,tid):
        rows=self.db.execute('SELECT payload FROM observations WHERE session=? AND track=? ORDER BY seconds DESC,id DESC',
                             (self.session,tid)).fetchall()
        accepted=[json.loads(row[0]) for row in rows if json.loads(row[0])['state'] in ACCEPTED and json.loads(row[0]).get('object_id')]
        old=self.records.get(tid)
        track=self.tracks.get(tid)
        if not accepted:
            if old:
                old.update(rejected=True,status='History disputed; no confirmed identity')
            if track:
                previous=track.pop('personal_id',None)
                if previous:track.setdefault('denied_objects',set()).add(previous)
                track.update(override=None,override_guard=False)
                track.pop('auto_votes',None)
            return
        row=accepted[0];item=self.registry.get(row['object_id'])
        if not item or not item.get('active',True):return
        source=self.directory.parent/row['evidence']
        snapshot=''
        if source.is_file():
            destination=self.directory/'snapshots'/self.session/f'repaired_{tid}_{uuid.uuid4().hex[:8]}.jpg'
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
            snapshot=str(destination.relative_to(self.directory.parent))
        record=dict(id=tid,name=item['name'],nickname=item['name'],personal_id=item['id'],box=row['box'],
            seconds=row['seconds'],last_seen=row['stamp'],last_detected=track['last_stamp'] if track else row['stamp'],
            status='Repaired historical sighting',confidence=row['detector_confidence'],agreement=0,
            location=zone_for(row['box'],self.settings.get('zones',[])),snapshot=snapshot,best_snapshot='',best_seen='',best_box=[],best_quality=-1,
            user_labelled=row['state']=='User-confirmed',identity_basis=row['basis'],match_score=row['score'],event_box=row['box'])
        self.records[tid]=record
        if track:
            track.update(personal_id=item['id'],override=item['name'],override_guard=True,
                         identity_manual=record['user_labelled'],identity_hits=deque(maxlen=self.settings['personal_confirmations']))
            track.setdefault('denied_objects',set()).discard(item['id'])
        self.log(record,'Historical memory rebuilt from reviewed evidence',row['stamp'])

    def repair_audit(self):
        return self.db.execute('SELECT observation,reason,before,after FROM repairs WHERE session=? ORDER BY id DESC',(self.session,)).fetchall()
