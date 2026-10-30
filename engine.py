"""Conservative, dependency-free short-term tracking and persistent memory."""
from collections import Counter, deque
import json
import math
from pathlib import Path
import sqlite3


def centre(box):
    return ((box[0] + box[2])/2, (box[1] + box[3])/2)


def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return intersection/max(1e-9, area_a+area_b-intersection)


def zone_for(box, zones):
    x, y = centre(box)
    for zone in zones:
        a, b, c, d = zone['box']
        if a <= x <= c and b <= y <= d:
            return zone['name']
    return f"{('upper', 'middle', 'lower')[min(2, max(0, int(y*3)))]} {('left', 'centre', 'right')[min(2, max(0, int(x*3)))]} of image"


class Engine:
    def __init__(self, directory, session, settings):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session = session
        self.settings = settings
        minimum = settings['minimum_observations']
        if not 1 <= minimum <= settings['vote_window']:
            raise ValueError('minimum_observations must be between 1 and vote_window.')
        if not 0 < settings['label_agreement'] <= 1 or not 0 < settings['memory_confidence'] <= 1:
            raise ValueError('Confidence and agreement must be in (0, 1].')
        if settings['missing_after_seconds'] <= 0 or settings['evidence_gap_seconds'] <= 0:
            raise ValueError('Time tolerances must be positive.')
        self.tracks = {}
        self.records = {}
        self.next_id = 1
        self.db = sqlite3.connect(self.directory/'memory_final.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS sightings(session TEXT, track INTEGER, payload TEXT, PRIMARY KEY(session, track))')
        self.db.execute('CREATE TABLE IF NOT EXISTS events(session TEXT, stamp TEXT, track INTEGER, name TEXT, event TEXT, zone TEXT)')

    def log(self, record, event, stamp):
        self.db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',
                        (self.session, stamp, record['id'], record['name'], event, record['location']))

    def stable_label(self, track):
        votes = list(track['votes'])
        if len(votes) < self.settings['minimum_observations']:
            return None
        winner, count = Counter(name for name, _ in votes).most_common(1)[0]
        if winner=='unclassified object':
            return None
        confidences = [conf for name, conf in votes if name == winner]
        mean = sum(confidences)/len(confidences)
        if count < self.settings['minimum_observations'] or count/len(votes) < self.settings['label_agreement']:
            return None
        if mean < self.settings['memory_confidence']:
            return None
        # The latest observation must itself support the consensus.
        if votes[-1][0] != winner or votes[-1][1] < self.settings['memory_confidence']:
            return None
        return winner, mean, count/len(votes)

    def identity_check(self, track, detection):
        pass

    def association_score(self,track,detection,iou,distance,ratio):
        if ratio > .2 and (iou > .05 or distance < .12):
            return iou + max(0, .12-distance)
        return None

    def update(self, detections, seconds, stamp):
        roi = self.settings.get('roi')
        if roi:
            detections = [d for d in detections if roi[0] <= centre(d['box'])[0] <= roi[2]
                          and roi[1] <= centre(d['box'])[1] <= roi[3]]
        active = [t for t in self.tracks.values()
                  if seconds-t['last_detection'] <= self.settings['missing_after_seconds']]
        candidates = []
        for index, detection in enumerate(detections):
            b = detection['box']
            for track in active:
                a = track['box']
                distance = math.dist(centre(a), centre(b))
                iou = overlap(a, b)
                area_a = max(1e-8, (a[2]-a[0])*(a[3]-a[1]))
                area_b = max(1e-8, (b[2]-b[0])*(b[3]-b[1]))
                ratio = min(area_a, area_b)/max(area_a, area_b)
                score=self.association_score(track,detection,iou,distance,ratio)
                if score is not None:
                    candidates.append((score, track['id'], index))
        assigned_tracks, assigned_detections, assignments = set(), set(), {}
        for _, tid, index in sorted(candidates, reverse=True):
            if tid not in assigned_tracks and index not in assigned_detections:
                assigned_tracks.add(tid)
                assigned_detections.add(index)
                assignments[index] = tid
        confirmed = []
        for index, detection in enumerate(detections):
            tid = assignments.get(index)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = dict(id=tid, box=detection['box'], last_detection=seconds,
                    votes=deque(maxlen=self.settings['vote_window']), state='Uncertain',
                    raw_name=detection['name'], override=None)
            track = self.tracks[tid]
            # Do not call widely separated sightings consecutive evidence.
            if seconds-track['last_detection'] > self.settings['evidence_gap_seconds']:
                track['votes'].clear()
            track.update(box=detection['box'], last_detection=seconds,
                         raw_name=detection['name'], raw_confidence=detection['confidence'], last_stamp=stamp)
            track['observation'] = detection
            track['votes'].append((detection['name'], detection['confidence']))
            self.identity_check(track,detection)
            if track.get('rejected'):
                track['state'] = 'Rejected by user'
                continue
            consensus = self.stable_label(track)
            old = self.records.get(tid)
            if detection.get('detector_conflict'):
                track['state'] = 'Models disagree'
                if old:
                    old['status'] = 'Models disagree; last confirmed evidence retained'
                continue
            if track.get('override_guard') and not detection.get('identity_verified',False):
                track['state'] = 'Identity uncertain'
                if old:
                    old['status'] = 'Identity uncertain'
                continue
            if track['override']:
                name = track['override']
                consensus = (name, detection['confidence'], 1.0)
                state = 'User-labelled'
            elif consensus is None:
                state = 'Uncertain'
            else:
                name = consensus[0]
                state = 'Confirmed'
                if old and old['name'] != name:
                    state = 'Label conflict'
            track['state'] = state
            if state in ('Uncertain', 'Label conflict'):
                if old:
                    old['status'] = state
                continue
            name, score, agreement = consensus
            record = dict(id=tid, name=name, nickname=old.get('nickname', '') if old else '',
                location=zone_for(detection['box'], self.settings.get('zones', [])),
                box=detection['box'], confidence=score, agreement=agreement,
                last_seen=stamp, seconds=seconds, status=state,
                snapshot=old.get('snapshot', '') if old else '',
                best_snapshot=old.get('best_snapshot', '') if old else '',
                best_seen=old.get('best_seen', '') if old else '',
                best_box=old.get('best_box', []) if old else [],
                best_quality=old.get('best_quality', -1) if old else -1,
                personal_id=old.get('personal_id', '') if old else '',
                user_labelled=bool(track['override']))
            self.records[tid] = record
            if old is None:
                self.log(record, 'Confirmed sighting' if state == 'Confirmed' else 'User-labelled sighting', stamp)
                record['event_box'] = record['box']
            else:
                record['event_box'] = old.get('event_box', old['box'])
                if old['status'] in ('Not detected', 'Briefly not detected'):
                    self.log(record, 'Seen again within short-term track', stamp)
                if math.dist(centre(record['event_box']), centre(record['box'])) > 0.15:
                    self.log(record, 'Detected position changed', stamp)
                    record['event_box'] = record['box']
            confirmed.append(record)
        # Unmatched detections must not extend the lifespan of a stored sighting.
        for tid, record in self.records.items():
            track = self.tracks[tid]
            if track.get('rejected'):
                continue
            if seconds-track['last_detection'] > self.settings['missing_after_seconds']:
                if record['status'] != 'Not detected':
                    record['status'] = 'Not detected'
                    track['state'] = 'Not detected'
                    self.log(record, 'No longer detected; cause unknown', stamp)
            elif tid not in assignments.values() and tid not in [r['id'] for r in confirmed]:
                # A recently seen object is not necessarily visible in this frame.
                if seconds-track['last_detection'] > 0.5 and record['status'] in ('Confirmed', 'User-labelled'):
                    record['status'] = 'Briefly not detected'
        return confirmed

    def correct(self, tid, name, nickname, stamp):
        if tid not in self.tracks:
            raise ValueError('Choose a track in the current session.')
        track = self.tracks[tid]
        track['rejected'] = False
        track['override'] = name
        track['votes'].clear()
        old = self.records.get(tid)
        record = dict(id=tid, name=name, nickname=nickname.strip(), location=zone_for(track['box'], self.settings.get('zones', [])),
                      box=track['box'], confidence=track.get('raw_confidence', 0), agreement=0,
                      last_seen=track['last_stamp'], seconds=track['last_detection'], status='User-labelled',
                      snapshot=old.get('snapshot', '') if old else '', user_labelled=True, event_box=track['box'])
        self.records[tid] = record
        track['state'] = 'User-labelled'
        self.log(record, f"User correction: {old['name'] if old else track['raw_name']} -> {name}; not model training", stamp)
        self.flush()
        return record

    def reject(self, tid, stamp):
        if tid not in self.tracks:
            raise ValueError('Choose a track in the current session.')
        track = self.tracks[tid]
        track['rejected'] = True
        track['override'] = None
        track['state'] = 'Rejected by user'
        record = self.records.get(tid)
        if record:
            record['rejected'] = True
            record['status'] = 'Rejected by user'
        else:
            record = dict(id=tid, name=track['raw_name'], location=zone_for(track['box'], self.settings.get('zones', [])))
        self.log(record, 'User rejected detection; not model training', stamp)
        self.flush()

    def flush(self):
        for tid, record in self.records.items():
            self.db.execute('INSERT OR REPLACE INTO sightings VALUES(?,?,?)',
                            (self.session, tid, json.dumps(record)))
        self.db.commit()

    def search(self, query):
        self.flush()
        query = query.lower().strip()
        found = []
        for session, payload in self.db.execute('SELECT session,payload FROM sightings ORDER BY rowid DESC'):
            r = json.loads(payload)
            if r.get('rejected'):
                continue
            if query and query not in r['name'].lower() and query not in r.get('nickname', '').lower():
                continue
            r['session'] = session
            r['current_session'] = session == self.session
            if not r['current_session']:
                r['status'] = 'Historical'
            found.append(r)
        # Source times cannot be compared between sessions. Preserve newest session first,
        # but sort sightings within a session by last confirmed source time.
        order = {}
        for r in found:
            order.setdefault(r['session'], len(order))
        return sorted(found, key=lambda r: (order[r['session']], -r['seconds']))

    def timeline(self):
        return self.db.execute('SELECT session,stamp,track,name,event,zone FROM events ORDER BY rowid DESC').fetchall()

    def close(self):
        self.flush()
        self.db.close()
