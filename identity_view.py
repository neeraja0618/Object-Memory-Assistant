"""Shared display rules. A possible personal name is never a confirmed identity."""
import math
from engine import zone_for


def valid_box(box):
    return (isinstance(box,(list,tuple)) and len(box)==4
            and all(isinstance(v,(int,float)) and math.isfinite(v) for v in box)
            and 0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1)


def position_description(record):
    box=record.get('box')
    location=record.get('location') or (zone_for(box,[]) if valid_box(box) else 'Position unavailable')
    if valid_box(box):
        x=(box[0]+box[2])*50
        y=(box[1]+box[3])*50
        location+=f' | centre: {x:.0f}% from left, {y:.0f}% from top'
    return location


def display_identity(track,record=None):
    if track.get('model_disputed'):
        return track['raw_name']+'? | models disagree',(0,160,255)
    state=track['state']
    if record and state in ('Confirmed','Reference-confirmed','User-labelled'):
        if record.get('category_only') and not record.get('personal_id'):
            suggestion=track.get('suggestion',{})
            if suggestion.get('display_allowed') and suggestion.get('name'):
                return record['name']+' | possible: '+suggestion['name'],(0,160,255)
            return record['name']+' (category)',(220,190,60)
        color=(230,160,40) if state=='User-labelled' else (60,210,70)
        return record['name'],color
    suggestion=track.get('suggestion',{})
    if suggestion.get('object_id') and suggestion.get('name') and suggestion.get('display_allowed',True):
        return 'Possible: '+suggestion['name'],(0,160,255)
    if track.get('category_ready'):
        return 'Category: '+track['raw_name']+' | identity unverified',(0,160,255)
    return 'Unknown (guess: '+track['raw_name']+')',(0,160,255)
