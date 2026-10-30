"""Colour evidence from an object's central body; not a standalone identity detector."""
import numpy as np


def color_signature(image):
    image=image.convert('RGB')
    w,h=image.size
    # Exclude background-heavy boundaries and much of the cap/base.
    body=image.crop((int(w*.25),int(h*.30),max(int(w*.25)+1,int(w*.75)),max(int(h*.30)+1,int(h*.85))))
    body.thumbnail((96,96))
    hsv=np.asarray(body.convert('HSV'),dtype=np.float32)/255
    hue=hsv[:,:,0].ravel()
    saturation=hsv[:,:,1].ravel()
    weight=np.clip((saturation-.12)/.30,0,1)
    bins=np.minimum(15,(hue*16).astype(int))
    histogram=np.bincount(bins,weights=weight,minlength=16)
    histogram=np.append(histogram,np.sum(1-weight))
    histogram=np.sqrt(histogram/max(1e-8,float(histogram.sum())))
    return histogram.astype(np.float32)


def reference_scores(vector,features,color=None,colors=None):
    """Pair each embedding with the colour of that SAME reference view."""
    scores=np.asarray(features,dtype=np.float32)@np.asarray(vector,dtype=np.float32)
    if color is None or colors is None or len(colors)!=len(features):
        return scores,scores,None
    chroma=np.clip(np.asarray(colors)@np.asarray(color),0,1)
    return .80*scores+.20*chroma,scores,chroma


def cross_scores(a,b):
    fa=np.asarray(a['features'],dtype=np.float32)
    fb=np.asarray(b['features'],dtype=np.float32)
    fa=fa/np.maximum(1e-8,np.linalg.norm(fa,axis=1,keepdims=True))
    fb=fb/np.maximum(1e-8,np.linalg.norm(fb,axis=1,keepdims=True))
    deep=fa@fb.T
    if 'colors' in a and 'colors' in b:
        return .80*deep+.20*np.clip(np.asarray(a['colors'])@np.asarray(b['colors']).T,0,1)
    return deep
