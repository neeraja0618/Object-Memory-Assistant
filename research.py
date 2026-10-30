"""Experimental heuristics for confusing references; not calibrated probabilities."""
import numpy as np
from personal import normalize,PersonalVision
from appearance import cross_scores


def select_views(features,negative_features,budget):
    features=normalize(features)
    budget=min(max(1,int(budget)),len(features))
    if len(negative_features):
        negative=normalize(negative_features)
        distinctiveness=1-np.max(features@negative.T,axis=1)
        first=int(np.argmax(distinctiveness))
    else:
        distinctiveness=np.zeros(len(features))
        first=int(np.argmax(np.mean(features@features.T,axis=1)))
    chosen=[first]
    while len(chosen)<budget:
        diversity=1-np.max(features@features[chosen].T,axis=1)
        values=diversity+.35*distinctiveness
        values[chosen]=-np.inf
        chosen.append(int(np.argmax(values)))
    return chosen


def gallery_report(gallery,settings):
    result={'objects':[],'pairs':[]}
    for item in gallery:
        positive=normalize(item['features'])
        others=[o for o in gallery if o['id']!=item['id']]
        negative=np.concatenate([normalize(o['features']) for o in others]) if others else np.empty((0,positive.shape[1]))
        if len(negative):
            cross=np.max(np.concatenate([cross_scores(item,o) for o in others],axis=1),axis=1)
            confusion=float(np.max(cross))
            hard_negative=float(np.percentile(cross,95))
        else:
            confusion=0;hard_negative=0
        threshold=max(settings['similarity_threshold'],min(.97,hard_negative+.02))
        margin=max(settings['similarity_margin'],min(.14,.10*confusion))
        selected=select_views(positive,negative,min(5,len(positive)))
        advice=[]
        if len(positive)<settings['minimum_reference_views']:advice.append('Add distinct front, back and side views.')
        if confusion>=settings['confusion_threshold']:
            advice.append('Collect a tight view of a distinctive detail and test against the similar registered item; changing names alone cannot separate appearances.')
        if not advice:advice.append('Test these views on a separate recording; reference separation does not guarantee recognition.')
        result['objects'].append(dict(id=item['id'],name=item['name'],confusion=confusion,
                    threshold=threshold,margin=margin,selected_views=selected,advice=' '.join(advice)))
    for i,a in enumerate(gallery):
        for b in gallery[i+1:]:
            score=float(np.max(cross_scores(a,b)))
            result['pairs'].append(dict(first=a['name'],second=b['name'],score=score))
    result['pairs'].sort(key=lambda p:p['score'],reverse=True)
    return result


def describe_report(report):
    lines=['Reference analysis — visual/colour heuristic similarities, not measured recognition accuracy.','']
    for item in report['objects']:
        lines.extend([item['name'],f"Confusion score: {item['confusion']:.3f}; proposed update threshold: {item['threshold']:.3f}; margin: {item['margin']:.3f}",
                      'Useful existing view indices: '+', '.join(str(i+1) for i in item['selected_views']),item['advice'],''])
    lines.append('Most similar registered pairs:')
    lines.extend(f"{p['first']} ↔ {p['second']}: {p['score']:.3f}" for p in report['pairs'][:10])
    return '\n'.join(lines)


class ResearchVision(PersonalVision):
    def __init__(self,registry,settings):
        super().__init__(registry,settings)
        self.report=gallery_report(self.gallery,settings)
        rows={row['id']:row for row in self.report['objects']}
        for item in self.gallery:
            row=rows[item['id']]
            item.update(threshold=row['threshold'],margin=row['margin'],confused=row['confusion']>=settings['confusion_threshold'])
