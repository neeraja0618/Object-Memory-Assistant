"""Actual appearance-component measurements on downloaded public images. No invented results."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from personal import Embedder,rank_match
from research import select_views,gallery_report


def image_features(embedder,paths):
    return np.stack([embedder.reference(path) for path in paths])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path)
    parser.add_argument('--budget',type=int,default=5);parser.add_argument('--output',type=Path,default=Path('evaluation/results'))
    args=parser.parse_args()
    if args.budget<3:parser.error('Use at least three reference images for these settings.')
    root=Path(__file__).resolve().parent;config=json.loads((root/'config.json').read_text())
    manifest=json.loads(args.manifest.read_text());args.output.mkdir(parents=True,exist_ok=True)
    embedder=Embedder(config['embedding_model'],args.output/'features')
    all_gallery=[]
    for item in manifest['objects']:
        if not item['references']:raise ValueError('Each object needs reference examples.')
        all_gallery.append(dict(id=item['id'],name=item['name'],features=image_features(embedder,item['references'])))
    galleries={}
    for strategy in ['uniform','guided']:
        gallery=[]
        for item in all_gallery:
            count=min(args.budget,len(item['features']))
            negatives=[o['features'] for o in all_gallery if o['id']!=item['id']]
            negative=np.concatenate(negatives) if negatives else np.empty((0,item['features'].shape[1]))
            indices=np.linspace(0,len(item['features'])-1,count,dtype=int).tolist() if strategy=='uniform' else select_views(item['features'],negative,count)
            gallery.append(dict(id=item['id'],name=item['name'],features=item['features'][indices]))
        if strategy=='guided':
            report=gallery_report(gallery,config);rows={r['id']:r for r in report['objects']}
            for item in gallery:item.update(threshold=rows[item['id']]['threshold'],margin=rows[item['id']]['margin'])
            (args.output/'reference_analysis.json').write_text(json.dumps(report,indent=2))
        galleries[strategy]=gallery
    records=[];inference_times=[]
    for item in manifest['objects']:
        for query in item['queries']:
            started=time.perf_counter()
            with Image.open(query['path']) as image:feature=embedder.encode([image.convert('RGB')])[0]
            inference_times.append(time.perf_counter()-started)
            truth=item['id'] if query.get('object_present',True) else ''
            for mode in ['nearest','uniform_threshold','guided_global','guided_adaptive']:
                gallery=galleries['guided' if mode.startswith('guided') else 'uniform']
                if mode=='guided_global':gallery=[dict(o,threshold=config['similarity_threshold'],margin=config['similarity_margin']) for o in gallery]
                if mode=='nearest':
                    values=[(float(np.max(o['features']@feature)),o['id']) for o in gallery]
                    score,prediction=max(values);accepted=True
                else:
                    match=rank_match(feature,gallery,config);score=match['score'];accepted=match['strong'];prediction=match['object_id'] if accepted else ''
                records.append(dict(mode=mode,path=query['path'],truth=truth,prediction=prediction,score=score,accepted=accepted,
                                    correct=bool(prediction==truth and accepted),presence_annotation_available=query.get('presence_annotation_available',False)))
    summary={}
    for mode in ['nearest','uniform_threshold','guided_global','guided_adaptive']:
        rows=[r for r in records if r['mode']==mode];accepted=[r for r in rows if r['accepted']]
        correct=sum(r['correct'] for r in accepted);wrong=len(accepted)-correct
        summary[mode]=dict(query_frames=len(rows),accepted_frames=len(accepted),correct_acceptances=correct,wrong_acceptances=wrong,
                           coverage=len(accepted)/max(1,len(rows)),accepted_precision=correct/max(1,len(accepted)))
    result=dict(protocol=manifest['protocol'],source=manifest['source'],collector=manifest.get('collector'),reference_budget=args.budget,
                queries_without_presence_annotation=sum(not q.get('presence_annotation_available',False) for o in manifest['objects'] for q in o['queries']),
                model=config['embedding_model'],component='Full-frame appearance matching; no target detection, room location, review workload or history-repair metric is measured here',
                mean_feature_inference_seconds=sum(inference_times)/max(1,len(inference_times)),results=summary)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    with (args.output/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]) if records else ['mode']);writer.writeheader();writer.writerows(records)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
