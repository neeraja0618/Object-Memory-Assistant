"""Score exported memory observations using independent human ground-truth labels."""
import argparse
import csv
import json
from pathlib import Path

ACCEPTED={'User-confirmed','Reference-confirmed'}


def score(rows,labels):
    considered=[r for r in rows if r['observation_id'] in labels]
    accepted=[r for r in considered if r['state'] in ACCEPTED and r.get('object_id')]
    wrong=sum(r['object_id']!=labels[r['observation_id']]['object_id'] for r in accepted)
    automatic=[r for r in accepted if r['state']=='Reference-confirmed']
    manual=[r for r in accepted if r['state']=='User-confirmed']
    truth_latest={};memory_latest={}
    for r in sorted(considered,key=lambda r:(r['seconds'],r['observation_id'])):
        label=labels[r['observation_id']]
        if label['object_id']:truth_latest[label['object_id']]=(r,label)
        if r in accepted:memory_latest[r['object_id']]=r
    time_errors=[];zone_pairs=[]
    for oid,(truth,label) in truth_latest.items():
        prediction=memory_latest.get(oid)
        if prediction:
            time_errors.append(abs(prediction['seconds']-truth['seconds']))
            if label.get('zone'):
                # The export's image boxes are not independently labelled physical locations.
                from engine import zone_for
                zone_pairs.append(zone_for(prediction['box'],[])==label['zone'])
    return dict(labelled_observations=len(considered),accepted_observations=len(accepted),wrong_accepted_identities=wrong,
                update_coverage=len(accepted)/max(1,len(considered)),accepted_identity_precision=(len(accepted)-wrong)/max(1,len(accepted)),
                automatic_acceptances=len(automatic),manual_acceptances=len(manual),
                automatic_wrong=sum(r['object_id']!=labels[r['observation_id']]['object_id'] for r in automatic),
                manual_wrong=sum(r['object_id']!=labels[r['observation_id']]['object_id'] for r in manual),
                final_identity_memory_coverage=sum(oid in memory_latest for oid in truth_latest)/max(1,len(truth_latest)),
                last_seen_time_mean_absolute_error_seconds=sum(time_errors)/len(time_errors) if time_errors else None,
                nine_zone_accuracy=sum(zone_pairs)/len(zone_pairs) if zone_pairs else None,
                note='Observation-level diagnostics on labelled rows; not a complete independent room-search benchmark.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('export',type=Path);parser.add_argument('--labels',type=Path)
    parser.add_argument('--make-label-template',type=Path);parser.add_argument('--output',type=Path,default=Path('evaluation/memory_summary.json'))
    args=parser.parse_args();data=json.loads(args.export.read_text());rows=data['observations']
    if args.make_label_template:
        with args.make_label_template.open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['observation_id','labelled','object_id','zone','stamp','candidate_for_reference_only'])
            writer.writerows([r['observation_id'],0,'','',r['stamp'],r['candidate_name']] for r in rows)
        print('Inspect evidence independently; set labelled=1 for evaluated rows. For such rows, blank object_id means no registered target. Unlabelled rows are skipped.')
        return
    if not args.labels:parser.error('Provide independent --labels or use --make-label-template first.')
    with args.labels.open(newline='',encoding='utf-8-sig') as f:
        labels={int(r['observation_id']):r for r in csv.DictReader(f) if r.get('labelled','1')=='1'}
    before={r['observation_id']:dict(r) for r in rows}
    # Audit is newest-first; walking reversed restores the earliest pre-repair state.
    restored=set()
    for repair in reversed(data['repairs']):
        oid=repair['observation']
        if oid not in restored and oid in before:before[oid]=dict(repair['before'],observation_id=oid);restored.add(oid)
    result=dict(before_repair=score(list(before.values()),labels),after_repair=score(rows,labels),
                review_requests_surfaced=data.get('review_requests'),repair_count=len(data['repairs']))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
