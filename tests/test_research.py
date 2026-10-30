import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from personal import Registry
from research import gallery_report,select_views
from research_engine import ResearchEngine
from public_data import build_manifest
from evaluate_memory import score


class GuidanceTests(unittest.TestCase):
    def setUp(self):self.settings=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
    def test_confusing_references_raise_threshold_and_advice(self):
        gallery=[dict(id='a',name='A',features=np.array([[1,0],[.99,.01],[.98,.02]])),
                 dict(id='b',name='B',features=np.array([[.99,.01],[.98,.02],[.97,.03]]))]
        report=gallery_report(gallery,self.settings)
        self.assertGreater(report['objects'][0]['threshold'],self.settings['similarity_threshold'])
        self.assertIn('distinctive',report['objects'][0]['advice'])
    def test_guided_selection_covers_distinct_views_without_duplicates(self):
        features=np.array([[1,0,0],[.99,.01,0],[0,1,0],[0,0,1]])
        selected=select_views(features,np.array([[1,0,0]]),3)
        self.assertEqual(len(set(selected)),3)
        self.assertTrue(2 in selected or 3 in selected)


class ResearchMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.registry=Registry(self.root/'data');self.a=self.registry.create('Phone A');self.b=self.registry.create('Phone B')
        self.config=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
        self.e=ResearchEngine(self.root/'data','session',self.config,self.registry)
    def tearDown(self):self.e.close();self.tmp.cleanup()
    def match(self,item=None,confused=False,strong=True):
        item=item or self.a
        return dict(object_id=item['id'],name=item['name'],score=.94,margin=.2,strong=strong,confused=confused,threshold=.8)
    def feed(self,match=None,n=6,start=0):
        for i in range(n):
            self.e.update([dict(name='cell phone',confidence=.9,box=[.1,.1,.4,.4],personal_match=dict(match or self.match()))],start+i*.5,str(start+i*.5))
    def attach(self):
        folder=self.root/'data'/'history';folder.mkdir(exist_ok=True)
        path=folder/'frame.jpg';Image.new('RGB',(100,100),'blue').save(path)
        self.e.attach_history_image(list(self.e.pending_history),str(path.relative_to(self.root)))
    def test_easy_stable_item_autoassigned_without_review_request(self):
        self.feed()
        self.assertEqual(self.e.records[1]['personal_id'],self.a['id'])
        self.assertFalse(self.e.records[1]['user_labelled'])
        self.assertEqual(self.e.review_requests,0)
    def test_ambiguous_pair_stays_tentative_and_requests_review(self):
        self.feed(self.match(confused=True))
        self.assertEqual(self.e.search('Phone A'),[])
        self.assertEqual(self.e.review_requests,1)
        self.assertTrue(all(r['state']=='Tentative' for r in self.e.history(False)))
    def test_always_review_baseline_does_not_assign_automatically(self):
        self.e.settings['review_policy']='always';self.feed()
        self.assertEqual(self.e.search('Phone A'),[])
        self.assertEqual(self.e.review_requests,1)
    def test_request_budget_never_turns_unknown_into_accepted(self):
        self.e.settings['max_review_requests']=0;self.feed(self.match(confused=True))
        self.assertEqual(self.e.review_requests,0)
        self.assertEqual(self.e.search('Phone A'),[])
    def test_conflicting_new_identity_does_not_overwrite_memory(self):
        self.feed();old=self.e.records[1]['last_seen']
        self.feed(self.match(self.b),n=4,start=3)
        self.assertEqual(self.e.records[1]['personal_id'],self.a['id'])
        self.assertEqual(self.e.records[1]['last_seen'],old)
        self.assertEqual(self.e.records[1]['last_detected'],'4.5')
        self.assertTrue(self.e.tracks[1]['review_needed'])
    def test_explicit_scoped_repair_rebuilds_memory_and_preserves_originals(self):
        self.feed(self.match(self.b));self.attach()
        rows=[r for r in self.e.history(False) if r['state']=='Reference-confirmed']
        plan=self.e.preview_repair([r['observation_id'] for r in rows],self.a['id'],'manual')
        self.e.apply_repair(plan)
        self.assertEqual(self.e.search('Phone B'),[])
        self.assertEqual(self.e.search('Phone A')[0]['personal_id'],self.a['id'])
        self.assertEqual(len(self.e.repair_audit()),len(rows))
        original=json.loads(self.e.db.execute('SELECT original FROM observations WHERE id=?',(rows[0]['observation_id'],)).fetchone()[0])
        self.assertEqual(original['object_id'],self.b['id'])
    def test_unsupported_history_is_disputed_not_blindly_renamed(self):
        self.feed(self.match(self.b));self.attach()
        rows=[r for r in self.e.history(False) if r['state']=='Reference-confirmed']
        plan=self.e.preview_repair([r['observation_id'] for r in rows],self.a['id'],'evidence')
        self.assertTrue(all(p['after']['state']=='Disputed' for p in plan))
        self.e.apply_repair(plan)
        self.assertEqual(self.e.search('Phone B'),[]);self.assertEqual(self.e.search('Phone A'),[])
    def test_repair_scope_and_missing_evidence_are_enforced(self):
        self.feed();ids=[r['observation_id'] for r in self.e.history(False)]
        with self.assertRaises(ValueError):self.e.preview_repair(ids,self.a['id'])
        self.attach();self.e.max_seconds=200
        with self.assertRaises(ValueError):self.e.preview_repair(ids,self.a['id'])
    def test_repair_requires_fresh_preview(self):
        self.feed();self.attach();oid=self.e.history(False)[0]['observation_id']
        first=self.e.preview_repair([oid],self.b['id'],'manual')
        second=self.e.preview_repair([oid],self.a['id'],'manual')
        self.e.apply_repair(first)
        with self.assertRaises(ValueError):self.e.apply_repair(second)
    def test_repaired_evidence_is_copied_outside_retention_window(self):
        self.feed();self.attach();oid=self.e.history(False)[0]['observation_id']
        plan=self.e.preview_repair([oid],self.b['id'],'manual');self.e.apply_repair(plan)
        snapshot=self.root/self.e.records[1]['snapshot']
        self.assertTrue(snapshot.exists());self.assertIn('snapshots',snapshot.parts)
    def test_repair_cannot_apply_after_time_window_advances(self):
        self.feed();self.attach();oid=self.e.history(False)[0]['observation_id']
        plan=self.e.preview_repair([oid],self.b['id'],'manual')
        self.e.max_seconds=200
        with self.assertRaises(ValueError):self.e.apply_repair(plan)


class DatasetAndMetricTests(unittest.TestCase):
    def test_manifest_uses_separate_modes_and_presence_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);annotations={}
            for name in ['cup','bottle']:
                for mode in ['clean','clutter']:
                    folder=root/'Dataset'/'P1'/name/mode;folder.mkdir(parents=True)
                    image=folder/f'P1--{name}--{mode}--000001.jpeg';Image.new('RGB',(80,80),'red').save(image)
                    annotations[image.name]=dict(object_not_present_issue=mode=='clutter',pii_present_issue=False)
            folder=root/'Annotations';folder.mkdir();(folder/'labels.json').write_text(json.dumps(annotations))
            manifest=build_manifest(root,root/'manifest.json')
            self.assertEqual(len(manifest['objects']),2)
            self.assertFalse(manifest['objects'][0]['queries'][0]['object_present'])
            self.assertNotEqual(manifest['objects'][0]['references'][0],manifest['objects'][0]['queries'][0]['path'])
    def test_metric_reports_error_and_coverage_not_only_precision(self):
        rows=[dict(observation_id=1,seconds=0,object_id='wrong',state='Reference-confirmed',box=[0,0,.2,.2]),
              dict(observation_id=2,seconds=1,object_id='',state='Tentative',box=[0,0,.2,.2])]
        labels={1:dict(object_id='actual'),2:dict(object_id='actual')}
        result=score(rows,labels)
        self.assertEqual(result['wrong_accepted_identities'],1)
        self.assertEqual(result['update_coverage'],.5)
        self.assertEqual(result['automatic_wrong'],1)


if __name__=='__main__':unittest.main()
