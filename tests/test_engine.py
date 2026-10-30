import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import Engine, zone_for


def d(name='cell phone', conf=.9, box=None):
    return dict(name=name,confidence=conf,box=box or [.2,.2,.4,.4])


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.config=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
        self.config.update(missing_after_seconds=4,evidence_gap_seconds=2.5)
        self.e=Engine(self.folder.name,'a',self.config)
    def tearDown(self):
        self.e.close()
        self.folder.cleanup()
    def feed(self,items,n=6,start=0):
        for i in range(n):
            self.e.update(items,start+i*.1,str(start+i*.1))
    def test_reported_weak_wrong_labels_never_enter_memory(self):
        for name,conf in [('laptop',.41),('book',.43),('remote',.51)]:
            self.feed([d(name,conf)],12)
        self.assertEqual(self.e.records,{})
        self.assertEqual(self.e.search(''),[])
    def test_confirm_requires_sufficient_observations(self):
        self.feed([d()],5)
        self.assertFalse(self.e.records)
        self.e.update([d()],.6,'sixth')
        self.assertEqual(self.e.records[1]['name'],'cell phone')
    def test_class_changes_do_not_create_multiple_tracks(self):
        for i in range(12):
            self.e.update([d('book' if i%2 else 'cell phone')],i*.1,str(i))
        self.assertEqual(len(self.e.tracks),1)
        self.assertFalse(self.e.records)
    def test_occasional_wrong_label_does_not_replace_confirmed_name(self):
        self.feed([d()])
        self.e.update([d('book',.9)],.7,'wrong')
        self.assertEqual(self.e.records[1]['name'],'cell phone')
        self.assertEqual(self.e.records[1]['last_seen'],'0.5')
        self.assertEqual(self.e.records[1]['status'],'Uncertain')
    def test_persistent_label_conflict_is_not_silently_relabelled(self):
        self.feed([d()])
        self.feed([d('book')],12,start=1)
        self.assertEqual(self.e.records[1]['name'],'cell phone')
        self.assertEqual(self.e.tracks[1]['state'],'Label conflict')
    def test_two_same_category_objects_get_separate_short_term_ids(self):
        self.feed([d(box=[.1,.1,.2,.2]),d(box=[.7,.7,.8,.8])])
        self.assertEqual(len(self.e.records),2)
    def test_absence_preserves_last_seen(self):
        self.feed([d()])
        self.e.update([],5,'absence')
        self.assertEqual(self.e.records[1]['last_seen'],'0.5')
        self.assertEqual(self.e.records[1]['status'],'Not detected')
        self.assertIn('cause unknown',self.e.timeline()[0][4])
    def test_long_absence_does_not_claim_persistent_identity(self):
        self.feed([d()])
        self.e.update([],5,'absent')
        self.feed([d()],6,start=6)
        self.assertEqual(len(self.e.records),2)
        self.assertEqual(self.e.records[1]['status'],'Not detected')
    def test_manual_correction_explicit_and_searchable(self):
        self.feed([d('book',.43)])
        self.e.correct(1,'cell phone','my black phone','correction')
        self.assertTrue(self.e.records[1]['user_labelled'])
        self.assertEqual(self.e.records[1]['last_seen'],'0.5')
        self.assertEqual(len(self.e.search('black phone')),1)
        self.assertEqual(self.e.search('book'),[])
        self.e.update([d('remote',.51)],.8,'later')
        self.assertEqual(self.e.records[1]['name'],'cell phone')
    def test_correction_removes_wrong_category_from_current_memory(self):
        self.feed([d('book')])
        self.e.correct(1,'cell phone','','correct')
        self.assertEqual(self.e.search('book'),[])
        self.assertEqual(len(self.e.search('cell phone')),1)
    def test_restart_retains_memory_and_marks_it_historical(self):
        self.feed([d()])
        self.e.close()
        self.e=Engine(self.folder.name,'b',self.config)
        r=self.e.search('cell phone')[0]
        self.assertEqual(r['status'],'Historical')
        self.assertFalse(r['current_session'])
    def test_roi_excludes_background_objects(self):
        self.e.settings['roi']=[0,0,.5,.5]
        self.feed([d(),d('bottle',box=[.7,.7,.9,.9])])
        self.assertEqual(len(self.e.records),1)
        self.assertEqual(self.e.search('bottle'),[])
    def test_named_zone_and_default_zone(self):
        self.e.settings['zones']=[dict(name='laptop area',box=[.1,.1,.5,.5])]
        self.feed([d()])
        self.assertEqual(self.e.records[1]['location'],'laptop area')
        self.assertEqual(zone_for([.8,.8,1,1],[]),'lower right of image')
    def test_sparse_observations_do_not_accumulate_confirmation(self):
        for i in range(8):
            self.e.update([d()],i*3,'sparse')
        self.assertFalse(self.e.records)
    def test_position_changes_recorded_without_jitter_events(self):
        self.feed([d()])
        self.e.update([d(box=[.21,.2,.41,.4])],.7,'jitter')
        self.assertEqual(len(self.e.timeline()),1)
        for i in range(5):
            x=.22+i*.05
            self.e.update([d(box=[x,.2,x+.2,.4])],.8+i*.1,'move')
        self.assertIn('position changed',self.e.timeline()[0][4])
    def test_latest_sighting_within_session_wins_search(self):
        self.feed([d(box=[.1,.1,.2,.2]),d(box=[.7,.7,.8,.8])])
        self.e.update([d(box=[.1,.1,.2,.2])],1,'latest')
        self.assertEqual(self.e.search('cell phone')[0]['id'],1)

    def test_rejection_hides_memory_and_blocks_same_track(self):
        self.feed([d('book')])
        self.e.reject(1,'reject')
        self.assertEqual(self.e.search('book'),[])
        self.feed([d('book')],start=1)
        self.assertEqual(self.e.search('book'),[])
        self.assertEqual(self.e.tracks[1]['state'],'Rejected by user')

    def test_correction_restores_rejected_track(self):
        self.feed([d('book')])
        self.e.reject(1,'reject')
        self.e.correct(1,'cell phone','','restore')
        self.assertEqual(len(self.e.search('cell phone')),1)
        self.assertFalse(self.e.tracks[1]['rejected'])

    def test_best_evidence_metadata_survives_new_observation(self):
        self.feed([d()])
        self.e.records[1].update(best_snapshot='best.jpg',best_seen='earlier',best_quality=123)
        self.e.update([d()],1,'later')
        r=self.e.records[1]
        self.assertEqual(r['best_snapshot'],'best.jpg')
        self.assertEqual(r['best_seen'],'earlier')
        self.assertEqual(r['last_seen'],'later')

    def test_brief_absence_recovery_logged(self):
        self.feed([d()])
        self.e.update([],1.5,'gap')
        self.assertEqual(self.e.records[1]['status'],'Briefly not detected')
        self.e.update([d()],1.6,'return')
        self.assertEqual(self.e.timeline()[0][4],'Seen again within short-term track')

    def test_invalid_confirmation_settings_rejected(self):
        self.config['minimum_observations']=20
        with self.assertRaises(ValueError):
            Engine(self.folder.name,'invalid',self.config)


if __name__=='__main__':
    unittest.main()
