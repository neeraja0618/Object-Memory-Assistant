import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from personal import Registry,rank_match
from personal_engine import PersonalEngine
from scene_math import displacement_warning


def detection(match=None):
    return dict(name='cell phone',confidence=.9,box=[.2,.2,.4,.4],personal_match=match or {})


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.r=Registry(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_references_and_names_survive_restart(self):
        item=self.r.create('My calculator')
        self.r.add_reference(item['id'],Image.new('RGB',(80,60),'blue'))
        r=Registry(self.temp.name)
        self.assertEqual(r.active()[0]['name'],'My calculator')
        self.assertTrue((r.directory/r.active()[0]['references'][0]).exists())
    def test_duplicate_name_and_identical_photo_rejected(self):
        item=self.r.create('Bottle')
        with self.assertRaises(ValueError):self.r.create('bottle')
        image=Image.new('RGB',(80,80),'red')
        self.assertTrue(self.r.add_reference(item['id'],image))
        self.assertFalse(self.r.add_reference(item['id'],image))
    def test_archive_and_reference_exclusion(self):
        item=self.r.create('Book')
        self.r.add_reference(item['id'],Image.new('RGB',(80,80),'red'))
        path=item['references'][0]
        self.r.remove_reference(item['id'],path)
        self.assertFalse(item['references'])
        self.assertTrue((self.r.directory/path).exists())
        self.r.archive(item['id']);self.assertEqual(self.r.active(),[])
    def test_tiny_crops_rejected(self):
        item=self.r.create('Keys')
        with self.assertRaises(ValueError):self.r.add_reference(item['id'],Image.new('RGB',(10,10)))


class MatchTests(unittest.TestCase):
    def setUp(self):
        self.settings=dict(possible_threshold=.65,similarity_threshold=.8,similarity_margin=.06,minimum_reference_views=3)
    def gallery(self,vector,name='Calculator',oid='a',count=3):
        return dict(id=oid,name=name,features=np.tile(vector,(count,1)))
    def test_registered_match_identifies_gallery_name(self):
        result=rank_match([1,0],[self.gallery([1,0])],self.settings)
        self.assertEqual(result['name'],'Calculator');self.assertTrue(result['strong'])
    def test_unknown_is_not_forced_to_nearest_name(self):
        result=rank_match([0,1],[self.gallery([1,0])],self.settings)
        self.assertEqual(result['object_id'],'');self.assertFalse(result['strong'])
    def test_similar_registrations_require_review(self):
        result=rank_match([1,0],[self.gallery([1,0]),self.gallery([.99,.01],oid='b')],self.settings)
        self.assertFalse(result['strong']);self.assertIn('Similar',result['reason'])
    def test_insufficient_views_prevent_strong_match(self):
        result=rank_match([1,0],[self.gallery([1,0],count=1)],self.settings)
        self.assertFalse(result['strong']);self.assertIn('more',result['reason'])
    def test_empty_gallery_returns_unknown(self):
        self.assertEqual(rank_match([1,0],[],self.settings)['object_id'],'')


class PersonalMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.registry=Registry(self.tmp.name)
        self.item=self.registry.create('My phone')
        self.config=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
        self.e=PersonalEngine(self.tmp.name,'a',self.config,self.registry)
        self.match=dict(object_id=self.item['id'],name='My phone',score=.9,strong=True,margin=.2)
    def tearDown(self):self.e.close();self.tmp.cleanup()
    def feed(self,match,start=0,n=6):
        for i in range(n):self.e.update([detection(dict(match))],start+i*.1,str(start+i*.1))
    def approve(self):
        self.feed(self.match)
        return self.e.approve(1,self.item['id'],'approve')
    def test_personal_matches_do_not_silently_assign_identity(self):
        self.feed(self.match)
        self.assertEqual(self.e.search('My phone'),[])
        self.assertEqual(self.e.tracks[1]['suggestion']['name'],'My phone')
    def test_approval_persists_registered_identity(self):
        self.approve();self.e.close()
        self.e=PersonalEngine(self.tmp.name,'b',self.config,self.registry)
        rows=self.e.search('My phone')
        self.assertEqual(rows[0]['personal_id'],self.item['id']);self.assertFalse(rows[0]['current_session'])
    def test_lost_identity_keeps_last_identified_but_updates_detection_time(self):
        record=self.approve();old=record['last_seen']
        self.e.update([detection()],1,'uncertain')
        record=self.e.records[1]
        self.assertEqual(record['last_seen'],old)
        self.assertEqual(record['last_detected'],'uncertain')
        self.assertEqual(record['status'],'Identity uncertain')
    def test_two_strong_observations_required_after_approval(self):
        self.approve();self.feed(self.match,start=1,n=1)
        self.assertEqual(self.e.records[1]['last_seen'],'0.5')
        self.feed(self.match,start=1.1,n=1)
        self.assertEqual(self.e.records[1]['last_seen'],'1.1')
    def test_deny_blocks_identity_updates(self):
        self.approve();self.e.deny(1,self.item['id'],'deny')
        self.feed(self.match,start=1)
        self.assertEqual(self.e.records[1]['last_seen'],'0.5')
    def test_camera_warning_freezes_memory_and_is_logged_once(self):
        self.approve();self.e.warn_camera('movement');self.e.warn_camera('again')
        before=self.e.records[1]['last_seen']
        self.feed(self.match,start=2)
        self.assertEqual(self.e.records[1]['last_seen'],before)
        self.assertEqual(sum(row[4].startswith('Camera-view') for row in self.e.timeline()),1)
        with self.assertRaises(ValueError):self.e.approve(1,self.item['id'],'blocked')
    def test_rename_keeps_searchable_registration_identity(self):
        self.approve();self.registry.rename(self.item['id'],'Black phone')
        self.assertEqual(self.e.search('My phone'),[])
        self.assertEqual(self.e.search('Black phone')[0]['personal_id'],self.item['id'])
    def test_archived_object_excluded_from_personal_search(self):
        self.approve();self.registry.archive(self.item['id'])
        self.assertEqual(self.e.search('My phone'),[])
    def test_same_identity_in_two_regions_requires_review(self):
        self.approve()
        other=detection(dict(self.match));other['box']=[.7,.7,.9,.9]
        self.e.update([detection(dict(self.match)),other],1,'two')
        self.assertEqual(self.e.records[1]['last_seen'],'0.5')
        self.assertFalse(self.e.tracks[1]['suggestion']['strong'])
    def test_reference_proposal_is_not_saved_as_generic_unknown(self):
        for i in range(8):
            observation=detection(dict(self.match));observation['name']='unclassified object'
            self.e.update([observation],i*.1,str(i))
        self.assertEqual(self.e.records,{})
        self.e.approve(1,self.item['id'],'approve')
        self.assertEqual(self.e.search('My phone')[0]['personal_id'],self.item['id'])


class SceneMathTests(unittest.TestCase):
    def test_identity_and_small_jitter_do_not_trigger(self):
        self.assertFalse(displacement_warning(np.eye(3),640,360))
        matrix=np.eye(3);matrix[0,2]=2
        self.assertFalse(displacement_warning(matrix,640,360))
    def test_large_shift_and_scale_trigger(self):
        matrix=np.eye(3);matrix[0,2]=100
        self.assertTrue(displacement_warning(matrix,640,360))
        self.assertTrue(displacement_warning(np.diag([1.25,1.25,1]),640,360))


if __name__=='__main__':unittest.main()
