import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from appearance import color_signature
from personal import rank_match,Registry
from personal_engine import PersonalEngine
from research import gallery_report


class AppearanceTests(unittest.TestCase):
    def test_gray_and_violet_are_separable_despite_same_embedding(self):
        gray=color_signature(Image.new('RGB',(100,200),(160,160,160)))
        violet=color_signature(Image.new('RGB',(100,200),(150,90,200)))
        gallery=[dict(id='gray',name='Gray bottle',features=np.tile([1.,0.],(3,1)),colors=np.tile(gray,(3,1))),
                 dict(id='violet',name='Violet bottle',features=np.tile([1.,0.],(3,1)),colors=np.tile(violet,(3,1)))]
        settings=dict(possible_threshold=.65,similarity_threshold=.8,similarity_margin=.06,minimum_reference_views=3,confusion_threshold=.85)
        result=rank_match([1,0],gallery,settings,violet)
        self.assertEqual(result['object_id'],'violet')
        self.assertTrue(result['strong'])
        report=gallery_report(gallery,settings)
        self.assertLess(report['pairs'][0]['score'],.85)

    def test_colour_alone_cannot_confirm_weak_embedding(self):
        color=color_signature(Image.new('RGB',(100,200),'purple'))
        gallery=[dict(id='a',name='Bottle',features=np.tile([1.,0.],(3,1)),colors=np.tile(color,(3,1)))]
        settings=dict(possible_threshold=.65,similarity_threshold=.8,similarity_margin=.06,minimum_reference_views=3)
        result=rank_match([.76,np.sqrt(1-.76**2)],gallery,settings,color)
        self.assertFalse(result['strong'])

    def test_background_border_has_little_effect(self):
        image=Image.new('RGB',(100,200),'green')
        image.paste(Image.new('RGB',(60,180),'purple'),(20,10))
        pure=color_signature(Image.new('RGB',(100,200),'purple'))
        self.assertGreater(float(pure@color_signature(image)),.99)

    def test_color_evidence_cannot_override_insufficient_views(self):
        color=color_signature(Image.new('RGB',(100,200),'purple'))
        result=rank_match([1,0],[dict(id='a',name='Bottle',features=np.array([[1.,0.]]),colors=np.array([color]))],
             dict(possible_threshold=.65,similarity_threshold=.8,similarity_margin=.06,minimum_reference_views=3),color)
        self.assertFalse(result['strong'])

    def test_two_identical_items_remain_uncertain(self):
        color=color_signature(Image.new('RGB',(100,200),'purple'))
        gallery=[dict(id=i,name=i,features=np.tile([1.,0.],(3,1)),colors=np.tile(color,(3,1))) for i in ['a','b']]
        result=rank_match([1,0],gallery,dict(possible_threshold=.65,similarity_threshold=.8,similarity_margin=.06,minimum_reference_views=3),color)
        self.assertFalse(result['strong'])


class AssociationTests(unittest.TestCase):
    def test_crossing_positions_prefers_consistent_visual_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            registry=Registry(folder)
            a=registry.create('A');b=registry.create('B')
            config=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
            engine=PersonalEngine(folder,'test',config,registry)
            def detection(item,box):return dict(name='bottle',confidence=.9,box=box,
                personal_match=dict(object_id=item['id'],name=item['name'],score=.95,strong=True,margin=.2))
            left=[.1,.1,.25,.7];right=[.3,.1,.45,.7]
            try:
                engine.update([detection(a,left),detection(b,right)],0,'first')
                engine.update([detection(b,left),detection(a,right)],1,'crossed')
                self.assertEqual(len(engine.tracks),2)
                self.assertEqual(engine.tracks[1]['suggestion']['object_id'],a['id'])
                self.assertEqual(engine.tracks[1]['box'],right)
                self.assertEqual(engine.tracks[2]['suggestion']['object_id'],b['id'])
            finally:engine.close()
