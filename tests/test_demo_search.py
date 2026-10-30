import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from demo_search import possible_sightings


class TentativeSearchTests(unittest.TestCase):
    def row(self,**changes):
        row=dict(state='Tentative',track=1,candidate_id='grey',candidate_name='grey waterbottle',detector_name='bottle',seconds=2)
        row.update(changes)
        return row

    def test_search_returns_candidate_without_promoting_its_state(self):
        row=self.row()
        self.assertEqual(possible_sightings([row],' Grey waterbottle '),[row])
        self.assertEqual(row['state'],'Tentative')

    def test_rejected_identity_and_disputed_evidence_are_not_returned(self):
        self.assertEqual(possible_sightings([self.row()],'grey',{1:{'grey'}}),[])
        self.assertEqual(possible_sightings([self.row()],'grey',rejected={1}),[])
        self.assertEqual(possible_sightings([self.row(state='Disputed')],'grey'),[])
        self.assertEqual(possible_sightings([self.row(state='User-confirmed')],'grey'),[])

    def test_category_detection_does_not_invent_a_personal_identity(self):
        row=self.row(candidate_id='',candidate_name='')
        self.assertEqual(possible_sightings([row],'bottle'),[row])
        self.assertEqual(possible_sightings([row],'grey waterbottle'),[])
        self.assertEqual(possible_sightings([row],''),[])

    def test_latest_matching_observation_is_first(self):
        earlier=self.row(seconds=1)
        later=self.row(seconds=5)
        self.assertEqual(possible_sightings([earlier,later],'grey'),[later,earlier])
