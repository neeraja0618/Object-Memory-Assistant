"""Checks camera acquisition continues during slow or failed recognition, with no backlog."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import test_worker as worker_fakes
from live_capture import LatestFrame
import unittest


class FrameSlotTests(unittest.TestCase):
    def test_slow_consumer_gets_newest_sample_without_backlog(self):
        slot=LatestFrame()
        for i in range(100):slot.publish(i)
        sequence,sample=slot.after(0)
        self.assertEqual((sequence,sample),(100,99))
        self.assertEqual(slot.after(sequence),(100,None))


class LiveWorkerTests(unittest.TestCase):
    setUp = worker_fakes.WorkerTests.setUp
    def camera_run(self, model_factory):
        stop=threading.Event()
        state={'reads':0,'released':False}
        def read():
            time.sleep(.005)
            state['reads']+=1
            return True,SimpleNamespace(shape=(100,200,3),sequence=state['reads'])
        cap=SimpleNamespace(isOpened=lambda:True,read=read,set=lambda *a:None,
                            release=lambda:state.update(released=True))
        self.module.cv2.VideoCapture=lambda *a:cap
        with patch.dict('sys.modules',{'ultralytics':SimpleNamespace(YOLO=lambda *a:model_factory(stop,state))}):
            worker=threading.Thread(target=self.app.webcam_worker,args=(0,7,stop,self.config))
            worker.start();worker.join(2)
            if worker.is_alive():
                stop.set();worker.join(2)
                self.fail('Worker failed to finish')
        self.assertTrue(state['released'])
        self.assertEqual(list(self.app.queue.queue)[-1][1],'done')
        return state

    def test_camera_keeps_updating_while_detector_is_busy(self):
        def factory(stop,state):
            def predict(*args,**kwargs):
                before=state['reads']
                time.sleep(.12)
                self.assertGreater(state['reads'],before+2)
                stop.set()
                return [SimpleNamespace(names={0:'cell phone'},boxes=None)]
            return SimpleNamespace(names={0:'cell phone'},predict=predict)
        state=self.camera_run(factory)
        self.assertGreater(state['reads'],3)
        self.assertEqual(self.app.live_preview[0],7)

    def test_loading_failure_still_releases_camera_and_finishes(self):
        def factory(stop,state):
            time.sleep(.04)
            raise RuntimeError('model load failed')
        self.camera_run(factory)
        self.assertIn((7,'error','model load failed'),list(self.app.queue.queue))
