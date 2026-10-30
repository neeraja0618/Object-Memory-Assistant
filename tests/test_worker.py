"""Source/inference integration tests using fakes; not a webcam or real detector test."""
import importlib.util
from pathlib import Path
import queue
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class Value:
    def __init__(self,value): self.value=value
    def item(self): return self.value
    def tolist(self): return self.value


class Boxes(list):
    def cpu(self): return self


class Capture:
    def __init__(self,count=7,opened=True):
        self.remaining=count
        self.opened=opened
        self.released=False
    def isOpened(self): return self.opened
    def read(self):
        self.remaining-=1
        return (True,SimpleNamespace(shape=(100,200,3))) if self.remaining>=0 else (False,None)
    def get(self,key): return 8
    def set(self,*args): pass
    def release(self): self.released=True


class Stop:
    def __init__(self,stopped=False): self.stopped=stopped
    def is_set(self): return self.stopped
    def wait(self,seconds): return self.stopped


class Messages(queue.Queue):
    # No UI runs during the fake worker test; do not exercise queue backpressure here.
    def qsize(self): return 0


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.capture=Capture()
        fakecv=SimpleNamespace(VideoCapture=lambda *args:self.capture,CAP_DSHOW=1,
                               CAP_PROP_FRAME_WIDTH=2,CAP_PROP_FRAME_HEIGHT=3,CAP_PROP_FPS=4)
        spec=importlib.util.spec_from_file_location('app_worker_test',ROOT/'base_app.py')
        self.module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'cv2':fakecv}):
            spec.loader.exec_module(self.module)
        self.app=self.module.App.__new__(self.module.App)
        self.app.queue=Messages()
        self.model=SimpleNamespace(names={0:'cell phone'})
        box=SimpleNamespace(cls=Value(0),conf=Value(.9),xyxy=[Value([20,10,80,40])])
        self.model.predict=lambda *args,**kwargs:[SimpleNamespace(names=self.model.names,boxes=Boxes([box]))]
        self.config=dict(model='yolo11s.pt',remember_classes=['cell phone'],detection_confidence=.3,image_size=640)

    def run_worker(self,stop=None):
        with patch.dict(sys.modules,{'ultralytics':SimpleNamespace(YOLO=lambda name:self.model)}):
            self.app.worker('sample.mp4',True,7,stop or Stop(),Stop(),self.config)
        messages=[]
        while not self.app.queue.empty():
            messages.append(self.app.queue.get_nowait())
        return messages

    def test_video_frames_produce_normalized_detections_and_release_source(self):
        messages=self.run_worker()
        frames=[payload for token,kind,payload in messages if kind=='frame']
        self.assertEqual(len(frames),7)
        self.assertEqual(frames[0][1][0]['box'],[.1,.1,.4,.4])
        self.assertEqual(frames[0][3],'Video 00:00:00')
        self.assertTrue(self.capture.released)
        self.assertEqual(messages[-1][1],'done')
    def test_bad_video_reports_error_and_releases_capture(self):
        self.capture.opened=False
        messages=self.run_worker()
        self.assertIn('error',[kind for _,kind,_ in messages])
        self.assertTrue(self.capture.released)
        self.assertEqual(messages[-1][1],'done')
    def test_inference_error_releases_capture(self):
        def fail(*args,**kwargs): raise RuntimeError('inference failure')
        self.model.predict=fail
        messages=self.run_worker()
        self.assertIn((7,'error','inference failure'),messages)
        self.assertTrue(self.capture.released)
    def test_stop_during_loading_prevents_source_open(self):
        messages=self.run_worker(Stop(True))
        self.assertEqual(messages,[(7,'done',None)])
        self.assertEqual(self.capture.remaining,7)
    def test_custom_model_missing_categories_reports_specific_error(self):
        self.config['remember_classes']=['scissors']
        messages=self.run_worker()
        errors=[payload for _,kind,payload in messages if kind=='error']
        self.assertIn('scissors',errors[0])
        self.assertEqual(self.capture.remaining,7)
    def test_personal_pipeline_fields_and_camera_warning_reach_queue(self):
        self.app.build_personal_pipeline=lambda config:'personal pipeline'
        def augment(pipeline,frame,detections,seconds):
            self.assertEqual(pipeline,'personal pipeline')
            detections[0]['personal_match']=dict(object_id='registered-id',name='My phone',score=.9,strong=True)
            return detections,seconds==0
        self.app.process_frame=augment
        messages=self.run_worker()
        self.assertIn('camera_moved',[kind for _,kind,_ in messages])
        frame=next(payload for _,kind,payload in messages if kind=='frame')
        self.assertEqual(frame[1][0]['personal_match']['object_id'],'registered-id')


if __name__=='__main__': unittest.main()
