"""Check imports and settings without opening the camera or downloading a model."""
import importlib
import json
from pathlib import Path
import sys


def main():
    print('Python:',sys.version.split()[0])
    errors=[]
    for name in ['tkinter','cv2','PIL','numpy','scipy','torch','torchvision','ultralytics','transformers']:
        try:
            module=importlib.import_module(name)
            print(name+': OK '+str(getattr(module,'__version__','')))
        except Exception as exc:
            errors.append(name+': '+str(exc))
    try:
        import torch
        from torchvision.ops import nms
        nms(torch.tensor([[0.,0.,10.,10.]]),torch.tensor([.9]),.5)
        print('Torchvision NMS operation: OK')
    except Exception as exc:
        errors.append('Torch/Torchvision mismatch: '+str(exc)+'; run install_runtime.py using the project Python.')
    try:
        config=json.loads((Path(__file__).resolve().parent/'config.json').read_text())
        assert 0<config['detection_confidence']<=config['memory_confidence']<=1
        assert 1<=config['minimum_observations']<=config['vote_window']
        assert config['image_size']>=160 and config['image_size']%32==0
        assert config['missing_after_seconds']>0 and config['evidence_gap_seconds']>0
        assert 0<config['possible_threshold']<=config['similarity_threshold']<=1
        assert 0<=config['similarity_margin']<=1
        assert config['minimum_reference_views']>=1 and config['personal_confirmations']>=1
        assert config['video_sample_fps']>0 and config['max_personal_candidates']>=1
        assert config['review_policy'] in ('selective','always')
        assert config['auto_identity_observations']>=1 and config['history_window_seconds']>0
        assert config['history_image_retention_seconds']>=config['history_window_seconds']
        assert config['max_history_images']>=1 and config['history_sample_seconds']>0
        assert config['review_cooldown_seconds']>=0 and config['max_review_requests']>=0
        print('Settings: OK')
    except Exception as exc:
        errors.append('Invalid config.json: '+str(exc))
    for error in errors:
        print('ERROR:',error)
    if errors:
        return 1
    print('Imports and settings passed. Webcam, GUI display, model download and detection still need a live test.')
    return 0


if __name__=='__main__':
    sys.exit(main())
