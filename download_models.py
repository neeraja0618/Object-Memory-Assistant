"""Pre-download models before a demonstration; user reference images are never uploaded."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--personal',action='store_true',help='Download the DINOv2 appearance model')
    parser.add_argument('--reference-search',action='store_true',help='Also download OWLv2 for unfamiliar-object proposals')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    config=json.loads((root/'config.json').read_text())
    from ultralytics import YOLO
    path=Path(config['model'])
    YOLO(str(path if path.is_absolute() else root/path))
    print('YOLO detector ready.')
    if args.personal or args.reference_search:
        from transformers import AutoImageProcessor,AutoModel
        AutoImageProcessor.from_pretrained(config['embedding_model'])
        AutoModel.from_pretrained(config['embedding_model'])
        print('Personal appearance model ready.')
    if args.reference_search:
        from transformers import Owlv2Processor,Owlv2ForObjectDetection
        Owlv2Processor.from_pretrained(config['reference_model'])
        Owlv2ForObjectDetection.from_pretrained(config['reference_model'])
        print('Reference-guided detector ready.')
    print('Run the app on your laptop to check actual inference and camera performance.')


if __name__=='__main__':main()
