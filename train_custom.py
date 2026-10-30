"""Fine-tune on a user-supplied, labelled YOLO dataset; no dataset or trained model is bundled."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--model',default='yolo11s.pt')
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if not args.data.is_file():
        parser.error('Provide your dataset YAML file, with real labelled images.')
    if args.epochs<1:
        parser.error('Epochs must be positive.')
    model=YOLO(args.model)
    results=model.train(data=str(args.data.resolve()),epochs=args.epochs,imgsz=args.imgsz,
                        device=args.device,project=str(Path(__file__).resolve().parent/'training_runs'),name='desk_objects')
    print('Training outputs:',results.save_dir)
    print('Choose weights/best.pt in the app, and set remember_classes in config.json to the model category names.')


if __name__=='__main__':
    main()
