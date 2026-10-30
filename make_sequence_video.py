"""Reconstruct ONE real frame sequence; chosen playback rate is not original recording timing."""
import argparse
from pathlib import Path
import re
import cv2


def sequence_key(path):
    return re.sub(r'(--|_)?\d+$','',path.stem)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--prefix')
    parser.add_argument('--fps',type=float,default=10);parser.add_argument('--output',type=Path,default=Path('public_demo_sequence.mp4'))
    args=parser.parse_args()
    if args.fps<=0:parser.error('FPS must be positive.')
    files=sorted(p for p in args.folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'))
    groups={}
    for p in files:groups.setdefault(sequence_key(p),[]).append(p)
    if not args.prefix:
        print('Choose ONE sequence with --prefix from:');print('\n'.join(f'{key}: {len(paths)} frames' for key,paths in groups.items()));return
    selected=groups.get(args.prefix)
    if not selected:raise ValueError('Prefix must exactly match one listed recording sequence.')
    first=cv2.imread(str(selected[0]))
    if first is None:raise ValueError('Cannot read first frame.')
    h,w=first.shape[:2];args.output.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(args.output),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(w,h))
    if not writer.isOpened():raise RuntimeError('Video encoder unavailable.')
    try:
        for p in selected:
            frame=cv2.imread(str(p))
            if frame is None:raise ValueError('Cannot read frame: '+str(p))
            writer.write(cv2.resize(frame,(w,h)))
    finally:writer.release()
    print('Reconstructed sequence saved:',args.output)
    print('Original frame order retained; chosen playback FPS does not establish original real-world timing. Do not concatenate different recordings into a real memory test.')


if __name__=='__main__':main()
